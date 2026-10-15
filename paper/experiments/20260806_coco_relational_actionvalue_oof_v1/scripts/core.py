"""Pure contracts for the COCO Relational Action-Value diagnostic."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


METHODS = (
    "CurrentActionValue",
    "GlobalRank",
    "RelSetValue",
    "RelActionRank",
)
GLOBAL_METHODS = frozenset(("GlobalRank", "RelActionRank"))
RELATIONAL_METHODS = frozenset(("RelSetValue", "RelActionRank"))
EXPECTED_ROLES = {"fit": 9678, "calibration": 571}
EXPECTED_ACTIONS = ("edge", "original", "q90")
EXPECTED_CANDIDATES = ("original", "q90")
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
EXPECTED_MODES = ("fixed_call", "fixed_byte")
ACTION_COST_INDICES = (6, 7, 8, 9, 10, 11, 12)
SCENE_PAYLOAD_INDEX = 1
EDGE_ACTION_TYPE = 0


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("This diagnostic is COCO-only")
    if dict(protocol.get("roles", {})) != EXPECTED_ROLES:
        raise ValueError("Canonical fit/calibration populations changed")
    if tuple(map(str, protocol.get("actions", ()))) != EXPECTED_ACTIONS:
        raise ValueError("The No-ROI action space changed")
    if tuple(map(str, protocol.get("candidate_actions", ()))) != EXPECTED_CANDIDATES:
        raise ValueError("ROI or another candidate action is forbidden")
    if tuple(map(str, protocol.get("methods", ()))) != METHODS:
        raise ValueError("The 2x2 method matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("The budget grid changed")
    if tuple(map(str, protocol.get("modes", ()))) != EXPECTED_MODES:
        raise ValueError("The evaluation modes changed")
    if int(protocol.get("fold_count", 0)) != 5:
        raise ValueError("Five OOF folds are required")
    if tuple(map(int, protocol.get("seeds", ()))) != (20260806, 20260807, 20260808):
        raise ValueError("The frozen three-seed set changed")
    if protocol.get("primary_mode") != "fixed_byte":
        raise ValueError("Fixed-byte must remain primary")
    if protocol.get("reward", {}).get("fold_contained") is not True:
        raise ValueError("Fold-contained reward contexts are required")
    if protocol.get("reward_context_may_cross_oof_folds") is not False:
        raise ValueError("Cross-fold reward contexts are forbidden")
    if protocol.get("fit_only_parameter_selection") is not True:
        raise ValueError("Parameter selection must be fit-OOF-only")
    if protocol.get("calibration_used_for_training_or_selection") is not False:
        raise ValueError("Calibration cannot train or select parameters")
    for key in ("audit_accessed", "final_accessed", "registered_final_accessed"):
        if protocol.get(key) is not False:
            raise ValueError(f"Forbidden role flag changed: {key}")


def deterministic_folds(
    image_ids: Iterable[int], *, fold_count: int, seed: int
) -> list[list[int]]:
    values = sorted(map(int, image_ids))
    if len(values) != len(set(values)):
        raise ValueError("Image IDs must be unique")
    if not 2 <= int(fold_count) <= len(values):
        raise ValueError("Invalid fold count")
    ordered = sorted(
        values,
        key=lambda image_id: (
            hashlib.sha256(f"{int(seed)}:{image_id}".encode("ascii")).digest(),
            image_id,
        ),
    )
    folds = [[] for _ in range(int(fold_count))]
    for index, image_id in enumerate(ordered):
        folds[index % int(fold_count)].append(image_id)
    return [sorted(fold) for fold in folds]


def deterministic_validation_split(
    image_ids: Iterable[int], *, seed: int, fraction: float
) -> tuple[list[int], list[int]]:
    values = sorted(map(int, image_ids))
    if len(values) != len(set(values)) or len(values) < 2:
        raise ValueError("Validation split requires unique image IDs")
    count = max(1, min(len(values) - 1, int(round(len(values) * float(fraction)))))
    ordered = sorted(
        values,
        key=lambda image_id: (
            hashlib.sha256(f"validation:{int(seed)}:{image_id}".encode("ascii")).digest(),
            image_id,
        ),
    )
    validation = sorted(ordered[:count])
    validation_set = set(validation)
    return sorted(value for value in values if value not in validation_set), validation


def fold_membership(folds: Sequence[Sequence[int]]) -> dict[int, int]:
    membership: dict[int, int] = {}
    for fold_index, fold in enumerate(folds):
        for image_id in map(int, fold):
            if image_id in membership:
                raise ValueError("An image occurs in multiple folds")
            membership[image_id] = int(fold_index)
    return membership


def fold_contained_contexts(
    image_ids: Iterable[int],
    *,
    folds: Sequence[Sequence[int]],
    context_size: int,
    seed: int,
) -> dict[int, list[int]]:
    ids = list(map(int, image_ids))
    membership = fold_membership(folds)
    if set(ids) != set(membership):
        raise ValueError("Fold population differs from image IDs")
    output: dict[int, list[int]] = {}
    for target in ids:
        candidates = [
            int(value)
            for value in folds[membership[target]]
            if int(value) != target
        ]
        count = min(max(0, int(context_size)), len(candidates))
        generator = random.Random(int(seed) + target * 1_000_003)
        output[target] = sorted(generator.sample(candidates, count))
    return output


def assert_oof_reward_isolation(
    contexts: Mapping[int, Sequence[int]], folds: Sequence[Sequence[int]]
) -> None:
    membership = fold_membership(folds)
    if set(map(int, contexts)) != set(membership):
        raise RuntimeError("Reward target population differs from folds")
    for target_raw, context_raw in contexts.items():
        target = int(target_raw)
        context = list(map(int, context_raw))
        if target in context or len(context) != len(set(context)):
            raise RuntimeError("Reward context contains target or duplicates")
        if any(value not in membership for value in context):
            raise RuntimeError("Reward context contains an unknown image")
        if any(membership[value] != membership[target] for value in context):
            raise RuntimeError("Reward context crosses OOF folds")
    for heldout in folds:
        heldout_ids = set(map(int, heldout))
        for target, context in contexts.items():
            if int(target) not in heldout_ids and heldout_ids & set(map(int, context)):
                raise RuntimeError("Training reward accesses held-out fold")


def strip_cost_inputs(
    data: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output = dict(data)
    scene = data["scene"].clone()
    actions = data["actions"].clone()
    action_type = data["action_type"].clone()
    allowed = (action_type == 0) | (action_type == 1) | (action_type == 3)
    if not bool(allowed.all()):
        raise ValueError("Cost stripping received ROI or an unknown action type")
    action_type[action_type == 3] = 2
    scene[:, SCENE_PAYLOAD_INDEX] = 0.0
    actions[:, :, list(ACTION_COST_INDICES)] = 0.0
    output["scene"] = scene
    output["actions"] = actions
    output["action_type"] = action_type
    return output


def assert_no_roi_records(records: Sequence[Mapping[str, Any]]) -> None:
    allowed = set(EXPECTED_ACTIONS)
    for record in records:
        actions = {str(action["action_id"]) for action in record["actions"]}
        if not actions <= allowed or "edge" not in actions or "original" not in actions:
            raise RuntimeError(f"Invalid No-ROI actions for image {record['image_id']}")


def filter_no_roi_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    allowed = set(EXPECTED_ACTIONS)
    for source in records:
        record = copy.deepcopy(dict(source))
        record["actions"] = [
            action
            for action in record["actions"]
            if str(action["action_id"]) in allowed
        ]
        output.append(record)
    assert_no_roi_records(output)
    return output


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Rank inputs must be finite vectors")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = _rankdata(left), _rankdata(right)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denominator <= 0.0 else float(np.dot(x, y) / denominator)


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    return 1.0 if not a and not b else len(a & b) / max(1, len(a | b))


def _find_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    seed: Any,
    mode: str,
    budget: float,
) -> Mapping[str, Any]:
    found = [
        row
        for row in rows
        if str(row["method"]) == str(method)
        and str(row["seed"]) == str(seed)
        and str(row["mode"]) == str(mode)
        and math.isclose(float(row["budget"]), float(budget), abs_tol=1e-12)
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {method}/{seed}/{mode}/{budget:g}")
    return found[0]


def evaluate_candidate_gate(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    config = protocol["gate"]
    seeds = tuple(map(int, protocol["seeds"]))
    byte_rows, call_rows = [], []
    for mode, output in (("fixed_byte", byte_rows), ("fixed_call", call_rows)):
        for budget in EXPECTED_BUDGETS:
            baseline = _find_row(
                rows,
                method="CurrentActionValue",
                seed="ensemble",
                mode=mode,
                budget=budget,
            )
            candidate = _find_row(
                rows,
                method="RelActionRank",
                seed="ensemble",
                mode=mode,
                budget=budget,
            )
            seed_deltas = []
            for seed in seeds:
                seed_base = _find_row(
                    rows,
                    method="CurrentActionValue",
                    seed=seed,
                    mode=mode,
                    budget=budget,
                )
                seed_candidate = _find_row(
                    rows,
                    method="RelActionRank",
                    seed=seed,
                    mode=mode,
                    budget=budget,
                )
                seed_deltas.append(
                    float(seed_candidate["official_ap50_95"])
                    - float(seed_base["official_ap50_95"])
                )
            output.append(
                {
                    "budget": float(budget),
                    "delta_ap": float(candidate["official_ap50_95"])
                    - float(baseline["official_ap50_95"]),
                    "supporting_seeds": sum(
                        value >= float(config["minimum_fixed_byte_gain"])
                        for value in seed_deltas
                    ),
                    "seed_deltas": seed_deltas,
                    "capacity_ok": bool(candidate.get("capacity_ok", False)),
                    "capacity_utilization": float(
                        candidate.get("capacity_utilization", 1.0)
                    ),
                    "roi_count": int(candidate.get("roi_count", 0)),
                }
            )
    gain_points = sum(
        row["delta_ap"] >= float(config["minimum_fixed_byte_gain"])
        and row["supporting_seeds"] >= int(config["minimum_supporting_seeds"])
        for row in byte_rows
    )
    byte_noninferior = all(
        row["delta_ap"] >= -float(config["maximum_other_loss"])
        for row in byte_rows
    )
    call_noninferior = all(
        row["delta_ap"] >= -float(config["maximum_fixed_call_loss"])
        for row in call_rows
    )
    capacity_ok = all(
        row["capacity_ok"]
        and row["capacity_utilization"]
        >= float(config["minimum_capacity_utilization"])
        and row["roi_count"] == 0
        for row in byte_rows
    )
    return {
        "passed": bool(
            gain_points >= int(config["required_fixed_byte_gain_points"])
            and byte_noninferior
            and call_noninferior
            and capacity_ok
        ),
        "gain_points": int(gain_points),
        "byte_noninferior": bool(byte_noninferior),
        "call_noninferior": bool(call_noninferior),
        "capacity_ok": bool(capacity_ok),
        "fixed_byte": byte_rows,
        "fixed_call": call_rows,
    }


def should_run_calibration(
    oof_gate: Mapping[str, Any], protocol: Mapping[str, Any]
) -> bool:
    if protocol.get("calibration_conditioned_on_oof_gate") is not True:
        raise ValueError("Calibration must remain conditioned on the OOF gate")
    return bool(oof_gate.get("passed", False))


def validate_result_matrix(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    expected = {
        (str(method), str(seed), str(mode), float(budget))
        for method in METHODS
        for seed in (*tuple(map(int, protocol["seeds"])), "ensemble")
        for mode in EXPECTED_MODES
        for budget in EXPECTED_BUDGETS
    }
    observed = {
        (
            str(row["method"]),
            str(row["seed"]),
            str(row["mode"]),
            float(row["budget"]),
        )
        for row in rows
    }
    if len(rows) != len(observed) or observed != expected:
        raise RuntimeError(
            f"Incomplete or duplicate result matrix: {len(rows)}/{len(expected)}"
        )
    for row in rows:
        if int(row.get("roi_count", 0)) != 0:
            raise RuntimeError("A result row contains ROI")
        if not bool(row.get("capacity_ok", False)):
            raise RuntimeError("A result row violates its budget")


class MeanMaxActionValueRouter(nn.Module):
    """Existing permutation-invariant mean/max pooling value encoder."""

    def __init__(
        self,
        *,
        category_count: int,
        scene_dim: int,
        action_dim: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        box_hidden = int(config["box_hidden_dim"])
        action_hidden = int(config["action_hidden_dim"])
        joint_hidden = int(config["joint_hidden_dim"])
        class_dim = int(config["class_embedding_dim"])
        dropout = float(config["dropout"])
        self.category_count = int(category_count)
        self.class_embedding = nn.Embedding(self.category_count, class_dim)
        self.box_encoder = nn.Sequential(
            nn.Linear(6 + class_dim, box_hidden),
            nn.LayerNorm(box_hidden),
            nn.SiLU(),
            nn.Linear(box_hidden, box_hidden),
            nn.SiLU(),
        )
        self.scene_encoder = nn.Sequential(
            nn.Linear(int(scene_dim), box_hidden), nn.LayerNorm(box_hidden), nn.SiLU()
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(box_hidden * 3, joint_hidden),
            nn.LayerNorm(joint_hidden),
            nn.SiLU(),
        )
        self.action_type_embedding = nn.Embedding(3, 8)
        self.action_encoder = nn.Sequential(
            nn.Linear(int(action_dim) + 8, action_hidden),
            nn.LayerNorm(action_hidden),
            nn.SiLU(),
            nn.Linear(action_hidden, action_hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(joint_hidden + action_hidden, joint_hidden),
            nn.LayerNorm(joint_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden, 1),
        )

    def _image(
        self,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        box_mask: torch.Tensor,
        scene: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.class_embedding(classes.clamp(0, self.category_count - 1))
        encoded = self.box_encoder(torch.cat((boxes, embedded), dim=-1))
        valid = box_mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid, -torch.inf).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return self.image_encoder(
            torch.cat((mean, maximum, self.scene_encoder(scene)), dim=-1)
        )

    def forward(
        self,
        *,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        box_mask: torch.Tensor,
        scene: torch.Tensor,
        actions: torch.Tensor,
        action_type: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        image = self._image(boxes, classes, box_mask, scene)
        action = self.action_encoder(
            torch.cat(
                (
                    actions,
                    self.action_type_embedding(action_type.clamp(0, 2)),
                ),
                dim=-1,
            )
        )
        image = image[:, None, :].expand(-1, actions.shape[1], -1)
        scores = self.head(torch.cat((image, action), dim=-1)).squeeze(-1)
        return scores.masked_fill(~action_mask, -1e9)


class RelationalActionValueRouter(MeanMaxActionValueRouter):
    """Self-attention over detection tokens with invariant masked pooling."""

    def __init__(
        self,
        *,
        category_count: int,
        scene_dim: int,
        action_dim: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__(
            category_count=category_count,
            scene_dim=scene_dim,
            action_dim=action_dim,
            config=config,
        )
        relation_hidden = int(config["relation_hidden_dim"])
        class_dim = int(config["class_embedding_dim"])
        joint_hidden = int(config["joint_hidden_dim"])
        self.box_encoder = nn.Sequential(
            nn.Linear(6 + class_dim, relation_hidden),
            nn.LayerNorm(relation_hidden),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=relation_hidden,
            nhead=int(config["relation_heads"]),
            dim_feedforward=int(config["relation_ff_dim"]),
            dropout=float(config["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_encoder = nn.TransformerEncoder(
            layer, num_layers=int(config["relation_layers"]), enable_nested_tensor=False
        )
        self.scene_encoder = nn.Sequential(
            nn.Linear(int(scene_dim), relation_hidden),
            nn.LayerNorm(relation_hidden),
            nn.SiLU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(relation_hidden * 3, joint_hidden),
            nn.LayerNorm(joint_hidden),
            nn.SiLU(),
        )

    def _image(
        self,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        box_mask: torch.Tensor,
        scene: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.class_embedding(classes.clamp(0, self.category_count - 1))
        encoded = self.box_encoder(torch.cat((boxes, embedded), dim=-1))
        safe_mask = box_mask.to(torch.bool).clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
            encoded = encoded.clone()
            encoded[empty, 0] = 0.0
        related = self.relation_encoder(
            encoded, src_key_padding_mask=~safe_mask
        )
        valid = safe_mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (related * valid).sum(dim=1) / count
        maximum = related.masked_fill(~valid, -torch.inf).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return self.image_encoder(
            torch.cat((mean, maximum, self.scene_encoder(scene)), dim=-1)
        )


def build_model(
    method: str,
    *,
    category_count: int,
    scene_dim: int,
    action_dim: int,
    config: Mapping[str, Any],
) -> nn.Module:
    name = str(method)
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}")
    model_type = (
        RelationalActionValueRouter
        if name in RELATIONAL_METHODS
        else MeanMaxActionValueRouter
    )
    return model_type(
        category_count=int(category_count),
        scene_dim=int(scene_dim),
        action_dim=int(action_dim),
        config=config,
    )


def _within_image_pairwise_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    minimum_target_gap: float,
) -> torch.Tensor:
    difference = targets.unsqueeze(2) - targets.unsqueeze(1)
    score_difference = scores.unsqueeze(2) - scores.unsqueeze(1)
    count = scores.shape[1]
    upper = torch.triu(
        torch.ones((count, count), dtype=torch.bool, device=scores.device),
        diagonal=1,
    ).unsqueeze(0)
    valid = (
        upper
        & action_mask.unsqueeze(2)
        & action_mask.unsqueeze(1)
        & (difference.abs() >= float(minimum_target_gap))
    )
    if not valid.any():
        return scores.sum() * 0.0
    weights = difference.abs()[valid]
    values = F.softplus(-difference.sign() * score_difference)[valid]
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def cross_image_pairwise_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    action_type: torch.Tensor,
    *,
    minimum_target_gap: float,
) -> torch.Tensor:
    if scores.shape != targets.shape or scores.shape != action_mask.shape:
        raise ValueError("Score, target, and mask shapes differ")
    cloud = action_mask.to(torch.bool) & (action_type != EDGE_ACTION_TYPE)
    image_index = torch.arange(
        scores.shape[0], device=scores.device
    )[:, None].expand_as(scores)
    flat_scores = scores[cloud]
    flat_targets = targets[cloud]
    flat_images = image_index[cloud]
    count = len(flat_scores)
    if count < 2:
        return scores.sum() * 0.0
    target_difference = flat_targets[:, None] - flat_targets[None, :]
    score_difference = flat_scores[:, None] - flat_scores[None, :]
    valid = (
        torch.triu(
            torch.ones((count, count), dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        & (flat_images[:, None] != flat_images[None, :])
        & (target_difference.abs() >= float(minimum_target_gap))
    )
    if not valid.any():
        return scores.sum() * 0.0
    weights = target_difference.abs()[valid]
    values = F.softplus(-target_difference.sign() * score_difference)[valid]
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def action_value_objective(
    scores: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    action_type: torch.Tensor,
    *,
    cross_image_weight: float,
    reward_weight: float,
    within_weight: float,
    minimum_target_gap: float,
) -> dict[str, torch.Tensor]:
    valid = action_mask.to(torch.bool)
    reward = F.smooth_l1_loss(scores[valid], targets[valid])
    within = _within_image_pairwise_loss(
        scores,
        targets,
        valid,
        minimum_target_gap=float(minimum_target_gap),
    )
    cross = cross_image_pairwise_loss(
        scores,
        targets,
        valid,
        action_type,
        minimum_target_gap=float(minimum_target_gap),
    )
    loss = (
        float(reward_weight) * reward
        + float(within_weight) * within
        + float(cross_image_weight) * cross
    )
    return {
        "loss": loss,
        "reward_loss": reward,
        "within_loss": within,
        "cross_image_loss": cross,
    }
