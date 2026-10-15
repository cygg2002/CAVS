"""Pure contracts for Budget-Set SharedAction and CostEligible."""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ACTION_TYPES = {"edge": 0, "full": 1, "roi": 2, "pair": 3, "full_region": 4}
SCENE_DIM = 9
ACTION_DIM = 18


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco_train2017_cached":
        raise ValueError("This diagnostic is frozen to cached COCO train2017")
    if list(map(float, protocol.get("budgets", ()))) != [0.1, 0.25, 0.4]:
        raise ValueError("Budgets must remain 10%, 25%, and 40%")
    if int(protocol.get("train_images", -1)) + int(
        protocol.get("holdout_images", -1)
    ) != int(protocol.get("expected_images", -2)):
        raise ValueError("Train and holdout sizes do not cover the population")
    if protocol.get("holdout_used_for_training") is not False:
        raise ValueError("Holdout cannot train the model")
    if protocol.get("holdout_used_for_selection") is not False:
        raise ValueError("Holdout cannot select models or epochs")
    if protocol.get("val2017_final_accessed") is not False:
        raise ValueError("The consumed COCO val2017 final must stay sealed")
    if protocol.get("transductive_reward_context_disclosed") is not True:
        raise ValueError("The cached reward-context limitation must be disclosed")
    if int(protocol.get("max_actions", 0)) < 2 or int(protocol.get("max_boxes", 0)) < 1:
        raise ValueError("Action and box capacities are invalid")


def deterministic_split(
    image_ids: Iterable[int], *, train_count: int, seed: int
) -> dict[str, list[int] | int]:
    ids = sorted(map(int, image_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("Image IDs must be unique")
    if not 0 < int(train_count) < len(ids):
        raise ValueError("Train count must create two non-empty splits")

    def key(image_id: int) -> tuple[str, int]:
        digest = hashlib.sha256(f"{int(seed)}:{image_id}".encode("ascii")).hexdigest()
        return digest, image_id

    shuffled = sorted(ids, key=key)
    return {
        "train_image_ids": sorted(shuffled[: int(train_count)]),
        "holdout_image_ids": sorted(shuffled[int(train_count) :]),
        "split_seed": int(seed),
    }


def validate_split_manifest(
    manifest: Mapping[str, Any], *, expected_images: int, expected_train: int
) -> None:
    train = list(map(int, manifest.get("train_image_ids", ())))
    holdout = list(map(int, manifest.get("holdout_image_ids", ())))
    if len(train) != int(expected_train) or len(train) + len(holdout) != int(
        expected_images
    ):
        raise ValueError("Split population or train count drifted")
    if len(set(train)) != len(train) or len(set(holdout)) != len(holdout):
        raise ValueError("A split contains duplicate image IDs")
    if set(train) & set(holdout):
        raise ValueError("Train and holdout image IDs overlap")


def assert_holdout_allowed(
    training_completion: Mapping[str, Any], holdout_completion_exists: bool
) -> None:
    if training_completion.get("complete") is not True:
        raise RuntimeError("Training must complete before holdout evaluation")
    if training_completion.get("holdout_accessed", False) is not False:
        raise RuntimeError("Training completion reports holdout access")
    if holdout_completion_exists:
        raise RuntimeError("Holdout evaluation is one-time and already complete")


def _score(box: Mapping[str, Any]) -> float:
    return float(box.get("conf", box.get("score", 0.0)))


def build_box_set(
    record: Mapping[str, Any], *, max_boxes: int, confidence_floor: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(1.0, float(record["width"]))
    height = max(1.0, float(record["height"]))
    boxes = [
        dict(box)
        for box in record.get("edge_boxes", ())
        if _score(box) >= float(confidence_floor)
    ]
    boxes.sort(
        key=lambda box: (
            -_score(box),
            int(box.get("cls", 0)),
            tuple(map(float, box["xyxy"])),
        )
    )
    boxes = boxes[: int(max_boxes)]
    numeric = torch.zeros((int(max_boxes), 6), dtype=torch.float32)
    classes = torch.zeros(int(max_boxes), dtype=torch.long)
    mask = torch.zeros(int(max_boxes), dtype=torch.bool)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box["xyxy"])
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        numeric[index] = torch.tensor(
            (
                (x1 + x2) / 2.0 / width,
                (y1 + y2) / 2.0 / height,
                box_width / width,
                box_height / height,
                box_width * box_height / (width * height),
                _score(box),
            ),
            dtype=torch.float32,
        )
        classes[index] = int(box.get("cls", 0))
        mask[index] = True
    return numeric, classes, mask


def deployable_scene_features(
    record: Mapping[str, Any], *, deploy_confidence: float
) -> torch.Tensor:
    width = max(1.0, float(record["width"]))
    height = max(1.0, float(record["height"]))
    boxes = [
        box
        for box in record.get("edge_boxes", ())
        if _score(box) >= float(deploy_confidence)
    ]
    scores = [_score(box) for box in boxes]
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = map(float, box["xyxy"])
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height))
    image_stats = record.get("image_stats", {})
    values = (
        math.log(width / height),
        math.log1p(max(0, int(record.get("full_payload_bytes", 0))) / 1024.0),
        min(1.0, len(boxes) / 64.0),
        max(scores, default=0.0),
        statistics.fmean(scores) if scores else 0.0,
        min(1.0, sum(areas)),
        float(image_stats.get("luma_mean", 0.0)),
        float(image_stats.get("luma_std", 0.0)),
        float(image_stats.get("edge_density", 0.0)),
    )
    return torch.tensor(values, dtype=torch.float32)


def deployable_action_features(
    action: Mapping[str, Any], record: Mapping[str, Any]
) -> torch.Tensor:
    resolutions = [float(value) / 640.0 for value in action.get("resolutions", ())]
    regions_by_id = {
        str(region["region_uid"]): region for region in record.get("regions", ())
    }
    regions = [
        regions_by_id[str(uid)]
        for uid in action.get("region_uids", ())
        if str(uid) in regions_by_id
    ]
    width = max(1.0, float(record["width"]))
    height = max(1.0, float(record["height"]))
    crop_ratios = []
    source_counts: dict[str, int] = {}
    for region in regions:
        x1, y1, x2, y2 = map(float, region["crop_xyxy"])
        crop_ratios.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height))
        source = str(region.get("source", ""))
        source_counts[source] = source_counts.get(source, 0) + 1
    region_count = max(1, len(regions))
    full_bytes = max(1.0, float(record.get("full_payload_bytes", 1)))
    payload = max(0.0, float(action.get("payload_bytes", 0)))
    response = max(0.0, float(action.get("response_bytes", 0)))
    costs = action.get("resource_costs", {})
    values = (
        min(resolutions, default=0.0),
        max(resolutions, default=0.0),
        statistics.fmean(resolutions) if resolutions else 0.0,
        max(resolutions, default=0.0) - min(resolutions, default=0.0),
        min(1.0, float(action.get("cloud_calls", 0)) / 2.0),
        min(1.0, len(regions) / 2.0),
        payload / full_bytes,
        response / full_bytes,
        float(costs.get("compute", 0.0)),
        float(costs.get("upload", 0.0)),
        float(costs.get("e2e", 0.0)),
        math.log1p(payload / 1024.0),
        math.log1p(response / 1024.0),
        min(2.0, sum(crop_ratios)),
        max(crop_ratios, default=0.0),
        source_counts.get("miss_tile", 0) / region_count,
        source_counts.get("lowconf_proposal", 0) / region_count,
        source_counts.get("deployed_box", 0) / region_count,
    )
    return torch.tensor(values, dtype=torch.float32)


class BudgetSetRouter(nn.Module):
    """Permutation-invariant image set encoder with action/budget interaction."""

    def __init__(
        self,
        *,
        category_count: int,
        scene_dim: int,
        action_dim: int,
        class_embedding_dim: int,
        box_hidden_dim: int,
        action_hidden_dim: int,
        joint_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.category_count = int(category_count)
        self.class_embedding = nn.Embedding(category_count, class_embedding_dim)
        self.box_encoder = nn.Sequential(
            nn.Linear(6 + class_embedding_dim, box_hidden_dim),
            nn.LayerNorm(box_hidden_dim),
            nn.SiLU(),
            nn.Linear(box_hidden_dim, box_hidden_dim),
            nn.SiLU(),
        )
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, box_hidden_dim),
            nn.LayerNorm(box_hidden_dim),
            nn.SiLU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(box_hidden_dim * 3, joint_hidden_dim),
            nn.LayerNorm(joint_hidden_dim),
            nn.SiLU(),
        )
        self.action_type_embedding = nn.Embedding(len(ACTION_TYPES), 8)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim + 8 + 1, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
            nn.SiLU(),
            nn.Linear(action_hidden_dim, action_hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(joint_hidden_dim + action_hidden_dim, joint_hidden_dim),
            nn.LayerNorm(joint_hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(joint_hidden_dim, 1),
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
        budget: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.class_embedding(classes.clamp(0, self.category_count - 1))
        encoded = self.box_encoder(torch.cat((boxes, embedded), dim=-1))
        valid = box_mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid, -torch.inf).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        scene_encoded = self.scene_encoder(scene)
        image = self.image_encoder(torch.cat((mean, maximum, scene_encoded), dim=-1))

        action_count = actions.shape[1]
        budget_column = budget[:, None, None].expand(-1, action_count, 1)
        action = self.action_encoder(
            torch.cat(
                (
                    actions,
                    self.action_type_embedding(action_type.clamp(0, len(ACTION_TYPES) - 1)),
                    budget_column,
                ),
                dim=-1,
            )
        )
        image = image[:, None, :].expand(-1, action_count, -1)
        scores = self.head(torch.cat((image, action), dim=-1)).squeeze(-1)
        return scores.masked_fill(~action_mask, -1e9)


def boundary_pairs(
    selected: np.ndarray,
    valid: np.ndarray,
    *,
    pair_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_values = np.asarray(selected, dtype=bool).reshape(-1)
    valid_values = np.asarray(valid, dtype=bool).reshape(-1)
    if selected_values.shape != valid_values.shape:
        raise ValueError("Selected and valid masks differ")
    positive = np.flatnonzero(selected_values & valid_values)
    negative = np.flatnonzero(~selected_values & valid_values)
    if not len(positive) or not len(negative) or int(pair_count) <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return (
        rng.choice(positive, int(pair_count), replace=True),
        rng.choice(negative, int(pair_count), replace=True),
    )


def budget_set_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    boundary_selected: torch.Tensor,
    boundary_pairs_count: int,
    seed: int,
    reward_weight: float = 0.5,
    within_weight: float = 0.5,
    boundary_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    valid = action_mask.to(torch.bool)
    reward = F.smooth_l1_loss(scores[valid], targets[valid])
    target_difference = targets.unsqueeze(2) - targets.unsqueeze(1)
    score_difference = scores.unsqueeze(2) - scores.unsqueeze(1)
    action_count = scores.shape[1]
    upper = torch.triu(
        torch.ones((action_count, action_count), dtype=torch.bool, device=scores.device),
        diagonal=1,
    ).unsqueeze(0)
    pair_valid = (
        upper
        & valid.unsqueeze(2)
        & valid.unsqueeze(1)
        & (target_difference.abs() > 1e-8)
    )
    if pair_valid.any():
        within = (
            F.softplus(-target_difference.sign() * score_difference)[pair_valid]
            * target_difference.abs()[pair_valid]
        ).sum() / target_difference.abs()[pair_valid].sum().clamp_min(1e-8)
    else:
        within = scores.sum() * 0.0

    positive, negative = boundary_pairs(
        boundary_selected.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
        pair_count=int(boundary_pairs_count),
        seed=int(seed),
    )
    if len(positive):
        flat = scores.reshape(-1)
        positive_index = torch.as_tensor(positive, dtype=torch.long, device=scores.device)
        negative_index = torch.as_tensor(negative, dtype=torch.long, device=scores.device)
        boundary = F.softplus(-(flat[positive_index] - flat[negative_index])).mean()
    else:
        boundary = scores.sum() * 0.0
    total = float(reward_weight) * reward + float(within_weight) * within + float(
        boundary_weight
    ) * boundary
    return {
        "loss": total,
        "reward_loss": reward.detach(),
        "within_loss": within.detach(),
        "boundary_loss": boundary.detach(),
    }


def _top_count(total: int, budget: float) -> int:
    count = int(math.floor(int(total) * float(budget) + 0.5))
    if not 0 < count <= int(total):
        raise ValueError("Budget selects an invalid number of images")
    return count


def fixed_call_indices(
    scores: np.ndarray,
    action_mask: np.ndarray,
    action_types: np.ndarray,
    *,
    budget: float,
    full_only: bool,
) -> list[tuple[int, int]]:
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(action_mask, dtype=bool)
    kinds = np.asarray(action_types, dtype=np.int64)
    if values.shape != mask.shape or values.shape != kinds.shape:
        raise ValueError("Score, action-mask, and type shapes differ")
    candidates = []
    for image_index in range(len(values)):
        allowed = mask[image_index] & (
            kinds[image_index] == ACTION_TYPES["full"]
            if full_only
            else kinds[image_index] != ACTION_TYPES["edge"]
        )
        indices = np.flatnonzero(allowed)
        if not len(indices):
            continue
        action_index = min(
            map(int, indices),
            key=lambda index: (-values[image_index, index], index),
        )
        candidates.append((float(values[image_index, action_index]), image_index, action_index))
    count = _top_count(len(values), float(budget))
    if len(candidates) < count:
        raise ValueError("Not enough non-edge actions for fixed-call selection")
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [(row[1], row[2]) for row in candidates[:count]]


def _resource_cost(
    action: Mapping[str, Any], resource_profile: str, edge_e2e_ms: float
) -> float:
    if resource_profile == "upload":
        return float(action.get("payload_bytes", 0.0))
    if resource_profile == "compute":
        return float(action.get("cloud_gflops", 0.0))
    if resource_profile == "e2e":
        return max(0.0, float(action.get("e2e_ms", edge_e2e_ms)) - edge_e2e_ms)
    raise ValueError(f"Unknown resource profile: {resource_profile}")


def select_costeligible(
    records: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    *,
    resource_profile: str,
    capacity: float,
    full_only: bool = False,
) -> dict[str, Any]:
    from paper.experiments.GAIR_v5.gair.selector import select_actions_for_budget

    values = np.asarray(scores, dtype=np.float64)
    if values.shape[0] != len(records):
        raise ValueError("Record and score populations differ")
    groups = []
    for image_index, record in enumerate(records):
        actions = list(record["actions"])
        if values.shape[1] < len(actions):
            raise ValueError("Score matrix has fewer columns than actions")
        edge = next(
            action for action in actions if str(action.get("action_type")) == "edge"
        )
        edge_ms = float(edge.get("e2e_ms", 0.0))
        group = []
        for action_index, action in enumerate(actions):
            kind = str(action.get("action_type"))
            if full_only and kind not in {"edge", "full"}:
                continue
            group.append(
                {
                    "image_id": int(record["image_id"]),
                    "action_index": int(action_index),
                    "action_id": str(action["action_id"]),
                    "action_type": kind,
                    "action_record": action,
                    "value": 0.0 if kind == "edge" else float(values[image_index, action_index]),
                    "cost": _resource_cost(action, resource_profile, edge_ms),
                }
            )
        groups.append(group)
    return select_actions_for_budget(groups, float(capacity))


def selection_mask(
    records: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]], max_actions: int
) -> np.ndarray:
    by_image = {
        int(item["image_id"]): int(item.get("action_index", -1)) for item in selected
    }
    mask = np.zeros((len(records), int(max_actions)), dtype=bool)
    for image_index, record in enumerate(records):
        action_index = by_image.get(int(record["image_id"]), -1)
        if 0 <= action_index < int(max_actions):
            action = list(record["actions"])[action_index]
            if str(action.get("action_type")) != "edge":
                mask[image_index, action_index] = True
    return mask
