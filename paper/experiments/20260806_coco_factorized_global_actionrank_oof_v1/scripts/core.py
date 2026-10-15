"""Pure contracts and model for the factorized COCO action-rank diagnostic."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE_ROOT = REPO / PROTOCOL["source_experiments"]["relational_source"]
SOURCE_CORE_PATH = SOURCE_ROOT / "scripts/core.py"


def _load_source_core():
    spec = importlib.util.spec_from_file_location(
        "factorized_actionrank_relational_source_core", SOURCE_CORE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source_core()
METHODS = (
    "CurrentActionValue",
    "GlobalRank",
    "RelActionRank",
    "FactorizedGlobalActionRank",
)
TRAINED_METHODS = ("FactorizedGlobalActionRank",)
REUSED_METHODS = ("CurrentActionValue", "GlobalRank", "RelActionRank")
EXPECTED_ACTIONS = ("edge", "original", "q90")
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
EXPECTED_MODES = ("fixed_call", "fixed_byte")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("This diagnostic is COCO-only")
    if dict(protocol.get("roles", {})) != {"fit": 9678, "calibration": 571}:
        raise ValueError("Canonical fit/calibration populations changed")
    if tuple(map(str, protocol.get("actions", ()))) != EXPECTED_ACTIONS:
        raise ValueError("The No-ROI action space changed")
    if tuple(map(str, protocol.get("candidate_actions", ()))) != (
        "original",
        "q90",
    ):
        raise ValueError("Only Original/Q90 candidates are allowed")
    if tuple(map(str, protocol.get("methods", ()))) != METHODS:
        raise ValueError("The frozen method matrix changed")
    if tuple(map(str, protocol.get("trained_oof_methods", ()))) != TRAINED_METHODS:
        raise ValueError("Only the factorized model may be trained in OOF")
    if tuple(map(str, protocol.get("reused_oof_methods", ()))) != REUSED_METHODS:
        raise ValueError("The frozen baseline reuse set changed")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("The budget grid changed")
    if tuple(map(str, protocol.get("modes", ()))) != EXPECTED_MODES:
        raise ValueError("The evaluation modes changed")
    if int(protocol.get("fold_count", 0)) != 5:
        raise ValueError("Five OOF folds are required")
    if tuple(map(int, protocol.get("seeds", ()))) != (
        20260806,
        20260807,
        20260808,
    ):
        raise ValueError("The frozen seed set changed")
    if protocol.get("primary_mode") != "fixed_byte":
        raise ValueError("Fixed-byte must remain primary")
    if protocol.get("reward", {}).get("fit_source_read_only") is not True:
        raise ValueError("Fit rewards must be reused read-only")
    if protocol.get("fit_only_parameter_selection") is not True:
        raise ValueError("Parameter selection must be fit-OOF-only")
    if protocol.get("calibration_used_for_training_or_selection") is not False:
        raise ValueError("Calibration cannot train or select parameters")
    for key in ("audit_accessed", "final_accessed", "registered_final_accessed"):
        if protocol.get(key) is not False:
            raise ValueError(f"Forbidden role flag changed: {key}")


def factorized_targets(
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    action_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if targets.shape != action_mask.shape or targets.shape != action_type.shape:
        raise ValueError("Target, mask, and action type shapes differ")
    valid = action_mask.to(torch.bool)
    safe = targets.masked_fill(~valid, -torch.inf)
    global_target = safe.amax(dim=1)
    if not torch.isfinite(global_target).all():
        raise ValueError("Every image requires at least one valid action")
    advantage_target = targets - global_target[:, None]
    cloud = valid & (action_type != 0)
    return global_target, advantage_target, cloud


def scalar_pairwise_rank_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    minimum_target_gap: float,
) -> torch.Tensor:
    if scores.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("Scalar ranking expects equal one-dimensional tensors")
    count = len(scores)
    if count < 2:
        return scores.sum() * 0.0
    target_difference = targets[:, None] - targets[None, :]
    score_difference = scores[:, None] - scores[None, :]
    valid = torch.triu(
        torch.ones((count, count), dtype=torch.bool, device=scores.device),
        diagonal=1,
    ) & (target_difference.abs() >= float(minimum_target_gap))
    if not valid.any():
        return scores.sum() * 0.0
    weights = target_difference.abs()[valid]
    values = F.softplus(-target_difference.sign() * score_difference)[valid]
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class FactorizedGlobalActionRankRouter(SOURCE.MeanMaxActionValueRouter):
    """Separate global image value from relative cloud-action advantage."""

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
        joint_hidden = int(config["joint_hidden_dim"])
        action_hidden = int(config["action_hidden_dim"])
        dropout = float(config["dropout"])
        del self.head
        self.global_head = nn.Sequential(
            nn.Linear(joint_hidden, joint_hidden),
            nn.LayerNorm(joint_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(joint_hidden + action_hidden, joint_hidden),
            nn.LayerNorm(joint_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden, 1),
        )

    def forward_components(
        self,
        *,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        box_mask: torch.Tensor,
        scene: torch.Tensor,
        actions: torch.Tensor,
        action_type: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        global_value = self.global_head(image).squeeze(-1)
        expanded = image[:, None, :].expand(-1, actions.shape[1], -1)
        advantage = self.advantage_head(
            torch.cat((expanded, action), dim=-1)
        ).squeeze(-1)
        scores = global_value[:, None] + advantage
        scores = torch.where(action_type == 0, torch.zeros_like(scores), scores)
        scores = scores.masked_fill(~action_mask, -1e9)
        return {
            "global_value": global_value,
            "advantage": advantage,
            "scores": scores,
        }

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_components(**inputs)["scores"]


def factorized_objective(
    model: FactorizedGlobalActionRankRouter,
    inputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    action_type: torch.Tensor,
    *,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    components = model.forward_components(**inputs)
    global_target, advantage_target, cloud = factorized_targets(
        targets, action_mask, action_type
    )
    global_regression = F.smooth_l1_loss(
        components["global_value"], global_target
    )
    global_rank = scalar_pairwise_rank_loss(
        components["global_value"],
        global_target,
        minimum_target_gap=float(config["minimum_pair_target_gap"]),
    )
    advantage_regression = F.smooth_l1_loss(
        components["advantage"][cloud], advantage_target[cloud]
    )
    action_rank = SOURCE._within_image_pairwise_loss(
        components["scores"],
        targets,
        cloud,
        minimum_target_gap=float(config["minimum_pair_target_gap"]),
    )
    loss = (
        float(config["factor_global_regression_weight"]) * global_regression
        + float(config["factor_global_rank_weight"]) * global_rank
        + float(config["factor_advantage_regression_weight"])
        * advantage_regression
        + float(config["factor_action_rank_weight"]) * action_rank
    )
    return {
        "loss": loss,
        "global_regression_loss": global_regression,
        "global_rank_loss": global_rank,
        "advantage_regression_loss": advantage_regression,
        "action_rank_loss": action_rank,
    }


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
        if str(row["method"]) == method
        and str(row["seed"]) == str(seed)
        and str(row["mode"]) == mode
        and math.isclose(float(row["budget"]), budget, abs_tol=1e-12)
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {method}/{seed}/{mode}/{budget:g}")
    return found[0]


def evaluate_candidate_gate(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    config = protocol["gate"]
    baseline_name = str(config["baseline_method"])
    candidate_name = str(config["candidate_method"])
    seeds = tuple(map(int, protocol["seeds"]))
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in EXPECTED_MODES}
    for mode in EXPECTED_MODES:
        for budget in EXPECTED_BUDGETS:
            baseline = _find_row(
                rows,
                method=baseline_name,
                seed="ensemble",
                mode=mode,
                budget=budget,
            )
            candidate = _find_row(
                rows,
                method=candidate_name,
                seed="ensemble",
                mode=mode,
                budget=budget,
            )
            seed_deltas = [
                float(
                    _find_row(
                        rows,
                        method=candidate_name,
                        seed=seed,
                        mode=mode,
                        budget=budget,
                    )["official_ap50_95"]
                )
                - float(
                    _find_row(
                        rows,
                        method=baseline_name,
                        seed=seed,
                        mode=mode,
                        budget=budget,
                    )["official_ap50_95"]
                )
                for seed in seeds
            ]
            by_mode[mode].append(
                {
                    "budget": budget,
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
    byte = by_mode["fixed_byte"]
    call = by_mode["fixed_call"]
    gain_points = sum(
        row["delta_ap"] >= float(config["minimum_fixed_byte_gain"])
        and row["supporting_seeds"] >= int(config["minimum_supporting_seeds"])
        for row in byte
    )
    byte_noninferior = all(
        row["delta_ap"] >= -float(config["maximum_other_loss"]) for row in byte
    )
    call_noninferior = all(
        row["delta_ap"] >= -float(config["maximum_fixed_call_loss"])
        for row in call
    )
    capacity_ok = all(
        row["capacity_ok"]
        and row["capacity_utilization"]
        >= float(config["minimum_capacity_utilization"])
        and row["roi_count"] == 0
        for row in byte
    )
    return {
        "passed": bool(
            gain_points >= int(config["required_fixed_byte_gain_points"])
            and byte_noninferior
            and call_noninferior
            and capacity_ok
        ),
        "baseline": baseline_name,
        "candidate": candidate_name,
        "gain_points": int(gain_points),
        "byte_noninferior": bool(byte_noninferior),
        "call_noninferior": bool(call_noninferior),
        "capacity_ok": bool(capacity_ok),
        "fixed_byte": byte,
        "fixed_call": call,
    }


def should_run_calibration(
    oof_gate: Mapping[str, Any], protocol: Mapping[str, Any]
) -> bool:
    if protocol.get("calibration_conditioned_on_oof_gate") is not True:
        raise ValueError("Calibration must remain conditioned on OOF")
    return bool(oof_gate.get("passed", False))


def validate_result_matrix(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    expected = {
        (method, str(seed), mode, budget)
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


validate_protocol(PROTOCOL)
