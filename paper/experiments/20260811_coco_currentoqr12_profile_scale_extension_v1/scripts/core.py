"""Pure contracts for the COCO OQR12 profile-scale extension."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


PROFILES = ("current", "moderate", "historical_cost", "fixed_1_8")
IMAGE_SIZES = (160, 320, 640)


def collapse_static_budget_scores(
    matrix: np.ndarray, *, expected_budgets: int
) -> np.ndarray:
    """Remove a redundant budget axis without hiding budget-conditioned scores."""
    values = np.asarray(matrix)
    if values.ndim == 2:
        return values
    if values.ndim != 3 or values.shape[0] != int(expected_budgets):
        raise ValueError(f"unexpected score shape: {values.shape}")
    if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
        raise ValueError(
            "budget-conditioned scores require budget-specific evaluation"
        )
    return values[0]


def protocol_for_reused_fusion_source(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    adapted = dict(protocol)
    adapted["reversible_fusion"] = {
        "primary_parameters": dict(protocol["reversible_fusion"])
    }
    return adapted


def roi_actions(protocol: Mapping[str, Any]) -> list[str]:
    return [
        f"{profile}@{size}"
        for profile in map(str, protocol["profiles"])
        for size in map(int, protocol["imgsz"])
    ]


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("dataset must be COCO")
    if tuple(protocol.get("profiles", ())) != PROFILES:
        raise ValueError("ROI Profiles changed")
    if tuple(map(int, protocol.get("imgsz", ()))) != IMAGE_SIZES:
        raise ValueError("ROI image-size grid changed")
    expected = ["edge", "original", "q90", *roi_actions(protocol)]
    if list(protocol.get("roi_actions", ())) != expected[3:]:
        raise ValueError("top-level ROI action order changed")
    if list(protocol["action_spaces"]["OQR12"]) != expected:
        raise ValueError("OQR12 action order changed")
    if list(protocol["action_spaces"]["OQR"]) != [
        "edge",
        "original",
        "q90",
        "current@640",
    ]:
        raise ValueError("single ROI is not COCO current@640")
    if protocol.get("roles") != {"fit": 9678, "calibration": 571, "final": 5000}:
        raise ValueError("canonical COCO populations changed")
    if bool(protocol.get("confirmatory")) or bool(protocol.get("final_is_hidden")):
        raise ValueError("val2017 reuse cannot be confirmatory")


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    spaces = {
        str(protocol["methods"]["single"]): list(protocol["action_spaces"]["OQR"]),
        str(protocol["methods"]["multi"]): list(protocol["action_spaces"]["OQR12"]),
    }
    folds: list[int | str] = [*range(int(protocol["fold_count"])), "full"]
    return [
        {
            "id": f"{method}|seed={int(seed)}|{'full' if fold == 'full' else f'fold_{fold}'}",
            "method": method,
            "action_space": actions,
            "seed": int(seed),
            "fold": fold,
        }
        for method, actions in spaces.items()
        for seed in protocol["model_seeds"]
        for fold in folds
    ]


def expected_point_rows(protocol: Mapping[str, Any]) -> int:
    return 2 * len(protocol["methods"]) * len(protocol["modes"]) * len(protocol["budgets"])


def bootstrap_jobs(protocol: Mapping[str, Any]) -> int:
    return len(protocol["budgets"]) * 2


def cache_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    return [
        {
            "id": f"{role}|{action}",
            "role": role,
            "action_id": action,
            "profile": action.split("@", 1)[0],
            "imgsz": int(action.split("@", 1)[1]),
        }
        for role in protocol["roles"]
        for action in roi_actions(protocol)
    ]


class GpuGateState:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.consecutive = 0

    def observe(self, *, free_mib: int, utilization: int) -> bool:
        eligible = (
            int(free_mib) >= int(self.config["minimum_free_gpu_mib"])
            and int(utilization)
            <= int(self.config["maximum_gpu_utilization_percent"])
        )
        self.consecutive = self.consecutive + 1 if eligible else 0
        return self.consecutive >= int(self.config["stable_polls"])
