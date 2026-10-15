"""Pure contracts for the three-dataset ROI gate diagnostic."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


DATASETS = {
    "coco": {"canonical_run": "canonical_v1", "fit": 9678, "calibration": 571},
    "ena24": {"canonical_run": "canonical_v1", "fit": 1400, "calibration": 400},
    "oregon": {
        "canonical_run": "canonical_oregon_event_v1",
        "fit": 9923,
        "calibration": 4961,
    },
}
METHODS = (
    "ActionValue-OQ-Joint",
    "ActionValue-OQR-Naive",
    "ActionValue-OQR-AdvantageGate",
)
ACTIONS = ("edge", "original", "q90", "roi")
ROLES = ("fit_oof", "calibration")
MODES = ("fixed_byte", "fixed_call")
BUDGETS = (0.1, 0.25, 0.4)


def oqr_score_filename(role: str) -> str:
    if role not in ROLES:
        raise ValueError(role)
    return f"{role}_oqr.pkl"


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != "three_dataset_actionvalue_roi_gate_v1":
        raise ValueError("Protocol ID changed")
    if set(protocol.get("datasets", {})) != set(DATASETS):
        raise ValueError("Dataset set changed")
    for dataset, expected in DATASETS.items():
        observed = protocol["datasets"][dataset]
        if observed.get("canonical_run") != expected["canonical_run"]:
            if dataset == "oregon":
                raise ValueError("Oregon canonical run must be canonical_oregon_event_v1")
            raise ValueError(f"Canonical run changed for {dataset}")
        roles = observed.get("roles", {})
        if int(roles.get("fit", -1)) != expected["fit"] or int(
            roles.get("calibration", -1)
        ) != expected["calibration"]:
            raise ValueError(f"Role population changed for {dataset}")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError("Method matrix changed")
    if tuple(protocol.get("actions", ())) != ACTIONS:
        raise ValueError("Action space changed")
    if tuple(protocol.get("roles", ())) != ROLES:
        raise ValueError("Role matrix changed")
    if tuple(protocol.get("modes", ())) != MODES:
        raise ValueError("Mode matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != BUDGETS:
        raise ValueError("Budget matrix changed")
    if protocol.get("final_accessed") is not False:
        raise ValueError("Final access must remain false")
    if protocol.get("detector_inference") is not False:
        raise ValueError("Detector inference is forbidden")
    if protocol.get("advantage_gate", {}).get("threshold_source") != "fit_oof_only":
        raise ValueError("Gate threshold must be fit-OOF only")
    if protocol.get("advantage_gate", {}).get("calibration_used_for_threshold") is not False:
        raise ValueError("Calibration cannot select gate thresholds")
    if tuple(map(int, protocol.get("model", {}).get("seeds", ()))) != (
        20260804,
        20260805,
        20260806,
    ):
        raise ValueError("Seed matrix changed")
    source_text = "\n".join(map(str, protocol.get("source_experiments", {}).values()))
    for forbidden in protocol.get("forbidden_source_substrings", ()):
        if str(forbidden).lower() in source_text.lower():
            raise ValueError(f"Forbidden source configured: {forbidden}")


def expected_result_keys(
    protocol: Mapping[str, Any], dataset: str
) -> set[tuple[str, str, str, float]]:
    if dataset not in protocol.get("datasets", {}):
        raise ValueError(dataset)
    return {
        (str(role), str(method), str(mode), float(budget))
        for role in protocol["roles"]
        for method in protocol["methods"]
        for mode in protocol["modes"]
        for budget in protocol["budgets"]
    }


def apply_roi_advantage_threshold(
    scores: np.ndarray, *, threshold: float
) -> np.ndarray:
    """Mask ROI while leaving Edge/Original/Q90 values byte-for-byte intact."""
    values = np.asarray(scores, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected [N,4] OQR scores, got {values.shape}")
    advantage = values[:, 3] - np.maximum(values[:, 1], values[:, 2])
    values[advantage < float(threshold), 3] = -np.inf
    return values


def select_advantage_threshold(
    candidates: Sequence[Mapping[str, Any]], *, maximum_harmful_rate: float
) -> dict[str, Any]:
    """Choose the highest-reward safe candidate with deterministic conservative ties."""
    eligible = [
        dict(candidate)
        for candidate in candidates
        if float(candidate["harmful_rate"]) <= float(maximum_harmful_rate) + 1e-12
    ]
    if not eligible:
        return {
            "threshold": float("inf"),
            "reward": 0.0,
            "harmful_rate": 0.0,
            "roi_count": 0,
            "reason": "no_safe_candidate",
        }
    return max(
        eligible,
        key=lambda candidate: (
            float(candidate["reward"]),
            -int(candidate.get("roi_count", 0)),
            float(candidate["threshold"]),
        ),
    )


def merge_fold_scores(
    target: np.ndarray,
    *,
    seed_index: int,
    heldout_indices: Sequence[int],
    fold_scores: np.ndarray,
) -> None:
    """Place [budget, heldout, action] scores without NumPy axis reordering."""
    destination = target[int(seed_index)]
    expected = (destination.shape[0], len(heldout_indices), destination.shape[2])
    values = np.asarray(fold_scores, dtype=target.dtype)
    if values.shape != expected:
        raise ValueError(f"Fold score shape changed: {values.shape} != {expected}")
    destination[:, list(map(int, heldout_indices)), :] = values


def validate_result_rows(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], dataset: str) -> None:
    expected = expected_result_keys(protocol, dataset)
    observed = {
        (
            str(row["role"]),
            str(row["method"]),
            str(row["mode"]),
            float(row["budget"]),
        )
        for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise ValueError(f"Result matrix incomplete for {dataset}: {len(rows)} rows")
    for row in rows:
        if str(row["mode"]) == "fixed_byte" and not bool(row["capacity_ok"]):
            raise ValueError(f"Fixed-byte capacity exceeded: {row}")
        if str(row["mode"]) == "fixed_call":
            expected_count = int(math.floor(int(DATASETS[dataset]["calibration" if row["role"] == "calibration" else "fit"]) * float(row["budget"]) + 0.5))
            if int(row["selected_images"]) != expected_count:
                raise ValueError(f"Fixed-call count changed: {row}")
