"""Pure contracts for the COCO DualHead-v2 No-ROI diagnostic."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_ROLES = {"fit": 9678, "calibration": 571}
EXPECTED_ACTIONS = ("edge", "original", "q90")
EXPECTED_CANDIDATES = ("original", "q90")
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
EXPECTED_MODES = ("fixed_call", "fixed_byte")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("This diagnostic is COCO-only")
    if dict(protocol.get("roles", {})) != EXPECTED_ROLES:
        raise ValueError("Canonical fit/calibration role populations changed")
    if tuple(map(str, protocol.get("actions", ()))) != EXPECTED_ACTIONS:
        raise ValueError("The No-ROI action space changed")
    if tuple(map(str, protocol.get("candidate_actions", ()))) != EXPECTED_CANDIDATES:
        raise ValueError("ROI or another candidate action is forbidden")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("The canonical budget grid changed")
    if tuple(map(str, protocol.get("modes", ()))) != EXPECTED_MODES:
        raise ValueError("The evaluation modes changed")
    if protocol.get("primary_mode") != "fixed_byte":
        raise ValueError("Fixed-byte must remain primary")
    alphas = tuple(map(float, protocol.get("alpha_grid", ())))
    if not alphas or alphas[0] != 0.0 or any(
        not math.isfinite(value) or value < 0.0 for value in alphas
    ):
        raise ValueError("The alpha grid must start at zero and remain non-negative")
    if len(alphas) != len(set(alphas)):
        raise ValueError("The alpha grid contains duplicates")
    if not 0.0 < float(protocol.get("minimum_capacity_utilization", 0.0)) <= 1.0:
        raise ValueError("Invalid minimum byte-capacity utilization")
    if protocol.get("fit_only_parameter_selection") is not True:
        raise ValueError("Alpha selection must remain fit-OOF-only")
    if protocol.get("calibration_used_for_training_or_selection") is not False:
        raise ValueError("Calibration cannot train or select parameters")
    for key in ("audit_accessed", "final_accessed", "registered_final_accessed"):
        if protocol.get(key) is not False:
            raise ValueError(f"Forbidden role flag changed: {key}")


def blend_mean_lower(
    mean: np.ndarray, lower: np.ndarray, alpha: float
) -> np.ndarray:
    """Penalize only estimated downside while preserving the mean at alpha=0."""
    mean_array = np.asarray(mean, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    if mean_array.shape != lower_array.shape:
        raise ValueError("Mean and lower-quantile arrays must align")
    if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    return mean_array - float(alpha) * np.maximum(mean_array - lower_array, 0.0)


def no_roi_score_map(
    records: Sequence[Mapping[str, Any]], scores: np.ndarray
) -> dict[int, dict[str, float]]:
    """Map score columns to Original/Q90 and hard-drop every other action."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(records):
        raise ValueError("Score matrix and record population differ")
    output: dict[int, dict[str, float]] = {}
    for row_index, record in enumerate(records):
        actions = list(record["actions"])
        if len(actions) > values.shape[1]:
            raise ValueError("Score matrix has fewer columns than actions")
        mapped = {
            str(action["action_id"]): float(values[row_index, action_index])
            for action_index, action in enumerate(actions)
            if str(action["action_id"]) in EXPECTED_CANDIDATES
            and math.isfinite(float(values[row_index, action_index]))
        }
        if "original" not in mapped:
            raise RuntimeError(f"Original action missing for image {record['image_id']}")
        output[int(record["image_id"])] = mapped
    return output


def assert_no_roi_selection(selection: Mapping[int, str]) -> None:
    invalid = [
        (int(image_id), str(action))
        for image_id, action in selection.items()
        if str(action) not in EXPECTED_CANDIDATES
    ]
    if invalid:
        raise RuntimeError(f"ROI or invalid action found in selection: {invalid[:3]}")


def _find_row(
    rows: Sequence[Mapping[str, Any]], method: str, mode: str, budget: float
) -> Mapping[str, Any]:
    found = [
        row
        for row in rows
        if str(row["method"]) == str(method)
        and str(row["mode"]) == str(mode)
        and abs(float(row["budget"]) - float(budget)) < 1e-12
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {method}/{mode}/{budget:g}")
    return found[0]


def _alpha_from_row(row: Mapping[str, Any]) -> float:
    if "alpha" in row:
        return float(row["alpha"])
    prefix = "DualHeadV2-alpha="
    method = str(row["method"])
    if not method.startswith(prefix):
        raise ValueError(f"Cannot read alpha from method {method}")
    return float(method[len(prefix) :])


def choose_alphas(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    budgets: Iterable[float] | None = None,
) -> dict[float, dict[str, Any]]:
    """Freeze the best eligible risk blend using fit OOF rows only."""
    selected: dict[float, dict[str, Any]] = {}
    minimum_utilization = float(protocol["minimum_capacity_utilization"])
    call_margin = float(protocol["dualheadv2_gate"]["maximum_fixed_call_loss"])
    for budget in map(float, EXPECTED_BUDGETS if budgets is None else budgets):
        baseline_call = float(
            _find_row(rows, "ActionValue-OQ", "fixed_call", budget)[
                "official_ap50_95"
            ]
        )
        candidates = []
        for byte_row in rows:
            method = str(byte_row.get("method", ""))
            if not method.startswith("DualHeadV2-alpha="):
                continue
            if str(byte_row.get("mode")) != "fixed_byte" or abs(
                float(byte_row.get("budget", -1.0)) - budget
            ) >= 1e-12:
                continue
            call_row = _find_row(rows, method, "fixed_call", budget)
            eligible = (
                bool(byte_row.get("capacity_ok", False))
                and float(byte_row.get("capacity_utilization", 0.0))
                >= minimum_utilization
                and int(byte_row.get("roi_count", 0)) == 0
                and float(call_row["official_ap50_95"])
                >= baseline_call - call_margin
            )
            if eligible:
                candidates.append(
                    (
                        float(byte_row["official_ap50_95"]),
                        -_alpha_from_row(byte_row),
                        byte_row,
                    )
                )
        constraints_passed = bool(candidates)
        if candidates:
            _, negative_alpha, best = max(
                candidates, key=lambda item: (item[0], item[1])
            )
            alpha = -negative_alpha
        else:
            best = _find_row(
                rows, "DualHeadV2-alpha=0", "fixed_byte", budget
            )
            alpha = 0.0
        selected[budget] = {
            "alpha": float(alpha),
            "oof_fixed_byte_ap": float(best["official_ap50_95"]),
            "oof_capacity_utilization": float(best["capacity_utilization"]),
            "selection_constraints_passed": constraints_passed,
            "selection_role": "fit_oof",
            "calibration_used": False,
        }
    return selected


def evaluate_selection_context_gate(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if sorted(float(row["budget"]) for row in rows) != list(EXPECTED_BUDGETS):
        raise ValueError("Selection-context gate requires every budget exactly once")
    config = protocol["selection_context_gate"]
    gains = [float(row["delta_ap"]) for row in rows]
    gain_points = sum(value >= float(config["minimum_gain"]) for value in gains)
    all_noninferior = all(
        value >= -float(config["maximum_other_loss"]) for value in gains
    )
    capacity_ok = all(bool(row.get("capacity_ok", False)) for row in rows)
    return {
        "passed": bool(
            gain_points >= int(config["required_gain_points"])
            and all_noninferior
            and capacity_ok
        ),
        "gain_points": int(gain_points),
        "all_noninferior": bool(all_noninferior),
        "capacity_ok": bool(capacity_ok),
        "budget_rows": list(rows),
    }


def evaluate_dualheadv2_gate(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    config = protocol["dualheadv2_gate"]
    byte_comparisons = []
    call_comparisons = []
    for budget in EXPECTED_BUDGETS:
        for mode, output in (
            ("fixed_byte", byte_comparisons),
            ("fixed_call", call_comparisons),
        ):
            baseline = _find_row(rows, "ActionValue-OQ", mode, budget)
            candidate = _find_row(rows, "DualHeadV2-OQ", mode, budget)
            output.append(
                {
                    "budget": budget,
                    "delta_ap": float(candidate["official_ap50_95"])
                    - float(baseline["official_ap50_95"]),
                    "capacity_ok": bool(candidate.get("capacity_ok", False)),
                    "capacity_utilization": float(
                        candidate.get("capacity_utilization", 1.0)
                    ),
                    "roi_count": int(candidate.get("roi_count", 0)),
                }
            )
    gain_points = sum(
        row["delta_ap"] >= float(config["minimum_gain"])
        for row in byte_comparisons
    )
    byte_noninferior = all(
        row["delta_ap"] >= -float(config["maximum_other_loss"])
        for row in byte_comparisons
    )
    call_noninferior = all(
        row["delta_ap"] >= -float(config["maximum_fixed_call_loss"])
        for row in call_comparisons
    )
    capacity_ok = all(
        row["capacity_ok"]
        and row["capacity_utilization"]
        >= float(protocol["minimum_capacity_utilization"])
        and row["roi_count"] == 0
        for row in byte_comparisons
    )
    return {
        "passed": bool(
            gain_points >= int(config["required_gain_points"])
            and byte_noninferior
            and call_noninferior
            and capacity_ok
        ),
        "gain_points": int(gain_points),
        "byte_noninferior": bool(byte_noninferior),
        "call_noninferior": bool(call_noninferior),
        "capacity_ok": bool(capacity_ok),
        "fixed_byte": byte_comparisons,
        "fixed_call": call_comparisons,
    }
