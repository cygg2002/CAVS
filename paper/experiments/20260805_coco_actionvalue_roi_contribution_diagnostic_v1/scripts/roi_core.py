"""Pure policy and decision contracts for the ROI contribution diagnostic."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


METHODS = (
    "ActionValue-OQ",
    "ActionValue-OQR",
    "ActionValue-OQR-OracleRelativeGate",
)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("roles") != {"fit_oof": 9678, "calibration": 571}:
        raise ValueError("Canonical role populations changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("Frozen budgets changed")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError("Frozen method matrix changed")
    if protocol.get("oracle_uses_contextual_reward") is not True:
        raise ValueError("Oracle-relative semantics changed")
    if protocol.get("calibration_used_for_selection") is not False:
        raise ValueError("Calibration cannot select the policy")
    for field in ("audit_accessed", "final_accessed", "registered_final_accessed"):
        if protocol.get(field) is not False:
            raise ValueError(f"{field} must remain false")


def apply_action_policy(
    scores: Mapping[int, Mapping[str, float]],
    rewards: Mapping[int, Mapping[str, float]],
    method: str,
) -> dict[int, dict[str, float]]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    output: dict[int, dict[str, float]] = {}
    for raw_image_id, raw_scores in scores.items():
        image_id = int(raw_image_id)
        values = {str(action): float(value) for action, value in raw_scores.items()}
        if method == "ActionValue-OQ":
            values.pop("roi", None)
        elif method == "ActionValue-OQR-OracleRelativeGate" and "roi" in values:
            truth = rewards[image_id]
            alternatives = [
                float(truth[action])
                for action in ("original", "q90")
                if action in values and action in truth
            ]
            roi_reward = truth.get("roi")
            if roi_reward is None or not alternatives or float(roi_reward) <= max(0.0, *alternatives):
                values.pop("roi", None)
        if not any(action in values for action in ("original", "q90", "roi")):
            raise ValueError(f"No cloud action remains for image {image_id}")
        output[image_id] = values
    return output


def build_action_score_map(
    records: Sequence[Mapping[str, Any]], value_scores: np.ndarray
) -> dict[int, dict[str, float]]:
    scores = np.asarray(value_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != len(records):
        raise ValueError("Score matrix does not match the record population")
    output: dict[int, dict[str, float]] = {}
    for row_index, record in enumerate(records):
        actions = record["actions"]
        if len(actions) > scores.shape[1]:
            raise ValueError("Score matrix has fewer slots than the action record")
        output[int(record["image_id"])] = {
            str(action["action_id"]): float(scores[row_index, action_index])
            for action_index, action in enumerate(actions)
            if str(action["action_id"]) != "edge"
        }
    return output


def roi_risk_rates(
    selection: Mapping[int, str],
    rewards: Mapping[int, Mapping[str, float]],
) -> dict[str, float | int]:
    roi_image_ids = [
        int(image_id) for image_id, action in selection.items() if action == "roi"
    ]
    if not roi_image_ids:
        return {
            "roi_count": 0,
            "harmful_roi_rate": 0.0,
            "relative_suboptimal_roi_rate": 0.0,
        }
    harmful = sum(float(rewards[image_id]["roi"]) <= 0.0 for image_id in roi_image_ids)
    relative_suboptimal = 0
    for image_id in roi_image_ids:
        truth = rewards[image_id]
        alternatives = [
            0.0,
            *(float(truth[action]) for action in ("original", "q90") if action in truth),
        ]
        relative_suboptimal += float(truth["roi"]) <= max(alternatives)
    return {
        "roi_count": len(roi_image_ids),
        "harmful_roi_rate": harmful / len(roi_image_ids),
        "relative_suboptimal_roi_rate": relative_suboptimal / len(roi_image_ids),
    }


def evaluate_gain_gate(
    deltas: Sequence[float], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    values = list(map(float, deltas))
    if len(values) != len(protocol["budgets"]):
        raise ValueError("Gain vector must match the frozen budgets")
    minimum = float(protocol["minimum_ap_gain"])
    maximum_loss = float(protocol["maximum_other_loss"])
    required = int(protocol["required_gain_points"])
    gain_points = sum(value >= minimum for value in values)
    passed = gain_points >= required and all(value >= -maximum_loss for value in values)
    return {
        "passed": bool(passed),
        "gain_points": int(gain_points),
        "deltas": values,
    }
