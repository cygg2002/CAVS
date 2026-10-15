"""Pure contracts shared by Cloud-V3 training, evaluation, and reporting."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SCALAR_METHODS = (
    "Random",
    "Confidence",
    "ClassGainPrior",
    "M2E",
    "Confidence-Residual M2E",
    "SemanticRank",
    "MORIC-style",
    "GEC-style",
    "Current-Controlled-32D",
    "ORIC-AP50",
    "MORIC-AP50",
    "ORIC-AP5095",
    "MORIC-AP5095",
    "Current-BudgetSet",
    "SetValue",
    "DualHead",
)

JOINT_METHODS = (
    "Raw-ActionValue-OQ",
    "PerAction-Isotonic-OQ",
    "ValueRisk-LCB",
    "GlobalRank",
    "RelSetValue",
    "RelActionRank",
    "FactorizedGlobalActionRank",
    "CurrentActionValue-OQ",
    "CurrentActionValue-OQR12",
    "AFPD-OQ",
    "AFPD-OQR12",
    "OccupancyFree-AFPD-OQR12",
    "OccupancyAware-AFPD-OQR12",
)


def result_key(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["method"]),
        str(row["policy"]),
        str(row["mode"]),
        float(row["budget"]),
    )


def evaluation_matrix(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in SCALAR_METHODS:
        for policy in protocol["scalar_action_policies"]:
            for mode in protocol["modes"]:
                for budget in protocol["budgets"]:
                    rows.append(
                        {
                            "method": method,
                            "policy": str(policy),
                            "mode": str(mode),
                            "budget": float(budget),
                        }
                    )
    for method in JOINT_METHODS:
        for mode in protocol["modes"]:
            for budget in protocol["budgets"]:
                rows.append(
                    {
                        "method": method,
                        "policy": "Joint",
                        "mode": str(mode),
                        "budget": float(budget),
                    }
                )
    keys = [result_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("evaluation matrix contains duplicate rows")
    return rows


def training_role_matrix(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    folds: list[int | str] = [*range(int(protocol["fold_count"])), "full"]
    for method in (*protocol["external_rankers"], *(
        method
        for method in protocol["internal_rankers"]
        if method != "PerAction-Isotonic-OQ"
    )):
        for seed in protocol["model_seeds"]:
            for fold in folds:
                rows.append(
                    {
                        "method": str(method),
                        "seed": int(seed),
                        "fold": fold,
                        "training_role": "fit",
                        "prediction_roles": ["calibration", "final"]
                        if fold == "full"
                        else ["fit_oof"],
                        "prediction_role": "fit_oof" if fold != "full" else None,
                    }
                )
    return rows


def action_score_map(
    records: Sequence[Mapping[str, Any]], values: np.ndarray
) -> dict[int, dict[str, float]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("action score matrix shape changed")
    output: dict[int, dict[str, float]] = {}
    for row, record in enumerate(records):
        actions = list(record["actions"])
        if len(actions) > matrix.shape[1]:
            raise ValueError("score matrix has fewer columns than actions")
        image_id = int(record["image_id"])
        scores = {}
        for column, action in enumerate(actions):
            action_id = str(action["action_id"])
            if action_id == "edge":
                continue
            value = float(matrix[row, column])
            if math.isfinite(value):
                scores[action_id] = value
        if not scores:
            raise ValueError(f"no cloud score for image {image_id}")
        output[image_id] = scores
    return output


def gate_roi_scores(
    scores: Mapping[int, Mapping[str, float]],
    probabilities: Mapping[int, Mapping[str, float]],
    *,
    threshold: float,
) -> dict[int, dict[str, float]]:
    output = {}
    for image_id, actions in scores.items():
        values = {}
        action_probability = probabilities.get(int(image_id), {})
        for action_id, value in actions.items():
            if "@" in str(action_id) and float(
                action_probability.get(str(action_id), 0.0)
            ) < float(threshold):
                values[str(action_id)] = -1e9
            else:
                values[str(action_id)] = float(value)
        output[int(image_id)] = values
    return output


def validate_selection(
    samples: Sequence[Mapping[str, Any]],
    selection: Mapping[int, str],
    *,
    mode: str,
    budget: float,
) -> dict[str, float]:
    sample_by_id = {int(sample["image_id"]): sample for sample in samples}
    if not set(map(int, selection)) <= set(sample_by_id):
        raise ValueError("selection contains unknown images")
    requested = 0.0
    for image_id, action_id in selection.items():
        sample = sample_by_id[int(image_id)]
        if not bool(sample.get("available", {}).get(str(action_id), True)):
            raise ValueError(f"unavailable action selected: {image_id}/{action_id}")
        requested += float(sample["payload_bytes"][str(action_id)])
    original_total = sum(
        float(sample["payload_bytes"]["original"]) for sample in samples
    )
    if mode == "fixed_byte":
        capacity = float(budget) * original_total
        if requested > capacity + max(1e-6, capacity * 1e-12):
            raise ValueError("fixed-byte capacity exceeded")
        utilization = requested / max(1.0, capacity)
    elif mode == "fixed_call":
        expected = int(math.floor(len(samples) * float(budget) + 0.5))
        if len(selection) != expected:
            raise ValueError("fixed-call selection count changed")
        capacity = float(expected)
        utilization = len(selection) / max(1, expected)
    else:
        raise ValueError(mode)
    return {
        "request_bytes": requested,
        "capacity": capacity,
        "utilization": utilization,
    }
