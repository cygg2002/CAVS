"""Pure contracts for Original/Q90 advantage calibration."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


METHODS = (
    "Raw-CAVS-OQR",
    "PerAction-Isotonic-OQR",
    "Advantage-Isotonic-Midpoint-OQR",
    "Advantage-Isotonic-QAnchor-OQR",
    "Calibration-Frozen-OQMargin-OQR",
    "Q90-Preferred-Diagnostic",
)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != "oregon_cloudv3_oq_advantage_calibration_diagnostic_v1":
        raise ValueError("protocol ID changed")
    if protocol.get("status") != "posthoc_exploratory_final_reuse":
        raise ValueError("diagnostic status changed")
    if protocol.get("fit_source") != "fit_oof_only":
        raise ValueError("calibrators must be fit from fit OOF only")
    if tuple(map(str, protocol.get("methods", ()))) != METHODS:
        raise ValueError("method matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("budget matrix changed")
    if tuple(map(int, protocol.get("seeds", ()))) != (20260810, 20260811, 20260812):
        raise ValueError("seed matrix changed")
    if int(protocol.get("folds", -1)) != 5:
        raise ValueError("fold count changed")
    if protocol.get("margin_selection_role") != "calibration":
        raise ValueError("Original margin must be selected on calibration")
    if tuple(map(float, protocol.get("margin_quantiles", ()))) != (
        0.0,
        0.25,
        0.5,
        0.65,
        0.75,
        0.85,
        0.9,
        0.95,
        0.99,
        1.0,
    ):
        raise ValueError("Original margin candidate grid changed")
    if protocol.get("final_already_consumed") is not True:
        raise ValueError("final reuse must be disclosed")
    if protocol.get("final_used_for_fit_or_selection") is not False:
        raise ValueError("final cannot fit or select a method")
    if protocol.get("source_modification_allowed") is not False:
        raise ValueError("source experiment must remain read-only")
    if protocol.get("detector_training") is not False or protocol.get("learner_training") is not False:
        raise ValueError("this diagnostic cannot retrain models")


def _score_matrix(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError("scores must be [images, actions] with Edge/Original/Q90")
    if not np.isfinite(matrix[:, :3]).all():
        raise ValueError("Edge/Original/Q90 scores must be finite")
    return matrix


def replace_oq_advantage(
    scores: np.ndarray, calibrated_advantage: np.ndarray, *, anchor: str
) -> np.ndarray:
    """Replace O-Q score differences without changing unrelated actions."""

    matrix = _score_matrix(scores)
    advantage = np.asarray(calibrated_advantage, dtype=np.float64)
    if advantage.shape != (len(matrix),) or not np.isfinite(advantage).all():
        raise ValueError("calibrated advantage must be a finite vector")
    output = matrix.copy()
    if anchor == "q90":
        output[:, 1] = matrix[:, 2] + advantage
    elif anchor == "midpoint":
        midpoint = (matrix[:, 1] + matrix[:, 2]) / 2.0
        output[:, 1] = midpoint + advantage / 2.0
        output[:, 2] = midpoint - advantage / 2.0
    else:
        raise ValueError(f"unknown O/Q anchor: {anchor}")
    return output


def replace_per_action(
    scores: np.ndarray, original: np.ndarray, q90: np.ndarray
) -> np.ndarray:
    matrix = _score_matrix(scores)
    original_values = np.asarray(original, dtype=np.float64)
    q90_values = np.asarray(q90, dtype=np.float64)
    if original_values.shape != (len(matrix),) or q90_values.shape != (len(matrix),):
        raise ValueError("per-action calibrated scores must match image count")
    if not np.isfinite(original_values).all() or not np.isfinite(q90_values).all():
        raise ValueError("per-action calibrated scores must be finite")
    output = matrix.copy()
    output[:, 1] = original_values
    output[:, 2] = q90_values
    return output


def prefer_q90(scores: np.ndarray, *, suppressed_value: float = -1e9) -> np.ndarray:
    """Diagnostic: suppress Original only where Q90 is a valid action."""

    matrix = _score_matrix(scores)
    output = matrix.copy()
    q90_available = matrix[:, 2] > -1e8
    output[q90_available, 1] = float(suppressed_value)
    return output


def apply_original_margin(scores: np.ndarray, *, margin: float) -> np.ndarray:
    """Subtract one frozen safety margin from Original and preserve all else."""

    matrix = _score_matrix(scores)
    if not np.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("Original margin must be finite and non-negative")
    output = matrix.copy()
    q90_available = matrix[:, 2] > -1e8
    output[q90_available, 1] = matrix[q90_available, 1] - float(margin)
    return output


def advantage_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 1 or not len(prediction):
        raise ValueError("advantage metrics require matching non-empty vectors")
    return {
        "mae": float(np.mean(np.abs(prediction - truth))),
        "bias": float(np.mean(prediction - truth)),
        "sign_accuracy": float(np.mean((prediction > 0.0) == (truth > 0.0))),
        "predicted_positive_rate": float(np.mean(prediction > 0.0)),
        "target_positive_rate": float(np.mean(truth > 0.0)),
    }
