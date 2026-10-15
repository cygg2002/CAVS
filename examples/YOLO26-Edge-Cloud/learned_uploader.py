# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight learned upload policy for edge-cloud object detection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from upload_strategies import box_area_ratio, detection_coverage, edge_density, edge_margin_ratio

ROI_FEATURES = [
    "conf",
    "inv_conf",
    "relative_conf",
    "area",
    "log_area",
    "aspect_log",
    "edge_margin",
    "is_small",
    "is_edge",
    "center_x",
    "center_y",
    "coverage_share",
    "image_complexity",
    "det_count",
    "coverage",
]

FRAME_FEATURES = [
    "image_complexity",
    "det_count",
    "coverage",
    "max_conf",
    "mean_conf",
    "min_conf",
    "conf_std",
    "empty",
    "sparse",
    "low_coverage",
    "mean_area",
    "max_area",
    "edge_box_ratio",
    "low_conf_ratio",
    "small_box_ratio",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def image_context(image: np.ndarray, edge_boxes: list[dict[str, Any]]) -> dict[str, float]:
    """Compute shared image-level context features once."""
    h, w = image.shape[:2]
    areas = [box_area_ratio(box, (h, w)) for box in edge_boxes]
    confs = [float(box.get("conf", 0.0)) for box in edge_boxes]
    edge_near = [1.0 if edge_margin_ratio(box, (h, w)) < 0.05 else 0.0 for box in edge_boxes]
    low_conf = [1.0 if float(box.get("conf", 0.0)) < 0.45 else 0.0 for box in edge_boxes]
    small_boxes = [1.0 if area < 0.015 else 0.0 for area in areas]
    coverage = detection_coverage(edge_boxes, (h, w))
    return {
        "image_complexity": edge_density(image),
        "det_count": min(1.0, len(edge_boxes) / 20.0),
        "coverage": coverage,
        "max_conf": max(confs) if confs else 0.0,
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "min_conf": min(confs) if confs else 0.0,
        "conf_std": float(np.std(confs)) if len(confs) > 1 else 0.0,
        "empty": 1.0 if not edge_boxes else 0.0,
        "sparse": 1.0 if len(edge_boxes) <= 2 else 0.0,
        "low_coverage": 1.0 if coverage <= 0.04 else 0.0,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "max_area": max(areas) if areas else 0.0,
        "edge_box_ratio": float(np.mean(edge_near)) if edge_near else 0.0,
        "low_conf_ratio": float(np.mean(low_conf)) if low_conf else 0.0,
        "small_box_ratio": float(np.mean(small_boxes)) if small_boxes else 0.0,
    }


def roi_feature_values(box: dict[str, Any], image_shape: tuple[int, int], context: dict[str, float]) -> dict[str, float]:
    """Extract ROI upload features for one edge detection as a name-value mapping."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box["xyxy"]]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    area = box_area_ratio(box, (h, w))
    margin = edge_margin_ratio(box, (h, w))
    conf = float(box.get("conf", 0.0))
    coverage_share = area / max(context.get("coverage", 0.0), area, 1e-6)
    return {
        "conf": conf,
        "inv_conf": 1.0 - conf,
        "relative_conf": conf - float(context.get("mean_conf", 0.0)),
        "area": area,
        "log_area": math.log(max(area, 1e-6)),
        "aspect_log": math.log(max(bw / bh, 1e-6)),
        "edge_margin": margin,
        "is_small": 1.0 if area < 0.015 else 0.0,
        "is_edge": 1.0 if margin < 0.05 else 0.0,
        "center_x": ((x1 + x2) * 0.5) / max(1.0, float(w)),
        "center_y": ((y1 + y2) * 0.5) / max(1.0, float(h)),
        "coverage_share": min(1.0, max(0.0, coverage_share)),
        "image_complexity": context["image_complexity"],
        "det_count": context["det_count"],
        "coverage": context["coverage"],
    }


def frame_feature_values(context: dict[str, float]) -> dict[str, float]:
    """Extract full-frame upload features as a name-value mapping."""
    return {name: float(context.get(name, 0.0)) for name in FRAME_FEATURES}


def roi_features(
    box: dict[str, Any],
    image_shape: tuple[int, int],
    context: dict[str, float],
    feature_names: list[str] | None = None,
) -> list[float]:
    """Extract ROI upload features in the requested schema order."""
    feature_names = feature_names or ROI_FEATURES
    values = roi_feature_values(box, image_shape, context)
    return [float(values.get(name, 0.0)) for name in feature_names]


def frame_features(context: dict[str, float], feature_names: list[str] | None = None) -> list[float]:
    """Extract full-frame upload features in the requested schema order."""
    feature_names = feature_names or FRAME_FEATURES
    values = frame_feature_values(context)
    return [float(values.get(name, 0.0)) for name in feature_names]


def _split_indices(n: int, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic train/validation split."""
    if n <= 1 or val_split <= 0.0:
        idx = np.arange(n, dtype=np.int32)
        return idx, idx
    rng = np.random.default_rng(seed)
    indices = np.arange(n, dtype=np.int32)
    rng.shuffle(indices)
    val_count = min(max(1, int(round(n * val_split))), max(1, n // 5))
    if val_count >= n:
        val_count = max(1, n - 1)
    return indices[val_count:], indices[:val_count]


def _standardize_train_val(
    x: list[list[float]],
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split and standardize features using train-only statistics."""
    x_arr = np.asarray(x, dtype=np.float64)
    train_idx, val_idx = _split_indices(len(x_arr), val_split, seed)
    x_train = x_arr[train_idx]
    x_val = x_arr[val_idx]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_val - mean) / std, mean, std, train_idx, val_idx


def _binary_metrics(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Return accuracy and Brier score for binary predictions."""
    preds = (probs >= 0.5).astype(np.float64)
    accuracy = float((preds == y_true).mean()) if len(y_true) else 0.0
    brier = float(np.mean((probs - y_true) ** 2)) if len(y_true) else 0.0
    return accuracy, brier


def _huber_grad(error: np.ndarray, delta: float) -> np.ndarray:
    """Huber gradient with respect to prediction error."""
    abs_error = np.abs(error)
    return np.where(abs_error <= delta, error, delta * np.sign(error))


def fit_logistic(
    x: list[list[float]],
    y: list[int],
    epochs: int = 800,
    lr: float = 0.08,
    l2: float = 0.01,
    val_split: float = 0.1,
    feature_noise_std: float = 0.0,
    class_balance: bool = False,
    seed: int = 42,
    patience: int = 80,
) -> dict[str, Any]:
    """Fit a standardized logistic model with optional validation split and noise regularization."""
    if not x:
        raise ValueError("No training samples")
    y_arr = np.asarray(y, dtype=np.float64)
    x_train, x_val, mean, std, train_idx, val_idx = _standardize_train_val(x, val_split, seed)
    y_train = y_arr[train_idx]
    y_val = y_arr[val_idx]
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = float(np.log((y_train.mean() + 1e-3) / (1.0 - y_train.mean() + 1e-3)))
    best = None
    best_loss = float("inf")
    stalled = 0
    rng = np.random.default_rng(seed)
    pos_weight = 1.0
    if class_balance:
        positives = max(1.0, float(y_train.sum()))
        negatives = max(1.0, float(len(y_train) - y_train.sum()))
        pos_weight = negatives / positives
    for _ in range(max(1, epochs)):
        x_epoch = x_train
        if feature_noise_std > 0.0:
            x_epoch = x_train + rng.normal(0.0, feature_noise_std, x_train.shape)
        logits = x_epoch @ weights + bias
        probs = sigmoid(logits)
        error = probs - y_train
        if class_balance:
            sample_weight = np.where(y_train > 0.5, pos_weight, 1.0)
            error = error * sample_weight
            denom = max(1.0, float(sample_weight.sum()))
        else:
            denom = max(1.0, float(len(y_train)))
        grad_w = (x_epoch.T @ error) / denom + l2 * weights
        grad_b = float(error.sum() / denom)
        weights -= lr * grad_w
        bias -= lr * grad_b
        val_logits = x_val @ weights + bias
        val_probs = sigmoid(val_logits)
        eps = 1e-8
        val_loss = float(
            -np.mean(y_val * np.log(val_probs + eps) + (1.0 - y_val) * np.log(1.0 - val_probs + eps))
            + 0.5 * l2 * np.sum(weights**2)
        )
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best = (weights.copy(), bias)
            stalled = 0
        else:
            stalled += 1
            if stalled >= max(10, patience):
                break
    if best is not None:
        weights, bias = best
    train_probs = sigmoid(x_train @ weights + bias)
    val_probs = sigmoid(x_val @ weights + bias)
    train_accuracy, train_brier = _binary_metrics(train_probs, y_train)
    val_accuracy, val_brier = _binary_metrics(val_probs, y_val)
    return {
        "kind": "binary_logistic",
        "weights": weights.tolist(),
        "bias": bias,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "samples": int(len(y_arr)),
        "positive_rate": float(y_arr.mean()),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "train_brier": train_brier,
        "val_brier": val_brier,
        "val_samples": int(len(y_val)),
        "feature_noise_std": float(feature_noise_std),
        "class_balance": bool(class_balance),
    }


def fit_value_regressor(
    x: list[list[float]],
    y: list[float],
    epochs: int = 500,
    lr: float = 0.05,
    l2: float = 0.02,
    val_split: float = 0.1,
    feature_noise_std: float = 0.0,
    seed: int = 42,
    patience: int = 80,
    huber_delta: float = 0.15,
) -> dict[str, Any]:
    """Fit a lightweight robust linear regressor for normalized utility gain."""
    if not x:
        raise ValueError("No training samples")
    y_arr = np.asarray(y, dtype=np.float64)
    x_train, x_val, mean, std, train_idx, val_idx = _standardize_train_val(x, val_split, seed)
    y_train = y_arr[train_idx]
    y_val = y_arr[val_idx]
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = float(y_train.mean()) if len(y_train) else 0.0
    best = None
    best_loss = float("inf")
    stalled = 0
    rng = np.random.default_rng(seed)
    for _ in range(max(1, epochs)):
        x_epoch = x_train
        if feature_noise_std > 0.0:
            x_epoch = x_train + rng.normal(0.0, feature_noise_std, x_train.shape)
        preds = x_epoch @ weights + bias
        error = preds - y_train
        grad = _huber_grad(error, huber_delta)
        denom = max(1.0, float(len(y_train)))
        grad_w = (x_epoch.T @ grad) / denom + l2 * weights
        grad_b = float(grad.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b

        val_preds = x_val @ weights + bias
        val_error = val_preds - y_val
        val_loss = float(
            np.mean(
                np.where(
                    np.abs(val_error) <= huber_delta,
                    0.5 * (val_error**2),
                    huber_delta * (np.abs(val_error) - 0.5 * huber_delta),
                )
            )
            + 0.5 * l2 * np.sum(weights**2)
        )
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best = (weights.copy(), bias)
            stalled = 0
        else:
            stalled += 1
            if stalled >= max(10, patience):
                break
    if best is not None:
        weights, bias = best
    train_preds = x_train @ weights + bias
    val_preds = x_val @ weights + bias
    train_mae = float(np.mean(np.abs(train_preds - y_train))) if len(y_train) else 0.0
    val_mae = float(np.mean(np.abs(val_preds - y_val))) if len(y_val) else 0.0
    return {
        "kind": "linear_value",
        "weights": weights.tolist(),
        "bias": bias,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "samples": int(len(y_arr)),
        "target_mean": float(y_arr.mean()),
        "target_abs_mean": float(np.mean(np.abs(y_arr))) if len(y_arr) else 0.0,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "val_samples": int(len(y_val)),
        "feature_noise_std": float(feature_noise_std),
        "huber_delta": float(huber_delta),
    }


def predict(model: dict[str, Any], features: list[float]) -> float:
    """Predict either utility probability or normalized gain from a learned head."""
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    value = ((x - mean) / std) @ weights + float(model["bias"])
    kind = str(model.get("kind", "binary_logistic"))
    if kind in {"linear_value", "regression", "value"}:
        return float(value)
    return float(sigmoid(value))


def decision_score(probability: float, estimated_kb: float, cost_weight: float, expected_gain: float | None = None) -> float:
    """Return resource-aware decision score."""
    gain_scale = 1.0 if expected_gain is None else max(0.0, float(expected_gain))
    return float(probability * gain_scale - cost_weight * (estimated_kb / 100.0))


def policy_decision(
    policy: dict[str, Any],
    branch: str,
    features: list[float],
    estimated_kb: float,
    cost_weight: float,
) -> dict[str, float]:
    """Run a policy branch and return probability, expected gain, and final decision score."""
    prob = predict(policy[f"{branch}_model"], features)
    value_model = policy.get(f"{branch}_value_model")
    expected_gain = max(0.0, predict(value_model, features)) if value_model is not None else 1.0
    return {
        "probability": float(prob),
        "expected_gain": float(expected_gain),
        "decision": float(decision_score(prob, estimated_kb, cost_weight, expected_gain)),
    }


def save_policy(
    path: str | Path,
    roi_model: dict[str, Any],
    frame_model: dict[str, Any],
    metadata: dict[str, Any],
    roi_value_model: dict[str, Any] | None = None,
    frame_value_model: dict[str, Any] | None = None,
) -> None:
    """Save a learned upload policy as JSON."""
    payload = {
        "version": 2 if roi_value_model is not None or frame_value_model is not None else 1,
        "roi_features": ROI_FEATURES,
        "frame_features": FRAME_FEATURES,
        "roi_model": roi_model,
        "frame_model": frame_model,
        "roi_value_model": roi_value_model,
        "frame_value_model": frame_value_model,
        "metadata": metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_policy(path: str | Path) -> dict[str, Any]:
    """Load a learned upload policy JSON."""
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    policy.setdefault("roi_features", ROI_FEATURES)
    policy.setdefault("frame_features", FRAME_FEATURES)
    return policy
