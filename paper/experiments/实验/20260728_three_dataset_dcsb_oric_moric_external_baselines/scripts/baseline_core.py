"""Pure method contracts for DCSB and ORIC/MORIC reimplementations."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class DetectionStats:
    """Per-image arrays required by the ORIC mAP calculation."""

    correct: np.ndarray
    confidence: np.ndarray
    predicted_class: np.ndarray
    target_class: np.ndarray


def expected_result_rows(protocol: dict[str, Any]) -> int:
    work_points = sum(len(values) for values in protocol["work_points"].values())
    continuous = len(protocol["continuous_methods"])
    payloads = len(protocol["payload_policies"])
    native = len(protocol["native_methods"]) * payloads
    systems = len(protocol["system_references"])
    return continuous * payloads * work_points + native + systems


def _result_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["method"]),
        str(row["payload_policy"]),
        str(row["mode"]),
        float(row["nominal_budget"]),
    )


def assert_result_matrix_complete(
    rows: Sequence[dict[str, Any]], protocol: dict[str, Any]
) -> None:
    expected = {
        (method, payload, mode, float(budget))
        for method in protocol["continuous_methods"]
        for payload in protocol["payload_policies"]
        for mode, budgets in protocol["work_points"].items()
        for budget in budgets
    }
    expected.update(
        (method, payload, "native_rule", -1.0)
        for method in protocol["native_methods"]
        for payload in protocol["payload_policies"]
    )
    expected.update(
        (method, "system", "system", 0.0)
        for method in protocol["system_references"]
    )
    observed = [_result_key(row) for row in rows]
    unique = set(observed)
    if len(unique) != len(observed):
        raise RuntimeError("Result matrix contains duplicate rows")
    missing = sorted(expected - unique)
    unexpected = sorted(unique - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Result matrix is incomplete: missing={missing}, unexpected={unexpected}"
        )


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("final_used_for_selection") is not False:
        raise ValueError("Final data must remain evaluation-only")
    if protocol.get("calibration_used_for_model_training") is not False:
        raise ValueError("Calibration labels cannot train an external baseline")
    if protocol.get("calibration_teacher_gain_used_for_fixed_byte_lambda") is not True:
        raise ValueError("Fixed-byte lambda must preserve the frozen calibration objective")
    if protocol.get("fit_only_training") is not True:
        raise ValueError("External baseline fitting must be fit-only")
    if tuple(protocol["datasets"]) != ("coco", "ena24", "oregon"):
        raise ValueError("The frozen protocol requires COCO, ENA24, and Oregon")
    if tuple(protocol["payload_policies"]) != ("Original", "SharedAction"):
        raise ValueError("Payload-policy semantics changed")
    if "MORIC-style" in protocol["continuous_methods"]:
        raise ValueError("The internal MORIC-style proxy is not an external baseline")
    if sum(len(values) for values in protocol["work_points"].values()) != 10:
        raise ValueError("The paper protocol requires ten registered work points")
    observed = expected_result_rows(protocol)
    if int(protocol["expected_rows_per_dataset"]) != observed:
        raise ValueError(f"Expected row count drifted: {observed}")


def _score(prediction: dict[str, Any]) -> float:
    return float(prediction.get("score", prediction.get("conf", 0.0)))


def build_output_feature(
    predictions: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
    category_ids: Sequence[int],
    top_k: int,
) -> np.ndarray:
    """Build the class-count plus top-K feature used by the ORIC paper."""
    if width <= 0 or height <= 0 or top_k <= 0:
        raise ValueError("Image dimensions and top_k must be positive")
    categories = tuple(map(int, category_ids))
    if len(categories) != len(set(categories)):
        raise ValueError("category_ids must be unique")
    category_index = {category_id: index for index, category_id in enumerate(categories)}
    ordered = sorted(
        enumerate(predictions),
        key=lambda item: (-_score(item[1]), item[0]),
    )[: int(top_k)]
    feature = np.zeros(len(categories) + 5 * int(top_k), dtype=np.float64)
    for rank, (_, prediction) in enumerate(ordered):
        category_id = int(prediction["category_id"])
        if category_id not in category_index:
            raise ValueError(f"Prediction category {category_id} is not in the GT schema")
        feature[category_index[category_id]] += 1.0
        x, y, box_width, box_height = map(float, prediction["bbox"])
        offset = len(categories) + rank * 5
        feature[offset : offset + 5] = (
            (x + box_width / 2.0) / float(width),
            (y + box_height / 2.0) / float(height),
            box_width / float(width),
            box_height / float(height),
            _score(prediction),
        )
    if not np.isfinite(feature).all():
        raise ValueError("ORIC output features must be finite")
    return feature


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Right-continuous empirical CDF; equal rewards retain equal targets."""
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    if ordered.ndim != 1 or not len(ordered):
        raise ValueError("CDF reference must be a non-empty vector")
    return np.searchsorted(ordered, np.asarray(values), side="right") / len(ordered)


def weighted_mse(
    predictions: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> float:
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape != weight.shape:
        raise ValueError("Weighted MSE inputs must have matching shapes")
    return float(np.mean(weight * np.square(prediction - target)))


def context_indices(
    total: int,
    target_index: int,
    *,
    context_size: int,
    seed: int,
) -> list[int]:
    if not 0 <= int(target_index) < int(total):
        raise ValueError("target_index is out of range")
    candidates = [index for index in range(int(total)) if index != int(target_index)]
    count = min(max(0, int(context_size)), len(candidates))
    rng = random.Random(int(seed) + int(target_index) * 1_000_003)
    return sorted(rng.sample(candidates, count))


def _detection_signal(
    predictions: Sequence[dict[str, Any]],
    width: int,
    height: int,
    confidence: float,
) -> tuple[int, float]:
    selected = [prediction for prediction in predictions if _score(prediction) > float(confidence)]
    if not selected:
        return 0, 0.0
    image_area = float(max(1, int(width)) * max(1, int(height)))
    areas = [
        max(0.0, float(prediction["bbox"][2]))
        * max(0.0, float(prediction["bbox"][3]))
        / image_area
        for prediction in selected
    ]
    return len(selected), min(areas)


def _fit_confidence_threshold(
    image_ids: Sequence[int],
    edge_predictions: dict[int, list[dict[str, Any]]],
    ground_truth_counts: dict[int, int],
) -> tuple[float, int]:
    all_scores = np.sort(
        np.asarray(
            [
                _score(prediction)
                for image_id in image_ids
                for prediction in edge_predictions.get(int(image_id), [])
            ],
            dtype=np.float64,
        )
    )
    candidates = [0.0, *np.unique(all_scores).tolist(), 1.0]
    truth_total = sum(int(ground_truth_counts[int(image_id)]) for image_id in image_ids)
    evaluated = []
    for threshold in candidates:
        predicted = len(all_scores) - int(
            np.searchsorted(all_scores, float(threshold), side="right")
        )
        evaluated.append((abs(predicted - truth_total), -float(threshold), threshold, predicted))
    _, _, threshold, predicted_total = min(evaluated)
    return float(threshold), int(predicted_total)


def fit_dcsb(
    *,
    image_ids: Sequence[int],
    edge_predictions: dict[int, list[dict[str, Any]]],
    cloud_predictions: dict[int, list[dict[str, Any]]],
    image_sizes: dict[int, tuple[int, int]],
    ground_truth_counts: dict[int, int],
    detector_confidence: float,
    count_thresholds: Sequence[int],
    area_thresholds: Sequence[float],
) -> dict[str, Any]:
    """Fit the three DCSB thresholds using only the fit role."""
    ids = tuple(map(int, image_ids))
    if not ids:
        raise ValueError("DCSB requires at least one fit image")
    confidence_threshold, estimated_total = _fit_confidence_threshold(
        ids, edge_predictions, ground_truth_counts
    )
    records = []
    for image_id in ids:
        width, height = image_sizes[image_id]
        edge = edge_predictions.get(image_id, [])
        cloud = cloud_predictions.get(image_id, [])
        estimated_count, estimated_area = _detection_signal(
            edge, width, height, confidence_threshold
        )
        detected_count, _ = _detection_signal(edge, width, height, detector_confidence)
        cloud_count, _ = _detection_signal(cloud, width, height, detector_confidence)
        records.append(
            {
                "estimated_count": estimated_count,
                "estimated_area": estimated_area,
                "detected_count": detected_count,
                "label": int(cloud_count > detected_count),
            }
        )
    best: tuple[float, int, float] | None = None
    for count_threshold in map(int, count_thresholds):
        for area_threshold in map(float, area_thresholds):
            decisions = [
                int(
                    record["estimated_count"] != record["detected_count"]
                    and (
                        record["estimated_count"] > count_threshold
                        or record["estimated_area"] < area_threshold
                    )
                )
                for record in records
            ]
            accuracy = float(
                np.mean(
                    [
                        decision == int(record["label"])
                        for decision, record in zip(decisions, records)
                    ]
                )
            )
            candidate = (accuracy, count_threshold, area_threshold)
            if best is None or accuracy > best[0]:
                best = candidate
    if best is None:
        raise ValueError("DCSB threshold grids cannot be empty")
    accuracy, count_threshold, area_threshold = best
    positives = sum(int(record["label"]) for record in records)
    return {
        "confidence_threshold": confidence_threshold,
        "count_threshold": int(count_threshold),
        "area_threshold": float(area_threshold),
        "detector_confidence": float(detector_confidence),
        "fit_accuracy": float(accuracy),
        "fit_images": len(ids),
        "positive_labels": positives,
        "positive_label_rate": positives / len(ids),
        "ground_truth_count_total": sum(ground_truth_counts[image_id] for image_id in ids),
        "estimated_count_total": estimated_total,
    }


def dcsb_decision_and_score(
    predictions: Sequence[dict[str, Any]],
    width: int,
    height: int,
    model: dict[str, Any],
) -> tuple[int, float]:
    estimated_count, minimum_area = _detection_signal(
        predictions, width, height, float(model["confidence_threshold"])
    )
    detected_count, _ = _detection_signal(
        predictions, width, height, float(model["detector_confidence"])
    )
    if estimated_count == detected_count:
        return 0, -1.0
    count_threshold = float(model["count_threshold"])
    area_threshold = float(model["area_threshold"])
    count_margin = (estimated_count - count_threshold) / max(1.0, count_threshold)
    area_margin = (area_threshold - minimum_area) / max(1e-12, area_threshold)
    decision = int(estimated_count > count_threshold or minimum_area < area_threshold)
    return decision, float(max(count_margin, area_margin))


def _iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0.0 else 0.0


def detection_stats(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    iou_thresholds: Sequence[float],
) -> DetectionStats:
    """Match one image using the public ORIC implementation's greedy contract."""
    thresholds = np.asarray(iou_thresholds, dtype=np.float64)
    ordered = sorted(predictions, key=_score, reverse=True)
    correct = np.zeros((len(ordered), len(thresholds)), dtype=bool)
    for column, threshold in enumerate(thresholds):
        matches = []
        for gt_index, target in enumerate(ground_truth):
            for pred_index, prediction in enumerate(ordered):
                if int(target["category_id"]) != int(prediction["category_id"]):
                    continue
                overlap = _iou_xywh(target["bbox"], prediction["bbox"])
                if overlap >= float(threshold):
                    matches.append((gt_index, pred_index, overlap))
        if matches:
            matched = np.asarray(matches, dtype=np.float64)
            matched = matched[matched[:, 2].argsort()[::-1]]
            if len(matched) > 1:
                matched = matched[
                    np.unique(matched[:, 1], return_index=True)[1]
                ]
                matched = matched[
                    np.unique(matched[:, 0], return_index=True)[1]
                ]
            correct[matched[:, 1].astype(int), column] = True
    return DetectionStats(
        correct=correct,
        confidence=np.asarray([_score(prediction) for prediction in ordered]),
        predicted_class=np.asarray(
            [int(prediction["category_id"]) for prediction in ordered], dtype=np.int64
        ),
        target_class=np.asarray(
            [int(target["category_id"]) for target in ground_truth], dtype=np.int64
        ),
    )


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    values = np.interp(x, mrec, mpre)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, x))


def ap_per_class(
    correct: np.ndarray,
    confidence: np.ndarray,
    predicted_class: np.ndarray,
    target_class: np.ndarray,
) -> np.ndarray:
    correct = np.asarray(correct, dtype=bool)
    if correct.ndim != 2:
        raise ValueError("correct must be a prediction-by-IoU matrix")
    confidence = np.asarray(confidence, dtype=np.float64)
    predicted_class = np.asarray(predicted_class, dtype=np.int64)
    target_class = np.asarray(target_class, dtype=np.int64)
    classes, counts = np.unique(target_class, return_counts=True)
    output = np.zeros((len(classes), correct.shape[1]), dtype=np.float64)
    if not len(confidence):
        return output
    order = np.argsort(-confidence, kind="mergesort")
    correct = correct[order]
    predicted_class = predicted_class[order]
    for class_index, (category_id, target_count) in enumerate(zip(classes, counts)):
        mask = predicted_class == int(category_id)
        if not mask.any() or not target_count:
            continue
        false_positive = (~correct[mask]).cumsum(axis=0)
        true_positive = correct[mask].cumsum(axis=0)
        recall = true_positive / max(1, int(target_count))
        precision = true_positive / np.maximum(1, true_positive + false_positive)
        for column in range(correct.shape[1]):
            output[class_index, column] = _compute_ap(
                recall[:, column], precision[:, column]
            )
    return output


def _merge_stats(items: Sequence[DetectionStats]) -> DetectionStats:
    if not items:
        raise ValueError("At least one detection-stat item is required")
    columns = items[0].correct.shape[1]
    if any(item.correct.shape[1] != columns for item in items):
        raise ValueError("Detection-stat IoU dimensions differ")
    return DetectionStats(
        correct=np.concatenate([item.correct for item in items], axis=0),
        confidence=np.concatenate([item.confidence for item in items]),
        predicted_class=np.concatenate([item.predicted_class for item in items]),
        target_class=np.concatenate([item.target_class for item in items]),
    )


def _mean_ap(stats: DetectionStats) -> float:
    values = ap_per_class(
        stats.correct,
        stats.confidence,
        stats.predicted_class,
        stats.target_class,
    )
    return float(values.mean()) if values.size else 0.0


def oric_reward(
    context_edge: Sequence[DetectionStats],
    target_edge: DetectionStats,
    target_cloud: DetectionStats,
) -> dict[str, float]:
    edge_map = _mean_ap(_merge_stats([*context_edge, target_edge]))
    cloud_map = _mean_ap(_merge_stats([*context_edge, target_cloud]))
    scale = len(context_edge) + 1
    return {
        "edge_map": edge_map,
        "cloud_map": cloud_map,
        "reward": float(scale * (cloud_map - edge_map)),
        "context_images": len(context_edge),
    }


def _build_torch_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> Any:
    import torch

    layers: list[Any] = []
    previous = int(input_dim)
    for width in map(int, hidden_dims):
        linear = torch.nn.Linear(previous, width)
        torch.nn.init.kaiming_uniform_(linear.weight)
        layers.extend(
            [
                linear,
                torch.nn.BatchNorm1d(width),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(dropout)),
            ]
        )
        previous = width
    output = torch.nn.Linear(previous, 1)
    torch.nn.init.kaiming_uniform_(output.weight)
    layers.append(output)
    return torch.nn.Sequential(*layers)


def fit_reward_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Select an epoch on fit-internal data, then retrain on all fit samples."""
    import torch

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32)
    if x.ndim != 2 or y.shape != (len(x),) or weights.shape != (len(x),):
        raise ValueError("MLP feature, target, or weight shape mismatch")
    if len(x) < 4:
        raise ValueError("MLP fitting requires at least four samples")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    validation_count = max(1, int(round(len(x) * float(config["fit_validation_fraction"]))))
    validation_count = min(validation_count, len(x) - 2)
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]

    def make_model(model_seed: int) -> Any:
        torch.manual_seed(model_seed)
        return _build_torch_mlp(x.shape[1], config["hidden_dims"], config["dropout"])

    model = make_model(seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(map(int, config["milestones"])),
        gamma=float(config["gamma"]),
    )
    batch_size = int(config["batch_size"])
    best_epoch = 1
    best_loss = math.inf
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x[training_indices]),
        torch.from_numpy(y[training_indices]),
        torch.from_numpy(weights[training_indices]),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    validation_x = torch.from_numpy(x[validation_indices])
    validation_y = torch.from_numpy(y[validation_indices])
    validation_weight = torch.from_numpy(weights[validation_indices])
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for batch_x, batch_y, batch_weight in loader:
            prediction = model(batch_x).squeeze(1)
            loss = (batch_weight * (prediction - batch_y).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            prediction = model(validation_x).squeeze(1)
            validation_loss = (
                validation_weight * (prediction - validation_y).square()
            ).mean()
        if float(validation_loss) < best_loss:
            best_loss = float(validation_loss)
            best_epoch = epoch

    final_model = make_model(seed + 1)
    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(map(int, config["milestones"])),
        gamma=float(config["gamma"]),
    )
    all_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weights)
    )
    all_loader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    for _ in range(best_epoch):
        final_model.train()
        for batch_x, batch_y, batch_weight in all_loader:
            prediction = final_model(batch_x).squeeze(1)
            loss = (batch_weight * (prediction - batch_y).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return {
        "state_dict": final_model.state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dims": list(map(int, config["hidden_dims"])),
        "dropout": float(config["dropout"]),
        "selected_epoch": int(best_epoch),
        "fit_validation_loss": float(best_loss),
        "seed": seed,
    }


def predict_reward_mlp(checkpoint: dict[str, Any], features: np.ndarray) -> np.ndarray:
    import torch

    model = _build_torch_mlp(
        int(checkpoint["input_dim"]),
        checkpoint["hidden_dims"],
        float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        values = model(torch.from_numpy(np.asarray(features, dtype=np.float32)))
    return values.squeeze(1).numpy().astype(np.float64)


def area_threshold_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(math.ceil((float(stop) - float(start)) / float(step)))
    return [float(start) + index * float(step) for index in range(count)]
