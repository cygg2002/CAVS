"""Source-oriented DCSB and Adaptive Feeding baseline primitives.

The functions in this module are deliberately independent of CAVS.  Both
baselines can only choose between the edge result and an Original upload.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _score(prediction: Mapping[str, Any]) -> float:
    return float(prediction.get("score", prediction.get("conf", 0.0)))


def _area_ratio(box: Sequence[float], width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return max(0.0, float(box[2])) * max(0.0, float(box[3])) / float(width * height)


def adaptive_feeding_feature(
    predictions: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    category_ids: Sequence[int],
    top_k: int,
) -> np.ndarray:
    """Return the paper's C + (confidence + xmin,ymin,w,h) x K feature.

    Coordinates are normalized to [approximately] zero-to-one.  Stable source
    order resolves tied scores, and missing proposals remain zero padded.
    """
    if width <= 0 or height <= 0 or top_k <= 0:
        raise ValueError("width, height, and top_k must be positive")
    categories = tuple(map(int, category_ids))
    if len(categories) != len(set(categories)):
        raise ValueError("category_ids must be unique")
    category_index = {category_id: index for index, category_id in enumerate(categories)}
    ordered = sorted(
        enumerate(predictions), key=lambda item: (-_score(item[1]), item[0])
    )[: int(top_k)]
    feature = np.zeros(len(categories) + 5 * int(top_k), dtype=np.float64)
    for rank, (_, prediction) in enumerate(ordered):
        category_id = int(prediction["category_id"])
        if category_id not in category_index:
            raise ValueError(f"prediction category {category_id} is outside the schema")
        feature[category_index[category_id]] += 1.0
        x, y, box_width, box_height = map(float, prediction["bbox"])
        offset = len(categories) + 5 * rank
        feature[offset : offset + 5] = (
            _score(prediction),
            x / float(width),
            y / float(height),
            box_width / float(width),
            box_height / float(height),
        )
    if not np.isfinite(feature).all():
        raise ValueError("Adaptive Feeding feature contains non-finite values")
    return feature


def _iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0.0 else 0.0


def _voc07_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    return float(
        sum(
            (float(np.max(precision[recall >= threshold])) if np.any(recall >= threshold) else 0.0)
            / 11.0
            for threshold in np.arange(0.0, 1.1, 0.1)
        )
    )


def per_image_voc07_map50(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
) -> float:
    """Compute Adaptive Feeding's image-level VOC07 mAP at IoU 0.5.

    AP is averaged over non-ignored classes present in this image, matching the
    mAPI definition in the Adaptive Feeding paper.  A prediction matched only
    to a difficult/crowd target is ignored rather than counted as a false
    positive.
    """
    valid_targets, ignored_targets = [], []
    for target in ground_truth:
        ignored = bool(target.get("difficult", target.get("ignore", False))) or bool(
            target.get("iscrowd", False)
        )
        (ignored_targets if ignored else valid_targets).append(target)
    category_ids = sorted({int(target["category_id"]) for target in valid_targets})
    if not category_ids:
        return 0.0
    values = []
    for category_id in category_ids:
        targets = [target for target in valid_targets if int(target["category_id"]) == category_id]
        ignored = [target for target in ignored_targets if int(target["category_id"]) == category_id]
        detected: set[int] = set()
        true_positive: list[float] = []
        false_positive: list[float] = []
        ordered = sorted(
            (
                (index, prediction)
                for index, prediction in enumerate(predictions)
                if int(prediction["category_id"]) == category_id
            ),
            key=lambda item: (-_score(item[1]), item[0]),
        )
        for _, prediction in ordered:
            overlaps = [_iou_xywh(prediction["bbox"], target["bbox"]) for target in targets]
            best = int(np.argmax(overlaps)) if overlaps else -1
            if best >= 0 and overlaps[best] >= 0.5 and best not in detected:
                detected.add(best)
                true_positive.append(1.0)
                false_positive.append(0.0)
                continue
            if any(_iou_xywh(prediction["bbox"], target["bbox"]) >= 0.5 for target in ignored):
                continue
            true_positive.append(0.0)
            false_positive.append(1.0)
        if not true_positive:
            values.append(0.0)
            continue
        tp = np.cumsum(np.asarray(true_positive, dtype=np.float64))
        fp = np.cumsum(np.asarray(false_positive, dtype=np.float64))
        recall = tp / max(1, len(targets))
        precision = tp / np.maximum(1.0, tp + fp)
        values.append(_voc07_ap(recall, precision))
    return float(np.mean(values))


def adaptive_feeding_labels(
    image_ids: Sequence[int],
    edge_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    cloud_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    ground_truth: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_values, cloud_values = [], []
    for raw_image_id in image_ids:
        image_id = int(raw_image_id)
        targets = ground_truth.get(image_id, ())
        edge_values.append(
            per_image_voc07_map50(edge_predictions.get(image_id, ()), targets)
        )
        cloud_values.append(
            per_image_voc07_map50(cloud_predictions.get(image_id, ()), targets)
        )
    edge = np.asarray(edge_values, dtype=np.float64)
    cloud = np.asarray(cloud_values, dtype=np.float64)
    return (cloud > edge).astype(np.int64), edge, cloud


def fit_adaptive_feeding(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    positive_weight: float = 3.0,
) -> Any:
    """Fit the paper's primal linear SVM (C=1, hard-class weighting)."""
    from sklearn.svm import LinearSVC

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or target.shape != (len(matrix),):
        raise ValueError("Adaptive Feeding training shapes do not align")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("Adaptive Feeding requires both easy and hard fit images")
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive")
    return LinearSVC(
        C=1.0,
        dual=False,
        class_weight={0: 1.0, 1: float(positive_weight)},
        tol=1e-4,
        max_iter=10000,
        random_state=0,
    ).fit(matrix, target)


def _filtered_signal(
    predictions: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    confidence: float,
) -> tuple[int, float]:
    selected = [prediction for prediction in predictions if _score(prediction) > confidence]
    if not selected:
        return 0, 0.0
    return len(selected), min(
        _area_ratio(prediction["bbox"], width, height) for prediction in selected
    )


def _fit_dcsb_confidence(
    image_ids: Sequence[int],
    edge_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    ground_truth_counts: Mapping[int, int],
) -> tuple[float, int]:
    scores = np.sort(
        np.asarray(
            [
                _score(prediction)
                for raw_image_id in image_ids
                for prediction in edge_predictions.get(int(raw_image_id), ())
            ],
            dtype=np.float64,
        )
    )
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    truth_total = sum(int(ground_truth_counts[int(image_id)]) for image_id in image_ids)
    predicted = len(scores) - np.searchsorted(scores, candidates, side="right")
    difference = np.abs(predicted - truth_total)
    # Highest threshold is the deterministic tie break used by the earlier
    # exact-breakpoint implementation.
    index = int(np.lexsort((-candidates, difference))[0])
    return float(candidates[index]), int(predicted[index])


def fit_dcsb(
    *,
    image_ids: Sequence[int],
    edge_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    cloud_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    image_sizes: Mapping[int, tuple[int, int]],
    ground_truth_counts: Mapping[int, int],
    ground_truth_min_areas: Mapping[int, float],
    detector_confidence: float = 0.5,
    count_thresholds: Sequence[int] = tuple(range(1, 11)),
    area_thresholds: Sequence[float] = tuple(np.arange(0.2, 0.9, 0.01)),
) -> dict[str, Any]:
    """Fit the three-threshold DCSB rule using only fit-role data.

    In accordance with Section V-D of the paper, the number/area threshold
    regression uses true fit annotations; test-time decisions use estimated
    values from the edge detector.
    """
    ids = tuple(map(int, image_ids))
    if not ids:
        raise ValueError("DCSB requires fit images")
    confidence_threshold, estimated_total = _fit_dcsb_confidence(
        ids, edge_predictions, ground_truth_counts
    )
    records = []
    for image_id in ids:
        width, height = image_sizes[image_id]
        edge_count, _ = _filtered_signal(
            edge_predictions.get(image_id, ()),
            width=width,
            height=height,
            confidence=float(detector_confidence),
        )
        cloud_count, _ = _filtered_signal(
            cloud_predictions.get(image_id, ()),
            width=width,
            height=height,
            confidence=float(detector_confidence),
        )
        records.append(
            {
                "true_count": int(ground_truth_counts[image_id]),
                "true_min_area": float(ground_truth_min_areas[image_id]),
                "edge_count": edge_count,
                "label": int(cloud_count > edge_count),
            }
        )
    best: tuple[float, int, float] | None = None
    for count_threshold in map(int, count_thresholds):
        for area_threshold in map(float, area_thresholds):
            decisions = [
                int(
                    record["true_count"] != record["edge_count"]
                    and (
                        record["true_count"] > count_threshold
                        or record["true_min_area"] < area_threshold
                    )
                )
                for record in records
            ]
            accuracy = float(
                np.mean([decision == record["label"] for decision, record in zip(decisions, records)])
            )
            candidate = (accuracy, count_threshold, area_threshold)
            # Match the source grid search: the first count/area pair wins a
            # tie, rather than introducing an unreported secondary objective.
            if best is None or accuracy > best[0]:
                best = candidate
    if best is None:
        raise ValueError("DCSB threshold grids cannot be empty")
    accuracy, count_threshold, area_threshold = best
    return {
        "confidence_threshold": float(confidence_threshold),
        "count_threshold": int(count_threshold),
        "area_threshold": float(area_threshold),
        "detector_confidence": float(detector_confidence),
        "fit_accuracy": float(accuracy),
        "fit_images": len(ids),
        "positive_labels": sum(record["label"] for record in records),
        "positive_label_rate": float(np.mean([record["label"] for record in records])),
        "ground_truth_count_total": sum(ground_truth_counts[image_id] for image_id in ids),
        "estimated_count_total": int(estimated_total),
        "threshold_training_features": "ground_truth_count_and_min_area",
    }


def dcsb_decision_and_margin(
    predictions: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    model: Mapping[str, Any],
) -> tuple[int, float]:
    estimated_count, minimum_area = _filtered_signal(
        predictions,
        width=width,
        height=height,
        confidence=float(model["confidence_threshold"]),
    )
    detected_count, _ = _filtered_signal(
        predictions,
        width=width,
        height=height,
        confidence=float(model["detector_confidence"]),
    )
    if estimated_count == detected_count:
        return 0, -1.0
    count_threshold = float(model["count_threshold"])
    area_threshold = float(model["area_threshold"])
    count_margin = (estimated_count - count_threshold) / max(1.0, count_threshold)
    area_margin = (area_threshold - minimum_area) / max(1e-12, area_threshold)
    margin = float(max(count_margin, area_margin))
    return int(margin > 0.0), margin


def select_original_by_score(
    samples: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    """Budget-match a scalar external score using Original uploads only."""
    if len(samples) != len(scores):
        raise ValueError("sample and score populations do not align")
    if not 0.0 <= float(budget) <= 1.0:
        raise ValueError("budget must be in [0, 1]")
    order = sorted(
        range(len(samples)),
        key=lambda index: (-float(scores[index]), int(samples[index]["image_id"])),
    )
    if mode == "fixed_call":
        count = int(math.floor(len(samples) * float(budget) + 0.5))
        return {int(samples[index]["image_id"]): "original" for index in order[:count]}
    if mode != "fixed_byte":
        raise ValueError(mode)
    capacity = int(
        math.floor(
            float(budget)
            * sum(int(sample["payload_bytes"]["original"]) for sample in samples)
        )
    )
    selected: dict[int, str] = {}
    spent = 0
    for index in order:
        sample = samples[index]
        cost = int(sample["payload_bytes"]["original"])
        if spent + cost <= capacity:
            selected[int(sample["image_id"])] = "original"
            spent += cost
    return selected


def validate_original_only_selection(
    samples: Sequence[Mapping[str, Any]],
    selection: Mapping[int, str],
    *,
    mode: str,
    budget: float,
) -> dict[str, Any]:
    by_id = {int(sample["image_id"]): sample for sample in samples}
    if set(selection.values()) - {"original"}:
        raise ValueError("external baselines may only select Original uploads")
    if set(map(int, selection)) - set(by_id):
        raise ValueError("selection contains unknown images")
    requested = sum(
        int(by_id[int(image_id)]["payload_bytes"]["original"])
        for image_id in selection
    )
    denominator = sum(int(sample["payload_bytes"]["original"]) for sample in samples)
    if mode == "fixed_call":
        expected = int(math.floor(len(samples) * float(budget) + 0.5))
        if len(selection) != expected:
            raise ValueError(f"fixed-call count changed: {len(selection)} != {expected}")
        capacity = expected
        utilization = 1.0 if expected else 0.0
    elif mode == "fixed_byte":
        capacity = int(math.floor(float(budget) * denominator))
        if requested > capacity:
            raise ValueError(f"fixed-byte capacity exceeded: {requested}>{capacity}")
        utilization = requested / capacity if capacity else 0.0
    elif mode in {"native", "reference"}:
        capacity = denominator
        utilization = requested / denominator if denominator else 0.0
    else:
        raise ValueError(mode)
    return {
        "request_bytes": int(requested),
        "all_original_bytes": int(denominator),
        "capacity": int(capacity),
        "capacity_utilization": float(utilization),
        "selected_images": len(selection),
        "call_ratio": len(selection) / len(samples) if samples else 0.0,
        "byte_ratio": requested / denominator if denominator else 0.0,
    }
