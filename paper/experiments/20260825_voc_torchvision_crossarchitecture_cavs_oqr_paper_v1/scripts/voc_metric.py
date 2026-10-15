"""Deterministic VOC detection metrics with correct difficult-object semantics."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_IOUS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def voc_box_to_xyxy(
    xmin: float, ymin: float, xmax: float, ymax: float
) -> list[float]:
    """Convert inclusive, one-based VOC coordinates to zero-based half-open xyxy."""
    x1 = float(xmin) - 1.0
    y1 = float(ymin) - 1.0
    x2 = float(xmax)
    y2 = float(ymax)
    return [x1, y1, max(x1, x2), max(y1, y2)]


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, width, height = map(float, box)
    return np.asarray((x, y, x + max(0.0, width), y + max(0.0, height)), dtype=np.float64)


def _iou(box: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    if not len(ground_truth):
        return np.empty(0, dtype=np.float64)
    left_top = np.maximum(ground_truth[:, :2], box[:2])
    right_bottom = np.minimum(ground_truth[:, 2:], box[2:])
    intersection_size = np.maximum(0.0, right_bottom - left_top)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    gt_size = np.maximum(0.0, ground_truth[:, 2:] - ground_truth[:, :2])
    gt_area = gt_size[:, 0] * gt_size[:, 1]
    union = box_area + gt_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray, points: int) -> float:
    thresholds = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    values = [float(np.max(precision[recall >= threshold])) if np.any(recall >= threshold) else 0.0 for threshold in thresholds]
    return float(np.mean(values))


def _integral_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


class VocMetricBackend:
    """Evaluate AP while ignoring detections whose best match is difficult ground truth."""

    def __init__(
        self,
        annotations: str | Path | Mapping[str, Any],
        *,
        iou_thresholds: Sequence[float] = DEFAULT_IOUS,
    ) -> None:
        if isinstance(annotations, (str, Path)):
            payload = json.loads(Path(annotations).read_text(encoding="utf-8"))
        else:
            payload = dict(annotations)
        self.category_ids = tuple(int(row["id"]) for row in payload.get("categories", ()))
        if not self.category_ids or len(self.category_ids) != len(set(self.category_ids)):
            raise ValueError("ground-truth category schema is empty or duplicated")
        self.image_ids = {int(row["id"]) for row in payload.get("images", ())}
        self.iou_thresholds = tuple(float(value) for value in iou_thresholds)
        if not self.iou_thresholds:
            raise ValueError("IoU threshold grid is empty")
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in payload.get("annotations", ()):
            image_id = int(row["image_id"])
            category_id = int(row["category_id"])
            if image_id not in self.image_ids or category_id not in self.category_ids:
                raise ValueError("ground-truth annotation references an unknown image/category")
            grouped[(image_id, category_id)].append(
                {
                    "box": _xywh_to_xyxy(row["bbox"]),
                    "difficult": bool(row.get("difficult", row.get("ignore", False))),
                }
            )
        self.ground_truth = dict(grouped)

    def _class_threshold(
        self,
        predictions: Sequence[Mapping[str, Any]],
        image_ids: set[int],
        category_id: int,
        threshold: float,
        detections: Sequence[tuple[int, Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any] | None:
        truth: dict[int, list[dict[str, Any]]] = {
            image_id: self.ground_truth.get((image_id, category_id), [])
            for image_id in image_ids
        }
        positives = sum(not row["difficult"] for rows in truth.values() for row in rows)
        if positives == 0:
            return None
        if detections is None:
            ordered_detections = [
                (index, row)
                for index, row in enumerate(predictions)
                if int(row["image_id"]) in image_ids
                and int(row["category_id"]) == category_id
            ]
            ordered_detections.sort(key=self._detection_sort_key)
        else:
            ordered_detections = detections
        matched: dict[int, np.ndarray] = {
            image_id: np.zeros(len(rows), dtype=bool) for image_id, rows in truth.items()
        }
        true_positive: list[float] = []
        false_positive: list[float] = []
        ignored = 0
        for _, detection in ordered_detections:
            image_id = int(detection["image_id"])
            rows = truth.get(image_id, [])
            overlaps = _iou(
                _xywh_to_xyxy(detection["bbox"]),
                np.asarray([row["box"] for row in rows], dtype=np.float64).reshape((-1, 4)),
            )
            best = int(np.argmax(overlaps)) if len(overlaps) else -1
            if best >= 0 and float(overlaps[best]) >= float(threshold):
                if rows[best]["difficult"]:
                    ignored += 1
                    continue
                if not matched[image_id][best]:
                    matched[image_id][best] = True
                    true_positive.append(1.0)
                    false_positive.append(0.0)
                else:
                    true_positive.append(0.0)
                    false_positive.append(1.0)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)
        tp = np.cumsum(np.asarray(true_positive, dtype=np.float64))
        fp = np.cumsum(np.asarray(false_positive, dtype=np.float64))
        recall = tp / float(positives) if len(tp) else np.empty(0, dtype=np.float64)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return {
            "ap101": _interpolated_ap(recall, precision, 101),
            "ap11": _interpolated_ap(recall, precision, 11),
            "ap_integral": _integral_ap(recall, precision),
            "positives": positives,
            "true_positives": int(tp[-1]) if len(tp) else 0,
            "false_positives": int(fp[-1]) if len(fp) else 0,
            "ignored_detections": ignored,
        }

    @staticmethod
    def _detection_sort_key(
        item: tuple[int, Mapping[str, Any]],
    ) -> tuple[float, int, tuple[float, ...], int]:
        return (
            -float(item[1].get("score", 0.0)),
            int(item[1]["image_id"]),
            tuple(map(float, item[1]["bbox"])),
            int(item[0]),
        )

    def evaluate(
        self,
        predictions: Sequence[Mapping[str, Any]],
        image_ids: Sequence[int],
        *,
        return_details: bool = False,
    ) -> dict[str, Any]:
        selected = set(map(int, image_ids))
        if not selected or not selected <= self.image_ids:
            raise ValueError("evaluation image population is empty or unknown")
        detections_by_category: dict[
            int, list[tuple[int, Mapping[str, Any]]]
        ] = {category_id: [] for category_id in self.category_ids}
        for index, row in enumerate(predictions):
            category_id = int(row["category_id"])
            if int(row["image_id"]) in selected and category_id in detections_by_category:
                detections_by_category[category_id].append((index, row))
        for detections in detections_by_category.values():
            detections.sort(key=self._detection_sort_key)
        by_threshold: dict[float, list[float]] = {threshold: [] for threshold in self.iou_thresholds}
        class_details: dict[str, Any] = {}
        counts = {
            "positives_at_50": 0,
            "true_positives_at_50": 0,
            "false_positives_at_50": 0,
            "ignored_detections_at_50": 0,
        }
        voc07, voc12 = [], []
        for category_id in self.category_ids:
            threshold_details = {}
            for threshold in self.iou_thresholds:
                result = self._class_threshold(
                    predictions,
                    selected,
                    category_id,
                    threshold,
                    detections_by_category[category_id],
                )
                if result is None:
                    continue
                by_threshold[threshold].append(float(result["ap101"]))
                threshold_details[f"{threshold:.2f}"] = result
                if abs(threshold - 0.5) < 1e-9:
                    voc07.append(float(result["ap11"]))
                    voc12.append(float(result["ap_integral"]))
                    for key in counts:
                        source = key.removesuffix("_at_50")
                        counts[key] += int(result[source])
            if threshold_details:
                class_details[str(category_id)] = threshold_details
        threshold_map = {
            threshold: float(np.mean(values)) if values else 0.0
            for threshold, values in by_threshold.items()
        }
        nearest_50 = min(self.iou_thresholds, key=lambda value: abs(value - 0.50))
        nearest_75 = min(self.iou_thresholds, key=lambda value: abs(value - 0.75))
        output: dict[str, Any] = {
            "map50_95": float(np.mean(list(threshold_map.values()))),
            "map50": threshold_map[nearest_50],
            "map75": threshold_map[nearest_75],
            "voc07_map50": float(np.mean(voc07)) if voc07 else 0.0,
            "voc12_map50": float(np.mean(voc12)) if voc12 else 0.0,
        }
        if return_details:
            output["counts"] = counts
            output["per_iou"] = {f"{key:.2f}": value for key, value in threshold_map.items()}
            output["per_class"] = class_details
        return output
