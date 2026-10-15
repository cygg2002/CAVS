"""Pure Current ROI geometry, crop mapping, and MAUR fusion helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def roi_expansion(area_ratio: float, profile: str) -> float:
    area = max(0.0, float(area_ratio))
    if profile != "current":
        raise ValueError(f"unsupported ROI profile: {profile}")
    if area < 0.01:
        return 1.8
    if area < 0.10:
        return 1.6
    return 1.4


def merged_roi_xyxy(
    width: int,
    height: int,
    boxes: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    confidence: float,
    max_area_ratio: float,
) -> tuple[int, int, int, int] | None:
    image_width, image_height = max(1, int(width)), max(1, int(height))
    expanded = []
    for box in boxes:
        if float(box.get("score", box.get("conf", 0.0))) < float(confidence):
            continue
        x, y, box_width, box_height = map(float, box["bbox"])
        factor = roi_expansion(
            box_width * box_height / (image_width * image_height), profile
        )
        center_x, center_y = x + box_width / 2.0, y + box_height / 2.0
        expanded.append(
            (
                max(0.0, center_x - box_width * factor / 2.0),
                max(0.0, center_y - box_height * factor / 2.0),
                min(float(image_width), center_x + box_width * factor / 2.0),
                min(float(image_height), center_y + box_height * factor / 2.0),
            )
        )
    if not expanded:
        return None
    x1 = max(0, int(min(row[0] for row in expanded)))
    y1 = max(0, int(min(row[1] for row in expanded)))
    x2 = min(image_width, int(max(row[2] for row in expanded) + 0.999999))
    y2 = min(image_height, int(max(row[3] for row in expanded) + 0.999999))
    if x2 <= x1 or y2 <= y1:
        return None
    ratio = (x2 - x1) * (y2 - y1) / (image_width * image_height)
    return None if ratio > float(max_area_ratio) else (x1, y1, x2, y2)


def map_crop_predictions(
    predictions: Sequence[Mapping[str, Any]],
    geometry: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for source in predictions:
        row = dict(source)
        image_id = int(row["image_id"])
        x1, y1, _, _ = map(float, geometry[image_id]["xyxy"])
        x, y, width, height = map(float, row["bbox"])
        row["bbox"] = [x + x1, y + y1, width, height]
        output.append(row)
    return output


def _iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0.0 else 0.0


def maur_candidates(
    edge_boxes: Sequence[Mapping[str, Any]],
    cloud_boxes: Sequence[Mapping[str, Any]],
    roi_xyxy: Sequence[float],
    *,
    duplicate_iou: float,
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = map(float, roi_xyxy)
    kept = []
    for source in edge_boxes:
        edge = dict(source)
        x, y, width, height = map(float, edge["bbox"])
        center_x, center_y = x + width / 2.0, y + height / 2.0
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            continue
        duplicate = any(
            int(edge["category_id"]) == int(cloud["category_id"])
            and _iou_xywh(edge["bbox"], cloud["bbox"]) >= float(duplicate_iou)
            for cloud in cloud_boxes
        )
        if not duplicate:
            kept.append(edge)
    return [*kept, *map(dict, cloud_boxes)]


def class_aware_nms(
    predictions: Sequence[Mapping[str, Any]], *, iou_threshold: float
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    ordered = sorted(
        map(dict, predictions),
        key=lambda row: (
            -float(row.get("score", 1.0)),
            int(row["category_id"]),
            tuple(map(float, row["bbox"])),
        ),
    )
    for row in ordered:
        if any(
            int(row["category_id"]) == int(existing["category_id"])
            and _iou_xywh(row["bbox"], existing["bbox"]) >= float(iou_threshold)
            for existing in kept
        ):
            continue
        kept.append(row)
    return kept
