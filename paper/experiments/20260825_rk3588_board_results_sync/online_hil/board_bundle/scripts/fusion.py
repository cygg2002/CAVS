"""Frozen MAUR ROI fusion for raw contiguous-COCO detection boxes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1) + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1) - intersection
    return intersection / union if union > 0.0 else 0.0


def maur_fuse_boxes(
    edge_boxes: Sequence[Mapping[str, Any]], cloud_boxes: Sequence[Mapping[str, Any]], roi_xyxy: Sequence[int], *,
    duplicate_iou: float = 0.5, nms_iou: float = 0.7,
) -> list[dict[str, Any]]:
    """Replace Edge boxes centered in the crop, then class-wise NMS the union."""
    x1, y1, x2, y2 = map(float, roi_xyxy)
    candidates: list[dict[str, Any]] = []
    for source in edge_boxes:
        box = dict(source)
        left, top, right, bottom = map(float, box["xyxy"])
        if x1 <= (left + right) / 2.0 <= x2 and y1 <= (top + bottom) / 2.0 <= y2:
            continue
        if any(int(box["cls"]) == int(cloud["cls"]) and iou_xyxy(box["xyxy"], cloud["xyxy"]) >= duplicate_iou for cloud in cloud_boxes):
            continue
        candidates.append(box)
    candidates.extend(dict(box) for box in cloud_boxes)
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda box: (-float(box.get("conf", box.get("score", 0.0))), int(box["cls"]), tuple(map(float, box["xyxy"])) )):
        if any(int(candidate["cls"]) == int(existing["cls"]) and iou_xyxy(candidate["xyxy"], existing["xyxy"]) >= nms_iou for existing in kept):
            continue
        kept.append(dict(candidate))
    return kept

