# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Result fusion utilities for the YOLO26 edge-cloud demo."""

from __future__ import annotations

from collections import Counter
from typing import Any

from protocol import greedy_match


def fuse_detections(
    edge_boxes: list[dict[str, Any]],
    cloud_boxes: list[dict[str, Any]],
    payload_type: str,
    iou_threshold: float = 0.5,
    cloud_conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compare edge and cloud detections and return correction events.

    The first runnable version keeps the policy interpretable: matching boxes
    confirm or correct classes, unmatched cloud boxes are potential edge misses,
    and unmatched edge boxes inside an ROI are potential false positives.
    """
    matches, matched_edge, matched_cloud = greedy_match(edge_boxes, cloud_boxes, iou_threshold)
    events: list[dict[str, Any]] = []

    for edge_idx, cloud_idx, iou in matches:
        edge_box = edge_boxes[edge_idx]
        cloud_box = cloud_boxes[cloud_idx]
        if int(edge_box["cls"]) == int(cloud_box["cls"]):
            kind = "confirmed"
        else:
            kind = "class_corrected"
        events.append(
            {
                "type": kind,
                "iou": round(float(iou), 4),
                "edge": edge_box,
                "cloud": cloud_box,
            }
        )

    for cloud_idx, cloud_box in enumerate(cloud_boxes):
        if cloud_idx in matched_cloud or float(cloud_box.get("conf", 0.0)) < cloud_conf_threshold:
            continue
        events.append({"type": "missed_by_edge", "cloud": cloud_box})

    if payload_type == "roi":
        for edge_idx, edge_box in enumerate(edge_boxes):
            if edge_idx not in matched_edge:
                events.append({"type": "possible_false_positive", "edge": edge_box})

    counts = Counter(event["type"] for event in events)
    return {"events": events, "counts": dict(counts)}
