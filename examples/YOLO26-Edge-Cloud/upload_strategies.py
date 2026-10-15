# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Upload strategy helpers for edge-cloud object detection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from protocol import box_iou, greedy_match


def parse_grid(value: str) -> tuple[int, int]:
    """Parse grid text such as '2x2' or '3,2' as columns, rows."""
    text = value.lower().replace(",", "x")
    cols, rows = [int(x.strip()) for x in text.split("x", 1)]
    return max(1, cols), max(1, rows)


def edge_density(image: np.ndarray, resize_width: int = 320) -> float:
    """Compute a lightweight image complexity proxy using Canny edge density."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > resize_width:
        new_h = max(1, int(gray.shape[0] * resize_width / gray.shape[1]))
        gray = cv2.resize(gray, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 80, 160)
    return float((edges > 0).mean())


def box_area_ratio(box: dict[str, Any], shape: tuple[int, int]) -> float:
    """Return a box area ratio in image coordinates."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box["xyxy"]
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))


def edge_margin_ratio(box: dict[str, Any], shape: tuple[int, int]) -> float:
    """Return normalized distance from a box to the nearest image edge."""
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box["xyxy"]]
    return min(x1, y1, w - x2, h - y2) / max(1.0, float(min(h, w)))


def adaptive_roi_gain(
    box: dict[str, Any],
    shape: tuple[int, int],
    base_gain: float = 1.35,
    small_gain: float = 2.0,
    edge_gain: float = 1.65,
    large_gain: float = 1.15,
    small_area_threshold: float = 0.015,
    large_area_threshold: float = 0.20,
    edge_margin_threshold: float = 0.05,
) -> float:
    """Choose ROI expansion by target scale and edge proximity."""
    area_ratio = box_area_ratio(box, shape)
    margin_ratio = edge_margin_ratio(box, shape)
    gain = base_gain
    if area_ratio < small_area_threshold:
        gain = max(gain, small_gain)
    if margin_ratio < edge_margin_threshold:
        gain = max(gain, edge_gain)
    if area_ratio > large_area_threshold:
        gain = min(gain, large_gain)
    return float(gain)


def detection_coverage(boxes: list[dict[str, Any]], shape: tuple[int, int]) -> float:
    """Approximate total detection coverage ratio.

    The sum is intentionally capped. Exact polygon union is unnecessary for this
    upload heuristic and would add avoidable overhead on edge devices.
    """
    return min(1.0, sum(box_area_ratio(box, shape) for box in boxes))


def center_in_region(box: dict[str, Any], xyxy: list[int]) -> bool:
    """Return whether the box center lies inside a region."""
    x1, y1, x2, y2 = box["xyxy"]
    rx1, ry1, rx2, ry2 = xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def boxes_in_region(boxes: list[dict[str, Any]], xyxy: list[int]) -> list[dict[str, Any]]:
    """Return boxes whose center lies inside a region."""
    return [box for box in boxes if center_in_region(box, xyxy)]


def should_complexity_probe(
    image: np.ndarray,
    edge_boxes: list[dict[str, Any]],
    complexity_threshold: float = 0.18,
    empty_complexity_threshold: float = 0.08,
    max_detections: int = 1,
    coverage_threshold: float = 0.02,
) -> tuple[bool, float, list[str]]:
    """Decide whether a complex image with sparse edge detections deserves full-frame cloud review."""
    complexity = edge_density(image)
    coverage = detection_coverage(edge_boxes, image.shape[:2])
    reasons: list[str] = []
    if not edge_boxes and complexity >= empty_complexity_threshold:
        reasons.append("empty_complex_scene")
    if complexity >= complexity_threshold and len(edge_boxes) <= max_detections:
        reasons.append("complex_sparse_detections")
    if complexity >= complexity_threshold and coverage <= coverage_threshold:
        reasons.append("complex_low_coverage")
    return bool(reasons), complexity, reasons


def stability_probe_score(
    primary_boxes: list[dict[str, Any]],
    secondary_boxes: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[float, list[str]]:
    """Score detection instability under an input-scale perturbation."""
    matches, matched_primary, matched_secondary = greedy_match(primary_boxes, secondary_boxes, iou_threshold)
    total = max(1, len(primary_boxes) + len(secondary_boxes))
    unmatched = (len(primary_boxes) - len(matched_primary)) + (len(secondary_boxes) - len(matched_secondary))
    unmatched_ratio = unmatched / total
    class_mismatches = 0
    if matches:
        for primary_idx, secondary_idx, _ in matches:
            if int(primary_boxes[primary_idx]["cls"]) != int(secondary_boxes[secondary_idx]["cls"]):
                class_mismatches += 1
    class_mismatch_ratio = class_mismatches / max(1, len(matches))
    score = min(1.0, 0.75 * unmatched_ratio + 0.25 * class_mismatch_ratio)
    reasons: list[str] = []
    if unmatched_ratio >= 0.35:
        reasons.append("scale_unmatched_boxes")
    if class_mismatch_ratio > 0.0:
        reasons.append("scale_class_disagreement")
    if primary_boxes and not secondary_boxes:
        reasons.append("scale_dropout")
    if secondary_boxes and not primary_boxes:
        reasons.append("scale_new_detections")
    return float(score), reasons or ["scale_stability_score"]


def tile_candidates(
    image: np.ndarray,
    edge_boxes: list[dict[str, Any]],
    grid: tuple[int, int] = (2, 2),
    max_tiles: int = 1,
    score_threshold: float = 0.08,
) -> list[dict[str, Any]]:
    """Select high-complexity, weakly covered tiles for cloud probing."""
    h, w = image.shape[:2]
    cols, rows = grid
    candidates: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(round(col * w / cols))
            y1 = int(round(row * h / rows))
            x2 = int(round((col + 1) * w / cols))
            y2 = int(round((row + 1) * h / rows))
            xyxy = [x1, y1, x2, y2]
            tile = image[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            tile_boxes = boxes_in_region(edge_boxes, xyxy)
            complexity = edge_density(tile)
            coverage = detection_coverage(tile_boxes, tile.shape[:2])
            # Reward local visual complexity and sparse edge coverage. This is
            # deliberately recall-oriented: cloud checks regions where the edge
            # detector may have missed small or cluttered objects.
            score = complexity * (1.0 - min(0.9, coverage * 4.0))
            if score >= score_threshold:
                candidates.append(
                    {
                        "xyxy": xyxy,
                        "score": round(float(score), 4),
                        "complexity": round(float(complexity), 4),
                        "coverage": round(float(coverage), 4),
                        "edge_boxes": len(tile_boxes),
                        "reason": ["uncovered_complex_tile"],
                    }
                )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[: max(0, max_tiles)]


def low_conf_proposals(
    candidate_boxes: list[dict[str, Any]],
    deployment_boxes: list[dict[str, Any]],
    shape: tuple[int, int],
    min_conf: float = 0.05,
    max_conf: float = 0.25,
    max_proposals: int = 2,
    overlap_iou: float = 0.5,
    score_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """Select suppressed edge detections as cloud-verifiable object proposals.

    The edge detector often emits object-like boxes below the deployment
    threshold. Those boxes are not reliable enough for local output, but they
    are spatially informative upload proposals and avoid the blind-search
    problem seen with image tiles.
    """
    h, w = shape[:2]
    image_area = max(1.0, float(h * w))
    proposals: list[dict[str, Any]] = []
    for box in candidate_boxes:
        conf = float(box.get("conf", 0.0))
        if conf < min_conf or conf >= max_conf:
            continue
        if any(box_iou(box["xyxy"], deployed["xyxy"]) >= overlap_iou for deployed in deployment_boxes):
            continue
        x1, y1, x2, y2 = [float(v) for v in box["xyxy"]]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
        edge_distance = min(x1, y1, w - x2, h - y2) / max(1.0, min(h, w))
        reasons: list[str] = ["suppressed_low_conf"]
        score = 0.65 * min(1.0, conf / max(max_conf, 1e-6))
        if area_ratio < 0.015:
            score += 0.2
            reasons.append("small_object_proposal")
        if edge_distance < 0.05:
            score += 0.15
            reasons.append("edge_area_proposal")
        score = min(1.0, score)
        if score >= score_threshold:
            item = dict(box)
            item["proposal_score"] = round(float(score), 4)
            item["proposal_reason"] = reasons
            item["area_ratio"] = round(float(area_ratio), 6)
            proposals.append(item)
    proposals.sort(key=lambda item: (float(item["proposal_score"]), float(item.get("conf", 0.0))), reverse=True)
    return proposals[: max(0, max_proposals)]
