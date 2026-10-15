# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Protocol helpers for the YOLO26 edge-cloud detection demo.

The module intentionally keeps the wire format simple: JSON metadata plus a
base64-encoded JPEG payload. This is not the most bandwidth-efficient protocol,
but it is easy to inspect and sufficient for a first runnable edge-cloud
prototype before switching to multipart, gRPC, or a message queue.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def boxes_from_result(result) -> list[dict[str, Any]]:
    """Convert an Ultralytics detection result to serializable boxes."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    data = result.boxes.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    boxes: list[dict[str, Any]] = []
    for row in np.asarray(data):
        x1, y1, x2, y2, conf, cls = row[:6]
        boxes.append(
            {
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "cls": int(cls),
                "name": result.names.get(int(cls), str(int(cls))) if hasattr(result, "names") else str(int(cls)),
            }
        )
    return boxes


def clamp_xyxy(xyxy: list[float] | tuple[float, float, float, float], shape: tuple[int, int]) -> list[int]:
    """Clamp an xyxy box to image shape and return integer pixel coordinates."""
    h, w = shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return [x1, y1, x2, y2]


def expand_xyxy(
    xyxy: list[float] | tuple[float, float, float, float],
    shape: tuple[int, int],
    gain: float = 1.35,
) -> list[int]:
    """Expand an xyxy box around its center and clamp it to image bounds."""
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, x2 - x1) * gain, max(1.0, y2 - y1) * gain
    return clamp_xyxy([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], shape)


def crop_image(image: np.ndarray, xyxy: list[int]) -> np.ndarray:
    """Return a copy of the image crop for the given integer xyxy box."""
    x1, y1, x2, y2 = xyxy
    return image[y1:y2, x1:x2].copy()


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR image as JPEG bytes."""
    quality = int(max(1, min(100, quality)))
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes into a BGR image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decoding failed")
    return image


def bytes_to_b64(data: bytes) -> str:
    """Encode bytes as an ASCII base64 string."""
    return base64.b64encode(data).decode("ascii")


def b64_to_bytes(data: str) -> bytes:
    """Decode an ASCII base64 string into bytes."""
    return base64.b64decode(data.encode("ascii"))


def shift_boxes(boxes: list[dict[str, Any]], offset_xy: tuple[int, int]) -> list[dict[str, Any]]:
    """Shift boxes by an x/y offset, used to map ROI predictions back to full-frame coordinates."""
    ox, oy = offset_xy
    shifted = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        item = dict(box)
        item["xyxy"] = [float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy)]
        shifted.append(item)
    return shifted


def box_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match(
    edge_boxes: list[dict[str, Any]],
    cloud_boxes: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Greedily match edge and cloud boxes by IoU."""
    pairs: list[tuple[float, int, int]] = []
    for ei, ebox in enumerate(edge_boxes):
        for ci, cbox in enumerate(cloud_boxes):
            iou = box_iou(ebox["xyxy"], cbox["xyxy"])
            if iou >= iou_threshold:
                pairs.append((iou, ei, ci))
    pairs.sort(reverse=True)
    matched_edge: set[int] = set()
    matched_cloud: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, ei, ci in pairs:
        if ei in matched_edge or ci in matched_cloud:
            continue
        matched_edge.add(ei)
        matched_cloud.add(ci)
        matches.append((ei, ci, iou))
    return matches, matched_edge, matched_cloud
