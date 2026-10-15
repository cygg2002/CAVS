"""JPEG, JSON, COCO, and HTTP contracts shared by the board and cloud."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


COCO80_TO_COCO91 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)


def native_jpeg(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def q90_jpeg(path: str | Path, *, crop_xyxy: Sequence[int] | None = None, long_side: int = 640) -> bytes:
    """Use the exact frozen quality=90, optimize=True, subsampling=2 action codec."""
    with Image.open(path) as opened:
        image = opened.crop(tuple(map(int, crop_xyxy))) if crop_xyxy is not None else opened.copy()
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(long_side) / max(1, width, height))
        size = max(1, round(width * scale)), max(1, round(height * scale))
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True, subsampling=2)
        return output.getvalue()


def compact_boxes(boxes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"xyxy": [float(value) for value in box["xyxy"]], "conf": float(box.get("conf", box.get("score", 0.0))), "cls": int(box["cls"])} for box in boxes]


def coco_predictions(image_id: int, boxes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for box in boxes:
        x1, y1, x2, y2 = map(float, box["xyxy"])
        rows.append({"image_id": int(image_id), "category_id": COCO80_TO_COCO91[int(box["cls"])], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(box.get("conf", box.get("score", 0.0)))})
    return rows


def encode_b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def decode_b64(payload: str) -> bytes:
    return base64.b64decode(payload.encode("ascii"), validate=True)


def decode_jpeg(payload: bytes, *, maximum_pixels: int = 100_000_000) -> np.ndarray:
    if not payload:
        raise ValueError("empty JPEG payload")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[0] * image.shape[1] > maximum_pixels:
        raise ValueError("invalid or oversized JPEG payload")
    return image


def shift_boxes(boxes: Sequence[Mapping[str, Any]], offset: Sequence[int]) -> list[dict[str, Any]]:
    dx, dy = map(float, offset)
    shifted = compact_boxes(boxes)
    for box in shifted:
        x1, y1, x2, y2 = box["xyxy"]
        box["xyxy"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
    return shifted
