"""JPEG, JSON, COCO, and HTTP contracts shared by the board and cloud."""

from __future__ import annotations

from io import BytesIO
import json
import math
from numbers import Integral
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


COCO80_TO_COCO91 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
WIRE_MAGIC = b"CAVSHIL2"
WIRE_SCHEMA = "cavs_hil_binary_request_v2"
WIRE_MEDIA_TYPE = "application/vnd.cavs-hil.v2+octet-stream"
WIRE_HEADER_BYTES = len(WIRE_MAGIC) + 4
MAX_METADATA_BYTES = 64 * 1024


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


def sanitize_boxes(
    boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    class_count: int = 80,
) -> list[dict[str, Any]]:
    """Clip valid detections to an image and drop malformed or degenerate rows."""
    width, height, class_count = int(width), int(height), int(class_count)
    if width <= 0 or height <= 0 or class_count <= 0:
        raise ValueError("invalid box-sanitization geometry")
    clean: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, Mapping):
            continue
        try:
            xyxy = box["xyxy"]
            if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                continue
            coordinates = [float(value) for value in xyxy]
            confidence = float(box.get("conf", box.get("score")))
            category = box["cls"]
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (
            not all(math.isfinite(value) for value in coordinates)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or isinstance(category, bool)
            or not isinstance(category, Integral)
            or not 0 <= int(category) < class_count
        ):
            continue
        x1, y1, x2, y2 = coordinates
        clipped = [
            min(float(width), max(0.0, x1)),
            min(float(height), max(0.0, y1)),
            min(float(width), max(0.0, x2)),
            min(float(height), max(0.0, y2)),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        clean.append({"xyxy": clipped, "conf": confidence, "cls": int(category)})
    return clean


def coco_predictions(image_id: int, boxes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for box in boxes:
        x1, y1, x2, y2 = map(float, box["xyxy"])
        rows.append({"image_id": int(image_id), "category_id": COCO80_TO_COCO91[int(box["cls"])], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(box.get("conf", box.get("score", 0.0)))})
    return rows


def wire_contract() -> dict[str, Any]:
    return {
        "schema": WIRE_SCHEMA,
        "media_type": WIRE_MEDIA_TYPE,
        "magic_ascii": WIRE_MAGIC.decode("ascii"),
        "header_bytes": WIRE_HEADER_BYTES,
        "maximum_metadata_bytes": MAX_METADATA_BYTES,
        "base64": False,
    }


def encode_inference_request(metadata: Mapping[str, Any], payload: bytes) -> bytes:
    if not isinstance(metadata, Mapping) or not isinstance(payload, bytes):
        raise TypeError("metadata must be a mapping and payload must be bytes")
    value = dict(metadata)
    if value.get("schema", WIRE_SCHEMA) != WIRE_SCHEMA:
        raise ValueError("unsupported inference request schema")
    value["schema"] = WIRE_SCHEMA
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("metadata is not canonical JSON") from exc
    if not 0 < len(encoded) <= MAX_METADATA_BYTES:
        raise ValueError("invalid metadata length")
    return WIRE_MAGIC + struct.pack(">I", len(encoded)) + encoded + payload


def decode_inference_request(
    body: bytes,
    *,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(body, bytes) or not WIRE_HEADER_BYTES < len(body) <= int(maximum_bytes):
        raise ValueError("invalid request length")
    if not body.startswith(WIRE_MAGIC):
        raise ValueError("invalid request magic")
    metadata_length = struct.unpack(">I", body[len(WIRE_MAGIC) : WIRE_HEADER_BYTES])[0]
    if not 0 < metadata_length <= MAX_METADATA_BYTES:
        raise ValueError("invalid metadata length")
    payload_offset = WIRE_HEADER_BYTES + metadata_length
    if payload_offset > len(body):
        raise ValueError("metadata length exceeds request body")
    try:
        metadata = json.loads(body[WIRE_HEADER_BYTES:payload_offset].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid metadata JSON") from exc
    if not isinstance(metadata, Mapping) or metadata.get("schema") != WIRE_SCHEMA:
        raise ValueError("unsupported inference request schema")
    return dict(metadata), body[payload_offset:]


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
