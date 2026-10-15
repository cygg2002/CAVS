"""Framework-independent image encoding and detection normalization helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


def map_resized_xyxy(
    box: Sequence[float],
    *,
    original_width: int,
    original_height: int,
    resized_width: int,
    resized_height: int,
) -> list[float]:
    if min(original_width, original_height, resized_width, resized_height) <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = map(float, box)
    scale_x = float(original_width) / float(resized_width)
    scale_y = float(original_height) / float(resized_height)
    return [
        min(float(original_width), max(0.0, x1 * scale_x)),
        min(float(original_height), max(0.0, y1 * scale_y)),
        min(float(original_width), max(0.0, x2 * scale_x)),
        min(float(original_height), max(0.0, y2 * scale_y)),
    ]


def map_affine_xyxy(
    box: Sequence[float],
    *,
    transform: Sequence[Sequence[float]],
    original_width: int,
    original_height: int,
) -> list[float]:
    """Map an xyxy box through a 2x3 affine transform and clip to the source image."""
    if min(original_width, original_height) <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = map(float, box)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("affine transform must have shape (2, 3)")
    corners = np.asarray(
        ((x1, y1, 1.0), (x1, y2, 1.0), (x2, y1, 1.0), (x2, y2, 1.0)),
        dtype=np.float64,
    )
    mapped = corners @ matrix.T
    if not np.isfinite(mapped).all():
        return [float("nan")] * 4
    return [
        min(float(original_width), max(0.0, float(mapped[:, 0].min()))),
        min(float(original_height), max(0.0, float(mapped[:, 1].min()))),
        min(float(original_width), max(0.0, float(mapped[:, 0].max()))),
        min(float(original_height), max(0.0, float(mapped[:, 1].max()))),
    ]


def normalize_detections(
    *,
    image_id: int,
    class_ids: Sequence[float],
    scores: Sequence[float],
    boxes: Sequence[Sequence[float]],
    original_width: int,
    original_height: int,
    resized_width: int,
    resized_height: int,
    category_count: int,
    score_floor: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    classes = np.asarray(class_ids).reshape(-1)
    confidence = np.asarray(scores).reshape(-1)
    coordinates = np.asarray(boxes).reshape((-1, 4))
    if len(classes) != len(confidence) or len(classes) != len(coordinates):
        raise ValueError("class, score and box outputs have different lengths")
    audit = {
        "raw": len(classes),
        "below_score": 0,
        "invalid_class": 0,
        "nonfinite": 0,
        "degenerate": 0,
        "kept": 0,
    }
    output: list[dict[str, Any]] = []
    for raw_class, raw_score, raw_box in zip(classes, confidence, coordinates):
        values = np.asarray([raw_class, raw_score, *raw_box], dtype=np.float64)
        if not np.isfinite(values).all():
            audit["nonfinite"] += 1
            continue
        score = float(raw_score)
        if score < float(score_floor):
            audit["below_score"] += 1
            continue
        rounded_class = int(round(float(raw_class)))
        if abs(float(raw_class) - rounded_class) > 1e-6 or not 0 <= rounded_class < int(category_count):
            audit["invalid_class"] += 1
            continue
        x1, y1, x2, y2 = map_resized_xyxy(
            raw_box,
            original_width=int(original_width),
            original_height=int(original_height),
            resized_width=int(resized_width),
            resized_height=int(resized_height),
        )
        width, height = x2 - x1, y2 - y1
        if not math.isfinite(width + height) or width <= 0.0 or height <= 0.0:
            audit["degenerate"] += 1
            continue
        output.append(
            {
                "image_id": int(image_id),
                "category_id": rounded_class + 1,
                "bbox": [x1, y1, width, height],
                "score": score,
            }
        )
    audit["kept"] = len(output)
    return output, audit


def encode_q90(
    source: str | Path,
    target: str | Path,
    *,
    quality: int = 90,
    optimize: bool = True,
    subsampling: int = 2,
) -> int:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with Image.open(source) as image:
        image.convert("RGB").save(
            temporary,
            format="JPEG",
            quality=int(quality),
            optimize=bool(optimize),
            subsampling=int(subsampling),
        )
    temporary.replace(destination)
    return destination.stat().st_size
