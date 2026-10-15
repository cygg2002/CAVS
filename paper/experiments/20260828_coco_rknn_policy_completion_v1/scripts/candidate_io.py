"""Validated RKNN candidate input and output records.

This module deliberately treats scores from previous board policies as
untrusted metadata.  Deployment-aligned training consumes only RKNN Edge
detections, image geometry, and the immutable role manifest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePath
from typing import Any, Mapping


BACKENDS = ("fp16", "int8-heads")
INPUT_SCHEMA = "rknn_candidate_input_v1"
BUNDLE_SCHEMA = "rknn_candidate_bundle_v1"
RECORD_SCHEMA = "rknn_candidate_record_v1"


def _sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"{field} must be a SHA-256 digest")
    return digest


def _positive_int(value: object, *, field: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{field} must be an integer") from error
    if integer <= 0:
        raise RuntimeError(f"{field} must be positive")
    return integer


def _file_name(value: object) -> str:
    name = str(value)
    if not name or PurePath(name).name != name or name in {".", ".."}:
        raise RuntimeError("file_name must be a plain filename")
    return name


def _finite(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise RuntimeError(f"{field} must be finite")
    return number


def current_roi_xyxy(width: int, height: int, edge_boxes: list[dict[str, Any]]) -> list[int] | None:
    """Reproduce the fixed Current@640 geometry from raw RKNN Edge boxes."""
    expanded: list[tuple[float, float, float, float]] = []
    for box in edge_boxes:
        if float(box["conf"]) < 0.25:
            continue
        x1, y1, x2, y2 = map(float, box["xyxy"])
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        ratio = box_width * box_height / float(width * height)
        expansion = 1.8 if ratio < 0.01 else 1.6 if ratio < 0.10 else 1.4
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        expanded.append((
            max(0.0, center_x - box_width * expansion / 2.0),
            max(0.0, center_y - box_height * expansion / 2.0),
            min(float(width), center_x + box_width * expansion / 2.0),
            min(float(height), center_y + box_height * expansion / 2.0),
        ))
    if not expanded:
        return None
    x1 = max(0, int(min(box[0] for box in expanded)))
    y1 = max(0, int(min(box[1] for box in expanded)))
    x2 = min(width, int(max(box[2] for box in expanded) + 0.999999))
    y2 = min(height, int(max(box[3] for box in expanded) + 0.999999))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) * (y2 - y1) / float(width * height) > 0.65:
        return None
    return [x1, y1, x2, y2]


def validate_input_manifest(
    manifest: Mapping[str, Any], *, expected_role: str, expected_images: int
) -> dict[int, dict[str, Any]]:
    if str(manifest.get("schema")) != INPUT_SCHEMA:
        raise RuntimeError("candidate input schema changed")
    if str(manifest.get("role")) != str(expected_role):
        raise RuntimeError("candidate input role does not match")
    if _positive_int(manifest.get("images"), field="images") != int(expected_images):
        raise RuntimeError("candidate input population changed")
    _sha256(manifest.get("source_manifest_sha256"), field="source_manifest_sha256")
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) != int(expected_images):
        raise RuntimeError("candidate input records are incomplete")
    records: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("candidate input record is invalid")
        image_id = _positive_int(raw.get("image_id"), field="image_id")
        if image_id in records:
            raise RuntimeError("candidate input contains duplicate image IDs")
        records[image_id] = {
            "image_id": image_id,
            "file_name": _file_name(raw.get("file_name")),
            "width": _positive_int(raw.get("width"), field="width"),
            "height": _positive_int(raw.get("height"), field="height"),
        }
    return records


def validate_candidate_bundle(
    bundle: Mapping[str, Any], *, backend: str, model_sha256: str, role: str, expected_images: int
) -> None:
    if str(bundle.get("schema")) != BUNDLE_SCHEMA:
        raise RuntimeError("candidate bundle schema changed")
    if backend not in BACKENDS:
        raise RuntimeError("unsupported RKNN backend")
    if str(bundle.get("backend")) != backend:
        raise RuntimeError("candidate bundle backend does not match")
    if _sha256(bundle.get("model_sha256"), field="bundle model hash") != _sha256(model_sha256, field="model hash"):
        raise RuntimeError("candidate bundle model hash does not match")
    if str(bundle.get("role")) != str(role):
        raise RuntimeError("candidate bundle role does not match")
    if _positive_int(bundle.get("images"), field="bundle images") != int(expected_images):
        raise RuntimeError("candidate bundle population changed")
    _sha256(bundle.get("input_manifest_sha256"), field="input_manifest_sha256")


def _canonical_boxes(raw_boxes: object, *, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(raw_boxes, list):
        raise RuntimeError("edge_boxes must be a list")
    boxes: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_boxes):
        if not isinstance(raw, Mapping):
            raise RuntimeError("edge box is invalid")
        xyxy = raw.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            raise RuntimeError("edge box xyxy is invalid")
        x1, y1, x2, y2 = [_finite(value, field=f"edge_boxes[{index}].xyxy") for value in xyxy]
        if not (0.0 <= x1 <= x2 <= float(width) and 0.0 <= y1 <= y2 <= float(height)):
            raise RuntimeError("edge box lies outside image geometry or has reversed coordinates")
        confidence = _finite(raw.get("conf"), field=f"edge_boxes[{index}].conf")
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError("edge confidence is out of range")
        class_id = int(raw.get("cls"))
        if not 0 <= class_id < 80:
            raise RuntimeError("edge class is out of range")
        boxes.append({"xyxy": [x1, y1, x2, y2], "conf": confidence, "cls": class_id})
    return boxes


def _canonical_record(
    raw: Mapping[str, Any], *, expected: Mapping[str, Any], backend: str, model_sha256: str, role: str
) -> dict[str, Any]:
    if str(raw.get("schema")) != RECORD_SCHEMA:
        raise RuntimeError("candidate record schema changed")
    if str(raw.get("backend")) != backend:
        raise RuntimeError("candidate record backend does not match")
    if _sha256(raw.get("model_sha256"), field="record model hash") != _sha256(model_sha256, field="model hash"):
        raise RuntimeError("candidate record model hash does not match")
    if str(raw.get("role")) != role:
        raise RuntimeError("candidate record role does not match")
    image_id = _positive_int(raw.get("image_id"), field="image_id")
    if image_id != int(expected["image_id"]):
        raise RuntimeError("candidate record image ID does not match input")
    file_name = _file_name(raw.get("file_name"))
    if file_name != str(expected["file_name"]):
        raise RuntimeError("candidate record filename does not match input")
    shape = raw.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise RuntimeError("candidate record shape is invalid")
    height, width = int(shape[0]), int(shape[1])
    if (width, height) != (int(expected["width"]), int(expected["height"])):
        raise RuntimeError("candidate record geometry does not match input")
    boxes = _canonical_boxes(raw.get("edge_boxes"), width=width, height=height)
    reported_roi = raw.get("roi_xyxy")
    expected_roi = current_roi_xyxy(width, height, boxes)
    if reported_roi is not None:
        if not isinstance(reported_roi, (list, tuple)) or len(reported_roi) != 4:
            raise RuntimeError("candidate ROI geometry is invalid")
        try:
            normalized_roi = list(map(int, reported_roi))
        except (TypeError, ValueError) as error:
            raise RuntimeError("candidate ROI geometry is invalid") from error
        if normalized_roi != expected_roi:
            raise RuntimeError("candidate ROI geometry does not derive from matching Edge boxes")
    return {
        "image_id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "edge_boxes": boxes,
        "roi_xyxy": expected_roi,
    }


def load_candidate_records(
    path: Path,
    *,
    input_manifest: Mapping[str, Any],
    backend: str,
    model_sha256: str,
    role: str,
) -> dict[int, dict[str, Any]]:
    expected = validate_input_manifest(
        input_manifest, expected_role=role, expected_images=int(input_manifest["images"])
    )
    records: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid candidate JSONL at line {number}") from error
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"invalid candidate record at line {number}")
            image_id = _positive_int(raw.get("image_id"), field="image_id")
            if image_id in records:
                raise RuntimeError("candidate records contain duplicate image IDs")
            if image_id not in expected:
                raise RuntimeError("candidate record is outside the frozen role")
            records[image_id] = _canonical_record(
                raw,
                expected=expected[image_id],
                backend=backend,
                model_sha256=model_sha256,
                role=role,
            )
    if set(records) != set(expected):
        missing = len(set(expected) - set(records))
        raise RuntimeError(f"candidate records are incomplete: missing {missing}")
    return records
