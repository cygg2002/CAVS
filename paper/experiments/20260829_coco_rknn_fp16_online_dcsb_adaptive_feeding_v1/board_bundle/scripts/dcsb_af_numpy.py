"""Pure-NumPy DCSB and Adaptive Feeding policies for RK3588 deployment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POLICY_TYPE = "dcsb_adaptive_feeding_original_only_rknn"
METHODS = frozenset(
    {
        "dcsb_native",
        "dcsb_budget",
        "adaptive_feeding_native",
        "adaptive_feeding_budget",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(box: Mapping[str, Any]) -> float:
    value = float(box.get("conf", box.get("score", 0.0)))
    if not math.isfinite(value):
        raise ValueError("box confidence must be finite")
    return value


def adaptive_feeding_feature(
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    category_ids: Sequence[int],
    top_k: int,
) -> np.ndarray:
    """Return class histogram + Top-K (confidence, xmin, ymin, w, h)."""
    width, height, top_k = int(width), int(height), int(top_k)
    categories = tuple(map(int, category_ids))
    if width <= 0 or height <= 0 or top_k <= 0:
        raise ValueError("image dimensions and top_k must be positive")
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("category_ids must be non-empty and unique")
    ordered = sorted(
        enumerate(edge_boxes), key=lambda item: (-_score(item[1]), item[0])
    )[:top_k]
    feature = np.zeros(len(categories) + 5 * top_k, dtype=np.float64)
    for rank, (_, box) in enumerate(ordered):
        class_index = int(box.get("cls", -1))
        if not 0 <= class_index < len(categories):
            raise ValueError(f"native class index is outside the schema: {class_index}")
        xyxy = box.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            raise ValueError("box xyxy must contain four coordinates")
        x1, y1, x2, y2 = map(float, xyxy)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("box coordinates must be finite")
        feature[class_index] += 1.0
        offset = len(categories) + 5 * rank
        feature[offset : offset + 5] = (
            _score(box),
            x1 / width,
            y1 / height,
            max(0.0, x2 - x1) / width,
            max(0.0, y2 - y1) / height,
        )
    if not np.isfinite(feature).all():
        raise ValueError("Adaptive Feeding feature is not finite")
    return feature


def adaptive_feeding_margin(
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    policy: Mapping[str, Any],
) -> float:
    feature = adaptive_feeding_feature(
        edge_boxes,
        width=width,
        height=height,
        category_ids=policy["category_ids"],
        top_k=int(policy["top_k"]),
    )
    coefficient = np.asarray(policy["adaptive_feeding_coefficient"], dtype=np.float64)
    intercept = float(policy["adaptive_feeding_intercept"])
    if coefficient.shape != feature.shape:
        raise ValueError("Adaptive Feeding parameter dimension changed")
    value = float(np.dot(coefficient, feature) + intercept)
    if not math.isfinite(value):
        raise ValueError("Adaptive Feeding margin is not finite")
    return value


def _filtered_signal(
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    confidence: float,
) -> tuple[int, float]:
    selected = [box for box in edge_boxes if _score(box) > float(confidence)]
    if not selected:
        return 0, 0.0
    denominator = float(int(width) * int(height))
    if denominator <= 0:
        raise ValueError("image dimensions must be positive")
    areas = []
    for box in selected:
        x1, y1, x2, y2 = map(float, box["xyxy"])
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) / denominator)
    return len(selected), float(min(areas))


def dcsb_margin(
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    model: Mapping[str, Any],
) -> float:
    estimated_count, minimum_area = _filtered_signal(
        edge_boxes,
        width=width,
        height=height,
        confidence=float(model["confidence_threshold"]),
    )
    detected_count, _ = _filtered_signal(
        edge_boxes,
        width=width,
        height=height,
        confidence=float(model["detector_confidence"]),
    )
    if estimated_count == detected_count:
        return -1.0
    count_threshold = float(model["count_threshold"])
    area_threshold = float(model["area_threshold"])
    count_value = (estimated_count - count_threshold) / max(1.0, count_threshold)
    area_value = (area_threshold - minimum_area) / max(1e-12, area_threshold)
    value = float(max(count_value, area_value))
    if not math.isfinite(value):
        raise ValueError("DCSB margin is not finite")
    return value


def empirical_percentile(value: float, sorted_reference: np.ndarray) -> float:
    reference = np.asarray(sorted_reference, dtype=np.float64).reshape(-1)
    if not math.isfinite(float(value)) or not len(reference):
        raise ValueError("percentile value and reference must be finite and non-empty")
    if not np.isfinite(reference).all() or np.any(reference[1:] < reference[:-1]):
        raise ValueError("percentile reference must be sorted and finite")
    return float(np.searchsorted(reference, float(value), side="right") / len(reference))


def _bound_file(manifest_path: Path, row: object, *, field: str) -> Path:
    if not isinstance(row, Mapping):
        raise ValueError(f"{field} binding is missing")
    relative = row.get("file")
    digest = str(row.get("sha256", "")).lower()
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} file is invalid")
    path = (manifest_path.parent / relative).resolve()
    if path.parent != manifest_path.parent or not path.is_file():
        raise FileNotFoundError(path)
    if len(digest) != 64 or _sha256(path) != digest:
        raise ValueError(f"{field} SHA-256 mismatch")
    return path


def load_policy(manifest_path: str | Path) -> dict[str, Any]:
    """Load a hash-bound policy without pickle, sklearn, or PyTorch."""
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read DCSB/Adaptive Feeding policy") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("policy manifest must be an object")
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported DCSB/Adaptive Feeding policy schema")
    if manifest.get("policy_type") != POLICY_TYPE:
        raise ValueError("unexpected DCSB/Adaptive Feeding policy type")
    rknn_hash = str(manifest.get("rknn_model_sha256", "")).lower()
    if len(rknn_hash) != 64 or any(char not in "0123456789abcdef" for char in rknn_hash):
        raise ValueError("policy requires a valid RKNN SHA-256")

    categories = tuple(map(int, manifest.get("category_ids", ())))
    top_k = int(manifest.get("top_k", 0))
    if len(categories) != 80 or len(categories) != len(set(categories)) or top_k != 50:
        raise ValueError("COCO policy requires 80 categories and Top-50 proposals")

    model_path = _bound_file(path, manifest.get("adaptive_feeding"), field="adaptive_feeding")
    reference_path = _bound_file(
        path, manifest.get("calibration_references"), field="calibration_references"
    )
    with np.load(model_path, allow_pickle=False) as archive:
        coefficient = np.asarray(archive["coefficient"], dtype=np.float64).reshape(-1)
        intercept_values = np.asarray(archive["intercept"], dtype=np.float64).reshape(-1)
    expected_dimension = len(categories) + 5 * top_k
    if coefficient.shape != (expected_dimension,) or intercept_values.shape != (1,):
        raise ValueError("Adaptive Feeding parameter shapes changed")
    with np.load(reference_path, allow_pickle=False) as archive:
        dcsb_reference = np.asarray(archive["dcsb_margin"], dtype=np.float64).reshape(-1)
        af_reference = np.asarray(
            archive["adaptive_feeding_margin"], dtype=np.float64
        ).reshape(-1)
    if (
        not len(dcsb_reference)
        or not len(af_reference)
        or not np.isfinite(dcsb_reference).all()
        or not np.isfinite(af_reference).all()
        or np.any(dcsb_reference[1:] < dcsb_reference[:-1])
        or np.any(af_reference[1:] < af_reference[:-1])
    ):
        raise ValueError("calibration references must be sorted, finite, and non-empty")

    dcsb = manifest.get("dcsb")
    required = {
        "confidence_threshold",
        "detector_confidence",
        "count_threshold",
        "area_threshold",
    }
    if not isinstance(dcsb, Mapping) or not required <= set(dcsb):
        raise ValueError("DCSB threshold model is incomplete")
    return {
        **dict(manifest),
        "category_ids": categories,
        "adaptive_feeding_coefficient": coefficient,
        "adaptive_feeding_intercept": float(intercept_values[0]),
        "dcsb_model": dict(dcsb),
        "dcsb_reference": dcsb_reference,
        "adaptive_feeding_reference": af_reference,
    }


def method_score(
    policy: Mapping[str, Any],
    method: str,
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> float:
    name = str(method).lower()
    if name not in METHODS:
        raise ValueError(f"unsupported DCSB/Adaptive Feeding method: {name}")
    if name.startswith("dcsb_"):
        raw = dcsb_margin(
            edge_boxes,
            width=int(width),
            height=int(height),
            model=policy["dcsb_model"],
        )
        reference = policy["dcsb_reference"]
    else:
        raw = adaptive_feeding_margin(
            edge_boxes,
            width=int(width),
            height=int(height),
            policy=policy,
        )
        reference = policy["adaptive_feeding_reference"]
    return empirical_percentile(raw, reference) if name.endswith("_budget") else float(raw)
