"""Pure-NumPy inference for the frozen three-seed CAVS Current+MAUR policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_IDS = ("edge", "original", "q90", "current@640")
ACTION_TYPES = {"edge": 0, "full": 1, "roi": 2, "pair": 3, "full_region": 4}
EXPECTED_SEEDS = (20260809, 20260810, 20260811)
EXPECTED_KEYS = {
    "class_embedding.weight", "box_encoder.0.weight", "box_encoder.0.bias",
    "box_encoder.1.weight", "box_encoder.1.bias", "box_encoder.3.weight",
    "box_encoder.3.bias", "scene_encoder.0.weight", "scene_encoder.0.bias",
    "scene_encoder.1.weight", "scene_encoder.1.bias", "image_encoder.0.weight",
    "image_encoder.0.bias", "image_encoder.1.weight", "image_encoder.1.bias",
    "action_type_embedding.weight", "action_encoder.0.weight", "action_encoder.0.bias",
    "action_encoder.1.weight", "action_encoder.1.bias", "action_encoder.3.weight",
    "action_encoder.3.bias", "head.0.weight", "head.0.bias", "head.1.weight",
    "head.1.bias", "head.4.weight", "head.4.bias",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    architecture = manifest.get("architecture", {})
    required_architecture = {
        "class_embedding_dim": 8, "box_hidden_dim": 32,
        "action_hidden_dim": 32, "joint_hidden_dim": 64, "max_boxes": 64,
    }
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported policy schema")
    if manifest.get("policy_type") != "cavs_current_maur" or manifest.get("method") != "CAVS-Current-MAUR":
        raise ValueError("not the frozen CAVS Current+MAUR policy")
    if tuple(manifest.get("action_ids", ())) != ACTION_IDS or int(manifest.get("category_count", -1)) != 80:
        raise ValueError("policy action or category contract changed")
    if manifest.get("profile") != "current" or manifest.get("fusion") != "MAUR":
        raise ValueError("policy profile or fusion contract changed")
    if not isinstance(architecture, Mapping) or any(int(architecture.get(k, -1)) != v for k, v in required_architecture.items()):
        raise ValueError("policy architecture changed")
    if tuple(int(row.get("seed", -1)) for row in manifest.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("policy seed ensemble changed")


def load_policy(manifest_path: str | Path) -> dict[str, Any]:
    """Load hash-pinned NPZ weights without pickle or PyTorch."""
    path = Path(manifest_path).resolve()
    if not path.is_file() or path.suffix != ".json":
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    seeds: list[dict[str, Any]] = []
    for row in manifest["seeds"]:
        filename = str(row["file"])
        seed_path = (path.parent / filename).resolve()
        if seed_path.parent != path.parent.resolve() or not seed_path.is_file():
            raise FileNotFoundError(filename)
        if _sha256(seed_path) != str(row["sha256"]).lower():
            raise ValueError(f"policy seed checksum mismatch: {filename}")
        with np.load(seed_path, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_KEYS:
                raise ValueError(f"unexpected policy tensors: {filename}")
            weights = {key: np.asarray(archive[key], dtype=np.float32).copy() for key in archive.files}
        mean, std = float(row["reward_mean"]), float(row["reward_std"])
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"invalid reward normalization: {filename}")
        seeds.append({"seed": int(row["seed"]), "weights": weights, "reward_mean": mean, "reward_std": std})
    return {**manifest, "seeds_loaded": seeds}


def _linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return value @ weight.T + bias


def _layer_norm(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = np.mean(value, axis=-1, keepdims=True)
    variance = np.mean((value - mean) ** 2, axis=-1, keepdims=True)
    return (value - mean) / np.sqrt(variance + np.float32(1e-5)) * weight + bias


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (np.float32(1.0) + np.exp(-value))


def _expansion(area_ratio: float) -> float:
    return 1.8 if area_ratio < 0.01 else 1.6 if area_ratio < 0.10 else 1.4


def current_roi_xyxy(
    width: int, height: int, edge_boxes: Sequence[Mapping[str, Any]], *, confidence: float = 0.25,
    max_area_ratio: float = 0.65,
) -> list[int] | None:
    """Build the frozen merged Current@640 crop from raw Edge detections."""
    image_width, image_height = max(1, int(width)), max(1, int(height))
    expanded: list[tuple[float, float, float, float]] = []
    for box in edge_boxes:
        if float(box.get("conf", box.get("score", 0.0))) < confidence:
            continue
        x1, y1, x2, y2 = map(float, box["xyxy"])
        box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        scale = _expansion(box_width * box_height / float(image_width * image_height))
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        expanded.append((
            max(0.0, center_x - box_width * scale / 2.0), max(0.0, center_y - box_height * scale / 2.0),
            min(float(image_width), center_x + box_width * scale / 2.0), min(float(image_height), center_y + box_height * scale / 2.0),
        ))
    if not expanded:
        return None
    x1, y1 = max(0, int(min(box[0] for box in expanded))), max(0, int(min(box[1] for box in expanded)))
    x2, y2 = min(image_width, int(max(box[2] for box in expanded) + 0.999999)), min(image_height, int(max(box[3] for box in expanded) + 0.999999))
    if x2 <= x1 or y2 <= y1:
        return None
    return None if (x2 - x1) * (y2 - y1) / float(image_width * image_height) > max_area_ratio else [x1, y1, x2, y2]


def _features(
    edge_boxes: Sequence[Mapping[str, Any]], *, width: int, height: int, max_boxes: int,
    confidence: float, roi_xyxy: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    image_width, image_height = max(1.0, float(width)), max(1.0, float(height))
    ordered = [dict(box) for box in edge_boxes if float(box.get("conf", box.get("score", 0.0))) >= confidence]
    ordered.sort(key=lambda box: (-float(box.get("conf", box.get("score", 0.0))), int(box.get("cls", 0)), tuple(map(float, box["xyxy"]))))
    numeric, classes = np.zeros((max_boxes, 6), np.float32), np.zeros(max_boxes, np.int64)
    for index, box in enumerate(ordered[:max_boxes]):
        x1, y1, x2, y2 = map(float, box["xyxy"])
        box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        numeric[index] = ((x1 + x2) / 2 / image_width, (y1 + y2) / 2 / image_height, box_width / image_width, box_height / image_height, box_width * box_height / (image_width * image_height), float(box.get("conf", box.get("score", 0.0))))
        classes[index] = int(box.get("cls", 0))
    count = min(len(ordered), max_boxes)
    scores = [float(box.get("conf", box.get("score", 0.0))) for box in ordered]
    areas = [max(0.0, float(box["xyxy"][2]) - float(box["xyxy"][0])) * max(0.0, float(box["xyxy"][3]) - float(box["xyxy"][1])) / (image_width * image_height) for box in ordered]
    scene = np.asarray((math.log(image_width / image_height), 0.0, min(1.0, len(ordered) / 64.0), max(scores, default=0.0), fmean(scores) if scores else 0.0, min(1.0, sum(areas)), 0.0, 0.0, 0.0), dtype=np.float32)
    actions = np.zeros((4, 18), dtype=np.float32)
    actions[1:, 0:3] = 1.0
    actions[1:, 4] = 0.5
    available = np.asarray((True, True, True, roi_xyxy is not None), dtype=bool)
    types = np.asarray((ACTION_TYPES["edge"], ACTION_TYPES["full"], ACTION_TYPES["pair"], ACTION_TYPES["roi"]), dtype=np.int64)
    if roi_xyxy is not None:
        x1, y1, x2, y2 = map(float, roi_xyxy)
        ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (image_width * image_height)
        actions[3, 5], actions[3, 13], actions[3, 14], actions[3, 17] = 0.5, ratio, ratio, 1.0
    return numeric, classes, count, scene, actions, types * available.astype(np.int64)


def _seed_scores(seed: Mapping[str, Any], numeric: np.ndarray, classes: np.ndarray, count: int, scene: np.ndarray, actions: np.ndarray, action_types: np.ndarray) -> np.ndarray:
    weights = seed["weights"]
    embedded = weights["class_embedding.weight"][np.clip(classes, 0, 79)]
    encoded = _silu(_linear(_silu(_layer_norm(_linear(np.concatenate((numeric, embedded), axis=-1), weights["box_encoder.0.weight"], weights["box_encoder.0.bias"]), weights["box_encoder.1.weight"], weights["box_encoder.1.bias"])), weights["box_encoder.3.weight"], weights["box_encoder.3.bias"]))
    mean = encoded[:count].mean(axis=0) if count else np.zeros(32, np.float32)
    maximum = encoded[:count].max(axis=0) if count else np.zeros(32, np.float32)
    scene_encoded = _silu(_layer_norm(_linear(scene, weights["scene_encoder.0.weight"], weights["scene_encoder.0.bias"]), weights["scene_encoder.1.weight"], weights["scene_encoder.1.bias"]))
    image = _silu(_layer_norm(_linear(np.concatenate((mean, maximum, scene_encoded)), weights["image_encoder.0.weight"], weights["image_encoder.0.bias"]), weights["image_encoder.1.weight"], weights["image_encoder.1.bias"]))
    action_input = np.concatenate((actions, weights["action_type_embedding.weight"][np.clip(action_types, 0, 4)], np.zeros((4, 1), dtype=np.float32)), axis=-1)
    action_encoded = _silu(_linear(_silu(_layer_norm(_linear(action_input, weights["action_encoder.0.weight"], weights["action_encoder.0.bias"]), weights["action_encoder.1.weight"], weights["action_encoder.1.bias"])), weights["action_encoder.3.weight"], weights["action_encoder.3.bias"]))
    head = _silu(_layer_norm(_linear(np.concatenate((np.broadcast_to(image, (4, 64)), action_encoded), axis=-1), weights["head.0.weight"], weights["head.0.bias"]), weights["head.1.weight"], weights["head.1.bias"]))
    return _linear(head, weights["head.4.weight"], weights["head.4.bias"]).reshape(-1)


def predict_scores(policy: Mapping[str, Any], edge_boxes: Sequence[Mapping[str, Any]], *, width: int, height: int, roi_xyxy: Sequence[int] | None) -> dict[str, float]:
    """Return the frozen ensemble value for each available Edge/O/Q/R action."""
    architecture = policy["architecture"]
    numeric, classes, count, scene, actions, action_types = _features(edge_boxes, width=width, height=height, max_boxes=int(architecture["max_boxes"]), confidence=float(policy["deploy_confidence"]), roi_xyxy=roi_xyxy)
    available = (True, True, True, roi_xyxy is not None)
    rows = []
    for seed in policy["seeds_loaded"]:
        values = _seed_scores(seed, numeric, classes, count, scene, actions, action_types).astype(np.float64)
        values = values * float(seed["reward_std"]) + float(seed["reward_mean"])
        values[0] = 0.0
        rows.append(values)
    ensemble = np.mean(np.stack(rows), axis=0)
    return {action: float(ensemble[index]) for index, action in enumerate(ACTION_IDS) if available[index] and math.isfinite(float(ensemble[index]))}
