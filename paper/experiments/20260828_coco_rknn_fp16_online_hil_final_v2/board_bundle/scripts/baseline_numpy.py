"""Pure-NumPy score functions for frozen ORIC, MORIC, and Confidence baselines."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_EXTERNAL_SEEDS = (20260809, 20260810, 20260811)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confidence_score(
    edge_boxes: Sequence[Mapping[str, Any]], *, threshold: float = 0.25
) -> float:
    """Frozen confidence-only priority: lower surviving confidence uploads first."""
    scores = [
        float(box.get("conf", box.get("score", 0.0)))
        for box in edge_boxes
        if float(box.get("conf", box.get("score", 0.0))) >= float(threshold)
    ]
    return 2.0 if not scores else 1.0 - min(scores)


def build_output_feature(
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    category_ids: Sequence[int],
    top_k: int,
) -> np.ndarray:
    """Reproduce the public ORIC class-count plus top-K output feature."""
    if width <= 0 or height <= 0 or top_k <= 0:
        raise ValueError("image dimensions and top_k must be positive")
    categories = tuple(map(int, category_ids))
    if len(categories) != len(set(categories)):
        raise ValueError("category_ids must be unique")
    category_index = {category_id: index for index, category_id in enumerate(categories)}
    ordered = sorted(
        enumerate(edge_boxes),
        key=lambda item: (-float(item[1].get("conf", item[1].get("score", 0.0))), item[0]),
    )[: int(top_k)]
    feature = np.zeros(len(categories) + 5 * int(top_k), dtype=np.float32)
    for rank, (_, box) in enumerate(ordered):
        class_index = int(box.get("cls", box.get("category_id", -1)))
        if not 0 <= class_index < len(categories):
            raise ValueError(f"invalid native COCO class index: {class_index}")
        category_id = categories[class_index]
        feature[category_index[category_id]] += 1.0
        x1, y1, x2, y2 = map(float, box["xyxy"])
        offset = len(categories) + rank * 5
        feature[offset : offset + 5] = (
            (x1 + x2) / 2.0 / float(width),
            (y1 + y2) / 2.0 / float(height),
            max(0.0, x2 - x1) / float(width),
            max(0.0, y2 - y1) / float(height),
            float(box.get("conf", box.get("score", 0.0))),
        )
    if not np.isfinite(feature).all():
        raise ValueError("ORIC feature is not finite")
    return feature


def predict_mlp(model: Mapping[str, Any], feature: np.ndarray) -> float:
    """Evaluate the exported PyTorch MLP in eval mode (BatchNorm + ReLU)."""
    value = np.asarray(feature, dtype=np.float32).reshape(1, -1)
    for layer in model["layers"]:
        value = value @ np.asarray(layer["weight"], dtype=np.float32).T
        value += np.asarray(layer["bias"], dtype=np.float32)
        variance = np.asarray(layer["bn_var"], dtype=np.float32)
        value = (value - np.asarray(layer["bn_mean"], dtype=np.float32)) / np.sqrt(
            variance + np.float32(1e-5)
        )
        value = value * np.asarray(layer["bn_weight"], dtype=np.float32)
        value += np.asarray(layer["bn_bias"], dtype=np.float32)
        value = np.maximum(value, np.float32(0.0))
    output = value @ np.asarray(model["output_weight"], dtype=np.float32).T
    output += np.asarray(model["output_bias"], dtype=np.float32)
    score = float(output.reshape(-1)[0])
    if not math.isfinite(score):
        raise ValueError("external MLP output is not finite")
    return score


def load_policy(manifest_path: str | Path) -> dict[str, Any]:
    """Load hash-pinned external policies without pickle or PyTorch."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = int(manifest.get("schema_version", -1))
    if schema not in {1, 2}:
        raise ValueError("unsupported external policy schema")
    if manifest.get("policy_type") not in {
        "oric_moric_original_only",
        "oric_moric_original_only_rknn",
    }:
        raise ValueError("unexpected external policy type")
    category_ids = tuple(map(int, manifest.get("category_ids", ())))
    if len(category_ids) != 80 or len(set(category_ids)) != 80:
        raise ValueError("external policy requires 80 unique COCO categories")
    if int(manifest.get("top_k", 0)) != 25:
        raise ValueError("external policy top_k changed")
    models: dict[str, list[dict[str, Any]]] = {"oric": [], "moric": []}
    observed_seeds: dict[str, set[int]] = {"oric": set(), "moric": set()}
    for row in manifest.get("models", ()):
        name = str(row.get("name", ""))
        if name not in models:
            raise ValueError("external policy model set changed")
        seed = int(row.get("seed", -1))
        if schema == 2 and (seed not in EXPECTED_EXTERNAL_SEEDS or seed in observed_seeds[name]):
            raise ValueError(f"external policy seed set changed: {name}/{seed}")
        if schema == 1 and models[name]:
            raise ValueError("legacy external policy must contain one model per method")
        model_path = (path.parent / str(row.get("file", ""))).resolve()
        if model_path.parent != path.parent or not model_path.is_file():
            raise FileNotFoundError(model_path)
        if _sha256(model_path) != str(row.get("sha256", "")).lower():
            raise ValueError(f"external policy checksum mismatch: {name}")
        with np.load(model_path, allow_pickle=False) as archive:
            layer_count = int(np.asarray(archive["layer_count"]).item())
            layers = []
            for index in range(layer_count):
                prefix = f"layer_{index}_"
                layers.append(
                    {
                        "weight": np.asarray(archive[prefix + "weight"], dtype=np.float32),
                        "bias": np.asarray(archive[prefix + "bias"], dtype=np.float32),
                        "bn_weight": np.asarray(archive[prefix + "bn_weight"], dtype=np.float32),
                        "bn_bias": np.asarray(archive[prefix + "bn_bias"], dtype=np.float32),
                        "bn_mean": np.asarray(archive[prefix + "bn_mean"], dtype=np.float32),
                        "bn_var": np.asarray(archive[prefix + "bn_var"], dtype=np.float32),
                    }
                )
            model = {
                "seed": seed,
                "layers": layers,
                "output_weight": np.asarray(archive["output_weight"], dtype=np.float32),
                "output_bias": np.asarray(archive["output_bias"], dtype=np.float32),
            }
        input_dim = int(row.get("input_dim", -1))
        if input_dim != len(category_ids) + 5 * int(manifest["top_k"]):
            raise ValueError(f"external policy input dimension changed: {name}")
        if not layers or layers[0]["weight"].shape[1] != input_dim:
            raise ValueError(f"external policy tensor shape changed: {name}")
        models[name].append(model)
        observed_seeds[name].add(seed)
    if any(not rows for rows in models.values()):
        raise ValueError("external policy requires ORIC and MORIC")
    if schema == 2:
        seeds = tuple(map(int, manifest.get("ensemble_seeds", ())))
        if seeds != EXPECTED_EXTERNAL_SEEDS or manifest.get("aggregation") != "mean_score":
            raise ValueError("external policy ensemble contract changed")
        expected = set(EXPECTED_EXTERNAL_SEEDS)
        if any(observed_seeds[name] != expected for name in models):
            raise ValueError("external policy ensemble matrix is incomplete")
    return {**manifest, "models_loaded": models}


def external_score(
    policy: Mapping[str, Any],
    method: str,
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> float:
    name = str(method)
    if name == "confidence":
        return confidence_score(edge_boxes)
    if name not in {"oric", "moric"}:
        raise ValueError(f"unknown baseline: {name}")
    feature = build_output_feature(
        edge_boxes,
        width=int(width),
        height=int(height),
        category_ids=policy["category_ids"],
        top_k=int(policy["top_k"]),
    )
    models = policy["models_loaded"][name]
    return float(fmean(predict_mlp(model, feature) for model in models))
