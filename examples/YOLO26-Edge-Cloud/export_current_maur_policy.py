#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Export trusted experiment pickles into a non-executable Current+MAUR package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from current_maur_uploader import EXPECTED_EDGE_MODEL, EXPECTED_EDGE_MODEL_SHA256

METHOD = "CAVS-Current-MAUR"
SEEDS = (20260809, 20260810, 20260811)
ACTION_IDS = ["edge", "original", "q90", "current@640"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_trusted_artifact(path: Path) -> dict[str, Any]:
    """Load only a locally generated, hash-verified experiment artifact."""
    with path.open("rb") as handle:
        # Pickle is confined to this trusted one-time exporter; the demo runtime accepts only NPZ.
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"invalid artifact object: {path}")
    return payload


def export(source_root: Path, output_dir: Path) -> Path:
    protocol = read_json(source_root / "configs/protocol.json")
    if protocol.get("cells", {}).get(METHOD) != {
        "profile": "current",
        "fusion": "MAUR",
        "reused_anchor": False,
    }:
        raise ValueError("source protocol is not the controlled Current+MAUR retrain")
    if read_json(source_root / "runs/training/completion.json") != {"complete": True, "jobs": 72}:
        raise ValueError("controlled retraining is incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_entries = []
    category_ids = None
    for seed in SEEDS:
        root = source_root / f"runs/training/jobs/{METHOD}/seed_{seed}/full"
        artifact_path = root / "artifact.pkl"
        report = read_json(root / "report.json")
        completion = read_json(root / "completion.json")
        source_hash = file_sha256(artifact_path)
        if report.get("artifact_sha256") != source_hash or completion.get("artifact_sha256") != source_hash:
            raise ValueError(f"source artifact checksum mismatch: seed={seed}")
        outer = load_trusted_artifact(artifact_path)
        expected_outer = {
            "method": METHOD,
            "profile": "current",
            "fusion": "MAUR",
            "seed": seed,
            "fold": "full",
            "cloud_model_sha256": protocol["cloud_model_sha256"],
        }
        if any(outer.get(key) != value for key, value in expected_outer.items()):
            raise ValueError(f"source artifact metadata mismatch: seed={seed}")
        if list(map(str, outer.get("action_space", ()))) != ACTION_IDS:
            raise ValueError(f"source action space mismatch: seed={seed}")
        current_categories = list(map(int, outer.get("category_ids", ())))
        if len(current_categories) != 80 or len(set(current_categories)) != 80:
            raise ValueError(f"source category schema mismatch: seed={seed}")
        if category_ids is None:
            category_ids = current_categories
        elif current_categories != category_ids:
            raise ValueError("seed category schemas differ")

        inner = outer.get("artifact", {})
        metadata = inner.get("metadata", {})
        state_dict = inner.get("state_dict", {})
        if inner.get("kind") != "current_action" or int(inner.get("category_count", -1)) != 80:
            raise ValueError(f"source learner kind mismatch: seed={seed}")
        reward_mean = float(metadata.get("reward_mean", math.nan))
        reward_std = float(metadata.get("reward_std", math.nan))
        if not math.isfinite(reward_mean) or not math.isfinite(reward_std) or reward_std <= 0.0:
            raise ValueError(f"source normalization mismatch: seed={seed}")
        arrays = {}
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"source state is not a tensor: {name}")
            array = tensor.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"source state contains non-finite values: {name}")
            arrays[str(name)] = array
        if not arrays:
            raise ValueError(f"source state dictionary is empty: seed={seed}")

        filename = f"seed_{seed}.npz"
        output_path = output_dir / filename
        atomic_npz(output_path, arrays)
        seed_entries.append(
            {
                "seed": seed,
                "file": filename,
                "sha256": file_sha256(output_path),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "selected_epoch": int(metadata.get("selected_epoch", -1)),
                "source_artifact_sha256": source_hash,
            }
        )

    model = protocol["model"]
    manifest = {
        "schema_version": 1,
        "policy_type": "cavs_current_maur",
        "policy_id": "cavs_current_maur_controlled_retrain_v2",
        "display_name": "CAVS-Current-MAUR (3-seed)",
        "method": METHOD,
        "profile": "current",
        "fusion": "MAUR",
        "action_ids": ACTION_IDS,
        "category_count": 80,
        "category_ids": category_ids,
        "edge_model": {"name": EXPECTED_EDGE_MODEL, "sha256": EXPECTED_EDGE_MODEL_SHA256},
        "deploy_confidence": float(model["deploy_confidence"]),
        "architecture": {
            "class_embedding_dim": int(model["class_embedding_dim"]),
            "box_hidden_dim": int(model["box_hidden_dim"]),
            "action_hidden_dim": int(model["action_hidden_dim"]),
            "joint_hidden_dim": int(model["joint_hidden_dim"]),
            "dropout": float(model["dropout"]),
            "max_boxes": int(model["max_boxes"]),
        },
        "roi": {"action": "current@640", "confidence": 0.25, "max_area_ratio": 0.65, "imgsz": 640},
        "maur": {"duplicate_iou": 0.5, "nms_iou": 0.7},
        "cloud_model_sha256": protocol["cloud_model_sha256"],
        "source_protocol": protocol["protocol_id"],
        "training_jobs_complete": 72,
        "seeds": seed_entries,
    }
    manifest_path = output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.source_root.resolve(), args.output_dir.resolve()))


if __name__ == "__main__":
    main()
