#!/usr/bin/env python3
"""Export backend-specific CAVS full-fit models for the RK3588 client."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

import contracts


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]

ACTION_IDS = ("edge", "original", "q90", "current@640")
EXPECTED_KEYS = frozenset(
    {
        "class_embedding.weight",
        "box_encoder.0.weight",
        "box_encoder.0.bias",
        "box_encoder.1.weight",
        "box_encoder.1.bias",
        "box_encoder.3.weight",
        "box_encoder.3.bias",
        "scene_encoder.0.weight",
        "scene_encoder.0.bias",
        "scene_encoder.1.weight",
        "scene_encoder.1.bias",
        "image_encoder.0.weight",
        "image_encoder.0.bias",
        "image_encoder.1.weight",
        "image_encoder.1.bias",
        "action_type_embedding.weight",
        "action_encoder.0.weight",
        "action_encoder.0.bias",
        "action_encoder.1.weight",
        "action_encoder.1.bias",
        "action_encoder.3.weight",
        "action_encoder.3.bias",
        "head.0.weight",
        "head.0.bias",
        "head.1.weight",
        "head.1.bias",
        "head.4.weight",
        "head.4.bias",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def tensor_arrays(state_dict: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert a learner state dict to the exact board tensor contract."""
    observed = frozenset(state_dict)
    if observed != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - observed)
        extra = sorted(observed - EXPECTED_KEYS)
        raise RuntimeError(f"policy tensor keys changed; missing={missing}, extra={extra}")
    arrays: dict[str, np.ndarray] = {}
    for key in sorted(EXPECTED_KEYS):
        value = state_dict[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite policy tensor: {key}")
        arrays[key] = np.array(array, dtype=np.float32, copy=True)
    return arrays


def build_manifest(
    protocol: Mapping[str, Any],
    backend: str,
    category_ids: Sequence[int],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = contracts.validate_backend(backend)
    model = dict(protocol["model"])
    rknn = dict(protocol["rknn_models"][value])
    digest = contracts.validate_model_sha256(str(rknn["sha256"]))
    if len(entries) != len(contracts.SEEDS):
        raise RuntimeError("policy export requires all three full-fit seeds")
    observed_seeds = tuple(int(entry["seed"]) for entry in entries)
    if observed_seeds != contracts.SEEDS:
        raise RuntimeError(f"policy seed order changed: {observed_seeds}")
    return {
        "schema_version": 1,
        "policy_type": "cavs_current_maur",
        "policy_id": f"cavs_current_maur_rknn_{value.replace('-', '_')}_v1",
        "display_name": f"CAVS-Current-MAUR RKNN {value} (3-seed)",
        "method": "CAVS-Current-MAUR",
        "backend": value,
        "rknn_model_sha256": digest,
        "profile": "current",
        "fusion": "MAUR",
        "action_ids": list(ACTION_IDS),
        "category_count": len(category_ids),
        "category_ids": list(map(int, category_ids)),
        "edge_model": {
            "name": str(rknn.get("name", f"yolo26n-rk3588-{value}.rknn")),
            "sha256": digest,
        },
        "deploy_confidence": float(model["deploy_confidence"]),
        "architecture": {
            "class_embedding_dim": int(model["class_embedding_dim"]),
            "box_hidden_dim": int(model["box_hidden_dim"]),
            "action_hidden_dim": int(model["action_hidden_dim"]),
            "joint_hidden_dim": int(model["joint_hidden_dim"]),
            "dropout": float(model["dropout"]),
            "max_boxes": int(model["max_boxes"]),
        },
        "roi": {
            "action": "current@640",
            "confidence": float(protocol["roi"]["confidence"]),
            "max_area_ratio": float(protocol["roi"]["max_area_ratio"]),
            "imgsz": int(protocol["inference"]["imgsz"]),
        },
        "maur": dict(protocol["maur"]),
        "cloud_model_sha256": str(protocol["cloud_model_sha256"]),
        "source_protocol": str(protocol["protocol_id"]),
        "training_jobs_complete": 18,
        "seeds": [dict(entry) for entry in entries],
    }


def export_backend(root: str | Path, backend: str) -> dict[str, Any]:
    root_path = Path(root).resolve()
    protocol = json.loads((root_path / "configs/protocol.json").read_text(encoding="utf-8"))
    value = contracts.validate_backend(backend)
    runs = root_path / "runs/backends" / value
    learner_report = json.loads((runs / "learners/report.json").read_text(encoding="utf-8"))
    if not learner_report.get("complete") or int(learner_report.get("jobs", -1)) != 18:
        raise RuntimeError(f"learner matrix is incomplete: {value}")

    output = runs / "export/policy"
    entries: list[dict[str, Any]] = []
    category_ids: list[int] | None = None
    for seed in contracts.SEEDS:
        job_root = runs / f"learners/jobs/seed_{seed}/full"
        report = json.loads((job_root / "report.json").read_text(encoding="utf-8"))
        if report.get("old_learner_checkpoint_reused"):
            raise RuntimeError(f"old learner checkpoint was reused: {value}/{seed}")
        with (job_root / "artifact.pkl").open("rb") as handle:
            outer = pickle.load(handle)
        if outer.get("backend") != value or outer.get("fold") != "full":
            raise RuntimeError(f"artifact contract changed: {value}/{seed}")
        expected_model = str(protocol["rknn_models"][value]["sha256"])
        if str(outer.get("rknn_model_sha256")) != expected_model:
            raise RuntimeError(f"artifact model hash changed: {value}/{seed}")
        current_categories = list(map(int, outer["category_ids"]))
        if category_ids is None:
            category_ids = current_categories
        elif current_categories != category_ids:
            raise RuntimeError("category schema differs across seeds")
        inner = outer["artifact"]
        atomic_npz(output / f"seed_{seed}.npz", tensor_arrays(inner["state_dict"]))
        metadata = dict(inner["metadata"])
        entries.append(
            {
                "seed": int(seed),
                "file": f"seed_{seed}.npz",
                "sha256": file_sha256(output / f"seed_{seed}.npz"),
                "reward_mean": float(metadata["reward_mean"]),
                "reward_std": float(metadata["reward_std"]),
                "selected_epoch": int(metadata["selected_epoch"]),
                "source_artifact_sha256": file_sha256(job_root / "artifact.pkl"),
            }
        )
    assert category_ids is not None
    manifest = build_manifest(protocol, value, category_ids, entries)
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)
    result = {
        "complete": True,
        "backend": value,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "training_jobs": 18,
        "old_learner_checkpoints_reused": False,
    }
    atomic_json(runs / "export/completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=(*contracts.BACKENDS, "both"), required=True)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    backends = contracts.BACKENDS if args.backend == "both" else (args.backend,)
    result = {backend: export_backend(args.root, backend) for backend in backends}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
