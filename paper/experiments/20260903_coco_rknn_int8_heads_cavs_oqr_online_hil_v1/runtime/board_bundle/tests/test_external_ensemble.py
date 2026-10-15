from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from baseline_numpy import external_score, load_policy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_constant_model(path: Path, value: float) -> None:
    np.savez_compressed(
        path,
        layer_count=np.asarray(1, dtype=np.int64),
        layer_0_weight=np.zeros((1, 205), dtype=np.float32),
        layer_0_bias=np.zeros(1, dtype=np.float32),
        layer_0_bn_weight=np.ones(1, dtype=np.float32),
        layer_0_bn_bias=np.zeros(1, dtype=np.float32),
        layer_0_bn_mean=np.zeros(1, dtype=np.float32),
        layer_0_bn_var=np.ones(1, dtype=np.float32),
        output_weight=np.zeros((1, 1), dtype=np.float32),
        output_bias=np.asarray([value], dtype=np.float32),
    )


def test_three_seed_external_policy_averages_scores(tmp_path: Path) -> None:
    rows = []
    values = {"oric": (1.0, 3.0, 5.0), "moric": (2.0, 4.0, 6.0)}
    seeds = (20260809, 20260810, 20260811)
    for method, scores in values.items():
        for seed, score in zip(seeds, scores):
            path = tmp_path / f"{method}_{seed}.npz"
            _write_constant_model(path, score)
            rows.append(
                {
                    "name": method,
                    "seed": seed,
                    "file": path.name,
                    "sha256": _sha256(path),
                    "input_dim": 205,
                }
            )
    manifest = {
        "schema_version": 2,
        "policy_type": "oric_moric_original_only_rknn",
        "rknn_model_sha256": "a" * 64,
        "aggregation": "mean_score",
        "ensemble_seeds": list(seeds),
        "top_k": 25,
        "category_ids": list(range(80)),
        "models": rows,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    policy = load_policy(manifest_path)

    assert len(policy["models_loaded"]["oric"]) == 3
    assert len(policy["models_loaded"]["moric"]) == 3
    assert external_score(policy, "oric", [], width=640, height=640) == 3.0
    assert external_score(policy, "moric", [], width=640, height=640) == 4.0
