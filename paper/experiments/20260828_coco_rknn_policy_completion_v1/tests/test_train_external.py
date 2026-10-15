from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "rknn_train_external", ROOT / "scripts" / "train_external.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_targets_preserves_oric_reward_and_ranks_moric() -> None:
    rewards = np.asarray([2.0, -1.0, 2.0, 0.0], dtype=np.float64)

    targets = MODULE.build_targets(rewards)

    np.testing.assert_allclose(targets["oric"][0], rewards)
    np.testing.assert_allclose(targets["oric"][1], np.ones(4))
    np.testing.assert_allclose(targets["moric"][0], [1.0, 0.25, 1.0, 0.5])
    np.testing.assert_allclose(targets["moric"][1], [1.0, 0.25, 1.0, 0.5])


def test_feature_matrix_uses_frozen_sample_order_and_top_k_contract() -> None:
    samples = [
        {"image_id": 7, "width": 100, "height": 50},
        {"image_id": 9, "width": 200, "height": 100},
    ]
    predictions = {
        7: [{"image_id": 7, "category_id": 1, "bbox": [10, 5, 20, 10], "score": 0.8}],
        9: [],
    }

    features = MODULE.feature_matrix(samples, predictions, [1, 2], top_k=1)

    assert features.shape == (2, 7)
    np.testing.assert_allclose(features[0], [1, 0, 0.2, 0.2, 0.2, 0.2, 0.8])
    np.testing.assert_allclose(features[1], np.zeros(7))


def test_manifest_requires_three_seeds_for_both_methods() -> None:
    rows = [
        {
            "name": method,
            "seed": seed,
            "file": f"{method}_{seed}.npz",
            "sha256": "a" * 64,
            "input_dim": 205,
            "hidden_dims": [16, 16, 16, 16],
            "selected_epoch": 1,
        }
        for method in ("oric", "moric")
        for seed in (20260809, 20260810, 20260811)
    ]

    manifest = MODULE.build_manifest(
        rows,
        category_ids=list(range(80)),
        source_hashes={"source": "b" * 64},
        protocol={
            "protocol_id": "test",
            "seeds": [20260809, 20260810, 20260811],
            "rknn_models": {"fp16": {"sha256": "c" * 64}},
            "cloud_model_sha256": "d" * 64,
            "external": {
                "top_k": 25,
                "context_size": 1000,
                "context_seed": 20260728,
                "context_metric_iou_thresholds": [0.5],
            },
        },
    )

    assert manifest["schema_version"] == 2
    assert manifest["aggregation"] == "mean_score"
    assert manifest["ensemble_seeds"] == [20260809, 20260810, 20260811]
    assert len(manifest["models"]) == 6
    with pytest.raises(RuntimeError, match="matrix"):
        MODULE.build_manifest(
            rows[:-1],
            category_ids=list(range(80)),
            source_hashes={"source": "b" * 64},
            protocol={
                "protocol_id": "test",
                "seeds": [20260809, 20260810, 20260811],
                "rknn_models": {"fp16": {"sha256": "c" * 64}},
                "cloud_model_sha256": "d" * 64,
                "external": {
                    "top_k": 25,
                    "context_size": 1000,
                    "context_seed": 20260728,
                    "context_metric_iou_thresholds": [0.5],
                },
            },
        )


def test_exported_numpy_model_matches_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    core = MODULE.EXTERNAL_CORE
    rng = np.random.default_rng(4)
    features = rng.normal(size=(16, 7)).astype(np.float32)
    targets = rng.normal(size=16).astype(np.float32)
    checkpoint = core.fit_reward_mlp(
        features,
        targets,
        np.ones(16, dtype=np.float32),
        {
            "hidden_dims": [4],
            "dropout": 0.0,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "milestones": [2],
            "gamma": 0.5,
            "fit_validation_fraction": 0.25,
            "seed": 8,
        },
    )

    arrays = MODULE.checkpoint_arrays(checkpoint)
    expected = core.predict_reward_mlp(checkpoint, features)
    actual = np.asarray([MODULE.predict_exported_mlp(arrays, row) for row in features])

    assert torch is not None
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
