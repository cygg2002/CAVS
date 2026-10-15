from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "rknn_export_policy_backend", ROOT / "scripts" / "export_policy_backend.py"
)
assert SPEC and SPEC.loader
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


class ExportPolicyBackendTests(unittest.TestCase):
    def test_tensor_arrays_require_the_board_policy_key_set(self) -> None:
        state = {key: np.zeros((1,), dtype=np.float32) for key in EXPORT.EXPECTED_KEYS}

        arrays = EXPORT.tensor_arrays(state)

        self.assertEqual(set(arrays), EXPORT.EXPECTED_KEYS)
        self.assertTrue(all(value.dtype == np.float32 for value in arrays.values()))

    def test_manifest_is_backend_and_model_bound(self) -> None:
        protocol = {
            "protocol_id": "test-protocol",
            "cloud_model_sha256": "c" * 64,
            "rknn_models": {"fp16": {"sha256": "a" * 64}},
            "model": {
                "deploy_confidence": 0.001,
                "class_embedding_dim": 8,
                "box_hidden_dim": 32,
                "action_hidden_dim": 32,
                "joint_hidden_dim": 64,
                "dropout": 0.05,
                "max_boxes": 64,
            },
            "roi": {"confidence": 0.25, "max_area_ratio": 0.65},
            "maur": {"duplicate_iou": 0.5, "nms_iou": 0.7},
            "inference": {"imgsz": 640},
        }
        entries = [
            {"seed": seed, "file": "seed.npz", "sha256": "d" * 64}
            for seed in (20260809, 20260810, 20260811)
        ]

        manifest = EXPORT.build_manifest(protocol, "fp16", list(range(80)), entries)

        self.assertEqual(manifest["backend"], "fp16")
        self.assertEqual(manifest["rknn_model_sha256"], "a" * 64)
        self.assertEqual(manifest["cloud_model_sha256"], "c" * 64)
        self.assertEqual(manifest["training_jobs_complete"], 18)
