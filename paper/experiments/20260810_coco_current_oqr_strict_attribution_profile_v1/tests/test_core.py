import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strict_attr_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "configs/protocol.json").read_text())

    def test_frozen_protocol_is_valid(self):
        CORE.validate_protocol(self.protocol)

    def test_protocol_rejects_hidden_final(self):
        changed = dict(self.protocol, final_is_hidden=True)
        with self.assertRaises(ValueError):
            CORE.validate_protocol(changed)


class ScoreTests(unittest.TestCase):
    def test_ensemble_requires_exact_frozen_seeds(self):
        values = {
            20260809: np.asarray([[1.0, 2.0]]),
            20260810: np.asarray([[3.0, 4.0]]),
            20260811: np.asarray([[5.0, 6.0]]),
        }
        result = CORE.ensemble_scores(values, [20260809, 20260810, 20260811])
        np.testing.assert_allclose(result, [[3.0, 4.0]])
        with self.assertRaises(ValueError):
            CORE.ensemble_scores({20260809: values[20260809]}, [20260809, 20260810])

    def test_original_only_selection_is_deterministic(self):
        image_ids = [4, 2, 3, 1]
        scores = {1: 0.5, 2: 0.9, 3: 0.9, 4: 0.1}
        self.assertEqual(
            CORE.original_only_selection(image_ids, scores, budget=0.5),
            {2: "original", 3: "original"},
        )


class ProfileTests(unittest.TestCase):
    def test_latency_summary_reports_median_and_p95(self):
        result = CORE.latency_summary_ns([1_000, 2_000, 3_000, 4_000, 5_000])
        self.assertEqual(result["median_ms"], 0.003)
        self.assertGreater(result["p95_ms"], result["median_ms"])

    def test_current_oqr_linear_flops_are_frozen(self):
        result = CORE.current_linear_flops(
            max_boxes=64,
            action_count=4,
            class_embedding_dim=8,
            box_hidden_dim=32,
            action_hidden_dim=32,
            joint_hidden_dim=64,
            scene_dim=9,
            action_dim=18,
            multiply_add_flops=2,
        )
        self.assertEqual(result["linear_macs"], 133024)
        self.assertAlmostEqual(result["linear_gflops"], 0.000266048)


if __name__ == "__main__":
    unittest.main()
