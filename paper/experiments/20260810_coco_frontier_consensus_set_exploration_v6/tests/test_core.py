from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fcs_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(CORE)


class FrontierConsensusTest(unittest.TestCase):
    def test_two_of_three_and_edge_retention(self) -> None:
        membership = np.array(
            [
                [[0, 1, 0], [1, 0, 1]],
                [[0, 1, 1], [1, 0, 0]],
                [[0, 0, 0], [1, 0, 1]],
            ],
            dtype=bool,
        )
        available = np.ones((2, 3), dtype=bool)
        mandatory = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
        result = CORE.majority_mask(
            membership, available, minimum_votes=2, mandatory_columns=mandatory
        )
        expected = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
        np.testing.assert_array_equal(result, expected)

    def test_pruning_resolves_action_ids_per_record(self) -> None:
        records = [
            {
                "image_id": 1,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "roi"},
                ],
            },
            {
                "image_id": 2,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "q90"},
                    {"action_id": "roi"},
                ],
            },
        ]
        scores = np.array([[0, 1, 2, 0], [0, 3, 4, 5]], dtype=float)
        keep = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=bool)
        result = CORE.prune_score_map(records, scores, keep, roi_only=True)
        self.assertEqual(result[1], {"original": 1.0})
        self.assertEqual(result[2], {"original": 3.0, "q90": 4.0})

    def test_gate_requires_two_material_byte_gains(self) -> None:
        baseline = {
            ("fixed_byte", 0.1): 1.0,
            ("fixed_byte", 0.25): 1.0,
            ("fixed_byte", 0.4): 1.0,
            ("fixed_call", 0.1): 1.0,
            ("fixed_call", 0.25): 1.0,
            ("fixed_call", 0.4): 1.0,
        }
        candidate = dict(baseline)
        candidate[("fixed_byte", 0.1)] += 0.002
        candidate[("fixed_byte", 0.25)] += 0.0011
        config = {
            "minimum_fixed_byte_gain": 0.001,
            "required_fixed_byte_gain_points": 2,
            "maximum_fixed_byte_loss": 0.001,
            "maximum_fixed_call_loss": 0.001,
        }
        self.assertTrue(CORE.screen_gate(baseline, candidate, config)["passed"])


if __name__ == "__main__":
    unittest.main()
