import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oric_current_oqr_hybrid_core", ROOT / "scripts/core.py"
)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class HybridCoreTest(unittest.TestCase):
    def test_protocol_matrix_is_frozen(self):
        summary = CORE.validate_protocol(PROTOCOL)
        self.assertEqual(summary["aggregate_rows"], 24)
        self.assertEqual(summary["bootstrap_comparisons"], 12)

    def test_oric_anchors_original_and_current_supplies_advantages(self):
        current = {
            11: {"original": 0.20, "q90": 0.25, "roi": 0.17},
            12: {"original": -0.10, "q90": -0.04, "roi": 0.02},
        }
        hybrid = CORE.compose_hybrid_scores(
            [11, 12], {11: 0.70, 12: 0.30}, current,
            actions=("original", "q90", "roi"),
        )
        self.assertAlmostEqual(hybrid[11]["original"], 0.70)
        self.assertAlmostEqual(hybrid[12]["original"], 0.30)
        for image_id in (11, 12):
            for action in ("q90", "roi"):
                self.assertAlmostEqual(
                    hybrid[image_id][action] - hybrid[image_id]["original"],
                    current[image_id][action] - current[image_id]["original"],
                )

    def test_oq_hybrid_cannot_leak_roi(self):
        hybrid = CORE.compose_hybrid_scores(
            [1], {1: 0.5},
            {1: {"original": 0.2, "q90": 0.3, "roi": 0.8}},
            actions=("original", "q90"),
        )
        self.assertEqual(tuple(hybrid[1]), ("original", "q90"))

    def test_unavailable_optional_action_remains_omitted(self):
        hybrid = CORE.compose_hybrid_scores(
            [1], {1: 0.5},
            {1: {"original": 0.2, "q90": 0.3}},
            actions=("original", "q90", "roi"),
        )
        self.assertEqual(tuple(hybrid[1]), ("original", "q90"))
        self.assertAlmostEqual(hybrid[1]["original"], 0.5)
        self.assertAlmostEqual(hybrid[1]["q90"], 0.6)

    def test_score_populations_must_align_exactly(self):
        with self.assertRaisesRegex(ValueError, "align"):
            CORE.compose_hybrid_scores(
                [1, 2], {1: 0.5},
                {1: {"original": 0.2}, 2: {"original": 0.3}},
                actions=("original",),
            )

    def test_nonfinite_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            CORE.compose_hybrid_scores(
                [1], {1: np.nan}, {1: {"original": 0.2}},
                actions=("original",),
            )

    def test_capacity_contract_is_strict(self):
        CORE.assert_capacity(1000, 1000)
        with self.assertRaisesRegex(ValueError, "capacity"):
            CORE.assert_capacity(1001, 1000)


if __name__ == "__main__":
    unittest.main()
