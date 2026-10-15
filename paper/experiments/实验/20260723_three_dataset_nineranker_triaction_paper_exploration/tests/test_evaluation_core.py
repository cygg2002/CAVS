from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation_core import (  # noqa: E402
    assert_compatible_controlled_features,
    expected_matrix_counts,
    pca32_matrix,
    validate_protocol,
)


class EvaluationCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_has_nine_rankers_and_eight_constraints(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(len(self.protocol["rankers"]), 9)
        self.assertEqual(len(self.protocol["constrained_rankers"]), 8)
        self.assertNotIn("Random", self.protocol["constrained_rankers"])
        self.assertNotIn("Legacy-Frozen-32D", self.protocol["rankers"])

    def test_expected_matrix_counts_match_frozen_contract(self) -> None:
        self.assertEqual(
            expected_matrix_counts(self.protocol),
            {
                "main": 384,
                "per_ranker_ablation": 120,
                "random_seed_rows": 1200,
                "bootstrap_comparisons": 24,
            },
        )

    def test_pca32_matrix_requires_exact_finite_dimension(self) -> None:
        samples = [{"pca32": [0.0] * 32}, {"pca32": [1.0] * 32}]
        self.assertEqual(pca32_matrix(samples).shape, (2, 32))
        with self.assertRaisesRegex(ValueError, "32"):
            pca32_matrix([{"pca32": [0.0] * 31}])
        invalid = [0.0] * 32
        invalid[4] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            pca32_matrix([{"pca32": invalid}])

    def test_controlled_feature_gate_rejects_wrong_current_layout(self) -> None:
        valid = {
            "structured": [0.0] * 17,
            "pca32": [1.0] * 32,
            "features": [0.0] * 17 + [1.0] * 32,
        }
        assert_compatible_controlled_features([valid])
        wrong = dict(valid)
        wrong["features"] = [0.0] * 49
        with self.assertRaisesRegex(RuntimeError, "ordering"):
            assert_compatible_controlled_features([wrong])


if __name__ == "__main__":
    unittest.main()
