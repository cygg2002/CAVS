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
    CONTROLLED_RANKER,
    LEGACY_STYLE_RANKER,
    apply_extra_32d_evaluation_metadata,
    assert_compatible_controlled_features,
    expected_matrix_counts,
    fit_legacy_style_ranker,
    pca32_matrix,
    predict_legacy_style_ranker,
    validate_protocol,
)


class EvaluationCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_has_ten_rankers_and_nine_constraints(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(len(self.protocol["rankers"]), 10)
        self.assertEqual(len(self.protocol["constrained_rankers"]), 9)
        self.assertNotIn("Random", self.protocol["constrained_rankers"])
        self.assertIn(LEGACY_STYLE_RANKER, self.protocol["rankers"])
        self.assertNotIn("Legacy-Frozen-32D", self.protocol["rankers"])

    def test_expected_matrix_counts_match_frozen_contract(self) -> None:
        self.assertEqual(
            expected_matrix_counts(self.protocol),
            {
                "main": 429,
                "per_ranker_ablation": 135,
                "random_seed_rows": 1200,
                "bootstrap_comparisons": 27,
            },
        )

    def test_legacy_style_ranker_is_finite_32d_anchored_residual(self) -> None:
        features = np.asarray(
            [[0.0] * 32, [1.0] * 32, [2.0] * 32, [3.0] * 32], dtype=np.float64
        )
        targets = np.asarray([0.0, 0.1, 0.3, 0.5], dtype=np.float64)
        anchors = np.asarray([0.2, 0.4, 0.3, 0.1], dtype=np.float64)
        model = fit_legacy_style_ranker(
            features,
            targets,
            anchors,
            budgets=(0.1, 0.25, 0.4),
            epochs=2,
            learning_rate=0.03,
            ridge=0.001,
            max_pairs=20,
            seed=7,
            score_scale=0.25,
        )
        scores = predict_legacy_style_ranker(model, features, anchors)
        self.assertEqual(len(model["weights"]), 32)
        self.assertEqual(model["score_scale"], 0.25)
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.isfinite(scores).all())

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

    def test_extra_32d_evaluation_metadata_preserves_requested_ranker(self) -> None:
        for method in (CONTROLLED_RANKER, LEGACY_STYLE_RANKER):
            with self.subTest(method=method):
                row = apply_extra_32d_evaluation_metadata(
                    {"method": "SemanticRank"},
                    method=method,
                    action_params=17,
                )
                self.assertEqual(row["method"], method)
                self.assertEqual(row["selector_ms_per_image"], 0.08)
                self.assertEqual(row["selector_params"], 49)


if __name__ == "__main__":
    unittest.main()
