from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_core import (  # noqa: E402
    DetectionStats,
    ap_per_class,
    build_output_feature,
    context_indices,
    dcsb_decision_and_score,
    detection_stats,
    empirical_cdf,
    fit_dcsb,
    oric_reward,
    weighted_mse,
)


class BaselineCoreTest(unittest.TestCase):
    def test_top25_feature_has_class_counts_and_normalized_boxes(self) -> None:
        predictions = [
            {"category_id": 9, "bbox": [10, 20, 30, 40], "score": 0.8},
            {"category_id": 4, "bbox": [20, 10, 10, 20], "score": 0.9},
            {"category_id": 4, "bbox": [0, 0, 20, 10], "score": 0.7},
        ]
        feature = build_output_feature(
            predictions,
            width=100,
            height=200,
            category_ids=[4, 9],
            top_k=2,
        )
        self.assertEqual(feature.shape, (12,))
        np.testing.assert_allclose(feature[:2], [1.0, 1.0])
        np.testing.assert_allclose(feature[2:7], [0.25, 0.1, 0.1, 0.1, 0.9])
        np.testing.assert_allclose(feature[7:12], [0.25, 0.2, 0.3, 0.2, 0.8])

    def test_empirical_cdf_preserves_ties(self) -> None:
        reference = np.asarray([-1.0, 0.0, 0.0, 2.0])
        values = np.asarray([-2.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(empirical_cdf(reference, values), [0.0, 0.75, 0.75, 1.0])

    def test_context_indices_are_deterministic_and_exclude_target(self) -> None:
        first = context_indices(20, 7, context_size=5, seed=42)
        second = context_indices(20, 7, context_size=5, seed=42)
        self.assertEqual(first, second)
        self.assertNotIn(7, first)
        self.assertEqual(len(first), 5)

    def test_dcsb_native_rule_and_continuous_score(self) -> None:
        model = {
            "confidence_threshold": 0.2,
            "count_threshold": 2,
            "area_threshold": 0.1,
            "detector_confidence": 0.5,
        }
        easy = [
            {"bbox": [0, 0, 40, 40], "score": 0.8},
            {"bbox": [50, 50, 40, 40], "score": 0.7},
        ]
        count_hard = [*easy, {"bbox": [10, 10, 20, 20], "score": 0.3}]
        area_hard = [*easy, {"bbox": [10, 10, 5, 5], "score": 0.3}]
        easy_decision, easy_score = dcsb_decision_and_score(easy, 100, 100, model)
        count_decision, count_score = dcsb_decision_and_score(count_hard, 100, 100, model)
        area_decision, area_score = dcsb_decision_and_score(area_hard, 100, 100, model)
        self.assertEqual(easy_decision, 0)
        self.assertEqual(count_decision, 1)
        self.assertEqual(area_decision, 1)
        self.assertGreater(count_score, easy_score)
        self.assertGreater(area_score, easy_score)

    def test_fit_dcsb_uses_fit_only_counts_and_cloud_label(self) -> None:
        edge = {
            1: [{"bbox": [0, 0, 50, 50], "score": 0.9}],
            2: [
                {"bbox": [0, 0, 10, 10], "score": 0.9},
                {"bbox": [20, 20, 10, 10], "score": 0.3},
            ],
        }
        cloud = {
            1: [{"bbox": [0, 0, 50, 50], "score": 0.9}],
            2: [
                {"bbox": [0, 0, 10, 10], "score": 0.9},
                {"bbox": [20, 20, 10, 10], "score": 0.8},
            ],
        }
        model = fit_dcsb(
            image_ids=[1, 2],
            edge_predictions=edge,
            cloud_predictions=cloud,
            image_sizes={1: (100, 100), 2: (100, 100)},
            ground_truth_counts={1: 1, 2: 2},
            detector_confidence=0.5,
            count_thresholds=[1, 2],
            area_thresholds=[0.01, 0.2],
        )
        self.assertEqual(model["fit_images"], 2)
        self.assertEqual(model["positive_labels"], 1)
        self.assertIn("confidence_threshold", model)

    def test_oric_reward_is_scaled_context_map_delta(self) -> None:
        edge_context = DetectionStats(
            correct=np.asarray([[True]]),
            confidence=np.asarray([0.9]),
            predicted_class=np.asarray([1]),
            target_class=np.asarray([1]),
        )
        target_edge = DetectionStats(
            correct=np.asarray([[False]]),
            confidence=np.asarray([0.8]),
            predicted_class=np.asarray([1]),
            target_class=np.asarray([1]),
        )
        target_cloud = DetectionStats(
            correct=np.asarray([[True]]),
            confidence=np.asarray([0.8]),
            predicted_class=np.asarray([1]),
            target_class=np.asarray([1]),
        )
        reward = oric_reward([edge_context], target_edge, target_cloud)
        self.assertGreater(reward["reward"], 0.0)
        self.assertAlmostEqual(
            reward["reward"],
            2.0 * (reward["cloud_map"] - reward["edge_map"]),
        )

    def test_detection_matching_reproduces_official_conflict_order(self) -> None:
        stats = detection_stats(
            [
                {"category_id": 1, "bbox": [1, 0, 10, 10], "score": 0.9},
                {"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
            ],
            [
                {"category_id": 1, "bbox": [0, 0, 10, 10]},
                {"category_id": 1, "bbox": [4, 0, 10, 10]},
            ],
            [0.5],
        )
        np.testing.assert_array_equal(stats.correct[:, 0], [True, False])

    def test_ap_per_class_handles_empty_predictions(self) -> None:
        result = ap_per_class(
            np.empty((0, 1), dtype=bool),
            np.empty((0,)),
            np.empty((0,), dtype=int),
            np.asarray([1]),
        )
        np.testing.assert_allclose(result, np.zeros((1, 1)))

    def test_moric_weighted_mse_matches_source_mean_reduction(self) -> None:
        loss = weighted_mse(
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 2.0]),
            np.asarray([0.5, 1.0]),
        )
        self.assertAlmostEqual(loss, 2.25)


if __name__ == "__main__":
    unittest.main()
