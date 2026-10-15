from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("test_dcsb_af_core", ROOT / "scripts/baseline_core.py")
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


def prediction(image_id: int, category: int, box, score: float):
    return {
        "image_id": image_id,
        "category_id": category,
        "bbox": list(map(float, box)),
        "score": float(score),
    }


class AdaptiveFeedingTests(unittest.TestCase):
    def test_feature_order_topk_histogram_and_padding(self):
        values = CORE.adaptive_feeding_feature(
            [
                prediction(1, 2, [20, 10, 10, 20], 0.2),
                prediction(1, 1, [10, 20, 20, 10], 0.9),
                prediction(1, 2, [0, 0, 5, 5], 0.1),
            ],
            width=100,
            height=50,
            category_ids=[1, 2],
            top_k=3,
        )
        np.testing.assert_allclose(
            values,
            np.asarray(
                [
                    1.0,
                    2.0,
                    0.9,
                    0.1,
                    0.4,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.1,
                    0.4,
                    0.1,
                    0.0,
                    0.0,
                    0.05,
                    0.1,
                ]
            ),
        )

    def test_mapi_label_is_strict_cloud_improvement(self):
        gt = {1: [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}]}
        edge = {1: [prediction(1, 1, [20, 20, 5, 5], 0.9)]}
        cloud = {1: [prediction(1, 1, [0, 0, 10, 10], 0.8)]}
        labels, edge_map, cloud_map = CORE.adaptive_feeding_labels(
            [1], edge, cloud, gt
        )
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(edge_map.tolist(), [0.0])
        self.assertAlmostEqual(float(cloud_map[0]), 1.0)
        tie, _, _ = CORE.adaptive_feeding_labels([1], cloud, cloud, gt)
        self.assertEqual(tie.tolist(), [0])

    def test_linear_svm_is_deterministic(self):
        features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0, 0, 1, 1])
        left = CORE.fit_adaptive_feeding(features, labels, positive_weight=3.0)
        right = CORE.fit_adaptive_feeding(features, labels, positive_weight=3.0)
        np.testing.assert_allclose(left.coef_, right.coef_)
        np.testing.assert_allclose(left.intercept_, right.intercept_)
        self.assertEqual(left.predict(features).tolist(), [0, 0, 1, 1])


class DcsbTests(unittest.TestCase):
    def test_source_rule_easy_gate_and_threshold_violations(self):
        model = {
            "confidence_threshold": 0.2,
            "detector_confidence": 0.5,
            "count_threshold": 2,
            "area_threshold": 0.2,
        }
        easy = [prediction(1, 1, [0, 0, 50, 50], 0.8)]
        decision, margin = CORE.dcsb_decision_and_margin(
            easy, width=100, height=100, model=model
        )
        self.assertEqual(decision, 0)
        self.assertEqual(margin, -1.0)
        uncertain_small = [
            prediction(1, 1, [0, 0, 50, 50], 0.8),
            prediction(1, 1, [0, 0, 10, 10], 0.3),
        ]
        decision, margin = CORE.dcsb_decision_and_margin(
            uncertain_small, width=100, height=100, model=model
        )
        self.assertEqual(decision, 1)
        self.assertGreater(margin, 0.0)

    def test_fit_records_ground_truth_threshold_features(self):
        edge = {
            1: [prediction(1, 1, [0, 0, 10, 10], 0.9)],
            2: [prediction(2, 1, [0, 0, 10, 10], 0.9)],
        }
        cloud = {
            1: edge[1],
            2: [*edge[2], prediction(2, 1, [20, 20, 10, 10], 0.8)],
        }
        model = CORE.fit_dcsb(
            image_ids=[1, 2],
            edge_predictions=edge,
            cloud_predictions=cloud,
            image_sizes={1: (100, 100), 2: (100, 100)},
            ground_truth_counts={1: 1, 2: 3},
            ground_truth_min_areas={1: 0.5, 2: 0.01},
            count_thresholds=[1, 2],
            area_thresholds=[0.1, 0.2],
        )
        self.assertEqual(model["threshold_training_features"], "ground_truth_count_and_min_area")
        self.assertEqual(model["fit_images"], 2)
        self.assertEqual(model["positive_labels"], 1)


class BudgetAndProtocolTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"image_id": 1, "payload_bytes": {"original": 60}},
            {"image_id": 2, "payload_bytes": {"original": 40}},
            {"image_id": 3, "payload_bytes": {"original": 20}},
            {"image_id": 4, "payload_bytes": {"original": 80}},
        ]

    def test_fixed_call_is_original_only_and_exact(self):
        selection = CORE.select_original_by_score(
            self.samples, [0.1, 0.9, 0.8, 0.2], mode="fixed_call", budget=0.5
        )
        self.assertEqual(selection, {2: "original", 3: "original"})
        audit = CORE.validate_original_only_selection(
            self.samples, selection, mode="fixed_call", budget=0.5
        )
        self.assertEqual(audit["selected_images"], 2)

    def test_fixed_byte_never_exceeds_capacity(self):
        selection = CORE.select_original_by_score(
            self.samples, [0.9, 0.8, 0.7, 0.6], mode="fixed_byte", budget=0.5
        )
        audit = CORE.validate_original_only_selection(
            self.samples, selection, mode="fixed_byte", budget=0.5
        )
        self.assertLessEqual(audit["request_bytes"], 100)
        self.assertLessEqual(audit["capacity_utilization"], 1.0)
        self.assertEqual(set(selection.values()), {"original"})

    def test_native_selection_is_not_forced_to_a_budget(self):
        selection = {1: "original"}
        audit = CORE.validate_original_only_selection(
            self.samples, selection, mode="native", budget=-1.0
        )
        self.assertAlmostEqual(audit["call_ratio"], 0.25)
        self.assertAlmostEqual(audit["byte_ratio"], 0.3)

    def test_protocol_forbids_every_cavs_component_and_has_32_rows(self):
        for key in (
            "cavs_components_allowed",
            "shared_action_allowed",
            "q90_allowed",
            "roi_allowed",
            "maur_allowed",
        ):
            self.assertFalse(PROTOCOL[key])
        per_role = 2 + 2 + 2 * 2 * 3
        self.assertEqual(per_role * 2, PROTOCOL["expected_rows_per_dataset"])


if __name__ == "__main__":
    unittest.main()
