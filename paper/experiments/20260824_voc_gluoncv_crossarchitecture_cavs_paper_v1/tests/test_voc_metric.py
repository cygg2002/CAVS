from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/voc_metric.py"
SPEC = importlib.util.spec_from_file_location("voc_gluoncv_metric", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(PATH)
METRIC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = METRIC
SPEC.loader.exec_module(METRIC)


class VocMetricTest(unittest.TestCase):
    def test_voc_inclusive_box_converts_to_zero_based_xyxy(self) -> None:
        self.assertEqual(METRIC.voc_box_to_xyxy(1, 1, 101, 51), [0.0, 0.0, 101.0, 51.0])

    def test_perfect_detection_scores_one(self) -> None:
        truth = {
            "images": [{"id": 1}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "difficult": 0}
            ],
        }
        predictions = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
        result = METRIC.VocMetricBackend(truth).evaluate(predictions, [1])
        self.assertAlmostEqual(result["map50_95"], 1.0)
        self.assertAlmostEqual(result["map50"], 1.0)
        self.assertAlmostEqual(result["voc07_map50"], 1.0)

    def test_detection_matching_only_difficult_truth_is_ignored(self) -> None:
        truth = {
            "images": [{"id": 1}, {"id": 2}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "difficult": 1},
                {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "difficult": 0},
            ],
        }
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.99},
            {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.90},
        ]
        result = METRIC.VocMetricBackend(truth).evaluate(predictions, [1, 2])
        self.assertAlmostEqual(result["map50"], 1.0)

    def test_duplicate_detection_is_false_positive(self) -> None:
        truth = {
            "images": [{"id": 1}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "difficult": 0}
            ],
        }
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
        ]
        details = METRIC.VocMetricBackend(truth).evaluate(predictions, [1], return_details=True)
        self.assertEqual(details["counts"]["false_positives_at_50"], 1)

    def test_empty_predictions_return_zero_without_dropping_categories(self) -> None:
        truth = {
            "images": [{"id": 1}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "difficult": 0}
            ],
        }
        result = METRIC.VocMetricBackend(truth).evaluate([], [1])
        self.assertEqual(result["map50_95"], 0.0)


if __name__ == "__main__":
    unittest.main()
