from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/roi_core.py"
SPEC = importlib.util.spec_from_file_location("voc_torchvision_roi_core", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RoiCoreTest(unittest.TestCase):
    def test_current_profile_builds_bounded_merged_roi(self):
        roi = MODULE.merged_roi_xyxy(
            100,
            80,
            [{"bbox": [20, 10, 10, 10], "score": 0.9}],
            confidence=0.25,
            max_area_ratio=0.65,
            profile="current",
        )
        self.assertEqual(roi, (17, 7, 33, 23))

    def test_roi_larger_than_maximum_area_is_unavailable(self):
        roi = MODULE.merged_roi_xyxy(
            100,
            80,
            [{"bbox": [0, 0, 100, 80], "score": 0.9}],
            confidence=0.25,
            max_area_ratio=0.65,
            profile="current",
        )
        self.assertIsNone(roi)

    def test_crop_predictions_are_mapped_to_original_coordinates(self):
        mapped = MODULE.map_crop_predictions(
            [{"image_id": 1, "category_id": 3, "bbox": [2, 3, 10, 12], "score": 0.8}],
            {1: {"xyxy": [20, 30, 60, 70]}},
        )
        self.assertEqual(mapped[0]["bbox"], [22.0, 33.0, 10.0, 12.0])

    def test_maur_replaces_inside_edge_and_keeps_outside_edge(self):
        edge = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [80, 70, 10, 8], "score": 0.7},
        ]
        cloud = [
            {"image_id": 1, "category_id": 1, "bbox": [11, 11, 11, 11], "score": 0.95}
        ]
        candidates = MODULE.maur_candidates(edge, cloud, (0, 0, 50, 50), duplicate_iou=0.5)
        self.assertEqual({row["category_id"] for row in candidates}, {1, 2})
        self.assertEqual(sum(row["category_id"] == 1 for row in candidates), 1)


if __name__ == "__main__":
    unittest.main()
