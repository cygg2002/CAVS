from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/train_export.py"
    spec = importlib.util.spec_from_file_location("test_dcsb_af_train_export", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load_module()


class TrainExportTests(unittest.TestCase):
    def test_native_rknn_class_indices_map_to_coco_category_ids(self):
        predictions = TRAIN.native_edge_predictions(
            {
                "image_id": 7,
                "edge_boxes": [
                    {"xyxy": [1.0, 2.0, 6.0, 10.0], "conf": 0.75, "cls": 1}
                ],
            },
            category_ids=(1, 3),
        )
        self.assertEqual(
            predictions,
            [
                {
                    "image_id": 7,
                    "category_id": 3,
                    "bbox": [1.0, 2.0, 5.0, 8.0],
                    "score": 0.75,
                }
            ],
        )

    def test_alignment_rejects_candidate_geometry_drift(self):
        samples = [
            {
                "image_id": 7,
                "file_name": "7.jpg",
                "width": 10,
                "height": 20,
                "payload_bytes": {"original": 123},
            }
        ]
        candidates = [
            {
                "image_id": 7,
                "file_name": "7.jpg",
                "width": 11,
                "height": 20,
                "native_jpeg_bytes": 123,
                "edge_boxes": [],
            }
        ]
        with self.assertRaises(RuntimeError):
            TRAIN.validate_candidate_alignment(samples, candidates, role="fit")

    def test_policy_features_preserve_raw_zero_area_rknn_boxes(self):
        role = {
            "samples": [{"image_id": 7, "width": 10, "height": 20}],
            "category_ids": (1, 3),
        }
        candidates = {
            7: {
                "image_id": 7,
                "edge_boxes": [
                    {"xyxy": [1.0, 2.0, 6.0, 2.0], "conf": 0.9, "cls": 1},
                    {"xyxy": [0.0, 0.0, 5.0, 5.0], "conf": 0.8, "cls": 0},
                ],
            }
        }
        matrix = TRAIN.runtime_feature_matrix(role, candidates)
        self.assertEqual(matrix.shape, (1, 2 + 5 * TRAIN.TOP_K))
        np.testing.assert_allclose(matrix[0, :2], [1.0, 1.0])
        np.testing.assert_allclose(matrix[0, 2:7], [0.9, 0.1, 0.1, 0.5, 0.0])


if __name__ == "__main__":
    unittest.main()
