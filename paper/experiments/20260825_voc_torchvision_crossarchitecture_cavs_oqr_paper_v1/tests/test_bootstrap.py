import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_bootstrap_under_test", ROOT / "scripts/bootstrap.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BootstrapTest(unittest.TestCase):
    def test_registered_matrix_contains_fairness_main_and_two_action_ablations(self):
        self.assertEqual(MODULE.BOOTSTRAP_JOBS_PER_BUDGET, 12)

    def test_resampled_duplicate_images_receive_distinct_ids(self):
        gt = {
            "images": [{"id": 7, "file_name": "x.jpg"}],
            "annotations": [
                {"id": 4, "image_id": 7, "category_id": 1, "bbox": [0, 0, 2, 2]}
            ],
            "categories": [{"id": 1, "name": "x"}],
        }
        predictions = [{"image_id": 7, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 1.0}]
        remapped_gt, (remapped_predictions,) = MODULE.bootstrap_remap(
            gt, [predictions], [7, 7]
        )
        self.assertEqual([row["id"] for row in remapped_gt["images"]], [1, 2])
        self.assertEqual([row["image_id"] for row in remapped_gt["annotations"]], [1, 2])
        self.assertEqual([row["image_id"] for row in remapped_predictions], [1, 2])

    def test_bootstrap_worker_limit_can_be_overridden_without_changing_protocol(self):
        with mock.patch.dict(os.environ, {"VOC_BOOTSTRAP_WORKERS": "3"}, clear=False):
            self.assertEqual(MODULE._bootstrap_worker_count(24, 36), 3)


if __name__ == "__main__":
    unittest.main()
