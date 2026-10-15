import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_bootstrap_under_test", ROOT / "scripts/bootstrap.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BootstrapTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
