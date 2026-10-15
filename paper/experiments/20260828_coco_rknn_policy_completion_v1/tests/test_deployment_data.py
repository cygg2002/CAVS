from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "rknn_deployment_data", SCRIPTS / "deployment_data.py"
)
assert SPEC and SPEC.loader
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


class DeploymentDataTests(unittest.TestCase):
    def test_maps_rknn_classes_and_preserves_zero_area_boxes(self) -> None:
        candidate = {
            "image_id": 7,
            "width": 640,
            "height": 480,
            "edge_boxes": [
                {"xyxy": [1.0, 2.0, 5.0, 8.0], "conf": 0.9, "cls": 0},
                {"xyxy": [4.0, 0.0, 9.0, 0.0], "conf": 0.01, "cls": 11},
            ],
        }

        predictions = DATA.edge_predictions(candidate, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])

        self.assertEqual(predictions[0]["category_id"], 1)
        self.assertEqual(predictions[0]["bbox"], [1.0, 2.0, 4.0, 6.0])
        self.assertEqual(predictions[1]["category_id"], 13)
        self.assertEqual(predictions[1]["bbox"], [4.0, 0.0, 5.0, 0.0])

    def test_builds_only_deployment_aligned_actions(self) -> None:
        base = {
            "image_id": 7,
            "identity": "/data/000000000007.jpg",
            "file_name": "000000000007.jpg",
            "width": 640,
            "height": 480,
            "event_id": "coco:7",
            "camera": "coco",
            "payload_bytes": {"original": 1000, "q90": 600, "historical@640": 300},
            "available": {"original": True, "q90": True, "historical@640": True},
        }
        roi = {
            "bytes": 250,
            "xyxy": [10, 20, 110, 220],
            "crop_width": 100,
            "crop_height": 200,
            "area_ratio": 0.065,
        }

        sample = DATA.deployment_sample(base, roi)

        self.assertEqual(
            [action["action_id"] for action in sample["actions"]],
            ["edge", "original", "q90", "current@640"],
        )
        self.assertEqual(sample["payload_bytes"], {"original": 1000, "q90": 600, "current@640": 250})
        self.assertNotIn("historical@640", sample["available"])


if __name__ == "__main__":
    unittest.main()
