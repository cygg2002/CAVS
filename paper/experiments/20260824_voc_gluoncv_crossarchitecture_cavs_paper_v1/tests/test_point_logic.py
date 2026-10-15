import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_point_under_test", ROOT / "scripts/evaluate_point.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PointLogicTest(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {
                "image_id": 1,
                "payload_bytes": {"original": 10, "q90": 5},
                "available": {"original": True, "q90": True},
            },
            {
                "image_id": 2,
                "payload_bytes": {"original": 10, "q90": 5},
                "available": {"original": True, "q90": True},
            },
        ]

    def test_fixed_byte_joint_selection_obeys_floor_capacity(self):
        scores = {1: {"original": 2.0, "q90": 1.5}, 2: {"original": 1.0, "q90": 0.9}}
        selection = MODULE.joint_selection(
            self.samples, scores, mode="fixed_byte", budget=0.25
        )
        spent = sum(
            self.samples[int(image_id) - 1]["payload_bytes"][action]
            for image_id, action in selection.items()
        )
        self.assertLessEqual(spent, 5)

    def test_fixed_call_counts_cloud_images_not_actions(self):
        scores = {1: {"original": 2.0, "q90": 3.0}, 2: {"original": 1.0, "q90": 0.5}}
        selection = MODULE.joint_selection(
            self.samples, scores, mode="fixed_call", budget=0.5
        )
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection, {1: "q90"})


if __name__ == "__main__":
    unittest.main()
