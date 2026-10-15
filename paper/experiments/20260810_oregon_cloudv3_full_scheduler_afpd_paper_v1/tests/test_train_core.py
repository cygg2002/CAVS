import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oregon_cloudv3_train_core", ROOT / "scripts/train_core.py"
)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class TrainCoreTest(unittest.TestCase):
    def test_real_r12_record_contains_exact_geometry(self):
        sample = {
            "image_id": 1,
            "identity": "/data/a.jpg",
            "event_id": "cam|1",
            "camera": "cam",
            "width": 400,
            "height": 200,
            "payload_bytes": {"original": 1000, "q90": 200, "moderate@320": 100},
            "available": {"original": True, "q90": True, "moderate@320": True},
            "actions": [
                {"action_id": "edge", "payload_bytes": 0, "available": True},
                {"action_id": "original", "payload_bytes": 1000, "available": True},
                {"action_id": "q90", "payload_bytes": 200, "available": True},
                {
                    "action_id": "moderate@320",
                    "payload_bytes": 100,
                    "available": True,
                    "profile": "moderate",
                    "imgsz": 320,
                    "xyxy": [10, 20, 210, 120],
                    "area_ratio": 0.25,
                },
            ],
        }
        edge = {1: [{"image_id": 1, "category_id": 3, "bbox": [1, 2, 10, 20], "score": 0.8}]}
        record = CORE.build_action_records([sample], edge, category_ids=[3])[0]
        roi = [item for item in record["actions"] if item["action_id"] == "moderate@320"][0]
        self.assertEqual(roi["region_uids"], ["moderate@320"])
        self.assertEqual(record["regions"][0]["crop_xyxy"], [10.0, 20.0, 210.0, 120.0])
        self.assertEqual(roi["resolutions"], [320])

    def test_training_job_matrix_is_complete(self):
        jobs = CORE.training_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 342)
        self.assertEqual(len({job["id"] for job in jobs}), len(jobs))
        full = [job for job in jobs if job["fold"] == "full"]
        self.assertEqual(len(full), 19 * 3)

    def test_action_spaces_are_strict(self):
        actions = ["edge", "original", "q90", *PROTOCOL["roi_actions"]]
        self.assertEqual(CORE.action_space("O", actions), ["edge", "original"])
        self.assertEqual(CORE.action_space("OQ", actions), ["edge", "original", "q90"])
        self.assertEqual(CORE.action_space("OQR12", actions), actions)


if __name__ == "__main__":
    unittest.main()
