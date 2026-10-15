import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oregon_cloudv3_cache_core", ROOT / "scripts/cache_core.py"
)
CACHE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CACHE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class CacheContractTest(unittest.TestCase):
    def test_real_crop_geometry_is_action_conditioned(self):
        record = {
            "image_id": 7,
            "identity": "/data/a.jpg",
            "xyxy": [10, 20, 210, 120],
            "image_width": 400,
            "image_height": 200,
            "crop_width": 200,
            "crop_height": 100,
            "area_ratio": 0.25,
            "bytes": 1234,
            "profile": "moderate",
        }
        action = CACHE.action_record(record, imgsz=320)
        self.assertEqual(action["action_id"], "moderate@320")
        self.assertEqual(action["xyxy"], [10, 20, 210, 120])
        self.assertEqual(action["payload_bytes"], 1234)
        self.assertEqual(action["imgsz"], 320)

    def test_mapped_roi_box_uses_both_offset_axes(self):
        mapped = CACHE.map_crop_bbox([1.0, 2.0, 11.0, 22.0], [100, 200, 300, 400])
        self.assertEqual(mapped, [101.0, 202.0, 10.0, 20.0])

    def test_cache_signature_changes_with_cloud_hash_or_role(self):
        first = CACHE.cache_signature(
            role="fit", action_id="q90", model_sha="a", source_sha="b", settings={"imgsz": 640}
        )
        second = CACHE.cache_signature(
            role="calibration", action_id="q90", model_sha="a", source_sha="b", settings={"imgsz": 640}
        )
        third = CACHE.cache_signature(
            role="fit", action_id="q90", model_sha="c", source_sha="b", settings={"imgsz": 640}
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_complete_cache_plan_has_all_roles_and_actions(self):
        jobs = CACHE.planned_cache_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 54)
        by_role = {role: [job for job in jobs if job["role"] == role] for role in PROTOCOL["roles"]}
        self.assertEqual({role: len(values) for role, values in by_role.items()}, {"fit": 18, "calibration": 18, "final": 18})
        for values in by_role.values():
            inference = {job["action_id"] for job in values if job["kind"] == "inference"}
            self.assertEqual(inference, {"original", "q90", *PROTOCOL["roi_actions"]})


if __name__ == "__main__":
    unittest.main()
