from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/cache_actions.py"
SPEC = importlib.util.spec_from_file_location("voc_torchvision_cache_actions", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(PATH)
CACHE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CACHE
SPEC.loader.exec_module(CACHE)


class CacheActionsTest(unittest.TestCase):
    def test_gate_rows_are_flattened_without_changing_predictions(self) -> None:
        rows = [
            {
                "action": "q90",
                "image_id": 2,
                "identity": "2007:2",
                "payload_bytes": 70,
                "predictions": [
                    {"image_id": 2, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.5}
                ],
            },
            {
                "action": "q90",
                "image_id": 1,
                "identity": "2007:1",
                "payload_bytes": 60,
                "predictions": [],
            },
        ]
        predictions, payload = CACHE.flatten_action_rows(rows, "q90", {1, 2})
        self.assertEqual(predictions, rows[0]["predictions"])
        self.assertEqual(payload, {1: 60, 2: 70})

    def test_q90_manifest_requires_bytes_below_original(self) -> None:
        images = [
            {"id": 1, "identity": "a", "absolute_path": "/a.jpg", "width": 10, "height": 8, "original_bytes": 100},
            {"id": 2, "identity": "b", "absolute_path": "/b.jpg", "width": 10, "height": 8, "original_bytes": 100},
        ]
        records = CACHE.q90_manifest_records(images, {1: 90, 2: 110})
        self.assertTrue(records[0]["available"])
        self.assertFalse(records[1]["available"])

    def test_roi_requires_smaller_positive_payload(self) -> None:
        self.assertTrue(CACHE.roi_payload_available(99, 100, required=True))
        self.assertFalse(CACHE.roi_payload_available(100, 100, required=True))
        self.assertFalse(CACHE.roi_payload_available(0, 100, required=True))


if __name__ == "__main__":
    unittest.main()
