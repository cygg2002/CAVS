from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inference_protocol.py"
SPEC = importlib.util.spec_from_file_location("paper_parity_protocol", SCRIPT)
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


class ProtocolTest(unittest.TestCase):
    def test_native_jpeg_is_not_reencoded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            payload = b"\xff\xd8native-paper-payload\xff\xd9"
            path.write_bytes(payload)
            self.assertEqual(PROTOCOL.native_jpeg(path), payload)

    def test_q90_encoding_is_deterministic_and_crop_aware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            Image.new("RGB", (40, 30), (25, 80, 140)).save(path, format="JPEG", quality=97)
            full_left = PROTOCOL.paper_q90_jpeg(path)
            full_right = PROTOCOL.paper_q90_jpeg(path)
            crop = PROTOCOL.paper_q90_jpeg(path, crop_xyxy=(5, 4, 25, 20))
            self.assertEqual(full_left, full_right)
            self.assertNotEqual(full_left, crop)

    def test_cloud_health_requires_action_specific_protocols(self) -> None:
        health = {
            "model_sha256": "abc",
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "action_protocols": {
                "original": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
                "q90": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
                "current@640": {"end2end": False, "rect": True, "scaleup": True, "half": True, "multi_label": False},
            },
        }
        PROTOCOL.validate_cloud_health(
            health,
            cloud_sha256="abc",
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
        )
        health["action_protocols"]["q90"]["scaleup"] = True
        with self.assertRaises(RuntimeError):
            PROTOCOL.validate_cloud_health(
                health,
                cloud_sha256="abc",
                imgsz=640,
                conf=0.001,
                iou=0.7,
                max_det=300,
            )

    def test_roi_record_reads_frozen_geometry(self) -> None:
        sample = {
            "actions": [
                {"action_id": "edge", "available": True},
                {"action_id": "current@640", "available": True, "xyxy": [1, 2, 8, 9]},
            ]
        }
        self.assertEqual(PROTOCOL.roi_record(sample)["xyxy"], [1, 2, 8, 9])


if __name__ == "__main__":
    unittest.main()
