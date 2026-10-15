from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud_server import (
    backend_format,
    parse_inference_request,
    shift_roi_boxes,
    validate_request_content_type,
)
from protocol import (
    WIRE_MEDIA_TYPE,
    encode_inference_request,
    sanitize_boxes,
)


class CloudServerV2Test(unittest.TestCase):
    def test_backend_format_supports_current_and_legacy_autobackend_apis(self) -> None:
        self.assertEqual(backend_format(SimpleNamespace(format="pt")), "pt")
        self.assertEqual(backend_format(SimpleNamespace(pt=True, imx=False)), "pt")
        self.assertEqual(backend_format(SimpleNamespace(pt=False, imx=True)), "imx")
        self.assertIsNone(backend_format(SimpleNamespace(pt=False, imx=False)))

    def _metadata(self, action: str = "original") -> dict:
        return {
            "task_id": f"online-17-{action}",
            "frame_id": 17,
            "action_id": action,
            "payload_type": "roi" if action == "current@640" else "keyframe",
            "payload_bytes": 4,
            "crop_xyxy": [90, 70, 100, 80] if action == "current@640" else None,
            "orig_shape": [80, 100],
            "cloud_imgsz": 640,
        }

    def test_parser_accepts_only_v2_binary_body_and_exact_content_type(self) -> None:
        body = encode_inference_request(self._metadata(), b"jpeg")
        metadata, payload = parse_inference_request(body, maximum_bytes=len(body))
        self.assertEqual(metadata["frame_id"], 17)
        self.assertEqual(payload, b"jpeg")
        validate_request_content_type(WIRE_MEDIA_TYPE)
        for stale in ("application/json", "application/octet-stream", WIRE_MEDIA_TYPE + "; charset=utf-8"):
            with self.subTest(stale=stale):
                with self.assertRaisesRegex(ValueError, "Content-Type"):
                    validate_request_content_type(stale)
        with self.assertRaises(ValueError):
            parse_inference_request(b'{"image_b64":"anBlZw=="}', maximum_bytes=1000)

    def test_parser_rejects_payload_drift_and_roi_outside_original_shape(self) -> None:
        metadata = self._metadata()
        metadata["payload_bytes"] = 5
        with self.assertRaisesRegex(ValueError, "payload byte"):
            parse_inference_request(
                encode_inference_request(metadata, b"jpeg"), maximum_bytes=1000
            )

        metadata = self._metadata("current@640")
        metadata["crop_xyxy"] = [90, 70, 101, 80]
        with self.assertRaisesRegex(ValueError, "crop"):
            parse_inference_request(
                encode_inference_request(metadata, b"jpeg"), maximum_bytes=1000
            )

    def test_box_sanitization_clips_boundaries_and_drops_invalid_rows(self) -> None:
        boxes = [
            {"xyxy": [-2, -3, 5, 6], "conf": 0.9, "cls": 0},
            {"xyxy": [10, 10, 10, 20], "conf": 0.8, "cls": 0},
            {"xyxy": [1, 1, 2, 2], "conf": float("inf"), "cls": 0},
            {"xyxy": [95, 70, 110, 90], "conf": 0.7, "cls": 79},
            {"xyxy": [1, 1, 2, 2], "conf": 0.6, "cls": 80},
        ]
        self.assertEqual(
            sanitize_boxes(boxes, width=100, height=80, class_count=80),
            [
                {"xyxy": [0.0, 0.0, 5.0, 6.0], "conf": 0.9, "cls": 0},
                {"xyxy": [95.0, 70.0, 100.0, 80.0], "conf": 0.7, "cls": 79},
            ],
        )

    def test_shifted_roi_boxes_are_clipped_to_the_original_image(self) -> None:
        boxes = [
            {"xyxy": [-2, -2, 20, 20], "conf": 0.9, "cls": 0},
            {"xyxy": [10, 10, 10, 11], "conf": 0.8, "cls": 0},
        ]
        self.assertEqual(
            shift_roi_boxes(boxes, [90, 70, 100, 80], [80, 100]),
            [{"xyxy": [88.0, 68.0, 100.0, 80.0], "conf": 0.9, "cls": 0}],
        )


if __name__ == "__main__":
    unittest.main()
