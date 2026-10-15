from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cloud_server import parse_inference_request, require_runtime_token, shift_roi_boxes
from protocol import encode_b64


class CloudRequestTest(unittest.TestCase):
    def test_server_requires_a_runtime_token(self) -> None:
        self.assertEqual(require_runtime_token("secret"), "secret")
        with self.assertRaisesRegex(RuntimeError, "token is required"):
            require_runtime_token("")

    def test_roi_boxes_are_shifted_by_crop_origin_only(self) -> None:
        boxes = [{"xyxy": [1.0, 2.0, 11.0, 12.0], "conf": 0.9, "cls": 0}]

        shifted = shift_roi_boxes(boxes, [100, 200, 400, 500])

        self.assertEqual(shifted[0]["xyxy"], [101.0, 202.0, 111.0, 212.0])

    def test_request_accepts_only_allowlisted_action_and_matching_payload_length(self) -> None:
        jpeg = b"not-a-jpeg-yet"
        body = json.dumps({
            "metadata": {"frame_id": 1, "action_id": "q90", "payload_bytes": len(jpeg), "crop_xyxy": None},
            "image_b64": encode_b64(jpeg),
        }).encode("utf-8")
        metadata, received = parse_inference_request(body, maximum_bytes=1024)
        self.assertEqual(metadata["action_id"], "q90")
        self.assertEqual(received, jpeg)

    def test_request_rejects_unknown_action(self) -> None:
        body = json.dumps({
            "metadata": {"frame_id": 1, "action_id": "shell", "payload_bytes": 1},
            "image_b64": encode_b64(b"x"),
        }).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported action"):
            parse_inference_request(body, maximum_bytes=1024)


if __name__ == "__main__":
    unittest.main()
