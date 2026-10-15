from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from protocol import (
    WIRE_HEADER_BYTES,
    WIRE_MAGIC,
    WIRE_MEDIA_TYPE,
    WIRE_SCHEMA,
    decode_inference_request,
    encode_inference_request,
    wire_contract,
)


class WireProtocolV2Test(unittest.TestCase):
    def _metadata(self) -> dict:
        return {
            "task_id": "online-17-q90",
            "frame_id": 17,
            "action_id": "q90",
            "payload_type": "keyframe",
            "payload_bytes": 20_000,
            "crop_xyxy": None,
            "orig_shape": [480, 640],
            "cloud_imgsz": 640,
        }

    def test_binary_request_round_trip_has_bounded_overhead_and_no_base64(self) -> None:
        payload = bytes(range(256)) * 78 + b"x" * 32
        metadata = self._metadata()
        metadata["payload_bytes"] = len(payload)

        body = encode_inference_request(metadata, payload)
        decoded_metadata, decoded_payload = decode_inference_request(
            body, maximum_bytes=len(body)
        )

        self.assertTrue(body.startswith(WIRE_MAGIC))
        self.assertEqual(decoded_payload, payload)
        self.assertEqual(decoded_metadata["schema"], WIRE_SCHEMA)
        self.assertEqual(decoded_metadata["frame_id"], 17)
        self.assertLess(len(body) / len(payload), 1.02)
        self.assertEqual(WIRE_HEADER_BYTES, len(WIRE_MAGIC) + 4)
        self.assertNotIn(b"image_b64", body[: len(body) - len(payload)])
        self.assertEqual(wire_contract()["media_type"], WIRE_MEDIA_TYPE)
        self.assertFalse(wire_contract()["base64"])

    def test_decoder_rejects_bad_magic_lengths_schema_and_trailing_mismatch(self) -> None:
        payload = b"jpeg"
        metadata = self._metadata()
        metadata["payload_bytes"] = len(payload)
        body = encode_inference_request(metadata, payload)

        corruptions = (
            b"BADMAGIC" + body[len(WIRE_MAGIC) :],
            body[: len(WIRE_MAGIC)] + (0).to_bytes(4, "big") + body[WIRE_HEADER_BYTES:],
            body[: len(WIRE_MAGIC)] + (2**24).to_bytes(4, "big") + body[WIRE_HEADER_BYTES:],
        )
        for corrupted in corruptions:
            with self.subTest(corrupted=corrupted[:16]):
                with self.assertRaises(ValueError):
                    decode_inference_request(corrupted, maximum_bytes=max(len(body), len(corrupted)))

        header = json.dumps({"schema": "wrong"}, separators=(",", ":")).encode()
        wrong_schema = WIRE_MAGIC + len(header).to_bytes(4, "big") + header + payload
        with self.assertRaisesRegex(ValueError, "schema"):
            decode_inference_request(wrong_schema, maximum_bytes=len(wrong_schema))


if __name__ == "__main__":
    unittest.main()
