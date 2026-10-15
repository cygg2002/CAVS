from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from board_client import (
    assert_required_requests_complete,
    candidate_for_image,
    ensure_run_contract,
)
from cavs_numpy import load_policy


class BoardCandidateTest(unittest.TestCase):
    def test_candidate_keeps_edge_fallback_and_exact_payload_sizes(self) -> None:
        config = json.loads((ROOT / "configs" / "protocol.json").read_text())
        policy = load_policy(ROOT / "policy" / "manifest.json")
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            pixels = np.random.default_rng(7).integers(0, 255, (480, 640, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image_path, quality=95)
            boxes = [{"xyxy": [100.0, 100.0, 180.0, 220.0], "conf": 0.9, "cls": 0}]
            candidate = candidate_for_image(image_path, 17, (480, 640), boxes, policy, config)
        self.assertEqual(candidate["frame_id"], 17)
        self.assertEqual(candidate["payload_bytes"]["edge"], 0)
        self.assertIn("edge", candidate["scores"])
        self.assertGreater(candidate["payload_bytes"]["original"], candidate["payload_bytes"]["q90"])
        self.assertIsNotNone(candidate["roi_xyxy"])


class BoardRunIntegrityTest(unittest.TestCase):
    def test_run_contract_rejects_reuse_with_different_policy(self) -> None:
        contract = {"protocol_id": "p1", "policy_sha256": "a" * 64}
        with TemporaryDirectory() as directory:
            runs = Path(directory)
            ensure_run_contract(runs, contract)

            with self.assertRaisesRegex(RuntimeError, "run ID contract mismatch"):
                ensure_run_contract(runs, {**contract, "policy_sha256": "b" * 64})

    def test_run_contract_rejects_artifacts_without_contract(self) -> None:
        with TemporaryDirectory() as directory:
            runs = Path(directory)
            (runs / "candidates.jsonl").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "missing its run contract"):
                ensure_run_contract(runs, {"protocol_id": "p1"})

    def test_required_requests_must_match_frame_action_model_and_boxes(self) -> None:
        plans = {"10": {1: "q90"}}
        valid = {
            (1, "q90"): {
                "status": "ok",
                "frame_id": 1,
                "action_id": "q90",
                "model_sha256": "c" * 64,
                "cloud_boxes": [],
            }
        }
        self.assertEqual(
            assert_required_requests_complete(plans, valid, "c" * 64), 1
        )

        invalid = {key: {**row, "frame_id": 2} for key, row in valid.items()}
        with self.assertRaisesRegex(RuntimeError, "cloud requests incomplete"):
            assert_required_requests_complete(plans, invalid, "c" * 64)

        invalid_type = {
            key: {**row, "frame_id": {"unexpected": True}}
            for key, row in valid.items()
        }
        with self.assertRaisesRegex(RuntimeError, "cloud requests incomplete"):
            assert_required_requests_complete(plans, invalid_type, "c" * 64)

        malformed = {
            key: {**row, "cloud_boxes": [{"xyxy": [0, 1], "conf": 0.9, "cls": 0}]}
            for key, row in valid.items()
        }
        with self.assertRaisesRegex(RuntimeError, "cloud requests incomplete"):
            assert_required_requests_complete(plans, malformed, "c" * 64)


if __name__ == "__main__":
    unittest.main()
