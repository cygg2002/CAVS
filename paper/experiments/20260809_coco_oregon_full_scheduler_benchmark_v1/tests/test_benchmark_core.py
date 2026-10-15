import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from full_benchmark_core import (  # noqa: E402
    assert_fixed_byte_capacity,
    assert_primary_actions,
    expected_primary_rows,
    source_tier,
    validate_protocol,
    validate_unique_rows,
)


class BenchmarkCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "configs/protocol.json").read_text())

    def test_protocol_is_frozen(self):
        validate_protocol(self.protocol)

    def test_expected_primary_rows(self):
        self.assertEqual(expected_primary_rows(self.protocol), 197)

    def test_primary_rejects_roi(self):
        with self.assertRaises(RuntimeError):
            assert_primary_actions({1: "q90", 2: "roi"})

    def test_capacity_is_strict(self):
        samples = [
            {"image_id": 1, "payload_bytes": {"original": 10, "q90": 4}},
            {"image_id": 2, "payload_bytes": {"original": 11, "q90": 5}},
        ]
        self.assertAlmostEqual(
            assert_fixed_byte_capacity(samples, {1: "q90", 2: "q90"}, capacity=9),
            1.0,
        )
        with self.assertRaises(RuntimeError):
            assert_fixed_byte_capacity(samples, {1: "original"}, capacity=9)

    def test_cloudv2_is_reference_only(self):
        self.assertEqual(source_tier("/x/cloudv2/scan.json"), "cloudv2_reference")
        self.assertEqual(source_tier("/x/canonical_v1/final.json"), "canonical_primary")

    def test_duplicate_rows_are_rejected(self):
        row = {"dataset": "coco", "method": "Confidence", "policy": "Original-only", "mode": "fixed_call", "budget": 0.1}
        with self.assertRaises(RuntimeError):
            validate_unique_rows([row, dict(row)])


if __name__ == "__main__":
    unittest.main()
