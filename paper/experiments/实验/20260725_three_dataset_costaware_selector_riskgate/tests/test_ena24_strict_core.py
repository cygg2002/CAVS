from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ena24_strict_core import (  # noqa: E402
    apply_strict_risk_gate,
    build_near_duplicate_split_manifest,
    dhash256,
    require_strict_provisional_marker,
    subset_coco_gt,
)


class Ena24StrictCoreTests(unittest.TestCase):
    def test_dhash256_is_256_bits_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gradient.png"
            image = Image.new("L", (17, 16))
            image.putdata(
                [column + 17 * row for row in range(16) for column in range(17)]
            )
            image.save(path)

            first = dhash256(path)
            second = dhash256(path)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**256)
        self.assertEqual(len(f"{first:064x}"), 64)

    def test_split_manifest_keeps_near_duplicate_components_disjoint(self) -> None:
        records = [
            {
                "image_id": 1,
                "identity": "/tmp/1.jpg",
                "class_ids": [1],
                "dhash256": 0b0000,
                "source_sha256": "a" * 64,
            },
            {
                "image_id": 2,
                "identity": "/tmp/2.jpg",
                "class_ids": [1],
                "dhash256": 0b0001,
                "source_sha256": "b" * 64,
            },
            {
                "image_id": 3,
                "identity": "/tmp/3.jpg",
                "class_ids": [2],
                "dhash256": 0b1111,
                "source_sha256": "c" * 64,
            },
            {
                "image_id": 4,
                "identity": "/tmp/4.jpg",
                "class_ids": [2],
                "dhash256": 0b1110,
                "source_sha256": "d" * 64,
            },
        ]

        first = build_near_duplicate_split_manifest(
            records,
            max_hamming=1,
            target_selector_images=2,
            seed=20260725,
        )
        second = build_near_duplicate_split_manifest(
            list(reversed(records)),
            max_hamming=1,
            target_selector_images=2,
            seed=20260725,
        )

        self.assertEqual(first, second)
        selector = set(first["selector_image_ids"])
        gate = set(first["gate_image_ids"])
        self.assertFalse(selector & gate)
        self.assertEqual(selector | gate, {1, 2, 3, 4})
        for component in first["components"]:
            members = set(component["image_ids"])
            self.assertTrue(members <= selector or members <= gate)
        self.assertEqual(first["split_kind"], "near_duplicate_group_disjoint")
        self.assertEqual(first["event_camera_disjoint"], "unknown")

    def test_strict_marker_rejects_superseded_legacy_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "PROVISIONAL_FROZEN").write_text("ok\n", encoding="ascii")

            with self.assertRaisesRegex(RuntimeError, "STRICT_PROVISIONAL_FROZEN"):
                require_strict_provisional_marker(root)

            (root / "STRICT_PROVISIONAL_FROZEN").write_text(
                "ok\n", encoding="ascii"
            )
            require_strict_provisional_marker(root)

    def test_fixed_call_gate_requires_ap_and_nonpositive_cost_upper_bound(self) -> None:
        reference = {"ranker": "Confidence", "policy": "SharedAction"}
        provisional = {"ranker": "M2E", "policy": "GBDT-Hierarchical"}

        failed = apply_strict_risk_gate(
            reference=reference,
            provisional=provisional,
            mode="fixed_call",
            ap_delta_ci_low=-0.0005,
            kb_delta_ci_high=0.01,
            ap_tolerance=0.001,
        )
        passed = apply_strict_risk_gate(
            reference=reference,
            provisional=provisional,
            mode="fixed_call",
            ap_delta_ci_low=-0.0005,
            kb_delta_ci_high=0.0,
            ap_tolerance=0.001,
        )

        self.assertFalse(failed["gate_passed"])
        self.assertEqual(failed["frozen"], reference)
        self.assertNotIn("second_candidate", failed)
        self.assertTrue(passed["gate_passed"])
        self.assertEqual(passed["frozen"], provisional)

    def test_fixed_byte_gate_uses_ap_only(self) -> None:
        reference = {"ranker": "Confidence", "policy": "SharedAction"}
        provisional = {"ranker": "M2E", "policy": "GBDT-Hierarchical"}
        result = apply_strict_risk_gate(
            reference=reference,
            provisional=provisional,
            mode="fixed_byte",
            ap_delta_ci_low=-0.001,
            kb_delta_ci_high=100.0,
            ap_tolerance=0.001,
        )
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["frozen"], provisional)

    def test_subset_coco_gt_preserves_categories_and_filters_annotations(self) -> None:
        gt = {
            "images": [{"id": 1}, {"id": 2}, {"id": 3}],
            "annotations": [
                {"id": 10, "image_id": 1},
                {"id": 20, "image_id": 2},
                {"id": 30, "image_id": 3},
            ],
            "categories": [{"id": 1, "name": "animal"}],
            "info": {"description": "fixture"},
        }
        subset = subset_coco_gt(gt, {1, 3})
        self.assertEqual([row["id"] for row in subset["images"]], [1, 3])
        self.assertEqual(
            [row["image_id"] for row in subset["annotations"]], [1, 3]
        )
        self.assertEqual(subset["categories"], gt["categories"])
        self.assertEqual(subset["info"], gt["info"])


if __name__ == "__main__":
    unittest.main()
