from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/inference_core.py"
SPEC = importlib.util.spec_from_file_location("voc_gluoncv_inference_core", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(PATH)
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class InferenceCoreTest(unittest.TestCase):
    def test_resized_box_is_mapped_and_clipped_to_original(self) -> None:
        mapped = CORE.map_resized_xyxy(
            [-10.0, 20.0, 220.0, 120.0],
            original_width=100,
            original_height=50,
            resized_width=200,
            resized_height=100,
        )
        self.assertEqual(mapped, [0.0, 10.0, 100.0, 50.0])

    def test_affine_box_is_mapped_back_to_original_coordinates(self) -> None:
        mapped = CORE.map_affine_xyxy(
            [75.264, 0.0, 436.736, 512.0],
            transform=[[0.9765625, 0.0, -73.5], [0.0, 0.9765625, 0.0]],
            original_width=353,
            original_height=500,
        )
        self.assertAlmostEqual(mapped[0], 0.0, places=5)
        self.assertAlmostEqual(mapped[1], 0.0, places=5)
        self.assertAlmostEqual(mapped[2], 353.0, places=5)
        self.assertAlmostEqual(mapped[3], 500.0, places=5)

    def test_affine_box_preserves_nonfinite_values_for_the_audit(self) -> None:
        mapped = CORE.map_affine_xyxy(
            [float("nan"), 0.0, 1.0, 1.0],
            transform=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            original_width=10,
            original_height=10,
        )
        self.assertTrue(any(value != value for value in mapped))

    def test_normalizer_maps_zero_based_model_class_to_one_based_voc_id(self) -> None:
        rows, audit = CORE.normalize_detections(
            image_id=9,
            class_ids=[0, 19, -1],
            scores=[0.9, 0.8, 0.7],
            boxes=[[0, 0, 20, 10], [20, 10, 40, 30], [0, 0, 1, 1]],
            original_width=20,
            original_height=15,
            resized_width=40,
            resized_height=30,
            category_count=20,
            score_floor=0.0001,
        )
        self.assertEqual([row["category_id"] for row in rows], [1, 20])
        self.assertEqual(rows[0]["bbox"], [0.0, 0.0, 10.0, 5.0])
        self.assertEqual(audit["invalid_class"], 1)

    def test_normalizer_rejects_nonfinite_and_degenerate_boxes(self) -> None:
        rows, audit = CORE.normalize_detections(
            image_id=1,
            class_ids=[0, 0],
            scores=[0.9, 0.8],
            boxes=[[float("nan"), 0, 1, 1], [4, 4, 4, 8]],
            original_width=10,
            original_height=10,
            resized_width=10,
            resized_height=10,
            category_count=20,
            score_floor=0.0001,
        )
        self.assertEqual(rows, [])
        self.assertEqual(audit["nonfinite"], 1)
        self.assertEqual(audit["degenerate"], 1)

    def test_q90_encoding_is_deterministic_and_reports_actual_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            left = Path(directory) / "left.jpg"
            right = Path(directory) / "right.jpg"
            Image.new("RGB", (31, 17), (10, 120, 240)).save(source)
            left_bytes = CORE.encode_q90(source, left, quality=90, optimize=True, subsampling=2)
            right_bytes = CORE.encode_q90(source, right, quality=90, optimize=True, subsampling=2)
            self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertEqual(left_bytes, left.stat().st_size)


if __name__ == "__main__":
    unittest.main()
