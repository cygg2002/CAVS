from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


ACCELERATOR = load_module(
    "server_completion_accelerator_test", ROOT / "scripts/accelerate_bootstrap.py"
)
VOC = load_module(
    "server_completion_voc_reference_test",
    REPO
    / "paper/experiments/20260825_voc_torchvision_crossarchitecture_cavs_oqr_paper_v1/scripts/bootstrap.py",
)


class ChunkingTest(unittest.TestCase):
    def test_chunk_ranges_cover_interval_without_overlap(self):
        self.assertEqual(
            ACCELERATOR.chunk_ranges(3, 11, 3),
            [(3, 6), (6, 9), (9, 11)],
        )

    def test_out_of_order_chunks_only_extend_contiguous_prefix(self):
        values = [1.0]
        pending = {}
        self.assertFalse(ACCELERATOR.merge_contiguous(values, pending, 3, [4.0]))
        self.assertEqual(values, [1.0])
        self.assertTrue(ACCELERATOR.merge_contiguous(values, pending, 1, [2.0, 3.0]))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(pending, {})

    def test_overlapping_chunk_is_rejected(self):
        with self.assertRaises(RuntimeError):
            ACCELERATOR.merge_contiguous([1.0], {}, 0, [2.0])


class FastVocMetricTest(unittest.TestCase):
    def test_preindexed_metric_matches_frozen_reference_with_duplicates_and_difficult(self):
        ground_truth = {
            "images": [
                {"id": 11, "file_name": "a.jpg"},
                {"id": 12, "file_name": "b.jpg"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 11,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "difficult": 0,
                },
                {
                    "id": 2,
                    "image_id": 11,
                    "category_id": 1,
                    "bbox": [20.0, 20.0, 10.0, 10.0],
                    "difficult": 1,
                },
                {
                    "id": 3,
                    "image_id": 12,
                    "category_id": 1,
                    "bbox": [5.0, 5.0, 8.0, 8.0],
                    "difficult": 0,
                },
            ],
            "categories": [{"id": 1, "name": "object"}],
        }
        predictions = [
            {"image_id": 11, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9},
            {"image_id": 11, "category_id": 1, "bbox": [20.0, 20.0, 10.0, 10.0], "score": 0.9},
            {"image_id": 11, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9},
            {"image_id": 12, "category_id": 1, "bbox": [5.0, 5.0, 8.0, 8.0], "score": 0.8},
            {"image_id": 12, "category_id": 1, "bbox": [30.0, 30.0, 2.0, 2.0], "score": 0.7},
        ]
        sampled = [11, 11, 12, 11]
        remapped_ground_truth, (remapped_predictions,) = VOC.bootstrap_remap(
            ground_truth, [predictions], sampled
        )
        expected = VOC._in_memory_ap(remapped_ground_truth, remapped_predictions)
        fast = ACCELERATOR.FastVocBootstrap(VOC, ground_truth)
        fast.compile(predictions)
        actual = fast.evaluate(predictions, sampled)
        self.assertAlmostEqual(actual, expected, places=14)


if __name__ == "__main__":
    unittest.main()
