from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/data_core.py"
SPEC = importlib.util.spec_from_file_location("voc_gluoncv_data_core", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(PATH)
DATA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DATA
SPEC.loader.exec_module(DATA)


class DataCoreTest(unittest.TestCase):
    def test_fold_contexts_never_cross_folds(self) -> None:
        folds = [[1, 2, 3, 4], [10, 11, 12, 13]]
        contexts = DATA.fold_contained_contexts(folds, context_size=2, seed=7)
        for fold in folds:
            allowed = set(fold)
            for image_id in fold:
                self.assertNotIn(image_id, contexts[image_id])
                self.assertEqual(len(contexts[image_id]), 2)
                self.assertLessEqual(set(contexts[image_id]), allowed)

    def test_action_records_support_independent_o_oq_and_oqr_views(self) -> None:
        samples = [
            {
                "image_id": 1,
                "identity": "2007:000001",
                "width": 100,
                "height": 80,
                "payload_bytes": {"original": 1000, "q90": 600, "current@official": 250},
                "available": {"original": True, "q90": True, "current@official": True},
                "roi": {"xyxy": [5, 10, 70, 75], "area_ratio": 0.528125},
            }
        ]
        edge = {
            1: [
                {"image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.9}
            ]
        }
        oq = DATA.build_action_records(samples, edge, category_ids=list(range(1, 21)), action_space="OQ")
        oqr = DATA.build_action_records(samples, edge, category_ids=list(range(1, 21)), action_space="OQR")
        only_o = DATA.build_action_records(samples, edge, category_ids=list(range(1, 21)), action_space="O")
        self.assertEqual([row["action_id"] for row in oq[0]["actions"]], ["edge", "original", "q90"])
        self.assertEqual(
            [row["action_id"] for row in oqr[0]["actions"]],
            ["edge", "original", "q90", "current@official"],
        )
        self.assertEqual([row["action_id"] for row in only_o[0]["actions"]], ["edge", "original"])
        self.assertEqual(oq[0]["edge_boxes"][0]["cls"], 0)

    def test_unavailable_roi_is_not_inserted_into_oqr(self) -> None:
        samples = [{
            "image_id": 1,
            "identity": "2007:000001",
            "width": 10,
            "height": 10,
            "payload_bytes": {"original": 100, "q90": 60},
            "available": {"original": True, "q90": True, "current@official": False},
        }]
        records = DATA.build_action_records(samples, {1: []}, category_ids=[1], action_space="OQR")
        self.assertEqual(
            [row["action_id"] for row in records[0]["actions"]],
            ["edge", "original", "q90"],
        )

    def test_q90_larger_than_original_is_unavailable(self) -> None:
        samples = [
            {
                "image_id": 1,
                "identity": "2007:000001",
                "width": 10,
                "height": 10,
                "payload_bytes": {"original": 100, "q90": 120},
                "available": {"original": True, "q90": False},
            }
        ]
        records = DATA.build_action_records(samples, {1: []}, category_ids=[1], action_space="OQ")
        self.assertEqual([row["action_id"] for row in records[0]["actions"]], ["edge", "original"])


if __name__ == "__main__":
    unittest.main()
