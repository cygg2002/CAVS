from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_point_data_core", ROOT / "scripts/data_core.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("data_core.py is unavailable")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class DataCoreTest(unittest.TestCase):
    def test_voc_box_conversion_matches_ultralytics_coordinate_contract(self) -> None:
        coco = CORE.voc_box_to_coco(1, 1, 101, 51)
        self.assertEqual(coco, [0.0, 0.0, 100.0, 50.0])
        yolo = CORE.voc_box_to_yolo(1, 1, 101, 51, width=200, height=100)
        self.assertAlmostEqual(yolo[0], 0.25)
        self.assertAlmostEqual(yolo[1], 0.25)
        self.assertAlmostEqual(yolo[2], 0.5)
        self.assertAlmostEqual(yolo[3], 0.5)

    def test_group_predictions_keeps_empty_images_explicit(self) -> None:
        grouped = CORE.group_predictions(
            [{"image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}],
            image_ids=[1, 2],
        )
        self.assertEqual(grouped[1], [])
        self.assertEqual(len(grouped[2]), 1)

    def test_source_basename_mapping_uses_resolved_image_path(self) -> None:
        images = [
            {
                "id": 7,
                "file_name": "2012_2008_001823.jpg",
                "absolute_path": "/voc/VOC2012/JPEGImages/2008_001823.jpg",
            }
        ]
        self.assertEqual(
            CORE.source_basename_to_id(images),
            {"2008_001823.jpg": 7},
        )

    def test_source_basename_mapping_rejects_collisions(self) -> None:
        images = [
            {"id": 1, "absolute_path": "/voc/VOC2007/JPEGImages/shared.jpg"},
            {"id": 2, "absolute_path": "/voc/VOC2012/JPEGImages/shared.jpg"},
        ]
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            CORE.source_basename_to_id(images)

    def test_fixed_call_count_uses_paper_rounding(self) -> None:
        self.assertEqual(CORE.fixed_call_count(4952, 0.10), 495)
        self.assertEqual(CORE.fixed_call_count(4952, 0.25), 1238)
        self.assertEqual(CORE.fixed_call_count(4952, 0.40), 1981)

    def test_crop_predictions_are_mapped_to_source_coordinates(self) -> None:
        mapped = CORE.map_crop_predictions(
            [
                {
                    "image_id": 7,
                    "category_id": 3,
                    "bbox": [2.0, 4.0, 10.0, 20.0],
                    "score": 0.8,
                }
            ],
            {7: {"xyxy": [100, 200, 300, 500]}},
        )
        self.assertEqual(mapped[0]["bbox"], [102.0, 204.0, 10.0, 20.0])

    def test_reward_contexts_stay_inside_each_fold(self) -> None:
        folds = [[1, 2, 3, 4], [10, 11, 12, 13]]
        contexts = CORE.fold_contained_contexts(
            folds, context_size=2, seed=20260816
        )
        self.assertEqual(
            contexts,
            CORE.fold_contained_contexts(folds, context_size=2, seed=20260816),
        )
        for fold in folds:
            allowed = set(fold)
            for image_id in fold:
                self.assertEqual(len(contexts[image_id]), 2)
                self.assertNotIn(image_id, contexts[image_id])
                self.assertLessEqual(set(contexts[image_id]), allowed)

    def test_reward_roles_reject_final(self) -> None:
        self.assertEqual(CORE.reward_roles("fit"), ("fit",))
        self.assertEqual(CORE.reward_roles("all"), ("fit", "calibration"))
        with self.assertRaisesRegex(ValueError, "forbidden"):
            CORE.reward_roles("final")

    def test_action_records_keep_edge_first_and_skip_unavailable_roi(self) -> None:
        samples = [
            {
                "image_id": 1,
                "identity": "2007:000001",
                "event_id": "2007:000001",
                "camera": "2007",
                "width": 100,
                "height": 80,
                "payload_bytes": {"original": 1000, "q90": 700},
                "available": {"original": True, "q90": True, "current@640": False},
                "actions": [
                    {"action_id": "edge", "payload_bytes": 0, "available": True},
                    {"action_id": "original", "payload_bytes": 1000, "available": True},
                    {"action_id": "q90", "payload_bytes": 700, "available": True},
                ],
            }
        ]
        predictions = {
            1: [
                {
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "score": 0.9,
                }
            ]
        }
        records = CORE.build_action_records(samples, predictions, category_ids=list(range(20)))
        self.assertEqual([row["action_id"] for row in records[0]["actions"]], ["edge", "original", "q90"])
        self.assertEqual(records[0]["edge_boxes"][0]["xyxy"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(records[0]["full_payload_bytes"], 1000)


if __name__ == "__main__":
    unittest.main()
