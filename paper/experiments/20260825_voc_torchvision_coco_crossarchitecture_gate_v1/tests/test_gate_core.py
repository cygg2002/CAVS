import importlib.util
import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("voc_torchvision_gate_core", ROOT / "scripts/gate_core.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProtocolContractTest(unittest.TestCase):
    def test_protocol_rejects_final_access(self):
        protocol = MODULE.read_protocol(ROOT / "configs/protocol.json")
        protocol["allowed_roles"].append("final")
        with self.assertRaisesRegex(ValueError, "final"):
            MODULE.validate_protocol(protocol)

    def test_protocol_rejects_voc_trained_detector(self):
        protocol = MODULE.read_protocol(ROOT / "configs/protocol.json")
        protocol["detectors"]["edge"]["training_dataset"] = "VOC07+12 trainval"
        with self.assertRaisesRegex(ValueError, "COCO train2017"):
            MODULE.validate_protocol(protocol)

    def test_category_mapping_covers_exact_voc_schema(self):
        protocol = MODULE.read_protocol(ROOT / "configs/protocol.json")
        voc = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ]
        mapped = MODULE.coco_to_voc_ids(protocol["category_mapping"], voc)
        self.assertEqual(len(mapped), 20)
        self.assertEqual(mapped["airplane"], 1)
        self.assertEqual(mapped["tv"], 20)


class DeterministicSelectionTest(unittest.TestCase):
    def test_folds_are_deterministic_complete_and_disjoint(self):
        identities = [f"2007:{index:06d}" for index in range(23)]
        first = MODULE.deterministic_folds(identities, count=5, seed=7)
        second = MODULE.deterministic_folds(identities, count=5, seed=7)
        self.assertEqual(first, second)
        flattened = [value for fold in first for value in fold]
        self.assertEqual(sorted(flattened), sorted(identities))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_smoke_selection_is_deterministic_and_covers_categories(self):
        images = [
            {"id": index, "identity": f"x:{index}", "category_ids": [1 + index % 20]}
            for index in range(100)
        ]
        first = MODULE.select_smoke_images(images, count=40, seed=9, category_ids=range(1, 21))
        second = MODULE.select_smoke_images(images, count=40, seed=9, category_ids=range(1, 21))
        self.assertEqual(first, second)
        covered = {category for row in first for category in row["category_ids"]}
        self.assertEqual(covered, set(range(1, 21)))


class PredictionSanitizationTest(unittest.TestCase):
    def test_sanitization_maps_classes_clamps_and_drops_invalid_boxes(self):
        output, audit = MODULE.sanitize_detections(
            image_id=9,
            width=100,
            height=80,
            boxes=[[-3, 2, 20, 30], [4, 4, 4, 8], [1, 1, 10, 10], [0, 0, float("nan"), 4]],
            labels=[1, 1, 99, 1],
            scores=[0.9, 0.8, 0.7, 0.6],
            label_names={1: "person", 99: "toothbrush"},
            category_ids={"person": 15},
            threshold=0.001,
        )
        self.assertEqual(output, [{"image_id": 9, "category_id": 15, "bbox": [0.0, 2.0, 20.0, 28.0], "score": 0.9}])
        self.assertEqual(audit["degenerate_boxes"], 1)
        self.assertEqual(audit["nonfinite_boxes"], 1)
        self.assertEqual(audit["filtered_non_voc"], 1)


class GateDecisionTest(unittest.TestCase):
    def test_calibration_preflight_stops_before_fit_when_cloud_gap_is_too_small(self):
        protocol = MODULE.read_protocol(ROOT / "configs/protocol.json")
        decision = MODULE.assess_calibration_preflight(
            protocol,
            metrics={
                "edge": {"map50_95": 0.40, "map75": 0.42},
                "original": {"map50_95": 0.41, "map75": 0.44},
                "q90": {"map50_95": 0.405},
            },
            original_bytes=1000,
            q90_bytes=700,
        )
        self.assertFalse(decision["proceed_to_fit"])

    def test_gate_passes_only_when_every_registered_condition_passes(self):
        protocol = MODULE.read_protocol(ROOT / "configs/protocol.json")
        decision = MODULE.assess_endpoint_gate(
            protocol,
            fit={"edge": {"map50_95": 0.30, "map75": 0.31}, "original": {"map50_95": 0.36, "map75": 0.38}, "q90": {"map50_95": 0.355}},
            calibration={"edge": {"map50_95": 0.31, "map75": 0.32}, "original": {"map50_95": 0.35, "map75": 0.37}, "q90": {"map50_95": 0.345}},
            fit_fold_deltas=[0.04, 0.05, 0.03, 0.04, 0.02],
            bootstrap_ci=[0.01, 0.07],
            original_bytes=1000,
            q90_bytes=700,
        )
        self.assertTrue(decision["proceed_to_reward"])
        self.assertTrue(all(row["passed"] for row in decision["checks"]))

    def test_gate_rejects_endpoint_direction_reversal(self):
        protocol = MODULE.read_protocol(ROOT / "configs/protocol.json")
        decision = MODULE.assess_endpoint_gate(
            protocol,
            fit={"edge": {"map50_95": 0.40, "map75": 0.41}, "original": {"map50_95": 0.38, "map75": 0.39}, "q90": {"map50_95": 0.37}},
            calibration={"edge": {"map50_95": 0.30, "map75": 0.31}, "original": {"map50_95": 0.34, "map75": 0.35}, "q90": {"map50_95": 0.33}},
            fit_fold_deltas=[-0.02, 0.04, 0.03, 0.02, 0.01],
            bootstrap_ci=[0.01, 0.06],
            original_bytes=1000,
            q90_bytes=700,
        )
        self.assertFalse(decision["proceed_to_reward"])


class BootstrapRemapTest(unittest.TestCase):
    def test_resampling_with_replacement_creates_unique_image_instances(self):
        ground_truth = {
            "categories": [{"id": 1, "name": "person"}],
            "images": [{"id": 7}, {"id": 8}],
            "annotations": [
                {"id": 10, "image_id": 7, "category_id": 1, "bbox": [0, 0, 2, 2]},
                {"id": 11, "image_id": 8, "category_id": 1, "bbox": [1, 1, 2, 2]},
            ],
        }
        predictions = [[{"image_id": 7, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 0.9}]]
        remapped_gt, remapped_predictions = MODULE.bootstrap_remap(
            copy.deepcopy(ground_truth), predictions, [7, 7, 8]
        )
        self.assertEqual([row["id"] for row in remapped_gt["images"]], [1, 2, 3])
        self.assertEqual([row["image_id"] for row in remapped_gt["annotations"]], [1, 2, 3])
        self.assertEqual([row["image_id"] for row in remapped_predictions[0]], [1, 2])


if __name__ == "__main__":
    unittest.main()
