import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unified_cavs_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class ProtocolContractTest(unittest.TestCase):
    def test_protocol_is_the_frozen_two_dataset_contract(self):
        CORE.validate_protocol(PROTOCOL)
        self.assertEqual(PROTOCOL["datasets"]["coco"]["roles"], {"fit": 9678, "calibration": 571, "final": 5000})
        self.assertEqual(PROTOCOL["datasets"]["oregon"]["roles"], {"fit": 9923, "calibration": 4961, "final": 4962})
        self.assertFalse(PROTOCOL["final_is_hidden"])
        self.assertFalse(PROTOCOL["final_used_for_training_or_selection"])

    def test_action_masks_are_nested_and_use_one_roi(self):
        masks = PROTOCOL["action_spaces"]
        self.assertLess(set(masks["O"]), set(masks["OQ"]))
        self.assertLess(set(masks["OQ"]), set(masks["OQR"]))
        self.assertEqual(set(masks["OQR"]) - set(masks["OQ"]), {"current@640"})
        self.assertEqual(PROTOCOL["primary_roi_fusion"], "Reversible")

    def test_scalar_policy_is_shared_not_method_specific(self):
        self.assertIsInstance(PROTOCOL["scalar_policy"], str)
        self.assertEqual(PROTOCOL["scalar_policy"], "SharedAction-OQR")
        self.assertNotIn("method_action_policies", PROTOCOL)

    def test_expected_point_matrix(self):
        self.assertEqual(CORE.expected_cavs_rows(PROTOCOL, roles=3), 54)
        self.assertEqual(CORE.expected_scalar_rows(PROTOCOL, roles=2), 84)

    def test_bootstrap_matrix_counts_all_units(self):
        self.assertEqual(CORE.expected_bootstrap_jobs(PROTOCOL, "coco"), 24)
        self.assertEqual(CORE.expected_bootstrap_jobs(PROTOCOL, "oregon"), 48)
        self.assertEqual(len(CORE.expected_bootstrap_job_ids(PROTOCOL, "coco")), 24)
        self.assertEqual(len(CORE.expected_bootstrap_job_ids(PROTOCOL, "oregon")), 48)

    def test_json_threshold_represents_infinity_as_null(self):
        self.assertIsNone(CORE.json_threshold(float("inf")))
        self.assertIsNone(CORE.json_threshold(float("-inf")))
        self.assertEqual(CORE.json_threshold(0.75), 0.75)


class SelectionContractTest(unittest.TestCase):
    def test_reference_action_maps_are_edge_original_and_q90_only(self):
        samples = [
            {
                "image_id": 1,
                "available": {"original": True, "q90": True},
            },
            {
                "image_id": 2,
                "available": {"original": True, "q90": True},
            },
        ]
        maps = CORE.reference_action_maps(samples)
        self.assertEqual(maps["Edge-only"], {})
        self.assertEqual(maps["Cloud-only Original"], {1: "original", 2: "original"})
        self.assertEqual(maps["Cloud-only Q90"], {1: "q90", 2: "q90"})

    def test_reference_q90_falls_back_to_original_when_unavailable(self):
        maps = CORE.reference_action_maps(
            [{"image_id": 1, "available": {"original": True, "q90": False}}]
        )
        self.assertEqual(maps["Cloud-only Q90"], {1: "original"})

    def test_grouped_folds_reject_event_leakage(self):
        with self.assertRaises(ValueError):
            CORE.validate_grouped_folds(
                [1, 2, 3],
                [[1], [2, 3]],
                {1: "event-a", 2: "event-a", 3: "event-b"},
            )
        report = CORE.validate_grouped_folds(
            [1, 2, 3],
            [[1, 2], [3]],
            {1: "event-a", 2: "event-a", 3: "event-b"},
        )
        self.assertEqual(report["group_leakage_count"], 0)
        self.assertEqual(report["groups_per_fold"], [1, 1])

    def test_action_mask_rejects_forbidden_actions(self):
        scores = {1: {"original": 0.2, "q90": 0.3, "current@640": 0.4}}
        masked = CORE.mask_scores(scores, PROTOCOL["action_spaces"]["OQ"])
        self.assertEqual(masked, {1: {"original": 0.2, "q90": 0.3}})

    def test_fixed_byte_capacity_is_strict(self):
        samples = [
            {"image_id": 1, "payload_bytes": {"original": 100, "q90": 50}},
            {"image_id": 2, "payload_bytes": {"original": 100, "q90": 50}},
        ]
        report = CORE.validate_selection(samples, {1: "q90"}, mode="fixed_byte", budget=0.25)
        self.assertEqual(report["request_bytes"], 50)
        with self.assertRaises(ValueError):
            CORE.validate_selection(samples, {1: "original"}, mode="fixed_byte", budget=0.25)

    def test_unique_rows_reject_duplicates(self):
        rows = [{"role": "final", "method": "x", "policy": "p", "mode": "fixed_byte", "budget": 0.1}] * 2
        with self.assertRaises(ValueError):
            CORE.validate_unique_rows(rows)

    def test_calibration_frozen_external_choice(self):
        rows = [
            {"role": "calibration", "method": "ORIC-AP5095", "mode": "fixed_byte", "budget": 0.1, "official_ap50_95": 0.5},
            {"role": "calibration", "method": "MORIC-AP5095", "mode": "fixed_byte", "budget": 0.1, "official_ap50_95": 0.6},
            {"role": "final", "method": "ORIC-AP5095", "mode": "fixed_byte", "budget": 0.1, "official_ap50_95": 0.9},
        ]
        self.assertEqual(CORE.strongest_external(rows, 0.1, PROTOCOL), "MORIC-AP5095")

    def test_shared_action_map_is_independent_of_ranker(self):
        samples = [
            {"image_id": 1, "available": {"q90": True, "current@640": True}},
            {"image_id": 2, "available": {"q90": True, "current@640": False}},
        ]
        actions = CORE.shared_action_map(
            samples,
            q90_probabilities=[0.8, 0.4],
            roi_probabilities={1: 0.9},
            q90_threshold=0.5,
            roi_threshold=0.7,
        )
        self.assertEqual(actions, {1: "current@640", 2: "original"})

    def test_scalar_byte_selection_obeys_capacity(self):
        samples = [
            {"image_id": 1, "payload_bytes": {"original": 100, "q90": 40}},
            {"image_id": 2, "payload_bytes": {"original": 100, "q90": 40}},
            {"image_id": 3, "payload_bytes": {"original": 100, "q90": 40}},
        ]
        selection = CORE.scalar_byte_selection(
            samples,
            scores=[0.9, 0.5, 0.1],
            action_map={1: "q90", 2: "q90", 3: "q90"},
            budget=0.2,
            lambda_value=0.0,
            reference_scores=[0.9, 0.5, 0.1],
            median_original_bytes=100.0,
        )
        self.assertEqual(selection, {1: "q90"})

    def test_assemble_oof_scores_restores_canonical_order(self):
        matrices = [
            ([3, 1], [[[30.0], [10.0]]]),
            ([2], [[[20.0]]]),
        ]
        result = CORE.assemble_oof_scores([1, 2, 3], matrices)
        self.assertEqual(result.shape, (1, 3, 1))
        self.assertEqual(result[:, :, 0].tolist(), [[10.0, 20.0, 30.0]])

    def test_action_score_map_uses_declared_action_columns_only(self):
        records = [
            {
                "image_id": 7,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "q90"},
                    {"action_id": "current@640"},
                ],
            }
        ]
        result = CORE.action_score_map(records, [[0.0, 1.0, 2.0, 3.0, 99.0]])
        self.assertEqual(result, {7: {"original": 1.0, "q90": 2.0, "current@640": 3.0}})

    def test_fixed_call_joint_selects_one_best_action_per_image(self):
        samples = [
            {"image_id": 1, "payload_bytes": {"original": 100, "q90": 40}},
            {"image_id": 2, "payload_bytes": {"original": 100, "q90": 40}},
        ]
        scores = {
            1: {"original": 0.8, "q90": 0.9},
            2: {"original": 0.7, "q90": 0.2},
        }
        self.assertEqual(
            CORE.fixed_call_joint(samples, scores, budget=0.5),
            {1: "q90"},
        )


if __name__ == "__main__":
    unittest.main()
