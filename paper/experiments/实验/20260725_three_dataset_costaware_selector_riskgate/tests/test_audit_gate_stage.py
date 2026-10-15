import importlib.util
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "costaware_audit_gate", ROOT / "scripts/run_audit_gate.py"
)
audit_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_gate
SPEC.loader.exec_module(audit_gate)


class AuditGateDecisionTests(unittest.TestCase):
    def test_fixed_call_requires_both_ap_and_nonpositive_kb_upper_bound(self):
        self.assertTrue(
            audit_gate.gate_passes(
                mode="fixed_call",
                ap_delta_ci_low=-0.001,
                kb_delta_ci_high=0.0,
                ap_tolerance=0.001,
            )
        )
        self.assertFalse(
            audit_gate.gate_passes(
                mode="fixed_call",
                ap_delta_ci_low=-0.0002,
                kb_delta_ci_high=0.0001,
                ap_tolerance=0.001,
            )
        )

    def test_online_uses_cost_gate_but_fixed_byte_uses_ap_only(self):
        self.assertFalse(
            audit_gate.gate_passes(
                mode="online",
                ap_delta_ci_low=0.0,
                kb_delta_ci_high=0.01,
                ap_tolerance=0.001,
            )
        )
        self.assertTrue(
            audit_gate.gate_passes(
                mode="fixed_byte",
                ap_delta_ci_low=-0.001,
                kb_delta_ci_high=99.0,
                ap_tolerance=0.001,
            )
        )

    def test_failed_gate_falls_directly_back_to_reference(self):
        choice = {
            "dataset": "coco",
            "mode": "fixed_call",
            "budget": 0.1,
            "reference_ranker": "SemanticRank",
            "reference_policy": "SharedAction",
            "reference_threshold": math.inf,
            "provisional_ranker": "SemanticRank",
            "provisional_policy": "GBDT-Hierarchical",
            "provisional_threshold": 0.9,
        }
        frozen = audit_gate.freeze_choice(
            choice,
            {
                "official_map50_95_delta_ci_low": -0.0011,
                "request_kb_per_image_delta_ci_high": -1.0,
            },
            ap_tolerance=0.001,
        )
        self.assertFalse(frozen["gate_passed"])
        self.assertTrue(frozen["used_reference_fallback"])
        self.assertEqual(frozen["frozen_ranker"], "SemanticRank")
        self.assertEqual(frozen["frozen_policy"], "SharedAction")
        self.assertEqual(frozen["candidates_attempted"], 1)

    def test_bootstrap_metadata_cannot_overwrite_frozen_work_point_identity(self):
        choice = {
            "dataset": "coco",
            "mode": "online",
            "budget": 0.1,
            "reference_ranker": "SemanticRank",
            "reference_policy": "SharedAction",
            "reference_threshold": math.inf,
            "provisional_ranker": "SemanticRank",
            "provisional_policy": "GBDT-Hierarchical",
            "provisional_threshold": 0.9,
        }
        frozen = audit_gate.freeze_choice(
            choice,
            {
                "dataset": "wrong-dataset",
                "mode": "fixed_call",
                "budget": 0.25,
                "nominal_budget": 0.25,
                "official_map50_95_delta_ci_low": 0.0,
                "request_kb_per_image_delta_ci_high": -1.0,
            },
            ap_tolerance=0.001,
        )

        self.assertEqual(frozen["dataset"], "coco")
        self.assertEqual(frozen["mode"], "online")
        self.assertEqual(frozen["budget"], 0.1)
        self.assertNotIn("nominal_budget", frozen)

    def test_completed_checkpoint_rows_restore_choice_identity_in_order(self):
        choices = [
            {
                "dataset": "coco",
                "mode": "online",
                "budget": 0.1,
                "reference_ranker": "SemanticRank",
                "reference_policy": "SharedAction",
                "reference_threshold": math.inf,
                "provisional_ranker": "SemanticRank",
                "provisional_policy": "GBDT-Hierarchical",
                "provisional_threshold": 0.9,
            },
            {
                "dataset": "coco",
                "mode": "fixed_byte",
                "budget": 0.1,
                "reference_ranker": "Confidence",
                "reference_policy": "SharedAction",
                "reference_threshold": math.inf,
                "provisional_ranker": "Confidence",
                "provisional_policy": "GBDT-Hierarchical",
                "provisional_threshold": 0.9,
            },
        ]
        corrupted_rows = [
            {
                **choice,
                "mode": "fixed_call",
                "nominal_budget": 0.25,
                "official_map50_95_delta_ci_low": 0.0,
                "request_kb_per_image_delta_ci_high": -1.0,
            }
            for choice in choices
        ]

        restored = audit_gate.restore_checkpoint_identities(
            choices, corrupted_rows, ap_tolerance=0.001
        )

        self.assertEqual([row["mode"] for row in restored], ["online", "fixed_byte"])
        self.assertTrue(all("nominal_budget" not in row for row in restored))


class AuditActionReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "samples": [
                {
                    "image_id": 1,
                    "payload_bytes": {"original": 100, "q90": 50, "roi": 25},
                    "available": {"original": True, "q90": True, "roi": True},
                },
                {
                    "image_id": 2,
                    "payload_bytes": {"original": 100, "q90": 50, "roi": 25},
                    "available": {"original": True, "q90": True, "roi": True},
                },
                {
                    "image_id": 3,
                    "payload_bytes": {"original": 100, "q90": 50, "roi": 25},
                    "available": {"original": True, "q90": True, "roi": True},
                },
                {
                    "image_id": 4,
                    "payload_bytes": {"original": 100, "q90": 50, "roi": 25},
                    "available": {"original": True, "q90": True, "roi": True},
                },
            ]
        }
        self.data["sample_by_id"] = {
            sample["image_id"]: sample for sample in self.data["samples"]
        }
        self.scores = [0.9, 0.8, 0.2, 0.1]
        self.base = {1: "original", 2: "q90", 3: "original", 4: "q90"}
        self.roi_scores = [0.95, 0.85, 0.2, 0.1]

    def test_fixed_call_reuses_frozen_ranker_and_roi_threshold(self):
        selected = audit_gate.reconstruct_actions(
            mode="fixed_call",
            budget=0.5,
            data=self.data,
            calibration_scores=self.scores,
            audit_scores=self.scores,
            base_actions=self.base,
            roi_scores=self.roi_scores,
            roi_threshold=0.9,
            online_threshold=None,
            lambda_value=None,
            median_original_bytes=None,
        )
        self.assertEqual(selected, {1: "roi", 2: "q90"})

    def test_online_reuses_frozen_frame_threshold(self):
        selected = audit_gate.reconstruct_actions(
            mode="online",
            budget=0.1,
            data=self.data,
            calibration_scores=self.scores,
            audit_scores=self.scores,
            base_actions=self.base,
            roi_scores=self.roi_scores,
            roi_threshold=math.inf,
            online_threshold=0.5,
            lambda_value=None,
            median_original_bytes=None,
        )
        self.assertEqual(selected, {1: "original", 2: "q90"})

    def test_fixed_byte_reuses_frozen_lambda_and_calibration_percentiles(self):
        selected = audit_gate.reconstruct_actions(
            mode="fixed_byte",
            budget=0.5,
            data=self.data,
            calibration_scores=self.scores,
            audit_scores=self.scores,
            base_actions=self.base,
            roi_scores=self.roi_scores,
            roi_threshold=math.inf,
            online_threshold=None,
            lambda_value=0.0,
            median_original_bytes=100.0,
        )
        self.assertEqual(selected, {1: "original", 2: "q90", 4: "q90"})


if __name__ == "__main__":
    unittest.main()
