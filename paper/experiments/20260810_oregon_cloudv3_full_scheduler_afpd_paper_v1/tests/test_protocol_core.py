import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oregon_cloudv3_protocol_core", ROOT / "scripts/protocol_core.py"
)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class ProtocolCoreTest(unittest.TestCase):
    def test_frozen_protocol_and_complete_action_matrix(self):
        report = CORE.validate_protocol(PROTOCOL)
        self.assertEqual(report["roi_actions"], 12)
        self.assertEqual(report["learned_methods"], 29)
        self.assertEqual(PROTOCOL["cloud_model"]["epoch"], 28)
        self.assertEqual(
            PROTOCOL["cloud_model"]["sha256"],
            "d3890a57fc7d567880a5b01b7eb949387d9b7058ce7ee25bdb9471d9bb4642d9",
        )

    def test_role_numeric_ids_may_repeat_but_identity_must_not(self):
        roles = {
            "fit": [{"image_id": 1, "identity": "/fit/a.jpg"}],
            "calibration": [{"image_id": 1, "identity": "/cal/a.jpg"}],
            "final": [{"image_id": 1, "identity": "/final/a.jpg"}],
        }
        self.assertEqual(CORE.validate_role_isolation(roles), 0)
        roles["final"][0]["identity"] = "/fit/a.jpg"
        with self.assertRaisesRegex(ValueError, "identity leakage"):
            CORE.validate_role_isolation(roles)

    def test_event_group_folds_are_disjoint_and_cover_all_rows(self):
        samples = [
            {"identity": f"/x/{index}.jpg", "event_id": f"event-{index // 3}"}
            for index in range(30)
        ]
        folds = CORE.event_group_folds(samples, fold_count=5, seed=17)
        flattened = [identity for fold in folds for identity in fold]
        self.assertEqual(set(flattened), {row["identity"] for row in samples})
        self.assertEqual(len(flattened), len(samples))
        membership = {
            identity: fold_index
            for fold_index, fold in enumerate(folds)
            for identity in fold
        }
        for event_id in {row["event_id"] for row in samples}:
            event_folds = {
                membership[row["identity"]]
                for row in samples
                if row["event_id"] == event_id
            }
            self.assertEqual(len(event_folds), 1)

    def test_fixed_byte_capacity_is_strict(self):
        CORE.assert_capacity(1000, 1000)
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            CORE.assert_capacity(1001, 1000)

    def test_bootstrap_plan_is_exactly_27_by_two_units(self):
        comparisons = CORE.bootstrap_comparisons(PROTOCOL)
        self.assertEqual(len(comparisons), 27)
        self.assertEqual(len({item["id"] for item in comparisons}), 27)
        jobs = CORE.bootstrap_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 54)
        self.assertEqual({job["unit"] for job in jobs}, {"event", "image"})

    def test_cloudv2_artifacts_are_rejected_but_edge_sources_are_allowed(self):
        allowed = [
            "/experiment/canonical_oregon_event_v1/actions/edge/fit/predictions.json",
            "/experiment/canonical_oregon_event_v1/actions/fpn/fit/features.json",
        ]
        CORE.assert_source_boundary(allowed)
        forbidden = allowed + [
            "/paper/experiments/20260728_oregon_cloudv2_joint_action/checkpoint.pt"
        ]
        with self.assertRaisesRegex(ValueError, "old cloud artifact"):
            CORE.assert_source_boundary(forbidden)

    def test_final_is_posthoc_and_never_a_selection_role(self):
        self.assertEqual(PROTOCOL["status"], "posthoc_exploratory_final_reuse")
        self.assertFalse(PROTOCOL["confirmatory"])
        self.assertFalse(PROTOCOL["final_used_for_training_or_selection"])
        self.assertTrue(PROTOCOL["full_final_unconditional"])


if __name__ == "__main__":
    unittest.main()

