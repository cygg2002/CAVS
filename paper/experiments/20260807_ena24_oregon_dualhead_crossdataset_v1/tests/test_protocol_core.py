from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "crossdataset_protocol_core", ROOT / "scripts/crossdataset_core.py"
)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


class ProtocolContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_frozen_protocol_is_valid(self) -> None:
        core.validate_protocol(self.protocol)

    def test_contract_module_does_not_shadow_canonical_protocol_core(self) -> None:
        self.assertFalse((ROOT / "scripts/protocol_core.py").exists())
        self.assertTrue((ROOT / "scripts/crossdataset_core.py").is_file())

    def test_expected_result_matrices_are_exact(self) -> None:
        self.assertEqual(len(core.expected_track_a_keys(self.protocol)), 96)
        self.assertEqual(len(core.expected_track_b_keys(self.protocol)), 18)

    def test_cloud_v2_source_is_rejected(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["source_experiments"]["canonical"] = (
            "paper/experiments/20260727_oregon_o2m_cloudv2_paper_evaluation"
        )
        with self.assertRaisesRegex(ValueError, "Cloud-V2|cloudv2"):
            core.validate_protocol(changed)

    def test_track_a_cannot_access_final(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["track_a"]["final_accessed"] = True
        with self.assertRaisesRegex(ValueError, "final"):
            core.validate_protocol(changed)

    def test_frozen_gbdt_thresholds_are_checked(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["datasets"]["ena24"]["gbdt_fixed_call_threshold"] += 0.01
        with self.assertRaisesRegex(ValueError, "GBDT"):
            core.validate_protocol(changed)


class ScoreAndSelectionContractsTest(unittest.TestCase):
    def test_align_scores_uses_role_local_identity_and_order(self) -> None:
        samples = [
            {"image_id": 7, "identity": "fit/a"},
            {"image_id": 7, "identity": "fit/b"},
            {"image_id": 9, "identity": "fit/c"},
        ]
        rows = [
            {"image_id": 9, "identity": "fit/c", "score": 0.3},
            {"image_id": 7, "identity": "fit/a", "score": 0.1},
            {"image_id": 7, "identity": "fit/b", "score": 0.2},
        ]
        self.assertEqual(core.align_scores(samples, rows), [0.1, 0.2, 0.3])

    def test_align_scores_rejects_missing_or_duplicate_identity(self) -> None:
        samples = [{"image_id": 1, "identity": "x"}]
        with self.assertRaises(ValueError):
            core.align_scores(samples, [])
        with self.assertRaises(ValueError):
            core.align_scores(
                samples,
                [
                    {"image_id": 1, "identity": "x", "score": 1.0},
                    {"image_id": 1, "identity": "x", "score": 2.0},
                ],
            )

    def test_filter_actions_builds_exact_o_and_oq_spaces(self) -> None:
        record = {
            "image_id": 1,
            "actions": [
                {"action_id": "edge"},
                {"action_id": "original"},
                {"action_id": "q90"},
                {"action_id": "roi"},
            ],
        }
        self.assertEqual(
            [row["action_id"] for row in core.filter_actions(record, ("edge", "original"))["actions"]],
            ["edge", "original"],
        )
        self.assertEqual(
            [row["action_id"] for row in core.filter_actions(record, ("edge", "original", "q90"))["actions"]],
            ["edge", "original", "q90"],
        )

    def test_capacity_overflow_is_rejected(self) -> None:
        core.assert_capacity_safe(99.0, 100.0)
        with self.assertRaises(RuntimeError):
            core.assert_capacity_safe(100.01, 100.0)

    def test_track_b_membership_must_match_across_policies(self) -> None:
        core.assert_same_membership(
            {"SharedAction": {1, 2}, "CostEligible-AllROI": {1, 2}, "GBDT-Hierarchical": {2, 1}}
        )
        with self.assertRaises(RuntimeError):
            core.assert_same_membership(
                {"SharedAction": {1, 2}, "CostEligible-AllROI": {1, 3}}
            )

    def test_result_matrix_rejects_duplicate_or_missing_rows(self) -> None:
        protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )
        rows = [
            {
                "dataset": dataset,
                "role": role,
                "method": method,
                "mode": mode,
                "nominal_budget": budget,
            }
            for dataset, role, method, mode, budget in core.expected_track_a_keys(protocol)
        ]
        core.assert_track_a_matrix(rows, protocol)
        with self.assertRaises(RuntimeError):
            core.assert_track_a_matrix(rows[:-1], protocol)
        with self.assertRaises(RuntimeError):
            core.assert_track_a_matrix(rows + [rows[0]], protocol)

    def test_track_b_bootstrap_metadata_comes_from_core_row(self) -> None:
        metadata = core.track_b_bootstrap_metadata(
            "CostEligible-AllROI|0.25",
            {
                "action_policy": "CostEligible-AllROI",
                "nominal_budget": 0.25,
                "method": "DualHead-Value",
                "mode": "fixed_call",
                "comparator": "HistoricalRanker",
            },
        )
        self.assertEqual(metadata["comparison"], "DualHead-Value - HistoricalRanker")
        self.assertEqual(metadata["candidate_method"], "DualHead-Value")
        self.assertEqual(metadata["baseline_method"], "HistoricalRanker")
        self.assertEqual(metadata["action_policy"], "CostEligible-AllROI")
        self.assertEqual(metadata["nominal_budget"], 0.25)

    def test_track_b_bootstrap_metadata_rejects_key_row_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            core.track_b_bootstrap_metadata(
                "SharedAction|0.1",
                {
                    "action_policy": "GBDT-Hierarchical",
                    "nominal_budget": 0.1,
                    "method": "DualHead-Value",
                    "mode": "fixed_call",
                    "comparator": "HistoricalRanker",
                },
            )


if __name__ == "__main__":
    unittest.main()
