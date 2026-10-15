from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fullaction_core import (  # noqa: E402
    assert_fixed_call_membership,
    assert_result_matrix_complete,
    expected_result_keys,
    validate_source_artifact_state,
    validate_checkpoint_fingerprint,
    validate_protocol,
)


class FullActionCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_locks_canonical_matrix(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(len(expected_result_keys(self.protocol)), 126)
        self.assertEqual(
            self.protocol["cpu_workers"], {"point": 16, "bootstrap": 24}
        )
        self.assertEqual(self.protocol["runtime"]["sklearn_version"], "1.7.1")

    def test_complete_matrix_rejects_missing_and_duplicates(self) -> None:
        rows = [
            {
                "method": method,
                "payload_policy": policy,
                "mode": mode,
                "nominal_budget": budget,
            }
            for method in self.protocol["methods"]
            for policy in self.protocol["payload_policies"]
            for mode, budgets in self.protocol["work_points"].items()
            for budget in budgets
        ]
        assert_result_matrix_complete(rows, self.protocol)
        with self.assertRaises(RuntimeError):
            assert_result_matrix_complete(rows[:-1], self.protocol)
        with self.assertRaises(RuntimeError):
            assert_result_matrix_complete([*rows, dict(rows[0])], self.protocol)

    def test_fixed_call_membership_must_match_across_policies(self) -> None:
        assert_fixed_call_membership(
            {
                "SharedAction": {1: "original", 2: "q90"},
                "CostEligible-AllROI": {1: "roi", 2: "roi"},
                "GBDT-Hierarchical": {1: "original", 2: "q90"},
            }
        )
        with self.assertRaises(RuntimeError):
            assert_fixed_call_membership(
                {
                    "SharedAction": {1: "original", 2: "q90"},
                    "CostEligible-AllROI": {1: "roi"},
                    "GBDT-Hierarchical": {1: "original", 2: "q90"},
                }
            )

    def test_mutable_training_or_final_selection_is_rejected(self) -> None:
        changed = dict(self.protocol)
        changed["ranker_checkpoint_retrained"] = True
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_checkpoint_allows_only_bootstrap_wrapper_patch(self) -> None:
        stored = {
            "canonical_core": "old-core",
            "canonical_runner": "old-runner",
            "legacy_runner": "scoring",
            "protocol": "data-contract",
        }
        current = {
            **stored,
            "canonical_core": "new-core",
            "canonical_runner": "new-runner",
        }
        self.assertEqual(
            set(validate_checkpoint_fingerprint(stored, current)),
            {"canonical_core", "canonical_runner"},
        )
        changed_scoring = {**current, "legacy_runner": "changed"}
        with self.assertRaises(RuntimeError):
            validate_checkpoint_fingerprint(stored, changed_scoring)

    def test_source_point_artifacts_are_sufficient_for_parallel_start(self) -> None:
        validate_source_artifact_state(
            {
                "train_complete": True,
                "checkpoint_hash_matches": True,
                "point_report_exists": True,
                "selections_exist": True,
            }
        )
        with self.assertRaises(RuntimeError):
            validate_source_artifact_state(
                {
                    "train_complete": True,
                    "checkpoint_hash_matches": True,
                    "point_report_exists": True,
                    "selections_exist": False,
                }
            )
        changed = dict(self.protocol)
        changed["final_used_for_selection"] = True
        with self.assertRaises(ValueError):
            validate_protocol(changed)


if __name__ == "__main__":
    unittest.main()
