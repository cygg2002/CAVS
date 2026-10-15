from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from selector_core import (  # noqa: E402
    assign_group_disjoint_split,
    apply_risk_gate,
    choose_provisional,
    choose_reference,
    connected_hash_components,
    extract_frozen_policy_candidates,
    validate_candidate_grid,
    validate_protocol,
)


class SelectorCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_has_24_deterministic_candidates_and_random_is_not_selectable(
        self,
    ) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(
            len(self.protocol["selectable_rankers"]) * len(self.protocol["policies"]),
            24,
        )
        self.assertNotIn("Random", self.protocol["selectable_rankers"])
        self.assertFalse(self.protocol["random_selectable"])

    def test_candidate_grid_rejects_missing_or_duplicate_candidate(self) -> None:
        rows = [
            {"ranker": "A", "policy": "SharedAction"},
            {"ranker": "A", "policy": "SharedAction"},
        ]
        with self.assertRaisesRegex(ValueError, "candidate grid"):
            validate_candidate_grid(
                rows,
                rankers=("A",),
                policies=("SharedAction", "GBDT-Hierarchical"),
            )

    def test_extracts_exact_shared_all_roi_and_frozen_gbdt_candidates(self) -> None:
        rows = []
        for ranker in ("Confidence", "M2E"):
            for threshold, ap, kb in (
                (float("inf"), 0.70, 20.0),
                (0.8, 0.6998, 18.0),
                (-1.0e300, 0.69, 10.0),
            ):
                rows.append(
                    {
                        "mode": "fixed_call",
                        "ranker": ranker,
                        "budget": 0.1,
                        "threshold": threshold,
                        "ap": ap,
                        "kb": kb,
                    }
                )
        candidates = extract_frozen_policy_candidates(
            rows,
            mode="fixed_call",
            budget=0.1,
            rankers=("Confidence", "M2E"),
            gbdt_threshold=0.8,
        )
        validate_candidate_grid(
            candidates,
            rankers=("Confidence", "M2E"),
            policies=(
                "SharedAction",
                "CostEligible-AllROI",
                "GBDT-Hierarchical",
            ),
        )
        by_policy = {
            (row["ranker"], row["policy"]): row for row in candidates
        }
        self.assertEqual(by_policy[("Confidence", "SharedAction")]["threshold"], float("inf"))
        self.assertEqual(
            by_policy[("Confidence", "CostEligible-AllROI")]["threshold"],
            -1.0e300,
        )
        self.assertEqual(
            by_policy[("M2E", "GBDT-Hierarchical")]["threshold"],
            0.8,
        )

    def test_near_duplicate_components_are_transitive(self) -> None:
        components = connected_hash_components(
            {1: 0b0000, 2: 0b0001, 3: 0b0011, 4: 0b1111},
            max_hamming=1,
        )
        self.assertEqual(components, [[1, 2, 3], [4]])

    def test_group_disjoint_split_is_deterministic_and_never_splits_component(
        self,
    ) -> None:
        components = [[1, 2], [3], [4, 5], [6], [7], [8]]
        classes = {
            1: [1],
            2: [1],
            3: [2],
            4: [2],
            5: [2],
            6: [3],
            7: [3],
            8: [],
        }
        first = assign_group_disjoint_split(
            components,
            class_ids_by_image=classes,
            target_selector_images=4,
            seed=20260725,
        )
        second = assign_group_disjoint_split(
            components,
            class_ids_by_image=classes,
            target_selector_images=4,
            seed=20260725,
        )
        self.assertEqual(first, second)
        selector = set(first["selector_image_ids"])
        gate = set(first["gate_image_ids"])
        self.assertFalse(selector & gate)
        self.assertEqual(selector | gate, set(range(1, 9)))
        for component in components:
            self.assertTrue(set(component) <= selector or set(component) <= gate)

    def test_reference_is_highest_ap_shared_action_with_deterministic_ties(
        self,
    ) -> None:
        rows = [
            {
                "ranker": "Confidence",
                "policy": "SharedAction",
                "ap": 0.70,
                "kb": 20.0,
            },
            {
                "ranker": "M2E",
                "policy": "SharedAction",
                "ap": 0.71,
                "kb": 22.0,
            },
            {
                "ranker": "SemanticRank",
                "policy": "SharedAction",
                "ap": 0.71,
                "kb": 21.0,
            },
            {
                "ranker": "Confidence",
                "policy": "GBDT-Hierarchical",
                "ap": 0.72,
                "kb": 19.0,
            },
        ]
        chosen = choose_reference(
            rows,
            ranker_order=("Confidence", "M2E", "SemanticRank"),
        )
        self.assertEqual(chosen["ranker"], "SemanticRank")
        self.assertEqual(chosen["policy"], "SharedAction")

    def test_fixed_call_chooses_minimum_kb_within_reference_ap_tolerance(
        self,
    ) -> None:
        reference = {
            "ranker": "SemanticRank",
            "policy": "SharedAction",
            "ap": 0.7100,
            "kb": 22.0,
        }
        rows = [
            reference,
            {
                "ranker": "M2E",
                "policy": "GBDT-Hierarchical",
                "ap": 0.7092,
                "kb": 17.0,
            },
            {
                "ranker": "Confidence",
                "policy": "CostEligible-AllROI",
                "ap": 0.7089,
                "kb": 14.0,
            },
        ]
        chosen = choose_provisional(
            rows,
            reference=reference,
            mode="fixed_call",
            ap_tolerance=0.001,
            ranker_order=("Confidence", "M2E", "SemanticRank"),
            policy_order=(
                "SharedAction",
                "GBDT-Hierarchical",
                "CostEligible-AllROI",
            ),
        )
        self.assertEqual(chosen["ranker"], "M2E")
        self.assertEqual(chosen["policy"], "GBDT-Hierarchical")

    def test_fixed_byte_chooses_maximum_ap_then_lower_kb(self) -> None:
        reference = {
            "ranker": "Confidence",
            "policy": "SharedAction",
            "ap": 0.7000,
            "kb": 20.0,
        }
        rows = [
            reference,
            {
                "ranker": "M2E",
                "policy": "GBDT-Hierarchical",
                "ap": 0.7050,
                "kb": 20.1,
            },
            {
                "ranker": "SemanticRank",
                "policy": "CostEligible-AllROI",
                "ap": 0.7050,
                "kb": 19.9,
            },
        ]
        chosen = choose_provisional(
            rows,
            reference=reference,
            mode="fixed_byte",
            ap_tolerance=0.001,
            ranker_order=("Confidence", "M2E", "SemanticRank"),
            policy_order=(
                "SharedAction",
                "GBDT-Hierarchical",
                "CostEligible-AllROI",
            ),
        )
        self.assertEqual(chosen["ranker"], "SemanticRank")
        self.assertEqual(chosen["policy"], "CostEligible-AllROI")

    def test_failed_gate_falls_back_directly_to_reference(self) -> None:
        reference = {"ranker": "Confidence", "policy": "SharedAction"}
        provisional = {"ranker": "M2E", "policy": "GBDT-Hierarchical"}
        result = apply_risk_gate(
            reference=reference,
            provisional=provisional,
            ap_delta_ci_low=-0.00101,
            ap_tolerance=0.001,
        )
        self.assertFalse(result["gate_passed"])
        self.assertTrue(result["used_reference_fallback"])
        self.assertEqual(result["frozen"], reference)
        self.assertNotIn("second_candidate", result)

    def test_identical_choice_has_exact_zero_gate_and_passes(self) -> None:
        reference = {"ranker": "Confidence", "policy": "SharedAction"}
        result = apply_risk_gate(
            reference=reference,
            provisional=dict(reference),
            ap_delta_ci_low=0.0,
            ap_tolerance=0.001,
        )
        self.assertTrue(result["gate_passed"])
        self.assertFalse(result["used_reference_fallback"])
        self.assertTrue(result["identical_configuration"])
        self.assertEqual(result["frozen"], reference)


if __name__ == "__main__":
    unittest.main()
