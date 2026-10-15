from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_core import (  # noqa: E402
    choose_final_oracle,
    choose_no_cost_candidate,
    index_unique_rows,
    select_shared_pair,
)
from submission_extensions_core import (  # noqa: E402
    calibrated_online_threshold,
    random_byte_actions,
    raw_checkpoint_was_written,
    select_unique_workpoint,
    select_online_actions,
    summarize_bootstrap_prefixes,
    summarize_random_band,
    validated_parallel_workers,
)


RANKERS = ("Confidence", "M2E")
POLICIES = ("SharedAction", "GBDT-Hierarchical")


def row(dataset: str, ranker: str, policy: str, ap: float, kb: float) -> dict:
    return {
        "dataset": dataset,
        "mode": "fixed_call",
        "budget": 0.1,
        "ranker": ranker,
        "policy": policy,
        "ap": ap,
        "kb": kb,
    }


class SharedPolicySelectionTests(unittest.TestCase):
    def test_fixed_call_uses_mean_normalized_cost_across_datasets(self):
        rows = {
            "coco": [
                row("coco", "Confidence", "SharedAction", 0.8000, 10.0),
                row("coco", "M2E", "SharedAction", 0.7995, 9.0),
                row("coco", "M2E", "GBDT-Hierarchical", 0.7995, 8.0),
            ],
            "ena24": [
                row("ena24", "Confidence", "SharedAction", 0.8000, 100.0),
                row("ena24", "M2E", "SharedAction", 0.7995, 99.0),
                row("ena24", "M2E", "GBDT-Hierarchical", 0.7995, 95.0),
            ],
            "oregon": [
                row("oregon", "Confidence", "SharedAction", 0.8000, 1000.0),
                row("oregon", "M2E", "SharedAction", 0.7995, 900.0),
                row("oregon", "M2E", "GBDT-Hierarchical", 0.7995, 990.0),
            ],
        }

        result = select_shared_pair(
            rows,
            mode="fixed_call",
            ap_tolerance=0.001,
            ranker_order=RANKERS,
            policy_order=POLICIES,
        )

        self.assertTrue(result["strict_feasible"])
        self.assertEqual(result["strict_feasible_pairs"], 3)
        self.assertEqual(result["selected_ranker"], "M2E")
        self.assertEqual(result["selected_policy"], "GBDT-Hierarchical")

    def test_fixed_byte_prioritizes_mean_ap_delta_before_cost(self):
        rows = {
            dataset: [
                row(dataset, "Confidence", "SharedAction", confidence_ap, reference_kb),
                row(dataset, "M2E", "SharedAction", ap, reference_kb),
                row(dataset, "M2E", "GBDT-Hierarchical", 0.7995, reference_kb * 0.5),
            ]
            for dataset, reference_kb, confidence_ap, ap in (
                ("coco", 10.0, 0.7980, 0.8001),
                ("ena24", 100.0, 0.8000, 0.7990),
                ("oregon", 1000.0, 0.7980, 0.8001),
            )
        }
        for dataset_rows in rows.values():
            for candidate in dataset_rows:
                candidate["mode"] = "fixed_byte"

        result = select_shared_pair(
            rows,
            mode="fixed_byte",
            ap_tolerance=0.001,
            ranker_order=RANKERS,
            policy_order=POLICIES,
        )

        self.assertTrue(result["strict_feasible"])
        self.assertEqual(result["selected_ranker"], "M2E")
        self.assertEqual(result["selected_policy"], "SharedAction")

    def test_infeasible_selection_does_not_relax_epsilon(self):
        rows = {
            "coco": [
                row("coco", "Confidence", "SharedAction", 0.800, 10.0),
                row("coco", "M2E", "SharedAction", 0.790, 9.0),
            ],
            "ena24": [
                row("ena24", "Confidence", "SharedAction", 0.790, 100.0),
                row("ena24", "M2E", "SharedAction", 0.800, 90.0),
            ],
            "oregon": [
                row("oregon", "Confidence", "SharedAction", 0.800, 1000.0),
                row("oregon", "M2E", "SharedAction", 0.790, 900.0),
            ],
        }

        result = select_shared_pair(
            rows,
            mode="fixed_call",
            ap_tolerance=0.001,
            ranker_order=RANKERS,
            policy_order=POLICIES,
        )

        self.assertFalse(result["strict_feasible"])
        self.assertEqual(result["strict_feasible_pairs"], 0)
        self.assertEqual(result["selection_status"], "least_violating_diagnostic")
        self.assertEqual(result["selected_ranker"], "Confidence")
        self.assertEqual(result["violating_datasets"], 1)


class LesionSelectionTests(unittest.TestCase):
    def test_no_cost_candidate_chooses_ap_then_frozen_tie_break(self):
        rows = [
            row("coco", "Confidence", "SharedAction", 0.80, 10.0),
            row("coco", "M2E", "SharedAction", 0.81, 100.0),
            row("coco", "M2E", "GBDT-Hierarchical", 0.81, 200.0),
        ]

        selected = choose_no_cost_candidate(
            rows,
            ranker_order=RANKERS,
            policy_order=POLICIES,
        )

        self.assertEqual(selected["ranker"], "M2E")
        self.assertEqual(selected["policy"], "SharedAction")
        self.assertEqual(selected["kb"], 100.0)

    def test_unique_index_rejects_duplicate_final_candidates(self):
        rows = [
            row("coco", "Confidence", "SharedAction", 0.80, 10.0),
            row("coco", "Confidence", "SharedAction", 0.81, 11.0),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate row key"):
            index_unique_rows(
                rows,
                key_fields=("dataset", "mode", "budget", "ranker", "policy"),
            )

    def test_final_oracle_uses_ap_then_lower_bytes_and_frozen_order(self):
        rows = [
            row("coco", "Confidence", "SharedAction", 0.81, 12.0),
            row("coco", "M2E", "GBDT-Hierarchical", 0.81, 10.0),
            row("coco", "M2E", "SharedAction", 0.80, 8.0),
        ]

        selected = choose_final_oracle(
            rows,
            ranker_order=RANKERS,
            policy_order=POLICIES,
        )

        self.assertEqual(selected["ranker"], "M2E")
        self.assertEqual(selected["policy"], "GBDT-Hierarchical")


class BootstrapSensitivityTests(unittest.TestCase):
    def test_parallel_worker_count_must_be_positive(self):
        self.assertEqual(validated_parallel_workers(1), 1)
        self.assertEqual(validated_parallel_workers(2), 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            validated_parallel_workers(0)

    def test_identical_action_shortcut_does_not_require_raw_checkpoint(self):
        self.assertFalse(
            raw_checkpoint_was_written(
                checkpoint_exists=False, identical_action_shortcut=True
            )
        )
        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            raw_checkpoint_was_written(
                checkpoint_exists=False, identical_action_shortcut=False
            )

    def test_unique_workpoint_selects_registered_mode_and_budget(self):
        comparisons = [
            {"mode": "fixed_call", "budget": 0.1},
            {"mode": "fixed_call", "budget": 0.25},
        ]

        selected = select_unique_workpoint(
            comparisons, mode="fixed_call", budget=0.25
        )

        self.assertEqual(selected, comparisons[1])

    def test_unique_workpoint_rejects_missing_or_duplicate_match(self):
        comparisons = [
            {"mode": "online", "budget": 0.1},
            {"mode": "online", "budget": 0.1},
        ]

        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_unique_workpoint(comparisons, mode="online", budget=0.1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_unique_workpoint(comparisons, mode="fixed_byte", budget=0.1)

    def test_prefix_summaries_use_only_registered_prefixes(self):
        rows = summarize_bootstrap_prefixes(
            ap_deltas=[-0.002, -0.001, 0.001, 0.004],
            kb_deltas=[-4.0, -2.0, 0.0, 2.0],
            checkpoints=(2, 4),
            mode="fixed_call",
            ap_tolerance=0.001,
        )

        self.assertEqual([row["replicates"] for row in rows], [2, 4])
        self.assertAlmostEqual(rows[0]["official_map50_95_delta_mean"], -0.0015)
        self.assertFalse(rows[0]["gate_passed"])
        self.assertTrue(rows[1]["request_cost_gate_applied"])

    def test_prefix_summary_rejects_unavailable_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "exceeds available replicates"):
            summarize_bootstrap_prefixes(
                ap_deltas=[0.0],
                kb_deltas=[0.0],
                checkpoints=(2,),
                mode="fixed_byte",
                ap_tolerance=0.001,
            )


class RandomBudgetModeTests(unittest.TestCase):
    def test_online_threshold_targets_calibration_count_without_final_access(self):
        threshold = calibrated_online_threshold([0.9, 0.7, 0.4, 0.1], 0.5)
        self.assertAlmostEqual(threshold, 0.55)

        samples = [{"image_id": 1}, {"image_id": 2}, {"image_id": 3}]
        selected = select_online_actions(
            samples,
            scores=[0.8, 0.55, 0.2],
            action_map={1: "q90", 2: "original", 3: "roi"},
            threshold=threshold,
        )
        self.assertEqual(selected, {1: "q90", 2: "original"})

    def test_random_byte_actions_respect_original_byte_capacity(self):
        samples = [
            {"image_id": 1, "payload_bytes": {"original": 100, "q90": 40}},
            {"image_id": 2, "payload_bytes": {"original": 100, "q90": 60}},
            {"image_id": 3, "payload_bytes": {"original": 100, "q90": 50}},
        ]
        selected = random_byte_actions(
            samples,
            scores=[0.8, 0.9, 0.7],
            action_map={1: "q90", 2: "q90", 3: "q90"},
            budget=0.5,
        )

        self.assertEqual(selected, {2: "q90", 1: "q90", 3: "q90"})
        self.assertEqual(sum(samples[image_id - 1]["payload_bytes"][action] for image_id, action in selected.items()), 150)

    def test_random_band_reports_seed_quantiles(self):
        row = summarize_random_band(
            [
                {"official_map50_95": 0.4, "request_kb_per_image": 10.0},
                {"official_map50_95": 0.6, "request_kb_per_image": 20.0},
            ],
            metrics=("official_map50_95", "request_kb_per_image"),
        )

        self.assertEqual(row["seeds"], 2)
        self.assertAlmostEqual(row["official_map50_95_mean"], 0.5)
        self.assertLess(row["official_map50_95_ci_low"], 0.41)
        self.assertGreater(row["official_map50_95_ci_high"], 0.59)


if __name__ == "__main__":
    unittest.main()
