from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from paper_core import (  # noqa: E402
    assert_final_allowed,
    audit_gate,
    batches,
    bootstrap_remap,
    category_index_to_id,
    chunk_signature,
    context_indices,
    deterministic_roles,
    dual_metric_context_reward,
    freeze_comparators,
    original_e2e_ms,
    resolve_result_image_id,
    registered_final_gate,
    select_fixed_call,
    summarize_bootstrap,
    validate_protocol,
    validate_prediction_population,
    validate_reward_isolation,
)


class FakeMetric:
    def evaluate(self, predictions, image_ids):
        cloud = sum(1 for item in predictions if item.get("source") == "cloud")
        return {
            "map50_95": 0.20 + cloud * 0.01,
            "map50": 0.40 + cloud * 0.02,
        }


def record(image_id: int, edge_score: float, cloud_score: float) -> dict:
    return {
        "image_id": image_id,
        "edge_predictions": [
            {
                "image_id": image_id,
                "category_id": 1,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "score": edge_score,
                "source": "edge",
            }
        ],
        "cloud_predictions": [
            {
                "image_id": image_id,
                "category_id": 1,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "score": cloud_score,
                "source": "cloud",
            }
        ],
    }


class PaperCoreTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "configs/protocol.json").read_text())

    def test_frozen_protocol_is_valid(self):
        validate_protocol(self.protocol)

    def test_deterministic_roles_are_exact_disjoint_and_exclude_old_ids(self):
        population = list(range(1, 80))
        excluded = {2, 3, 5, 7, 11}
        counts = {"development": 20, "audit": 10, "registered_final": 15}
        first = deterministic_roles(population, excluded, counts, seed=41)
        second = deterministic_roles(population, excluded, counts, seed=41)
        different = deterministic_roles(population, excluded, counts, seed=42)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual({key: len(value) for key, value in first.items()}, counts)
        observed = [value for values in first.values() for value in values]
        self.assertEqual(len(observed), len(set(observed)))
        self.assertFalse(set(observed) & excluded)

    def test_context_indices_exclude_target_and_other_roles(self):
        development_ids = [101, 103, 107, 109, 113, 127]
        values = context_indices(development_ids, 107, context_size=4, seed=9)
        self.assertEqual(len(values), 4)
        self.assertNotIn(107, values)
        self.assertTrue(set(values) <= set(development_ids))
        self.assertNotIn(999, values)

    def test_dual_metric_reward_uses_same_context_and_scales_delta(self):
        context = [record(1, 0.1, 0.9), record(2, 0.2, 0.8)]
        target = record(3, 0.3, 0.7)
        result = dual_metric_context_reward(FakeMetric(), context, target)

        self.assertEqual(result["context_image_ids"], [1, 2])
        self.assertAlmostEqual(result["reward_ap50_95"], 0.03)
        self.assertAlmostEqual(result["reward_ap50"], 0.06)
        self.assertEqual(result["scale"], 3)

    def test_fixed_call_selects_exact_k_with_stable_ties(self):
        image_ids = [20, 10, 30, 40]
        scores = np.asarray([0.5, 0.5, 0.1, 0.9])
        selected = select_fixed_call(image_ids, scores, budget=0.5)
        self.assertEqual(selected, {10, 40})

    def test_batches_enforce_requested_inference_batch_size(self):
        self.assertEqual(list(batches([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        with self.assertRaises(ValueError):
            list(batches([1], 0))

    def test_audit_freezes_best_external_comparator_per_point(self):
        rows = []
        for mode in ("fixed_call", "fixed_byte"):
            for budget in (0.1, 0.25, 0.4):
                rows.extend(
                    [
                        {"method": "ORIC-AP50", "mode": mode, "budget": budget, "map50_95": 0.5},
                        {"method": "MORIC-AP50", "mode": mode, "budget": budget, "map50_95": 0.51},
                        {"method": "ORIC-AP5095", "mode": mode, "budget": budget, "map50_95": 0.52},
                        {"method": "MORIC-AP5095", "mode": mode, "budget": budget, "map50_95": 0.515},
                        {"method": "BudgetSet", "mode": mode, "budget": budget, "map50_95": 0.53},
                    ]
                )
        frozen = freeze_comparators(rows, external_methods={
            "ORIC-AP50", "MORIC-AP50", "ORIC-AP5095", "MORIC-AP5095"
        })
        self.assertEqual(len(frozen), 6)
        self.assertTrue(all(value == "ORIC-AP5095" for value in frozen.values()))

    def test_audit_gate_uses_only_fixed_call_registered_thresholds(self):
        comparisons = [
            {"budget": 0.1, "ap_gain": 0.002},
            {"budget": 0.25, "ap_gain": 0.0015},
            {"budget": 0.4, "ap_gain": -0.0005},
        ]
        gate = audit_gate(
            comparisons,
            minimum_gain=0.001,
            required_gain_points=2,
            maximum_other_loss=0.001,
        )
        self.assertTrue(gate["passed"])
        comparisons[-1]["ap_gain"] = -0.0011
        self.assertFalse(
            audit_gate(
                comparisons,
                minimum_gain=0.001,
                required_gain_points=2,
                maximum_other_loss=0.001,
            )["passed"]
        )

    def test_registered_final_requires_passed_audit_and_is_one_time(self):
        assert_final_allowed({"complete": True, "passed": True}, final_exists=False)
        with self.assertRaises(RuntimeError):
            assert_final_allowed({"complete": True, "passed": False}, final_exists=False)
        with self.assertRaises(RuntimeError):
            assert_final_allowed({"complete": True, "passed": True}, final_exists=True)

    def test_bootstrap_remap_preserves_pairs_and_makes_duplicate_ids_unique(self):
        gt = {
            "images": [{"id": 10}, {"id": 20}],
            "annotations": [
                {"id": 1, "image_id": 10, "category_id": 1},
                {"id": 2, "image_id": 20, "category_id": 1},
            ],
            "categories": [{"id": 1, "name": "x"}],
        }
        left = [
            {"image_id": 10, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5},
            {"image_id": 20, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.6},
        ]
        right = [
            {"image_id": 10, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.7},
            {"image_id": 20, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.8},
        ]
        remapped_gt, remapped_left, remapped_right = bootstrap_remap(
            gt, left, right, sampled_image_ids=[10, 10, 20]
        )
        ids = [image["id"] for image in remapped_gt["images"]]
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual([row["image_id"] for row in remapped_left], [1, 2, 3])
        self.assertEqual([row["image_id"] for row in remapped_right], [1, 2, 3])
        self.assertEqual(len(remapped_gt["annotations"]), 3)

    def test_category_mapping_requires_exact_coco_names(self):
        categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
        self.assertEqual(category_index_to_id({0: "person", 1: "car"}, categories), {0: 1, 1: 3})
        with self.assertRaises(RuntimeError):
            category_index_to_id({0: "person", 1: "truck"}, categories)

    def test_chunk_signature_binds_role_detector_model_and_paths(self):
        first = chunk_signature(
            protocol_hash="p", role="audit", detector="edge", model_hash="m", image_ids=[1, 2], paths=["a", "b"]
        )
        self.assertEqual(
            first,
            chunk_signature(
                protocol_hash="p", role="audit", detector="edge", model_hash="m", image_ids=[1, 2], paths=["a", "b"]
            ),
        )
        self.assertNotEqual(
            first,
            chunk_signature(
                protocol_hash="p", role="audit", detector="cloud", model_hash="m", image_ids=[1, 2], paths=["a", "b"]
            ),
        )

    def test_prediction_population_rejects_unknown_or_duplicate_image_rows(self):
        validate_prediction_population(
            [{"image_id": 1}, {"image_id": 1}, {"image_id": 2}], expected_ids={1, 2}
        )
        with self.assertRaises(RuntimeError):
            validate_prediction_population([{"image_id": 1}, {"image_id": 3}], expected_ids={1, 2})

    def test_result_path_mapping_supports_real_and_generated_list_names(self):
        real = "/data/000000000010.jpg"
        self.assertEqual(
            resolve_result_image_id(
                real,
                path_to_id={real: 10},
                ordered_image_ids=[10, 20],
                result_index=0,
            ),
            10,
        )
        self.assertEqual(
            resolve_result_image_id(
                "/work/image1.jpg",
                path_to_id={real: 10},
                ordered_image_ids=[10, 20],
                result_index=1,
            ),
            20,
        )
        with self.assertRaises(RuntimeError):
            resolve_result_image_id(
                "/work/image0.jpg",
                path_to_id={real: 10},
                ordered_image_ids=[10, 20],
                result_index=1,
            )

    def test_reward_rows_are_strictly_development_only(self):
        development = {1, 2, 3}
        rows = [
            {"target_image_id": 1, "context_image_ids": [2, 3]},
            {"target_image_id": 2, "context_image_ids": [1, 3]},
        ]
        validate_reward_isolation(rows, development_ids=development)
        rows[0]["context_image_ids"].append(99)
        with self.assertRaises(RuntimeError):
            validate_reward_isolation(rows, development_ids=development)

    def test_original_e2e_increases_with_payload_and_response(self):
        network = self.protocol["network"]
        low = original_e2e_ms(1000, 100, network)
        high = original_e2e_ms(2000, 200, network)
        self.assertGreater(high, low)
        self.assertGreater(low, float(network["edge_ms"]))

    def test_bootstrap_summary_and_registered_final_gate(self):
        summary = summarize_bootstrap(np.asarray([-0.001, 0.001, 0.002, 0.003]), confidence=0.5)
        self.assertGreater(summary["ci_high"], summary["ci_low"])
        intervals = [
            {"budget": 0.1, "ci_low": 0.0002},
            {"budget": 0.25, "ci_low": 0.0001},
            {"budget": 0.4, "ci_low": -0.0005},
        ]
        costs = [
            {"budget": budget, "candidate_kb": 10.1, "comparator_kb": 10.0}
            for budget in (0.1, 0.25, 0.4)
        ]
        capacities = [{"used": 99.0, "capacity": 100.0}]
        result = registered_final_gate(
            intervals,
            costs,
            capacities,
            required_superior_points=2,
            noninferiority_margin=0.001,
            maximum_kb_increase_fraction=0.02,
        )
        self.assertTrue(result["passed"])
        intervals[-1]["ci_low"] = -0.0011
        self.assertFalse(
            registered_final_gate(
                intervals,
                costs,
                capacities,
                required_superior_points=2,
                noninferiority_margin=0.001,
                maximum_kb_increase_fraction=0.02,
            )["passed"]
        )


if __name__ == "__main__":
    unittest.main()
