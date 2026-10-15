import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("final_completion_core", ROOT / "scripts/core.py")
TRAINING = load_module("final_completion_training_test", ROOT / "scripts/training.py")
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class ProtocolTests(unittest.TestCase):
    def test_protocol_contract(self):
        CORE.validate_protocol(PROTOCOL)
        self.assertEqual(PROTOCOL["primary_method"], "CAVS-OQ-Core")
        self.assertFalse(PROTOCOL["final_is_hidden"])
        self.assertFalse(PROTOCOL["final_used_for_training_or_selection"])

    def test_unique_method_semantics(self):
        with self.assertRaises(ValueError):
            CORE.validate_method_name("CAVS-OQ-from-OQR", "CAVS-OQ-Core")
        CORE.validate_method_name("CAVS-OQ-from-OQR", "CAVS-OQ-from-OQR")

    def test_training_matrix(self):
        jobs = CORE.training_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len({job["id"] for job in jobs}), 60)
        self.assertEqual(
            sum(job["dataset"] == "oregon" and job["method"] == "CAVS-OQR-MAUR" for job in jobs),
            18,
        )
        self.assertEqual(
            sum(job["dataset"] == "coco" and job["method"] in {"ORIC-AP5095", "MORIC-AP5095"} for job in jobs),
            6,
        )
        self.assertTrue(all(job["predict_roles"] == ["fit_oof"] for job in jobs if job["fold"] != "full"))
        self.assertTrue(all("final" not in job["train_roles"] for job in jobs))

    def test_bootstrap_matrix(self):
        self.assertEqual(CORE.expected_bootstrap_jobs(PROTOCOL, "coco"), 36)
        self.assertEqual(CORE.expected_bootstrap_jobs(PROTOCOL, "oregon"), 72)

    def test_category_ids_come_from_frozen_gt_not_role_adapter(self):
        gt = {"categories": [{"id": 7}, {"id": 3}, {"id": 11}]}
        self.assertEqual(CORE.category_ids_from_gt(gt, expected_count=3), [7, 3, 11])
        with self.assertRaises(ValueError):
            CORE.category_ids_from_gt(gt, expected_count=2)
        with self.assertRaises(ValueError):
            CORE.category_ids_from_gt({"categories": [{"id": 7}, {"id": 7}]}, expected_count=2)

    def test_action_width_uses_role_maximum_not_first_image(self):
        records = [
            {"image_id": 1, "actions": [{"action_id": "edge"}, {"action_id": "original"}, {"action_id": "q90"}]},
            {"image_id": 2, "actions": [{"action_id": "edge"}, {"action_id": "original"}, {"action_id": "q90"}, {"action_id": "current@640"}]},
        ]
        self.assertEqual(CORE.maximum_action_width(records), 4)

    def test_runtime_worker_override_is_explicit_and_bounded(self):
        self.assertIsNone(CORE.runtime_worker_override(None, cpu_count=64))
        self.assertEqual(CORE.runtime_worker_override("32", cpu_count=64), 32)
        for value in ("0", "33", "not-an-integer"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CORE.runtime_worker_override(value, cpu_count=64)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"image_id": 1, "payload_bytes": {"original": 100, "q90": 30}},
            {"image_id": 2, "payload_bytes": {"original": 100, "q90": 30}},
            {"image_id": 3, "payload_bytes": {"original": 100, "q90": 30}},
        ]
        self.scores = {
            1: {"edge": 0.0, "original": 0.9, "q90": 0.8},
            2: {"edge": 0.0, "original": 0.7, "q90": 0.85},
            3: {"edge": 0.0, "original": 0.6, "q90": 0.5},
        }

    def test_two_stage_picks_one_cloud_action_before_selection(self):
        image_scores, action_map = CORE.two_stage_projection(self.scores)
        self.assertEqual(action_map, {1: "original", 2: "q90", 3: "original"})
        self.assertEqual(image_scores, {1: 0.9, 2: 0.85, 3: 0.6})

    def test_capacity_validation(self):
        report = CORE.validate_selection(
            self.samples, {1: "q90", 2: "q90"}, mode="fixed_byte", budget=0.25
        )
        self.assertEqual(report["request_bytes"], 60)
        self.assertLessEqual(report["request_bytes"], report["capacity_bytes"])
        with self.assertRaises(ValueError):
            CORE.validate_selection(
                self.samples, {1: "original"}, mode="fixed_byte", budget=0.25
            )

    def test_final_never_allowed_as_training_role(self):
        with self.assertRaises(ValueError):
            CORE.validate_training_roles(["fit", "final"])

    def test_cost_in_value_does_not_call_cost_stripping_helper(self):
        class Adapter:
            def tensorize(self, records, rewards, protocol):
                return {"actions": "payload-visible"}

            def _frontier_data(self, *args, **kwargs):
                raise AssertionError("cost-stripping helper must not be called")

        output = TRAINING.cost_in_value_data(Adapter(), [], np.empty((0, 0)), {})
        self.assertEqual(output["actions"], "payload-visible")


class StatisticsTests(unittest.TestCase):
    def test_nested_random_averages_metrics_not_selections(self):
        candidate = np.asarray([0.5, 0.7, 0.6])
        random_values = np.asarray([[0.4, 0.5, 0.6], [0.3, 0.6, 0.4]])
        delta = CORE.nested_random_delta(candidate, random_values)
        np.testing.assert_allclose(delta, [0.15, 0.15, 0.10])

    def test_unique_result_rows(self):
        rows = [
            {"dataset": "coco", "role": "final", "method": "a", "policy": "x", "mode": "fixed_byte", "budget": 0.1, "seed": "ensemble"},
            {"dataset": "coco", "role": "final", "method": "a", "policy": "x", "mode": "fixed_byte", "budget": 0.1, "seed": "ensemble"},
        ]
        with self.assertRaises(ValueError):
            CORE.validate_unique_rows(rows)


if __name__ == "__main__":
    unittest.main()
