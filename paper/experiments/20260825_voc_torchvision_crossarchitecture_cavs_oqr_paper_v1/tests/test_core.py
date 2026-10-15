from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load("voc_torchvision_oqr_core", "scripts/core.py")
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class ProtocolTest(unittest.TestCase):
    def test_frozen_protocol_is_valid(self) -> None:
        CORE.validate_protocol(PROTOCOL)

    def test_training_jobs_are_complete_and_unique(self) -> None:
        jobs = CORE.training_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 66)
        self.assertEqual(len({job["id"] for job in jobs}), 66)
        self.assertEqual(sum(job["method"].startswith("CAVS") for job in jobs), 54)
        self.assertFalse(any("final" in job["train_roles"] for job in jobs))

    def test_point_matrix_has_no_semantic_aliases(self) -> None:
        keys = CORE.expected_point_keys(PROTOCOL)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(any(key[1] == "CAVS-O-Core" for key in keys))
        self.assertTrue(any(key[1] == "CAVS-OQ-Core" for key in keys))
        self.assertTrue(any(key[1] == "CAVS-OQR-Core" for key in keys))

    def test_fixed_call_uses_paper_rounding(self) -> None:
        self.assertEqual(CORE.fixed_call_count(4952, 0.10), 495)
        self.assertEqual(CORE.fixed_call_count(4952, 0.25), 1238)
        self.assertEqual(CORE.fixed_call_count(4952, 0.40), 1981)

    def test_nested_random_averages_metrics_not_selections(self) -> None:
        candidate = [0.7, 0.8]
        random = [[0.4, 0.6], [0.6, 0.4]]
        values = CORE.nested_random_delta(candidate, random)
        self.assertAlmostEqual(float(values[0]), 0.2)
        self.assertAlmostEqual(float(values[1]), 0.3)

    def test_capacity_selection_never_exceeds_budget(self) -> None:
        selected = CORE.select_actions_for_capacity(
            [
                [
                    {"image_id": 1, "action_id": "edge", "value": 0.0, "cost": 0.0},
                    {"image_id": 1, "action_id": "original", "value": 2.0, "cost": 5.0},
                ],
                [
                    {"image_id": 2, "action_id": "edge", "value": 0.0, "cost": 0.0},
                    {"image_id": 2, "action_id": "original", "value": 1.0, "cost": 4.0},
                ],
            ],
            5.0,
        )
        self.assertLessEqual(sum(float(row["cost"]) for row in selected), 5.0)
        self.assertEqual(len({int(row["image_id"]) for row in selected}), len(selected))

    def test_final_is_forbidden_for_reward_and_training(self) -> None:
        self.assertEqual(CORE.reward_roles("all"), ("fit", "calibration"))
        with self.assertRaisesRegex(ValueError, "final"):
            CORE.reward_roles("final")
        with self.assertRaisesRegex(ValueError, "final"):
            CORE.validate_training_roles(["fit", "final"])


if __name__ == "__main__":
    unittest.main()
