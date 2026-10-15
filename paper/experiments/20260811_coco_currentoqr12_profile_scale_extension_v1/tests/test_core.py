from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("coco_oqr12_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class CoreContractTest(unittest.TestCase):
    def test_protocol_contains_exact_four_by_three_roi_matrix(self) -> None:
        CORE.validate_protocol(PROTOCOL)
        self.assertEqual(len(CORE.roi_actions(PROTOCOL)), 12)
        self.assertEqual(len(set(CORE.roi_actions(PROTOCOL))), 12)

    def test_single_roi_is_strict_coco_current640(self) -> None:
        self.assertEqual(
            PROTOCOL["action_spaces"]["OQR"],
            ["edge", "original", "q90", "current@640"],
        )

    def test_cache_matrix_has_thirty_six_jobs(self) -> None:
        jobs = CORE.cache_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len({job["id"] for job in jobs}), 36)

    def test_training_matrix_has_two_methods_and_three_seeds(self) -> None:
        jobs = CORE.training_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len({job["id"] for job in jobs}), 36)

    def test_result_and_bootstrap_contracts(self) -> None:
        self.assertEqual(CORE.expected_point_rows(PROTOCOL), 24)
        self.assertEqual(CORE.bootstrap_jobs(PROTOCOL), 6)

    def test_gpu_gate_requires_three_stable_polls(self) -> None:
        state = CORE.GpuGateState(PROTOCOL["resource_gate"])
        self.assertFalse(state.observe(free_mib=12000, utilization=99))
        self.assertFalse(state.observe(free_mib=12000, utilization=99))
        self.assertTrue(state.observe(free_mib=12000, utilization=99))
        self.assertFalse(state.observe(free_mib=9000, utilization=10))

    def test_reused_fusion_source_receives_its_expected_protocol_shape(self) -> None:
        adapted = CORE.protocol_for_reused_fusion_source(PROTOCOL)
        self.assertEqual(
            adapted["reversible_fusion"]["primary_parameters"],
            PROTOCOL["reversible_fusion"],
        )
        self.assertNotIn("primary_parameters", PROTOCOL["reversible_fusion"])

    def test_static_budget_axis_is_collapsed_before_action_mapping(self) -> None:
        scores = np.arange(8, dtype=np.float64).reshape(2, 4)
        repeated = np.repeat(scores[None, :, :], 3, axis=0)
        np.testing.assert_array_equal(
            CORE.collapse_static_budget_scores(repeated, expected_budgets=3),
            scores,
        )

    def test_nonidentical_budget_scores_are_not_silently_collapsed(self) -> None:
        scores = np.zeros((3, 2, 4), dtype=np.float64)
        scores[1, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "budget-conditioned"):
            CORE.collapse_static_budget_scores(scores, expected_budgets=3)


if __name__ == "__main__":
    unittest.main()
