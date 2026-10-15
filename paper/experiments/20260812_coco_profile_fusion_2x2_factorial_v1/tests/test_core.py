import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("factorial_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class FactorialCoreTest(unittest.TestCase):
    def test_exact_protocol_and_matrix_sizes(self):
        CORE.validate_protocol(PROTOCOL)
        self.assertEqual(CORE.expected_new_training_jobs(PROTOCOL), 54)
        self.assertEqual(CORE.expected_point_rows(PROTOCOL), 72)
        self.assertEqual(CORE.expected_bootstrap_jobs(PROTOCOL), 15)
        self.assertEqual(len(CORE.evaluation_jobs(PROTOCOL)), 12)

    def test_evaluation_jobs_partition_by_role_and_method(self):
        jobs = CORE.evaluation_jobs(PROTOCOL)
        self.assertEqual(len({job["id"] for job in jobs}), 12)
        self.assertEqual(
            [(job["point_role"], job["method"]) for job in jobs[:4]],
            [("fit_oof", method) for method in PROTOCOL["cells"]],
        )
        self.assertEqual(PROTOCOL["cpu"]["evaluation_workers"], 12)
        self.assertEqual(PROTOCOL["cpu"]["bootstrap_workers"], 12)

    def test_only_current_reversible_is_reused(self):
        reused = [
            method
            for method, cell in PROTOCOL["cells"].items()
            if cell["reused_anchor"]
        ]
        self.assertEqual(reused, ["CAVS-Current-Reversible"])
        jobs = CORE.training_jobs(PROTOCOL)
        self.assertNotIn("CAVS-Current-Reversible", {job["method"] for job in jobs})
        self.assertEqual(len({job["id"] for job in jobs}), 54)

    def test_action_space_contains_one_matching_roi(self):
        self.assertEqual(
            CORE.action_space(PROTOCOL, "CAVS-Current-MAUR"),
            ["edge", "original", "q90", "current@640"],
        )
        self.assertEqual(
            CORE.action_space(PROTOCOL, "CAVS-Historical-Reversible"),
            ["edge", "original", "q90", "historical_cost@640"],
        )

    def test_gate_requires_two_improvements_and_one_noninferior(self):
        config = PROTOCOL["teacher_gate"]
        self.assertTrue(CORE.cell_passes_gate([0.0012, 0.0011, -0.0009], config))
        self.assertFalse(CORE.cell_passes_gate([0.0012, 0.0009, -0.0009], config))
        self.assertFalse(CORE.cell_passes_gate([0.0012, 0.0011, -0.0011], config))

    def test_factorial_interaction(self):
        values = {
            "CAVS-Current-Reversible": 0.50,
            "CAVS-Current-MAUR": 0.51,
            "CAVS-Historical-Reversible": 0.52,
            "CAVS-Historical-MAUR": 0.55,
        }
        effects = CORE.factorial_effects(values)
        self.assertAlmostEqual(effects["fusion_at_current"], 0.01)
        self.assertAlmostEqual(effects["fusion_at_historical"], 0.03)
        self.assertAlmostEqual(effects["interaction"], 0.02)

    def test_bootstrap_contrast_population(self):
        contrasts = CORE.bootstrap_contrasts()
        self.assertEqual(len(contrasts), 5)
        self.assertIn("interaction", {item["name"] for item in contrasts})
        self.assertTrue(
            all(abs(sum(item["coefficients"].values())) < 1e-12 for item in contrasts)
        )

    def test_oof_scores_collapse_redundant_budget_axis_before_id_alignment(self):
        per_image = np.arange(8, dtype=np.float64).reshape(2, 4)
        saved = np.broadcast_to(per_image, (3, *per_image.shape)).copy()

        actual = CORE.normalize_oof_scores(
            saved, expected_images=2, expected_budgets=3
        )

        np.testing.assert_array_equal(actual, per_image)

    def test_oof_scores_reject_budget_conditioned_values(self):
        saved = np.zeros((3, 2, 4), dtype=np.float64)
        saved[1, 0, 0] = 1.0

        with self.assertRaisesRegex(ValueError, "budget-conditioned"):
            CORE.normalize_oof_scores(saved, expected_images=2, expected_budgets=3)


if __name__ == "__main__":
    unittest.main()
