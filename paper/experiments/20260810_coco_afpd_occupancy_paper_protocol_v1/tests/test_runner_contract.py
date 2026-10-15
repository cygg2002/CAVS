from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts/run_experiment.py"
    spec = importlib.util.spec_from_file_location("afpd_paper_runner_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunnerContractTests(unittest.TestCase):
    def test_action_method_specs_form_the_exact_two_by_two_matrix(self) -> None:
        runner = load_runner()
        self.assertEqual(
            runner.ACTION_METHOD_SPECS,
            {
                "Current-OQ": ("OQ", "CurrentActionValue"),
                "Current-OQR": ("OQR", "CurrentActionValue"),
                "AFPD-OQ": ("OQ", "BoundaryPathDistill"),
                "AFPD-OQR": ("OQR", "BoundaryPathDistill"),
            },
        )

    def test_occupancy_specs_are_secondary_and_frozen(self) -> None:
        runner = load_runner()
        self.assertEqual(
            runner.OCCUPANCY_SPECS,
            {
                "OccupancyFree": "CostFree-FrontierGate",
                "OccupancyAware": "CostAware-FrontierGate",
            },
        )
        self.assertEqual(runner.PROTOCOL["primary_method"], "AFPD-OQR")
        self.assertFalse(runner.PROTOCOL["occupancy_can_replace_primary"])

    def test_ensemble_requires_all_frozen_seeds(self) -> None:
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.ensemble_arrays({20260809: [1.0], 20260810: [2.0]})


if __name__ == "__main__":
    unittest.main()
