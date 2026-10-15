from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/run_parallel_nineranker.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_parallel_nineranker", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ParallelNinerankerTest(unittest.TestCase):
    def test_plan_reuses_one_shot_budget_iterable_for_every_ranker(self):
        runner = load_runner()
        selections = {
            f"fixed_call|{policy}|{ranker}|{budget:.2f}": {1: action}
            for ranker in ("A", "B")
            for budget in (0.1, 0.25)
            for policy, action in (
                ("SharedAction", "original"),
                ("GBDT-Hierarchical", "roi"),
            )
        }

        plan, _ = runner.plan_workpoints(
            rankers=("A", "B"),
            budgets=map(float, (0.1, 0.25)),
            selections=selections,
            digest_fn=lambda baseline, candidate: str((tuple(baseline.items()), tuple(candidate.items()))),
        )

        self.assertEqual(
            [(point["method"], point["nominal_budget"]) for point in plan],
            [("A", 0.1), ("A", 0.25), ("B", 0.1), ("B", 0.25)],
        )

    def test_plan_preserves_grid_order_and_deduplicates_action_pairs(self):
        runner = load_runner()
        selections = {
            "fixed_call|SharedAction|A|0.10": {1: "original", 2: "q90"},
            "fixed_call|GBDT-Hierarchical|A|0.10": {1: "roi", 2: "q90"},
            "fixed_call|SharedAction|A|0.25": {1: "original", 2: "q90"},
            "fixed_call|GBDT-Hierarchical|A|0.25": {1: "original", 2: "q90"},
            "fixed_call|SharedAction|B|0.10": {1: "original", 2: "q90"},
            "fixed_call|GBDT-Hierarchical|B|0.10": {1: "roi", 2: "q90"},
            "fixed_call|SharedAction|B|0.25": {1: "q90", 2: "original"},
            "fixed_call|GBDT-Hierarchical|B|0.25": {1: "q90", 2: "roi"},
        }

        plan, jobs = runner.plan_workpoints(
            rankers=("A", "B"),
            budgets=(0.1, 0.25),
            selections=selections,
            digest_fn=lambda baseline, candidate: str((tuple(baseline.items()), tuple(candidate.items()))),
        )

        self.assertEqual(
            [(point["method"], point["nominal_budget"]) for point in plan],
            [("A", 0.1), ("A", 0.25), ("B", 0.1), ("B", 0.25)],
        )
        self.assertEqual([point["kind"] for point in plan], ["compute", "identical", "reuse", "compute"])
        self.assertEqual(len(jobs), 2)

    def test_materialize_rows_matches_serial_zero_and_reuse_semantics(self):
        runner = load_runner()
        plan = [
            {"method": "A", "nominal_budget": 0.1, "digest": "d1", "kind": "compute"},
            {"method": "A", "nominal_budget": 0.25, "digest": "same", "kind": "identical"},
            {"method": "B", "nominal_budget": 0.1, "digest": "d1", "kind": "reuse"},
        ]
        computed = {
            "d1": {
                "replicates": 200,
                "official_map50_95_delta_mean": 0.002,
                "request_kb_per_image_delta_mean": -4.0,
                "bootstrap_unit": "image",
                "paired": True,
                "strategy_refit_in_bootstrap": False,
            }
        }

        rows = runner.materialize_rows(
            plan,
            computed,
            comparison="GBDT-Hierarchical - SharedAction",
            replicates=200,
            seed=20260720,
            unit="image",
        )

        self.assertEqual(rows[0]["official_map50_95_delta_mean"], 0.002)
        self.assertNotIn("reused_identical_action_pair", rows[0])
        self.assertEqual(rows[1]["official_map50_95_delta_mean"], 0.0)
        self.assertEqual(rows[1]["request_kb_per_image_delta_mean"], 0.0)
        self.assertTrue(rows[2]["reused_identical_action_pair"])
        self.assertEqual(
            [(row["method"], row["nominal_budget"]) for row in rows],
            [("A", 0.1), ("A", 0.25), ("B", 0.1)],
        )


if __name__ == "__main__":
    unittest.main()
