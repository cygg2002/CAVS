from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from online_control import CausalByteController


class CausalByteControllerTest(unittest.TestCase):
    def test_prefix_budget_never_exceeds_exact_fraction(self) -> None:
        controller = CausalByteController(
            budget="0.25",
            price=0.0,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        frames = [
            ({"edge": 0, "original": 100, "q90": 20}, {"edge": 0.0, "original": 2.0, "q90": 1.0}),
            ({"edge": 0, "original": 80, "q90": 30}, {"edge": 0.0, "original": 2.0, "q90": 1.0}),
            ({"edge": 0, "original": 120, "q90": 25}, {"edge": 0.0, "original": 2.0, "q90": 1.0}),
        ]
        for index, (payloads, scores) in enumerate(frames):
            decision = controller.decide(index, payloads, scores)
            self.assertLessEqual(
                decision.spent_bytes * controller.budget_denominator,
                decision.cumulative_original_bytes * controller.budget_numerator,
            )

    def test_unaffordable_high_value_action_falls_back_to_feasible_action(self) -> None:
        controller = CausalByteController(
            budget="0.1",
            price=0.0,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        decision = controller.decide(
            1,
            {"edge": 0, "original": 100, "q90": 20},
            {"edge": 0.0, "original": 100.0, "q90": 10.0},
        )
        self.assertEqual(decision.action, "edge")
        decision = controller.decide(
            2,
            {"edge": 0, "original": 100, "q90": 20},
            {"edge": 0.0, "original": 100.0, "q90": 10.0},
        )
        self.assertEqual(decision.action, "q90")

    def test_price_changes_action_without_changing_byte_feasibility(self) -> None:
        controller = CausalByteController(
            budget="0.4",
            price=2.0,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        decision = controller.decide(
            1,
            {"edge": 0, "original": 100, "q90": 30},
            {"edge": 0.0, "original": 1.5, "q90": 1.0},
        )
        self.assertEqual(decision.action, "q90")

    def test_identical_prefix_is_independent_of_future_suffix(self) -> None:
        def run(suffix_score: float):
            controller = CausalByteController(
                budget="0.25",
                price=0.5,
                median_original_bytes=100,
                action_ids=("edge", "original", "q90"),
            )
            rows = []
            for frame, score in enumerate((1.0, 1.2, suffix_score)):
                rows.append(
                    controller.decide(
                        frame,
                        {"edge": 0, "original": 100, "q90": 20},
                        {"edge": 0.0, "original": score, "q90": score / 2},
                    )
                )
            return rows

        left, right = run(1000.0), run(-1000.0)
        self.assertEqual(
            [(row.action, row.spent_bytes) for row in left[:2]],
            [(row.action, row.spent_bytes) for row in right[:2]],
        )

    def test_resume_snapshot_matches_uninterrupted_controller(self) -> None:
        kwargs = dict(
            budget="0.4",
            price=0.25,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        uninterrupted = CausalByteController(**kwargs)
        first = uninterrupted.decide(
            10,
            {"edge": 0, "original": 100, "q90": 30},
            {"edge": 0.0, "original": 1.0, "q90": 0.8},
        )
        resumed = CausalByteController.from_snapshot(uninterrupted.snapshot())
        expected = uninterrupted.decide(
            11,
            {"edge": 0, "original": 120, "q90": 35},
            {"edge": 0.0, "original": 1.1, "q90": 0.7},
        )
        actual = resumed.decide(
            11,
            {"edge": 0, "original": 120, "q90": 35},
            {"edge": 0.0, "original": 1.1, "q90": 0.7},
        )
        self.assertGreaterEqual(first.spent_bytes, 0)
        self.assertEqual(expected, actual)
        self.assertEqual(uninterrupted.snapshot(), resumed.snapshot())

    def test_rejects_missing_edge_or_invalid_payload(self) -> None:
        controller = CausalByteController(
            budget="0.1",
            price=0.0,
            median_original_bytes=100,
            action_ids=("edge", "original"),
        )
        with self.assertRaises(ValueError):
            controller.decide(1, {"original": 100}, {"original": 1.0})
        with self.assertRaises(ValueError):
            controller.decide(
                1,
                {"edge": 0, "original": -1},
                {"edge": 0.0, "original": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
