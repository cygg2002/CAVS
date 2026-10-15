from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from online_control import CausalByteController, PrimalDualByteController
from online_stream import execute_stream


class PrimalDualByteControllerTest(unittest.TestCase):
    def _frame(self, controller, image_id, *, original=100, q90=30, score=1.0):
        return controller.decide(
            image_id,
            {"edge": 0, "original": original, "q90": q90},
            {"edge": 0.0, "original": score + 0.1, "q90": score},
        )

    def test_price_decreases_after_underspend_and_increases_after_overspend(self) -> None:
        controller = PrimalDualByteController(
            budget="0.4",
            initial_price=0.5,
            step_size=0.2,
            minimum_price=0.0,
            maximum_price=2.0,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        underspend = self._frame(controller, 1, q90=30, score=10.0)
        self.assertEqual(underspend.action, "q90")
        self.assertAlmostEqual(underspend.price_before, 0.5)
        self.assertAlmostEqual(underspend.price_after, 0.48)

        # Bank prefix credit without looking ahead, then spend more than this
        # frame's 40-byte arrival while remaining prefix-feasible.
        for image_id in range(2, 5):
            controller.decide(
                image_id,
                {"edge": 0, "original": 100, "q90": 30},
                {"edge": 0.0, "original": -1.0, "q90": -1.0},
            )
        overspend = controller.decide(
            5,
            {"edge": 0, "original": 100, "q90": 80},
            {"edge": 0.0, "original": -1.0, "q90": 10.0},
        )
        self.assertEqual(overspend.action, "q90")
        self.assertGreater(overspend.price_after, overspend.price_before)

    def test_price_projection_respects_configured_bounds(self) -> None:
        lower = PrimalDualByteController(
            budget="0.4", initial_price=0.1, step_size=10.0,
            minimum_price=0.05, maximum_price=0.2,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        decision = lower.decide(
            1,
            {"edge": 0, "original": 100, "q90": 30},
            {"edge": 0.0, "original": -1.0, "q90": -1.0},
        )
        self.assertEqual(decision.price_after, 0.05)

        upper = PrimalDualByteController(
            budget="0.4", initial_price=0.1, step_size=10.0,
            minimum_price=0.05, maximum_price=0.2,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        for image_id in range(1, 5):
            upper.decide(
                image_id,
                {"edge": 0, "original": 100, "q90": 30},
                {"edge": 0.0, "original": -1.0, "q90": -1.0},
            )
        decision = upper.decide(
            5,
            {"edge": 0, "original": 100, "q90": 100},
            {"edge": 0.0, "original": 10.0, "q90": 10.0},
        )
        self.assertEqual(decision.price_after, 0.2)

    def test_zero_step_size_matches_fixed_price_actions_and_byte_state(self) -> None:
        fixed = CausalByteController(
            budget="0.25", price=0.3, median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        adaptive = PrimalDualByteController(
            budget="0.25", initial_price=0.3, step_size=0.0,
            minimum_price=0.0, maximum_price=2.0,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        frames = [
            ({"edge": 0, "original": 100, "q90": 20}, {"edge": 0.0, "original": 1.0, "q90": 0.8}),
            ({"edge": 0, "original": 80, "q90": 25}, {"edge": 0.0, "original": 1.2, "q90": 0.7}),
            ({"edge": 0, "original": 120, "q90": 30}, {"edge": 0.0, "original": 0.9, "q90": 1.1}),
        ]
        for image_id, (payloads, scores) in enumerate(frames, start=1):
            fixed_decision = fixed.decide(image_id, payloads, scores)
            adaptive_decision = adaptive.decide(image_id, payloads, scores)
            self.assertEqual(fixed_decision.action, adaptive_decision.action)
            self.assertEqual(fixed_decision.payload_bytes, adaptive_decision.payload_bytes)
            self.assertEqual(fixed_decision.spent_bytes, adaptive_decision.spent_bytes)
            self.assertEqual(
                fixed_decision.cumulative_original_bytes,
                adaptive_decision.cumulative_original_bytes,
            )
        self.assertEqual(adaptive.current_price, 0.3)

    def test_prefix_budget_and_resume_are_exact(self) -> None:
        kwargs = dict(
            budget="0.25", initial_price=0.3, step_size=0.05,
            minimum_price=0.0, maximum_price=2.0,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        uninterrupted = PrimalDualByteController(**kwargs)
        for image_id in range(1, 4):
            decision = self._frame(uninterrupted, image_id, q90=20, score=1.0)
            self.assertLessEqual(
                decision.spent_bytes * uninterrupted.budget_denominator,
                decision.cumulative_original_bytes * uninterrupted.budget_numerator,
            )
        resumed = PrimalDualByteController.from_snapshot(uninterrupted.snapshot())
        expected = self._frame(uninterrupted, 4, q90=25, score=1.2)
        actual = self._frame(resumed, 4, q90=25, score=1.2)
        self.assertEqual(expected, actual)
        self.assertEqual(uninterrupted.snapshot(), resumed.snapshot())

    def test_current_upload_completes_before_the_next_frame_is_observed(self) -> None:
        controller = PrimalDualByteController(
            budget="1.0", initial_price=0.0, step_size=0.1,
            minimum_price=0.0, maximum_price=2.0,
            median_original_bytes=100,
            action_ids=("edge", "original"),
        )
        events = []

        def observe(item):
            events.append(("open", item["image_id"]))
            return {
                "frame_id": item["image_id"],
                "payload_bytes": {"edge": 0, "original": 100},
                "scores": {"edge": 0.0, "original": 1.0},
            }

        def execute(item, observation, decision):
            events.append(("upload", item["image_id"]))
            events.append(("response", item["image_id"]))
            return {"resolved_boxes": []}

        execute_stream(
            [{"image_id": 1}, {"image_id": 2}],
            controller=controller,
            observe=observe,
            execute=execute,
            commit=lambda row: events.append(("commit", row["image_id"])),
        )
        self.assertEqual(
            events,
            [
                ("open", 1), ("upload", 1), ("response", 1), ("commit", 1),
                ("open", 2), ("upload", 2), ("response", 2), ("commit", 2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
