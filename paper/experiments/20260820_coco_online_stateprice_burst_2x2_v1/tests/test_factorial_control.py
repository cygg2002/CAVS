from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
REFERENCE = ROOT.parent / "20260818_coco_rknn_fp16_online_primal_dual_v1" / "board_bundle" / "scripts"
sys.path.insert(0, str(REFERENCE))

from online_control import CausalByteController
from factorial_control import (
    BudgetEnvelope,
    EmpiricalDemandCurve,
    ImmediateController,
    build_empirical_demand_curve,
)


ACTIONS = ("edge", "original", "q90")


def record(image_id: int, *, original: int, q90: int, original_score: float, q90_score: float):
    return {
        "image_id": image_id,
        "payload_bytes": {"edge": 0, "original": original, "q90": q90},
        "scores": {"edge": 0.0, "original": original_score, "q90": q90_score},
        "objective_values": {"edge": 0.0, "original": original_score, "q90": q90_score},
    }


class BudgetEnvelopeTest(unittest.TestCase):
    def test_paced_burst_is_monotone_and_ends_at_exact_total_capacity(self) -> None:
        envelope = BudgetEnvelope.paced_burst(
            total_capacity=1000,
            total_frames=4,
            initial_burst_bytes=200,
        )
        limits = [envelope.limit(position=i, cumulative_original_bytes=10 * i) for i in range(1, 5)]
        self.assertEqual(limits, [400, 600, 800, 1000])

    def test_strict_prefix_uses_exact_fraction_of_observed_original_bytes(self) -> None:
        envelope = BudgetEnvelope.strict_prefix(
            budget=Fraction(1, 4),
            total_capacity=1000,
            total_frames=4,
        )
        self.assertEqual(envelope.limit(position=1, cumulative_original_bytes=101), 25)
        self.assertEqual(envelope.limit(position=2, cumulative_original_bytes=199), 49)


class EmpiricalDemandCurveTest(unittest.TestCase):
    def test_price_for_target_never_selects_an_empirical_demand_above_target(self) -> None:
        records = [
            record(1, original=100, q90=40, original_score=1.0, q90_score=0.8),
            record(2, original=100, q90=20, original_score=0.5, q90_score=0.4),
        ]
        curve = build_empirical_demand_curve(records, action_ids=ACTIONS, median_original_bytes=100)
        entry = curve.for_target_normalized_bytes(0.25)
        self.assertLessEqual(entry.mean_normalized_bytes, 0.25 + 1e-12)

    def test_zero_target_selects_an_edge_only_price(self) -> None:
        records = [record(1, original=100, q90=40, original_score=1.0, q90_score=0.8)]
        curve = build_empirical_demand_curve(records, action_ids=ACTIONS, median_original_bytes=100)
        entry = curve.for_target_normalized_bytes(0.0)
        self.assertEqual(entry.mean_normalized_bytes, 0.0)


class ImmediateControllerTest(unittest.TestCase):
    def test_fixed_strict_matches_frozen_reference_decisions(self) -> None:
        frames = [
            record(1, original=100, q90=20, original_score=2.0, q90_score=1.0),
            record(2, original=80, q90=30, original_score=1.0, q90_score=0.9),
            record(3, original=120, q90=25, original_score=0.8, q90_score=1.1),
        ]
        reference = CausalByteController(
            budget="0.25", price=0.3, median_original_bytes=100, action_ids=ACTIONS
        )
        candidate = ImmediateController.fixed_price(
            action_ids=ACTIONS,
            median_original_bytes=100,
            price=0.3,
            envelope=BudgetEnvelope.strict_prefix(
                budget=Fraction(1, 4), total_capacity=75, total_frames=3
            ),
        )
        for frame in frames:
            expected = reference.decide(frame["image_id"], frame["payload_bytes"], frame["scores"])
            actual = candidate.decide(frame)
            self.assertEqual((actual.action, actual.spent_bytes), (expected.action, expected.spent_bytes))

    def test_identical_observed_prefix_is_independent_of_future_suffix(self) -> None:
        calibration = [record(1, original=100, q90=30, original_score=1.0, q90_score=0.8)]
        curve = build_empirical_demand_curve(calibration, action_ids=ACTIONS, median_original_bytes=100)

        def observed_prefix(unseen_score: float):
            controller = ImmediateController.state_aware(
                action_ids=ACTIONS,
                median_original_bytes=100,
                demand_curve=curve,
                envelope=BudgetEnvelope.paced_burst(
                    total_capacity=100, total_frames=3, initial_burst_bytes=20
                ),
            )
            frames = [
                record(10, original=100, q90=30, original_score=1.0, q90_score=0.8),
                record(11, original=100, q90=30, original_score=0.9, q90_score=0.7),
                record(12, original=100, q90=30, original_score=unseen_score, q90_score=unseen_score),
            ]
            return [controller.decide(frame).action for frame in frames[:2]]

        self.assertEqual(observed_prefix(1000.0), observed_prefix(-1000.0))

    def test_burst_controller_never_exceeds_envelope_or_final_capacity(self) -> None:
        frames = [record(i, original=100, q90=30, original_score=2.0, q90_score=1.0) for i in range(1, 6)]
        envelope = BudgetEnvelope.paced_burst(
            total_capacity=150, total_frames=5, initial_burst_bytes=30
        )
        controller = ImmediateController.fixed_price(
            action_ids=ACTIONS,
            median_original_bytes=100,
            price=0.0,
            envelope=envelope,
        )
        for position, frame in enumerate(frames, start=1):
            decision = controller.decide(frame)
            self.assertLessEqual(decision.spent_bytes, envelope.limit(
                position=position,
                cumulative_original_bytes=100 * position,
            ))
        self.assertLessEqual(controller.spent_bytes, 150)


if __name__ == "__main__":
    unittest.main()
