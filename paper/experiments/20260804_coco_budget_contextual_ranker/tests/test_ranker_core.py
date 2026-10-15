from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ranker_core import (  # noqa: E402
    budget_boundary_pairs,
    calibration_gate,
    fit_budget_ranker,
    pairwise_logistic_loss,
    predict_budget_scores,
)


class RankerCoreTest(unittest.TestCase):
    def test_budget_pairs_cross_the_exact_top_k_boundary(self) -> None:
        rewards = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        positive, negative = budget_boundary_pairs(
            rewards, budget=0.4, pair_count=30, seed=7
        )
        self.assertEqual(len(positive), 30)
        self.assertTrue(set(positive).issubset({3, 4}))
        self.assertTrue(set(negative).issubset({0, 1, 2}))
        second = budget_boundary_pairs(rewards, budget=0.4, pair_count=30, seed=7)
        np.testing.assert_array_equal(positive, second[0])
        np.testing.assert_array_equal(negative, second[1])

    def test_pairwise_loss_rewards_positive_scores_above_negative_scores(self) -> None:
        good = pairwise_logistic_loss(
            torch.tensor([2.0, 3.0]), torch.tensor([-1.0, 0.0])
        )
        bad = pairwise_logistic_loss(
            torch.tensor([-1.0, 0.0]), torch.tensor([2.0, 3.0])
        )
        self.assertLess(float(good), float(bad))

    def test_calibration_gate_requires_mean_gain_and_per_budget_guard(self) -> None:
        baseline = {0.1: 0.40, 0.25: 0.45, 0.4: 0.48}
        passed = calibration_gate(
            {0.1: 0.403, 0.25: 0.452, 0.4: 0.4815},
            baseline,
            minimum_mean_ap_gain=0.002,
            maximum_budget_ap_loss=0.001,
        )
        self.assertTrue(passed["passed"])
        failed = calibration_gate(
            {0.1: 0.406, 0.25: 0.455, 0.4: 0.478},
            baseline,
            minimum_mean_ap_gain=0.002,
            maximum_budget_ap_loss=0.001,
        )
        self.assertFalse(failed["passed"])
        self.assertIn("0.4", failed["failed_budgets"])

    def test_small_ranker_learns_budget_boundary_signal(self) -> None:
        rng = np.random.default_rng(13)
        signal = np.linspace(-2.0, 2.0, 160)
        features = np.column_stack(
            [signal, rng.normal(scale=0.2, size=len(signal))]
        )
        config = {
            "hidden_dims": [8, 8],
            "dropout": 0.0,
            "epochs": 12,
            "batch_size": 64,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "milestones": [8, 10],
            "gamma": 0.5,
            "fit_validation_fraction": 0.2,
            "pairs_per_budget": 320,
            "pointwise_weight": 0.2,
        }
        checkpoint = fit_budget_ranker(
            features, signal, [0.1, 0.25, 0.4], config, seed=19
        )
        scores = predict_budget_scores(checkpoint, features, [0.1, 0.25, 0.4])
        self.assertEqual(scores.shape, (3, len(signal)))
        for row in scores:
            selected = np.argsort(-row, kind="mergesort")[:16]
            self.assertGreater(float(signal[selected].mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
