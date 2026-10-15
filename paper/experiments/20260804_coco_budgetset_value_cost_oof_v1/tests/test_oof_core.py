from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from oof_core import (  # noqa: E402
    ACTION_COST_INDICES,
    DualHeadRouter,
    candidate_gate,
    deterministic_folds,
    dual_head_objective,
    ranking_diagnostics,
    strip_cost_inputs,
    value_objective,
)


class OofCoreTest(unittest.TestCase):
    def test_deterministic_folds_cover_every_image_once(self):
        image_ids = list(range(23))
        first = deterministic_folds(image_ids, fold_count=5, seed=19)
        second = deterministic_folds(reversed(image_ids), fold_count=5, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(sorted(value for fold in first for value in fold), image_ids)
        self.assertLessEqual(max(map(len, first)) - min(map(len, first)), 1)
        self.assertEqual(len(set(value for fold in first for value in fold)), 23)

    def test_strip_cost_inputs_does_not_mutate_and_removes_direct_cost(self):
        data = {
            "scene": torch.arange(18, dtype=torch.float32).reshape(2, 9),
            "actions": torch.arange(72, dtype=torch.float32).reshape(2, 2, 18),
            "other": torch.tensor([1.0, 2.0]),
        }
        original_scene = data["scene"].clone()
        original_actions = data["actions"].clone()
        stripped = strip_cost_inputs(data)

        self.assertTrue(torch.equal(data["scene"], original_scene))
        self.assertTrue(torch.equal(data["actions"], original_actions))
        self.assertTrue(torch.equal(stripped["scene"][:, 1], torch.zeros(2)))
        self.assertTrue(
            torch.equal(
                stripped["actions"][:, :, list(ACTION_COST_INDICES)],
                torch.zeros((2, 2, len(ACTION_COST_INDICES))),
            )
        )
        self.assertTrue(torch.equal(stripped["scene"][:, 0], original_scene[:, 0]))
        self.assertTrue(torch.equal(stripped["actions"][:, :, 0], original_actions[:, :, 0]))

    def test_dual_head_cost_changes_utility_but_not_value(self):
        torch.manual_seed(7)
        model = DualHeadRouter(
            category_count=5,
            scene_dim=9,
            action_dim=18,
            class_embedding_dim=4,
            box_hidden_dim=8,
            action_hidden_dim=8,
            joint_hidden_dim=12,
            dropout=0.0,
        ).eval()
        inputs = {
            "boxes": torch.rand(3, 4, 6),
            "classes": torch.zeros(3, 4, dtype=torch.long),
            "box_mask": torch.ones(3, 4, dtype=torch.bool),
            "scene": torch.rand(3, 9),
            "actions": torch.rand(3, 2, 18),
            "action_type": torch.tensor([[0, 1]] * 3),
            "action_mask": torch.ones(3, 2, dtype=torch.bool),
        }
        low = model(**inputs, log_payload=torch.zeros(3, 2), budget=torch.full((3,), 0.25))
        high = model(**inputs, log_payload=torch.full((3, 2), 5.0), budget=torch.full((3,), 0.25))
        self.assertTrue(torch.equal(low["value"], high["value"]))
        self.assertFalse(torch.allclose(low["utility"], high["utility"]))

    def test_ranking_diagnostics_reports_signal_cost_and_overlap(self):
        output = ranking_diagnostics(
            scores=np.asarray([4.0, 3.0, 2.0, 1.0]),
            rewards=np.asarray([40.0, 30.0, 20.0, 10.0]),
            payload_bytes=np.asarray([100.0, 400.0, 300.0, 200.0]),
            selected_indices=[0, 1],
            teacher_indices=[0, 2],
        )
        self.assertAlmostEqual(output["reward_spearman"], 1.0)
        self.assertAlmostEqual(output["topk_recall"], 0.5)
        self.assertAlmostEqual(output["topk_jaccard"], 1.0 / 3.0)
        self.assertLess(output["payload_spearman"], 0.0)

    def test_candidate_gate_requires_two_wins_and_limits_other_loss(self):
        passed = candidate_gate(
            [0.0012, 0.0011, -0.0008],
            minimum_gain=0.001,
            required_gain_points=2,
            maximum_other_loss=0.001,
        )
        failed = candidate_gate(
            [0.002, 0.0015, -0.0011],
            minimum_gain=0.001,
            required_gain_points=2,
            maximum_other_loss=0.001,
        )
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])

    def test_dual_objective_keeps_value_and_boundary_losses_separate(self):
        values = torch.tensor([[0.0, 0.2], [0.0, -0.1]], requires_grad=True)
        utilities = torch.tensor([[0.0, 0.1], [0.0, 0.3]], requires_grad=True)
        targets = torch.tensor([[0.0, 1.0], [0.0, -0.5]])
        valid = torch.ones((2, 2), dtype=torch.bool)
        selected = torch.tensor([[False, True], [False, False]])

        value = value_objective(values, targets, valid, reward_weight=0.5, within_weight=0.5)
        result = dual_head_objective(
            values,
            utilities,
            targets,
            valid,
            boundary_selected=selected,
            boundary_pairs_count=8,
            seed=3,
            reward_weight=0.5,
            within_weight=0.5,
            boundary_weight=1.0,
        )
        self.assertTrue(torch.equal(value["loss"], result["value_loss"]))
        result["loss"].backward()
        self.assertGreater(float(values.grad.abs().sum()), 0.0)
        self.assertGreater(float(utilities.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
