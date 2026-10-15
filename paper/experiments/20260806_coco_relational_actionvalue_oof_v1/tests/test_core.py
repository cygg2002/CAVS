from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/core.py"
SPEC = importlib.util.spec_from_file_location("relational_actionvalue_core", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(CORE_PATH)
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_is_canonical_coco_noroi(self) -> None:
        core.validate_protocol(self.protocol)
        self.assertEqual(self.protocol["roles"], {"fit": 9678, "calibration": 571})
        self.assertEqual(self.protocol["actions"], ["edge", "original", "q90"])
        self.assertNotIn("roi", self.protocol["candidate_actions"])
        self.assertFalse(self.protocol["audit_accessed"])
        self.assertFalse(self.protocol["final_accessed"])
        self.assertFalse(self.protocol["registered_final_accessed"])

    def test_protocol_rejects_forbidden_role_or_roi(self) -> None:
        invalid = dict(self.protocol)
        invalid["actions"] = ["edge", "original", "q90", "roi"]
        with self.assertRaises(ValueError):
            core.validate_protocol(invalid)


class FoldContainedRewardTests(unittest.TestCase):
    def test_folds_cover_population_once(self) -> None:
        folds = core.deterministic_folds(range(101), fold_count=5, seed=17)
        flattened = [value for fold in folds for value in fold]
        self.assertEqual(sorted(flattened), list(range(101)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertLessEqual(max(map(len, folds)) - min(map(len, folds)), 1)

    def test_contexts_stay_in_target_fold_and_exclude_target(self) -> None:
        ids = list(range(70))
        folds = core.deterministic_folds(ids, fold_count=5, seed=19)
        contexts = core.fold_contained_contexts(
            ids, folds=folds, context_size=7, seed=23
        )
        membership = core.fold_membership(folds)
        for target, context in contexts.items():
            self.assertNotIn(target, context)
            self.assertEqual(len(context), 7)
            self.assertTrue(
                all(membership[value] == membership[target] for value in context)
            )

    def test_training_rewards_exclude_each_holdout_fold(self) -> None:
        ids = list(range(90))
        folds = core.deterministic_folds(ids, fold_count=5, seed=29)
        contexts = core.fold_contained_contexts(
            ids, folds=folds, context_size=8, seed=31
        )
        core.assert_oof_reward_isolation(contexts, folds)
        broken = dict(contexts)
        target = folds[1][0]
        broken[target] = [folds[0][0], *broken[target][1:]]
        with self.assertRaises(RuntimeError):
            core.assert_oof_reward_isolation(broken, folds)

    def test_inner_split_is_deterministic_and_disjoint(self) -> None:
        train_a, valid_a = core.deterministic_validation_split(
            list(range(50)), seed=41, fraction=0.2
        )
        train_b, valid_b = core.deterministic_validation_split(
            list(range(50)), seed=41, fraction=0.2
        )
        self.assertEqual((train_a, valid_a), (train_b, valid_b))
        self.assertFalse(set(train_a) & set(valid_a))
        self.assertEqual(set(train_a) | set(valid_a), set(range(50)))
        self.assertEqual(len(valid_a), 10)


class InputAndModelTests(unittest.TestCase):
    @staticmethod
    def tensors(batch: int = 3, boxes: int = 6) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(5)
        box_values = torch.rand((batch, boxes, 6), generator=generator)
        classes = torch.randint(0, 8, (batch, boxes), generator=generator)
        mask = torch.ones((batch, boxes), dtype=torch.bool)
        scene = torch.rand((batch, 9), generator=generator)
        actions = torch.rand((batch, 3, 18), generator=generator)
        action_type = torch.tensor([[0, 1, 2]] * batch, dtype=torch.long)
        action_mask = torch.ones((batch, 3), dtype=torch.bool)
        return {
            "boxes": box_values,
            "classes": classes,
            "box_mask": mask,
            "scene": scene,
            "actions": actions,
            "action_type": action_type,
            "action_mask": action_mask,
        }

    def test_cost_inputs_are_removed_without_mutating_source(self) -> None:
        data = self.tensors()
        data["action_type"][:, 2] = 3
        original_scene = data["scene"].clone()
        original_actions = data["actions"].clone()
        stripped = core.strip_cost_inputs(data)
        self.assertTrue(torch.equal(data["scene"], original_scene))
        self.assertTrue(torch.equal(data["actions"], original_actions))
        self.assertTrue(torch.equal(stripped["scene"][:, 1], torch.zeros(3)))
        self.assertTrue(
            torch.equal(
                stripped["actions"][:, :, list(core.ACTION_COST_INDICES)],
                torch.zeros((3, 3, len(core.ACTION_COST_INDICES))),
            )
        )
        self.assertTrue(torch.equal(stripped["action_type"][:, 2], torch.full((3,), 2)))

    def test_no_roi_filter_retains_edge_original_q90_only(self) -> None:
        records = [
            {
                "image_id": 1,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "q90"},
                    {"action_id": "roi"},
                ],
            }
        ]
        filtered = core.filter_no_roi_records(records)
        self.assertEqual(
            [value["action_id"] for value in filtered[0]["actions"]],
            ["edge", "original", "q90"],
        )
        self.assertEqual(len(records[0]["actions"]), 4)
        core.assert_no_roi_records(filtered)

    def test_relational_encoder_is_invariant_to_valid_box_permutation(self) -> None:
        torch.manual_seed(7)
        config = {
            "class_embedding_dim": 8,
            "box_hidden_dim": 16,
            "relation_hidden_dim": 32,
            "relation_heads": 4,
            "relation_layers": 2,
            "relation_ff_dim": 64,
            "action_hidden_dim": 16,
            "joint_hidden_dim": 32,
            "dropout": 0.0,
        }
        model = core.RelationalActionValueRouter(
            category_count=8, scene_dim=9, action_dim=18, config=config
        ).eval()
        data = self.tensors(batch=2, boxes=6)
        first = model(**data)
        order = torch.tensor([4, 1, 5, 0, 3, 2])
        permuted = dict(data)
        permuted["boxes"] = data["boxes"][:, order]
        permuted["classes"] = data["classes"][:, order]
        permuted["box_mask"] = data["box_mask"][:, order]
        second = model(**permuted)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

    def test_every_method_has_finite_forward_and_backward(self) -> None:
        config = {
            "class_embedding_dim": 4,
            "box_hidden_dim": 8,
            "relation_hidden_dim": 16,
            "relation_heads": 4,
            "relation_layers": 1,
            "relation_ff_dim": 32,
            "action_hidden_dim": 8,
            "joint_hidden_dim": 16,
            "dropout": 0.0,
            "reward_loss_weight": 0.5,
            "within_action_weight": 0.5,
            "cross_image_weight": 0.5,
            "minimum_pair_target_gap": 1e-6,
        }
        data = self.tensors(batch=4, boxes=5)
        target = torch.tensor(
            [[0.0, 0.3, 0.1], [0.0, -0.1, 0.4], [0.0, 0.2, 0.5], [0.0, 0.0, -0.2]]
        )
        for method in core.METHODS:
            model = core.build_model(
                method, category_count=8, scene_dim=9, action_dim=18, config=config
            )
            scores = model(**data)
            result = core.action_value_objective(
                scores,
                target,
                data["action_mask"],
                data["action_type"],
                cross_image_weight=(
                    config["cross_image_weight"] if method in core.GLOBAL_METHODS else 0.0
                ),
                reward_weight=config["reward_loss_weight"],
                within_weight=config["within_action_weight"],
                minimum_target_gap=config["minimum_pair_target_gap"],
            )
            result["loss"].backward()
            self.assertTrue(torch.isfinite(scores).all())
            self.assertTrue(torch.isfinite(result["loss"]))


class ObjectiveAndMetricTests(unittest.TestCase):
    def test_cross_image_pairwise_loss_prefers_correct_order(self) -> None:
        targets = torch.tensor(
            [[0.0, 0.1, 0.2], [0.0, 0.8, 0.7], [0.0, -0.2, 0.4]]
        )
        action_mask = torch.ones_like(targets, dtype=torch.bool)
        action_type = torch.tensor([[0, 1, 2]] * 3)
        correct = targets * 4.0
        reversed_scores = -correct
        correct_loss = core.cross_image_pairwise_loss(
            correct, targets, action_mask, action_type, minimum_target_gap=1e-6
        )
        reversed_loss = core.cross_image_pairwise_loss(
            reversed_scores, targets, action_mask, action_type, minimum_target_gap=1e-6
        )
        self.assertLess(float(correct_loss), float(reversed_loss))

    def test_spearman_and_top_overlap(self) -> None:
        self.assertAlmostEqual(core.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(core.spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertAlmostEqual(core.jaccard({1, 2}, {2, 3}), 1 / 3)


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    @staticmethod
    def rows(byte_deltas: list[float], call_deltas: list[float]) -> list[dict]:
        rows = []
        for seed in (20260806, 20260807, 20260808, "ensemble"):
            for mode, deltas in (("fixed_byte", byte_deltas), ("fixed_call", call_deltas)):
                for budget, delta in zip((0.1, 0.25, 0.4), deltas):
                    rows.extend(
                        [
                            {
                                "method": "CurrentActionValue",
                                "seed": seed,
                                "mode": mode,
                                "budget": budget,
                                "official_ap50_95": 0.5,
                                "capacity_ok": True,
                                "capacity_utilization": 0.999,
                                "roi_count": 0,
                            },
                            {
                                "method": "RelActionRank",
                                "seed": seed,
                                "mode": mode,
                                "budget": budget,
                                "official_ap50_95": 0.5 + delta,
                                "capacity_ok": True,
                                "capacity_utilization": 0.999,
                                "roi_count": 0,
                            },
                        ]
                    )
        return rows

    def test_gate_passes_two_supported_byte_gains(self) -> None:
        report = core.evaluate_candidate_gate(
            self.rows([0.002, 0.0015, -0.0005], [0.0, -0.0005, 0.0002]),
            self.protocol,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["gain_points"], 2)

    def test_gate_rejects_fixed_call_regression(self) -> None:
        report = core.evaluate_candidate_gate(
            self.rows([0.002, 0.0015, 0.0], [0.0, -0.002, 0.0]),
            self.protocol,
        )
        self.assertFalse(report["passed"])

    def test_calibration_runs_only_after_oof_pass(self) -> None:
        self.assertTrue(
            core.should_run_calibration({"passed": True}, self.protocol)
        )
        self.assertFalse(
            core.should_run_calibration({"passed": False}, self.protocol)
        )

    def test_result_matrix_requires_all_methods_seeds_modes_and_budgets(self) -> None:
        rows = []
        for method in core.METHODS:
            for seed in (*self.protocol["seeds"], "ensemble"):
                for mode in self.protocol["modes"]:
                    for budget in self.protocol["budgets"]:
                        rows.append(
                            {
                                "method": method,
                                "seed": seed,
                                "mode": mode,
                                "budget": budget,
                                "capacity_ok": True,
                                "capacity_utilization": 0.999,
                                "roi_count": 0,
                            }
                        )
        core.validate_result_matrix(rows, self.protocol)
        with self.assertRaises(RuntimeError):
            core.validate_result_matrix(rows[:-1], self.protocol)


if __name__ == "__main__":
    unittest.main()
