from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/core.py"
SPEC = importlib.util.spec_from_file_location("factorized_actionrank_core", CORE_PATH)
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

    def test_protocol_freezes_one_new_model_and_three_reused_baselines(self) -> None:
        core.validate_protocol(self.protocol)
        self.assertEqual(
            self.protocol["trained_oof_methods"], ["FactorizedGlobalActionRank"]
        )
        self.assertEqual(len(self.protocol["reused_oof_methods"]), 3)
        self.assertNotIn("roi", self.protocol["actions"])
        self.assertFalse(self.protocol["final_accessed"])

    def test_protocol_rejects_payload_or_roi_action(self) -> None:
        invalid = dict(self.protocol)
        invalid["actions"] = ["edge", "original", "q90", "roi"]
        with self.assertRaises(ValueError):
            core.validate_protocol(invalid)


class FactorizedTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = torch.tensor(
            [[-0.2, 0.3, 0.1], [-0.2, -0.1, 0.4]], dtype=torch.float32
        )
        self.mask = torch.ones_like(self.targets, dtype=torch.bool)
        self.action_type = torch.tensor([[0, 1, 2], [0, 1, 2]])

    def test_targets_separate_global_value_and_relative_action_advantage(self) -> None:
        global_target, advantage_target, cloud = core.factorized_targets(
            self.targets, self.mask, self.action_type
        )
        torch.testing.assert_close(global_target, torch.tensor([0.3, 0.4]))
        torch.testing.assert_close(
            advantage_target,
            torch.tensor([[-0.5, 0.0, -0.2], [-0.6, -0.5, 0.0]]),
        )
        self.assertTrue(torch.equal(cloud, self.action_type != 0))
        reconstructed = global_target[:, None] + advantage_target
        torch.testing.assert_close(reconstructed[cloud], self.targets[cloud])

    def test_advantage_is_invariant_to_a_common_reward_shift(self) -> None:
        first_global, first_advantage, _ = core.factorized_targets(
            self.targets, self.mask, self.action_type
        )
        shifted_global, shifted_advantage, _ = core.factorized_targets(
            self.targets + 7.0, self.mask, self.action_type
        )
        torch.testing.assert_close(shifted_global, first_global + 7.0)
        torch.testing.assert_close(shifted_advantage, first_advantage)


class FactorizedModelTests(unittest.TestCase):
    @staticmethod
    def config() -> dict:
        return {
            "class_embedding_dim": 4,
            "box_hidden_dim": 8,
            "relation_hidden_dim": 16,
            "relation_heads": 4,
            "relation_layers": 1,
            "relation_ff_dim": 32,
            "action_hidden_dim": 8,
            "joint_hidden_dim": 16,
            "dropout": 0.0,
            "factor_global_regression_weight": 0.25,
            "factor_global_rank_weight": 0.25,
            "factor_advantage_regression_weight": 0.25,
            "factor_action_rank_weight": 0.25,
            "minimum_pair_target_gap": 1e-6,
        }

    @staticmethod
    def inputs() -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(11)
        return {
            "boxes": torch.rand((4, 5, 6), generator=generator),
            "classes": torch.randint(0, 8, (4, 5), generator=generator),
            "box_mask": torch.ones((4, 5), dtype=torch.bool),
            "scene": torch.rand((4, 9), generator=generator),
            "actions": torch.rand((4, 3, 18), generator=generator),
            "action_type": torch.tensor([[0, 1, 2]] * 4),
            "action_mask": torch.ones((4, 3), dtype=torch.bool),
        }

    def test_model_factorizes_cloud_scores_and_keeps_edge_fallback_zero(self) -> None:
        torch.manual_seed(13)
        model = core.FactorizedGlobalActionRankRouter(
            category_count=8, scene_dim=9, action_dim=18, config=self.config()
        )
        inputs = self.inputs()
        components = model.forward_components(**inputs)
        scores = model(**inputs)
        torch.testing.assert_close(scores[:, 0], torch.zeros(4))
        torch.testing.assert_close(
            scores[:, 1] - scores[:, 2],
            components["advantage"][:, 1] - components["advantage"][:, 2],
        )
        self.assertTrue(torch.isfinite(scores).all())

    def test_factorized_objective_has_finite_gradients_and_named_components(self) -> None:
        model = core.FactorizedGlobalActionRankRouter(
            category_count=8, scene_dim=9, action_dim=18, config=self.config()
        )
        inputs = self.inputs()
        targets = torch.tensor(
            [
                [0.0, 0.3, 0.1],
                [0.0, -0.1, 0.4],
                [0.0, 0.2, 0.5],
                [0.0, 0.0, -0.2],
            ]
        )
        result = core.factorized_objective(
            model,
            inputs,
            targets,
            inputs["action_mask"],
            inputs["action_type"],
            config=self.config(),
        )
        result["loss"].backward()
        self.assertEqual(
            set(result),
            {
                "loss",
                "global_regression_loss",
                "global_rank_loss",
                "advantage_regression_loss",
                "action_rank_loss",
            },
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


class GateAndMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def rows(self, byte_deltas: list[float], call_deltas: list[float]) -> list[dict]:
        rows = []
        for method in core.METHODS:
            for seed in (20260806, 20260807, 20260808, "ensemble"):
                for mode, deltas in (
                    ("fixed_byte", byte_deltas),
                    ("fixed_call", call_deltas),
                ):
                    for budget, delta in zip((0.1, 0.25, 0.4), deltas):
                        value = 0.5
                        if method == "FactorizedGlobalActionRank":
                            value += delta
                        rows.append(
                            {
                                "method": method,
                                "seed": seed,
                                "mode": mode,
                                "budget": budget,
                                "official_ap50_95": value,
                                "capacity_ok": True,
                                "capacity_utilization": 0.999,
                                "roi_count": 0,
                            }
                        )
        return rows

    def test_result_matrix_requires_all_96_rows(self) -> None:
        rows = self.rows([0.002, 0.0015, 0.0], [0.0, 0.0, 0.0])
        core.validate_result_matrix(rows, self.protocol)
        self.assertEqual(len(rows), 96)
        with self.assertRaises(RuntimeError):
            core.validate_result_matrix(rows[:-1], self.protocol)

    def test_gate_compares_factorized_candidate_to_current_baseline(self) -> None:
        rows = self.rows([0.002, 0.0015, -0.0005], [0.0, 0.0, 0.0])
        gate = core.evaluate_candidate_gate(rows, self.protocol)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["candidate"], "FactorizedGlobalActionRank")
        failed = core.evaluate_candidate_gate(
            self.rows([0.002, 0.0009, -0.0005], [0.0, 0.0, 0.0]),
            self.protocol,
        )
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
