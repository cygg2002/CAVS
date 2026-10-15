from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METHODS = load("oric_moric_ac_methods_test", ROOT / "scripts/methods.py")
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class MethodContractTest(unittest.TestCase):
    def test_router_outputs_one_value_per_action(self) -> None:
        model = METHODS.ActionConditionedORICRouter(
            image_feature_dim=205,
            action_dim=18,
            action_type_count=5,
            image_hidden_dims=[16, 16, 16, 16],
            action_hidden_dim=32,
            joint_hidden_dim=64,
            image_dropout=0.1,
            joint_dropout=0.05,
        )
        scores = model(
            image_features=torch.zeros((3, 205)),
            actions=torch.zeros((3, 4, 18)),
            action_type=torch.zeros((3, 4), dtype=torch.long),
            action_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        )
        self.assertEqual(tuple(scores.shape), (3, 4))
        self.assertLess(float(scores[1, 3].detach()), -1e8)

    def test_cost_columns_are_zeroed_before_training(self) -> None:
        actions = torch.arange(2 * 3 * 18, dtype=torch.float32).reshape(2, 3, 18)
        scene = torch.arange(2 * 9, dtype=torch.float32).reshape(2, 9)
        cleaned = METHODS.strip_value_cost_inputs({"actions": actions, "scene": scene})
        np.testing.assert_allclose(cleaned["actions"][:, :, METHODS.ACTION_COST_INDICES], 0.0)
        np.testing.assert_allclose(cleaned["scene"][:, METHODS.SCENE_PAYLOAD_INDEX], 0.0)
        self.assertTrue(torch.equal(actions, torch.arange(2 * 3 * 18, dtype=torch.float32).reshape(2, 3, 18)))

    def test_small_fit_predict_roundtrip(self) -> None:
        rng = np.random.default_rng(4)
        count, action_count = 12, 4
        features = rng.normal(size=(count, 205)).astype(np.float32)
        tensors = {
            "scene": torch.from_numpy(rng.normal(size=(count, 9)).astype(np.float32)),
            "actions": torch.from_numpy(rng.normal(size=(count, action_count, 18)).astype(np.float32)),
            "action_type": torch.tensor([[0, 1, 1, 2]] * count, dtype=torch.long),
            "action_mask": torch.ones((count, action_count), dtype=torch.bool),
            "target": torch.from_numpy(rng.normal(size=(count, action_count)).astype(np.float32)),
        }
        config = json.loads(json.dumps(PROTOCOL))
        config["model"].update({"epochs": 2, "patience": 2, "batch_size": 4})
        artifact = METHODS.fit_action_conditioned(
            "ORIC-AC-OQR", features, tensors, config, seed=7
        )
        scores = METHODS.predict_action_conditioned(artifact, features, tensors, config)
        self.assertEqual(scores.shape, (count, action_count))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
