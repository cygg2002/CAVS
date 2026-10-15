from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from oof_train import (  # noqa: E402
    fit_dual_router,
    fit_value_router,
    predict_dual_value,
    predict_value_router,
)


def synthetic_data(count: int = 10) -> tuple[list[dict], dict[str, torch.Tensor], np.ndarray]:
    torch.manual_seed(4)
    target = torch.zeros((count, 2), dtype=torch.float32)
    target[:, 1] = torch.linspace(-1.0, 1.0, count)
    data = {
        "boxes": torch.rand(count, 4, 6),
        "classes": torch.zeros(count, 4, dtype=torch.long),
        "box_mask": torch.ones(count, 4, dtype=torch.bool),
        "scene": torch.rand(count, 9),
        "actions": torch.rand(count, 2, 18),
        "action_type": torch.tensor([[0, 1]] * count),
        "action_mask": torch.ones(count, 2, dtype=torch.bool),
        "target": target,
    }
    records = []
    for index in range(count):
        payload = 1000 + index * 100
        records.append(
            {
                "image_id": index + 1,
                "actions": [
                    {
                        "action_id": "edge",
                        "action_type": "edge",
                        "payload_bytes": 0,
                        "e2e_ms": 1.0,
                    },
                    {
                        "action_id": "full",
                        "action_type": "full",
                        "payload_bytes": payload,
                        "e2e_ms": 2.0,
                    },
                ],
            }
        )
    return records, data, target.numpy()


def synthetic_protocol() -> dict:
    return {
        "budgets": [0.1, 0.25, 0.4],
        "max_boxes": 4,
        "max_actions": 2,
        "deploy_confidence": 0.001,
        "resource_profile": "upload",
        "model": {
            "class_embedding_dim": 4,
            "box_hidden_dim": 8,
            "action_hidden_dim": 8,
            "joint_hidden_dim": 12,
            "dropout": 0.0,
            "epochs": 2,
            "patience": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "validation_fraction": 0.2,
            "reward_loss_weight": 0.5,
            "within_action_weight": 0.5,
            "boundary_weight": 1.0,
            "pairs_per_budget": 16,
        },
    }


class OofTrainTest(unittest.TestCase):
    def test_value_and_dual_train_and_predict_on_cpu(self):
        records, data, raw = synthetic_data()
        protocol = synthetic_protocol()
        image_ids = [record["image_id"] for record in records]
        value_checkpoint = fit_value_router(
            image_ids,
            data,
            protocol,
            category_count=5,
            seed=11,
            device=torch.device("cpu"),
        )
        dual_checkpoint = fit_dual_router(
            records,
            data,
            raw,
            protocol,
            category_count=5,
            seed=11,
            device=torch.device("cpu"),
        )
        value_scores = predict_value_router(
            value_checkpoint, data, protocol, category_count=5
        )
        dual_scores = predict_dual_value(
            dual_checkpoint, records, data, protocol, category_count=5
        )
        self.assertEqual(value_scores.shape, (3, 10, 2))
        self.assertEqual(dual_scores.shape, (3, 10, 2))
        self.assertTrue(np.isfinite(value_scores[:, :, 1]).all())
        self.assertTrue(np.isfinite(dual_scores[:, :, 1]).all())
        self.assertTrue(np.equal(value_scores[:, :, 0], 0.0).all())
        self.assertTrue(np.equal(dual_scores[:, :, 0], 0.0).all())


if __name__ == "__main__":
    unittest.main()
