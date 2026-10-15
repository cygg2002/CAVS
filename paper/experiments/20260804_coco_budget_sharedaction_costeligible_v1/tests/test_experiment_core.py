from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))

from experiment_core import (  # noqa: E402
    BudgetSetRouter,
    boundary_pairs,
    budget_set_loss,
    build_box_set,
    deployable_action_features,
    deployable_scene_features,
    deterministic_split,
    fixed_call_indices,
    select_costeligible,
)


def _record(image_id: int = 1) -> dict:
    return {
        "image_id": image_id,
        "width": 200,
        "height": 100,
        "full_payload_bytes": 20_000,
        "edge_boxes": [
            {"xyxy": [10, 10, 30, 30], "conf": 0.8, "cls": 2},
            {"xyxy": [100, 20, 180, 70], "conf": 0.4, "cls": 4},
        ],
        "image_stats": {"luma_mean": 0.5, "luma_std": 0.2, "edge_density": 0.1},
        "frame_anchors": [99.0, 99.0, 99.0],
        "miss_summary": [99.0] * 8,
        "regions": [
            {
                "region_uid": "r1",
                "crop_xyxy": [0, 0, 100, 50],
                "source": "lowconf_proposal",
            }
        ],
    }


def _action(action_id: str, action_type: str, payload: int, value: float) -> dict:
    return {
        "action_id": action_id,
        "action_type": action_type,
        "payload_bytes": payload,
        "response_bytes": 100,
        "cloud_calls": int(action_type != "edge"),
        "cloud_gflops": 20.0 if action_type != "edge" else 0.0,
        "e2e_ms": 30.0 if action_type != "edge" else 18.0,
        "resolutions": [320] if action_type != "edge" else [],
        "region_uids": ["r1"] if action_type == "roi" else [],
        "resource_costs": {
            "upload": payload / 20_000,
            "compute": 0.1 if action_type != "edge" else 0.0,
            "e2e": 0.1 if action_type != "edge" else 0.0,
        },
        "anchors": [value + 100.0, value + 200.0, value + 300.0],
    }


class ExperimentCoreTest(unittest.TestCase):
    def test_deterministic_split_is_disjoint_and_seeded(self) -> None:
        image_ids = list(range(100))
        first = deterministic_split(image_ids, train_count=80, seed=7)
        repeat = deterministic_split(reversed(image_ids), train_count=80, seed=7)
        changed = deterministic_split(image_ids, train_count=80, seed=8)

        self.assertEqual(first, repeat)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first["train_image_ids"]), 80)
        self.assertEqual(len(first["holdout_image_ids"]), 20)
        self.assertFalse(
            set(first["train_image_ids"]) & set(first["holdout_image_ids"])
        )

    def test_box_set_is_permutation_invariant(self) -> None:
        record = _record()
        original = build_box_set(record, max_boxes=4, confidence_floor=0.001)
        shuffled = copy.deepcopy(record)
        shuffled["edge_boxes"].reverse()
        permuted = build_box_set(shuffled, max_boxes=4, confidence_floor=0.001)

        for left, right in zip(original, permuted):
            self.assertTrue(torch.equal(left, right))

    def test_forbidden_teacher_fields_do_not_change_deployable_features(self) -> None:
        record = _record()
        action = _action("roi", "roi", 1000, 0.3)
        scene = deployable_scene_features(record, deploy_confidence=0.001)
        action_features = deployable_action_features(action, record)

        changed = copy.deepcopy(record)
        changed["frame_anchors"] = [-999.0] * 3
        changed["miss_summary"] = [-999.0] * 8
        changed_action = copy.deepcopy(action)
        changed_action["anchors"] = [-999.0] * 3

        self.assertTrue(
            torch.equal(
                scene,
                deployable_scene_features(changed, deploy_confidence=0.001),
            )
        )
        self.assertTrue(
            torch.equal(
                action_features,
                deployable_action_features(changed_action, changed),
            )
        )

    def test_boundary_pairs_cross_selected_boundary(self) -> None:
        selected = np.array([True, False, True, False, False])
        valid = np.array([True, True, True, True, False])
        positive, negative = boundary_pairs(
            selected, valid, pair_count=50, seed=11
        )

        self.assertTrue(selected[positive].all())
        self.assertTrue((~selected[negative]).all())
        self.assertTrue(valid[negative].all())

    def test_fixed_call_selects_exact_images_and_requested_action_space(self) -> None:
        scores = np.array(
            [
                [0.0, 0.9, 0.8],
                [0.0, 0.7, 0.95],
                [0.0, 0.6, 0.5],
                [0.0, 0.4, 0.3],
            ]
        )
        mask = np.ones_like(scores, dtype=bool)
        types = np.array([[0, 1, 2]] * 4)

        full = fixed_call_indices(scores, mask, types, budget=0.5, full_only=True)
        shared = fixed_call_indices(
            scores, mask, types, budget=0.5, full_only=False
        )

        self.assertEqual(full, [(0, 1), (1, 1)])
        self.assertEqual(shared, [(1, 2), (0, 1)])

    def test_costeligible_never_exceeds_capacity(self) -> None:
        records = []
        score_rows = []
        for image_id in range(3):
            record = _record(image_id)
            record["actions"] = [
                _action(f"edge-{image_id}", "edge", 0, 0.0),
                _action(f"full-{image_id}", "full", 100, 1.0),
                _action(f"roi-{image_id}", "roi", 30, 0.8),
            ]
            records.append(record)
            score_rows.append([0.0, 1.0, 0.8])

        result = select_costeligible(
            records,
            np.asarray(score_rows),
            resource_profile="upload",
            capacity=90.0,
        )

        self.assertLessEqual(result["total_cost"], 90.0 + 1e-9)
        self.assertEqual(len(result["selected"]), 3)

    def test_model_and_loss_reward_correct_budget_order(self) -> None:
        model = BudgetSetRouter(
            category_count=5,
            scene_dim=9,
            action_dim=18,
            class_embedding_dim=4,
            box_hidden_dim=8,
            action_hidden_dim=8,
            joint_hidden_dim=16,
            dropout=0.0,
        )
        boxes = torch.zeros(2, 3, 6)
        classes = torch.zeros(2, 3, dtype=torch.long)
        box_mask = torch.tensor([[True, False, False], [True, True, False]])
        scene = torch.zeros(2, 9)
        action = torch.zeros(2, 2, 18)
        action_type = torch.tensor([[0, 1], [0, 2]])
        action_mask = torch.ones(2, 2, dtype=torch.bool)
        output = model(
            boxes=boxes,
            classes=classes,
            box_mask=box_mask,
            scene=scene,
            actions=action,
            action_type=action_type,
            action_mask=action_mask,
            budget=torch.tensor([0.25, 0.25]),
        )
        self.assertEqual(tuple(output.shape), (2, 2))

        target = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
        good = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
        bad = -good
        selected = torch.tensor([[False, True], [False, False]])
        good_loss = budget_set_loss(
            good,
            target,
            action_mask,
            boundary_selected=selected,
            boundary_pairs_count=8,
            seed=3,
        )["loss"]
        bad_loss = budget_set_loss(
            bad,
            target,
            action_mask,
            boundary_selected=selected,
            boundary_pairs_count=8,
            seed=3,
        )["loss"]
        self.assertLess(float(good_loss), float(bad_loss))


if __name__ == "__main__":
    unittest.main()
