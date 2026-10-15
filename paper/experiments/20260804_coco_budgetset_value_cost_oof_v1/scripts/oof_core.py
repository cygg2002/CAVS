"""Pure contracts for the development-only value/cost OOF diagnostic."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[4]
DIAGNOSTIC_SCRIPTS = (
    REPO
    / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts"
)
if str(DIAGNOSTIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTIC_SCRIPTS))

from experiment_core import ACTION_TYPES, BudgetSetRouter, boundary_pairs  # noqa: E402


SCENE_PAYLOAD_INDEX = 1
ACTION_COST_INDICES = (6, 7, 8, 9, 10, 11, 12)


def deterministic_folds(
    image_ids: Iterable[int], *, fold_count: int, seed: int
) -> list[list[int]]:
    """Return balanced deterministic folds with exact population coverage."""
    values = sorted(map(int, image_ids))
    if len(values) != len(set(values)):
        raise ValueError("Image IDs must be unique")
    if not 2 <= int(fold_count) <= len(values):
        raise ValueError("Fold count must create at least two non-empty folds")

    ordered = sorted(
        values,
        key=lambda image_id: (
            hashlib.sha256(f"{int(seed)}:{image_id}".encode("ascii")).hexdigest(),
            image_id,
        ),
    )
    folds = [[] for _ in range(int(fold_count))]
    for index, image_id in enumerate(ordered):
        folds[index % int(fold_count)].append(image_id)
    return [sorted(fold) for fold in folds]


def strip_cost_inputs(
    data: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove direct payload/resource inputs without mutating the source tensors."""
    output = dict(data)
    scene = data["scene"].clone()
    actions = data["actions"].clone()
    scene[:, SCENE_PAYLOAD_INDEX] = 0.0
    actions[:, :, list(ACTION_COST_INDICES)] = 0.0
    output["scene"] = scene
    output["actions"] = actions
    return output


class DualHeadRouter(BudgetSetRouter):
    """Cost-free value prediction plus a separate cost-aware utility head."""

    def __init__(
        self,
        *,
        category_count: int,
        scene_dim: int,
        action_dim: int,
        class_embedding_dim: int,
        box_hidden_dim: int,
        action_hidden_dim: int,
        joint_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(
            category_count=category_count,
            scene_dim=scene_dim,
            action_dim=action_dim,
            class_embedding_dim=class_embedding_dim,
            box_hidden_dim=box_hidden_dim,
            action_hidden_dim=action_hidden_dim,
            joint_hidden_dim=joint_hidden_dim,
            dropout=dropout,
        )
        self.utility_head = nn.Sequential(
            nn.Linear(joint_hidden_dim + action_hidden_dim + 2, joint_hidden_dim),
            nn.LayerNorm(joint_hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(joint_hidden_dim, 1),
        )

    def forward(
        self,
        *,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        box_mask: torch.Tensor,
        scene: torch.Tensor,
        actions: torch.Tensor,
        action_type: torch.Tensor,
        action_mask: torch.Tensor,
        log_payload: torch.Tensor,
        budget: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded = self.class_embedding(classes.clamp(0, self.category_count - 1))
        encoded = self.box_encoder(torch.cat((boxes, embedded), dim=-1))
        valid = box_mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid, -torch.inf).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        scene_encoded = self.scene_encoder(scene)
        image = self.image_encoder(torch.cat((mean, maximum, scene_encoded), dim=-1))

        action_count = actions.shape[1]
        zero_budget = torch.zeros(
            (len(actions), action_count, 1), dtype=actions.dtype, device=actions.device
        )
        action = self.action_encoder(
            torch.cat(
                (
                    actions,
                    self.action_type_embedding(
                        action_type.clamp(0, len(ACTION_TYPES) - 1)
                    ),
                    zero_budget,
                ),
                dim=-1,
            )
        )
        image = image[:, None, :].expand(-1, action_count, -1)
        joint = torch.cat((image, action), dim=-1)
        value = self.head(joint).squeeze(-1)
        budget_column = budget[:, None, None].expand(-1, action_count, 1)
        utility = self.utility_head(
            torch.cat((joint, log_payload.unsqueeze(-1), budget_column), dim=-1)
        ).squeeze(-1)
        return {
            "value": value.masked_fill(~action_mask, -1e9),
            "utility": utility.masked_fill(~action_mask, -1e9),
        }


def value_objective(
    scores: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    reward_weight: float,
    within_weight: float,
) -> dict[str, torch.Tensor]:
    """Contextual-value regression and within-image action ordering only."""
    valid = action_mask.to(torch.bool)
    reward = F.smooth_l1_loss(scores[valid], targets[valid])
    target_difference = targets.unsqueeze(2) - targets.unsqueeze(1)
    score_difference = scores.unsqueeze(2) - scores.unsqueeze(1)
    action_count = scores.shape[1]
    upper = torch.triu(
        torch.ones((action_count, action_count), dtype=torch.bool, device=scores.device),
        diagonal=1,
    ).unsqueeze(0)
    pair_valid = (
        upper
        & valid.unsqueeze(2)
        & valid.unsqueeze(1)
        & (target_difference.abs() > 1e-8)
    )
    if pair_valid.any():
        weights = target_difference.abs()[pair_valid]
        within = (
            F.softplus(-target_difference.sign() * score_difference)[pair_valid]
            * weights
        ).sum() / weights.sum().clamp_min(1e-8)
    else:
        within = scores.sum() * 0.0
    loss = float(reward_weight) * reward + float(within_weight) * within
    return {"loss": loss, "reward_loss": reward, "within_loss": within}


def dual_head_objective(
    values: torch.Tensor,
    utilities: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    boundary_selected: torch.Tensor,
    boundary_pairs_count: int,
    seed: int,
    reward_weight: float,
    within_weight: float,
    boundary_weight: float,
) -> dict[str, torch.Tensor]:
    """Train value from reward and utility from the cost-aware boundary only."""
    value = value_objective(
        values,
        targets,
        action_mask,
        reward_weight=reward_weight,
        within_weight=within_weight,
    )
    positive, negative = boundary_pairs(
        boundary_selected.detach().cpu().numpy(),
        action_mask.detach().cpu().numpy(),
        pair_count=int(boundary_pairs_count),
        seed=int(seed),
    )
    if len(positive):
        flat = utilities.reshape(-1)
        positive_index = torch.as_tensor(positive, dtype=torch.long, device=flat.device)
        negative_index = torch.as_tensor(negative, dtype=torch.long, device=flat.device)
        boundary = F.softplus(
            -(flat[positive_index] - flat[negative_index])
        ).mean()
    else:
        boundary = utilities.sum() * 0.0
    return {
        "loss": value["loss"] + float(boundary_weight) * boundary,
        "value_loss": value["loss"],
        "reward_loss": value["reward_loss"],
        "within_loss": value["within_loss"],
        "boundary_loss": boundary,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> float:
    left = _average_ranks(np.asarray(values, dtype=np.float64))
    right = _average_ranks(np.asarray(targets, dtype=np.float64))
    if len(left) != len(right) or not len(left):
        raise ValueError("Spearman inputs must be non-empty and aligned")
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def ranking_diagnostics(
    *,
    scores: Sequence[float],
    rewards: Sequence[float],
    payload_bytes: Sequence[float],
    selected_indices: Sequence[int],
    teacher_indices: Sequence[int],
) -> dict[str, float]:
    selected = set(map(int, selected_indices))
    teacher = set(map(int, teacher_indices))
    overlap = len(selected & teacher)
    union = len(selected | teacher)
    return {
        "reward_spearman": spearman(scores, rewards),
        "payload_spearman": spearman(
            scores, np.log1p(np.asarray(payload_bytes, dtype=np.float64))
        ),
        "topk_overlap": int(overlap),
        "topk_recall": float(overlap / max(1, len(teacher))),
        "topk_jaccard": float(overlap / max(1, union)),
    }


def candidate_gate(
    gains: Sequence[float],
    *,
    minimum_gain: float,
    required_gain_points: int,
    maximum_other_loss: float,
) -> dict[str, object]:
    values = list(map(float, gains))
    gain_points = sum(value >= float(minimum_gain) for value in values)
    worst = min(values)
    passed = gain_points >= int(required_gain_points) and worst >= -float(
        maximum_other_loss
    )
    return {
        "passed": bool(passed),
        "gain_points": int(gain_points),
        "worst_gain": float(worst),
        "gains": values,
    }
