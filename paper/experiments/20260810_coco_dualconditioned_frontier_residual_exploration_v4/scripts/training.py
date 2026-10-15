"""Model adapter for dual-conditioned ROI-value residual learning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
V1_ROOT = REPO / "paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V1_TRAINING = _load("residual_v1_training", V1_ROOT / "scripts/training.py")


def build_model(
    *, category_count: int, scene_dim: int, action_dim: int, config: Mapping[str, Any]
) -> torch.nn.Module:
    return V1_TRAINING.SET_CORE.BudgetSetRouter(
        category_count=int(category_count),
        scene_dim=int(scene_dim),
        action_dim=int(action_dim) + int(config["frontier_state_dim"]),
        class_embedding_dim=int(config["class_embedding_dim"]),
        box_hidden_dim=int(config["box_hidden_dim"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        joint_hidden_dim=int(config["joint_hidden_dim"]),
        dropout=float(config["dropout"]),
    )


def augmented_inputs(
    data: Mapping[str, torch.Tensor],
    budget: torch.Tensor,
    frontier_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if frontier_features.shape != (len(budget), 3):
        raise ValueError("Frontier features must have shape [batch, 3]")
    action_count = data["actions"].shape[1]
    repeated = frontier_features[:, None, :].expand(-1, action_count, -1)
    return {
        "boxes": data["boxes"],
        "classes": data["classes"],
        "box_mask": data["box_mask"],
        "scene": data["scene"],
        "actions": torch.cat((data["actions"], repeated), dim=2),
        "action_type": data["action_type"],
        "action_mask": data["action_mask"],
        "budget": budget,
    }


def residual_objective(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    budget: torch.Tensor,
    frontier_features: torch.Tensor,
    frozen_margin: torch.Tensor,
    true_margin: torch.Tensor,
    *,
    sign_loss_weight: float,
    sign_temperature: float,
) -> dict[str, torch.Tensor]:
    scores = model(**augmented_inputs(data, budget, frontier_features))
    roi = data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
    if not bool((roi.sum(dim=1) == 1).all()):
        raise ValueError("Every residual row must contain exactly one ROI")
    correction = scores[roi]
    predicted_margin = frozen_margin.to(correction.dtype) + correction
    target_margin = true_margin.to(correction.dtype)
    regression = F.smooth_l1_loss(predicted_margin, target_margin)
    sign = F.binary_cross_entropy_with_logits(
        predicted_margin / float(sign_temperature),
        (target_margin >= 0.0).to(predicted_margin.dtype),
    )
    loss = regression + float(sign_loss_weight) * sign
    accuracy = ((predicted_margin >= 0.0) == (target_margin >= 0.0)).to(
        predicted_margin.dtype
    ).mean()
    return {
        "loss": loss,
        "regression_loss": regression.detach(),
        "sign_loss": sign.detach(),
        "sign_accuracy": accuracy.detach(),
    }

