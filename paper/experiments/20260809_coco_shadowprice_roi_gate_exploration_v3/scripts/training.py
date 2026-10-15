"""Model adapter for a shadow-price-conditioned ROI gate."""

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


V1_TRAINING = _load("price_gate_v1_training", V1_ROOT / "scripts/training.py")


def build_model(
    *, category_count: int, scene_dim: int, action_dim: int, config: Mapping[str, Any]
) -> torch.nn.Module:
    return V1_TRAINING.SET_CORE.BudgetSetRouter(
        category_count=int(category_count),
        scene_dim=int(scene_dim),
        action_dim=int(action_dim),
        class_embedding_dim=int(config["class_embedding_dim"]),
        box_hidden_dim=int(config["box_hidden_dim"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        joint_hidden_dim=int(config["joint_hidden_dim"]),
        dropout=float(config["dropout"]),
    )


def model_inputs(
    data: Mapping[str, torch.Tensor], budget: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        **{
            key: data[key]
            for key in (
                "boxes",
                "classes",
                "box_mask",
                "scene",
                "actions",
                "action_type",
                "action_mask",
            )
        },
        "budget": budget,
    }


def conditional_roi_objective(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    budget: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logits = model(**model_inputs(data, budget))
    roi = data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
    if not bool((roi.sum(dim=1) == 1).all()):
        raise ValueError("Each price-gate row must contain exactly one ROI action")
    roi_logits = logits[roi]
    target = target.to(roi_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(roi_logits, target)
    prediction = roi_logits >= 0.0
    accuracy = (prediction == target.to(torch.bool)).to(roi_logits.dtype).mean()
    return {"loss": loss, "accuracy": accuracy.detach()}
