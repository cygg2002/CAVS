"""Cost-free/cost-aware frontier-occupancy model adapter."""

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


V1_TRAINING = _load(
    "occupancy_v1_training", V1_ROOT / "scripts/training.py"
)


def build_model(
    *, category_count: int, scene_dim: int, action_dim: int, config: Mapping[str, Any]
) -> torch.nn.Module:
    return V1_TRAINING.build_model(
        category_count=category_count,
        scene_dim=scene_dim,
        action_dim=action_dim,
        config=config,
    )


def model_inputs(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return V1_TRAINING.model_inputs(data)


def occupancy_objective(
    model: torch.nn.Module, data: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    logits = model(**model_inputs(data))
    roi = data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
    if not bool(roi.any()):
        raise ValueError("An occupancy batch must contain available ROI actions")
    loss = F.binary_cross_entropy_with_logits(
        logits[roi], data["occupancy_target"][roi].to(logits.dtype)
    )
    prediction = logits[roi] >= 0.0
    target = data["occupancy_target"][roi].to(torch.bool)
    accuracy = (prediction == target).to(logits.dtype).mean()
    return {"loss": loss, "accuracy": accuracy.detach()}
