"""Minimal ActionValue adapter with parametric frontier-path supervision."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FRONTIER = _load("frontier_training_core", ROOT / "scripts/core.py")
SET_CORE = _load(
    "frontier_source_budgetset_core",
    REPO
    / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts/experiment_core.py",
)
VALUE_CORE = _load(
    "frontier_source_value_core",
    REPO
    / "paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/scripts/oof_core.py",
)


class StaticActionValueRouter(SET_CORE.BudgetSetRouter):
    """Canonical ActionValue encoder with its budget input fixed to zero."""

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        budget = torch.zeros(
            len(inputs["boxes"]),
            dtype=inputs["boxes"].dtype,
            device=inputs["boxes"].device,
        )
        return super().forward(**inputs, budget=budget)


def build_model(
    *,
    category_count: int,
    scene_dim: int,
    action_dim: int,
    config: Mapping[str, Any],
) -> StaticActionValueRouter:
    return StaticActionValueRouter(
        category_count=int(category_count),
        scene_dim=int(scene_dim),
        action_dim=int(action_dim),
        class_embedding_dim=int(config["class_embedding_dim"]),
        box_hidden_dim=int(config["box_hidden_dim"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        joint_hidden_dim=int(config["joint_hidden_dim"]),
        dropout=float(config["dropout"]),
    )


def model_inputs(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
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
    }


def strip_cost_inputs(
    data: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove payload features while preserving all five action identities."""

    return VALUE_CORE.strip_cost_inputs(data)


def reward_and_cost_tensors(
    records: list[Mapping[str, Any]],
    reward_map: Mapping[int, Mapping[str, float]],
    *,
    max_actions: int,
    cost_scale_bytes: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not float(cost_scale_bytes) > 0.0:
        raise ValueError("A positive dataset-wide cost scale is required")
    rewards = torch.zeros((len(records), int(max_actions)), dtype=torch.float64)
    costs = torch.zeros_like(rewards)
    for row, record in enumerate(records):
        image_id = int(record["image_id"])
        if image_id not in reward_map:
            raise ValueError(f"Missing reward image: {image_id}")
        if float(record["full_payload_bytes"]) <= 0.0:
            raise ValueError(f"Invalid Original payload for image {image_id}")
        actions = list(record["actions"])
        if len(actions) > int(max_actions):
            raise ValueError(f"Too many actions for image {image_id}")
        for column, action in enumerate(actions):
            action_id = str(action["action_id"])
            if action_id != "edge":
                if action_id not in reward_map[image_id]:
                    raise ValueError(f"Missing reward for {image_id}/{action_id}")
                rewards[row, column] = float(reward_map[image_id][action_id])
            costs[row, column] = (
                float(action.get("payload_bytes", 0.0)) / float(cost_scale_bytes)
            )
    return rewards, costs


def training_objective(
    model: StaticActionValueRouter,
    method: str,
    data: Mapping[str, torch.Tensor],
    *,
    reward_mean: float,
    reward_std: float,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if method not in FRONTIER.EXPECTED_METHODS:
        raise ValueError(f"Unknown screen method: {method}")
    scores = model(**model_inputs(data))
    base = VALUE_CORE.value_objective(
        scores,
        data["target"],
        data["action_mask"],
        reward_weight=float(config["reward_loss_weight"]),
        within_weight=float(config["within_action_weight"]),
    )
    frontier = scores.sum() * 0.0
    if method != "CurrentActionValue":
        raw_scores = scores * float(reward_std) + float(reward_mean)
        raw_scores = torch.where(
            data["action_type"] == 0,
            torch.zeros_like(raw_scores),
            raw_scores,
        )
        prices, price_mask = FRONTIER.frontier_probe_prices(
            data["raw_target"],
            data["cost_ratio"],
            data["action_mask"],
            relative_offset=float(config["frontier_relative_offset"]),
            maximum_price=float(config["frontier_maximum_price"]),
        )
        boundary = method == "BoundaryPathDistill"
        frontier = FRONTIER.frontier_path_loss(
            raw_scores,
            data["raw_target"],
            data["cost_ratio"],
            data["action_mask"],
            prices,
            price_mask=price_mask,
            temperature=float(
                config["boundary_temperature"]
                if boundary
                else config["path_temperature"]
            ),
            boundary_scale=float(config["boundary_scale"]),
            boundary_boost=float(config["boundary_boost"] if boundary else 0.0),
        )
    weight = (
        0.0
        if method == "CurrentActionValue"
        else float(
            config["boundary_path_loss_weight"]
            if method == "BoundaryPathDistill"
            else config["path_loss_weight"]
        )
    )
    return {
        "loss": base["loss"] + weight * frontier,
        "base_loss": base["loss"],
        "reward_loss": base["reward_loss"],
        "within_loss": base["within_loss"],
        "frontier_loss": frontier,
    }
