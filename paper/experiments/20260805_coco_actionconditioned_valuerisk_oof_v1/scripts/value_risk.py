"""Shared-encoder mean and lower-quantile action value model."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
SET_SCRIPTS = (
    REPO
    / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts"
)
VALUE_SCRIPTS = (
    REPO / "paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/scripts"
)
sys.path[:0] = [str(SET_SCRIPTS), str(VALUE_SCRIPTS)]

from experiment_core import ACTION_TYPES, BudgetSetRouter, deterministic_split  # noqa: E402
from oof_core import strip_cost_inputs, value_objective  # noqa: E402


class ActionValueRiskRouter(BudgetSetRouter):
    """A cost-free action value mean plus conditional lower reward quantile."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        joint_hidden_dim = int(kwargs["joint_hidden_dim"])
        action_hidden_dim = int(kwargs["action_hidden_dim"])
        dropout = float(kwargs["dropout"])
        self.lower_head = nn.Sequential(
            nn.Linear(joint_hidden_dim + action_hidden_dim, joint_hidden_dim),
            nn.LayerNorm(joint_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
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
    ) -> dict[str, torch.Tensor]:
        embedded = self.class_embedding(classes.clamp(0, self.category_count - 1))
        encoded = self.box_encoder(torch.cat((boxes, embedded), dim=-1))
        valid = box_mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean_boxes = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid, -torch.inf).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        scene_encoded = self.scene_encoder(scene)
        image = self.image_encoder(
            torch.cat((mean_boxes, maximum, scene_encoded), dim=-1)
        )
        action_count = actions.shape[1]
        zero_budget = torch.zeros(
            (len(actions), action_count, 1),
            dtype=actions.dtype,
            device=actions.device,
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
        mean = self.head(joint).squeeze(-1)
        lower = self.lower_head(joint).squeeze(-1)
        return {
            "mean": mean.masked_fill(~action_mask, -1e9),
            "lower": lower.masked_fill(~action_mask, -1e9),
        }


def value_risk_objective(
    mean: torch.Tensor,
    lower: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    action_type: torch.Tensor,
    quantile: float,
    reward_weight: float,
    within_weight: float,
    quantile_weight: float,
    crossing_weight: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 < float(quantile) < 0.5:
        raise ValueError("quantile must be in (0, 0.5)")
    value = value_objective(
        mean,
        target,
        action_mask,
        reward_weight=float(reward_weight),
        within_weight=float(within_weight),
    )
    cloud = action_mask.to(torch.bool) & (
        action_type != int(ACTION_TYPES["edge"])
    )
    residual = target[cloud] - lower[cloud]
    pinball = torch.maximum(
        float(quantile) * residual,
        (float(quantile) - 1.0) * residual,
    ).mean()
    crossing = F.relu(lower[cloud] - mean[cloud]).mean()
    loss = (
        value["loss"]
        + float(quantile_weight) * pinball
        + float(crossing_weight) * crossing
    )
    return {
        "loss": loss,
        "value_loss": value["loss"],
        "reward_loss": value["reward_loss"],
        "within_loss": value["within_loss"],
        "quantile_loss": pinball,
        "crossing_loss": crossing,
    }


def _subset(
    data: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _device_data(
    data: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


def _model(protocol: Mapping[str, Any], category_count: int, seed: int) -> ActionValueRiskRouter:
    torch.manual_seed(int(seed))
    config = protocol["model"]
    return ActionValueRiskRouter(
        category_count=int(category_count),
        scene_dim=9,
        action_dim=18,
        class_embedding_dim=int(config["class_embedding_dim"]),
        box_hidden_dim=int(config["box_hidden_dim"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        joint_hidden_dim=int(config["joint_hidden_dim"]),
        dropout=float(config["dropout"]),
    )


def _normalize_targets(
    data: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], float, float]:
    cost_free = strip_cost_inputs(data)
    cloud = cost_free["action_mask"] & (
        cost_free["action_type"] != int(ACTION_TYPES["edge"])
    )
    values = cost_free["target"][cloud]
    mean = float(values.mean())
    std = max(1e-6, float(values.std(unbiased=False)))
    normalized = dict(cost_free)
    normalized["target"] = (cost_free["target"] - mean) / std
    return normalized, mean, std


def _epoch(
    model: ActionValueRiskRouter,
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    output = model(
        boxes=data["boxes"],
        classes=data["classes"],
        box_mask=data["box_mask"],
        scene=data["scene"],
        actions=data["actions"],
        action_type=data["action_type"],
        action_mask=data["action_mask"],
    )
    config = protocol["model"]
    result = value_risk_objective(
        output["mean"],
        output["lower"],
        data["target"],
        data["action_mask"],
        action_type=data["action_type"],
        quantile=float(config["quantile"]),
        reward_weight=float(config["reward_loss_weight"]),
        within_weight=float(config["within_action_weight"]),
        quantile_weight=float(config["quantile_loss_weight"]),
        crossing_weight=float(config["crossing_loss_weight"]),
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return {key: float(value.detach()) for key, value in result.items()}


def fit_value_risk_router(
    image_ids: Sequence[int],
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    *,
    category_count: int,
    seed: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    normalized, reward_mean, reward_std = _normalize_targets(data)
    validation_count = max(
        1,
        int(round(len(image_ids) * float(protocol["model"]["validation_fraction"]))),
    )
    split = deterministic_split(
        image_ids,
        train_count=len(image_ids) - validation_count,
        seed=int(seed),
    )
    index_by_id = {int(image_id): index for index, image_id in enumerate(image_ids)}
    fit_indices = [index_by_id[int(value)] for value in split["train_image_ids"]]
    validation_indices = [
        index_by_id[int(value)] for value in split["holdout_image_ids"]
    ]
    fit_data = _device_data(_subset(normalized, fit_indices), device)
    validation_data = _device_data(_subset(normalized, validation_indices), device)
    model = _model(protocol, int(category_count), int(seed)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["model"]["learning_rate"]),
        weight_decay=float(protocol["model"]["weight_decay"]),
    )
    best_epoch, best_loss, stale = 1, math.inf, 0
    history = []
    for epoch in range(1, int(protocol["model"]["epochs"]) + 1):
        train = _epoch(model, fit_data, protocol, optimizer)
        with torch.no_grad():
            validation = _epoch(model, validation_data, protocol, None)
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(
            f"[ValueRisk] seed={seed} epoch={epoch} "
            f"train={train['loss']:.6f} validation={validation['loss']:.6f}",
            flush=True,
        )
        if validation["loss"] < best_loss - 1e-6:
            best_epoch, best_loss, stale = epoch, validation["loss"], 0
        else:
            stale += 1
        if stale >= int(protocol["model"]["patience"]):
            break
    final_seed = int(seed) + 1
    model = _model(protocol, int(category_count), final_seed).to(device)
    all_data = _device_data(normalized, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["model"]["learning_rate"]),
        weight_decay=float(protocol["model"]["weight_decay"]),
    )
    for _ in range(best_epoch):
        _epoch(model, all_data, protocol, optimizer)
    return {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "seed": final_seed,
        "selected_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "history": history,
        "device": str(device),
    }


def predict_value_risk_router(
    checkpoint: Mapping[str, Any],
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    *,
    category_count: int,
) -> dict[str, np.ndarray]:
    model = _model(protocol, int(category_count), int(checkpoint["seed"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    cost_free = strip_cost_inputs(data)
    with torch.no_grad():
        output = model(
            boxes=cost_free["boxes"],
            classes=cost_free["classes"],
            box_mask=cost_free["box_mask"],
            scene=cost_free["scene"],
            actions=cost_free["actions"],
            action_type=cost_free["action_type"],
            action_mask=cost_free["action_mask"],
        )
    values = {}
    for name in ("mean", "lower"):
        score = output[name].numpy() * float(checkpoint["reward_std"]) + float(
            checkpoint["reward_mean"]
        )
        score[cost_free["action_type"].numpy() == ACTION_TYPES["edge"]] = 0.0
        score[~cost_free["action_mask"].numpy()] = -1e9
        values[name] = score
    return values
