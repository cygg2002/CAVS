"""Training helpers for cost-free value and separated dual-head routers."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[4]
DIAGNOSTIC_SCRIPTS = (
    REPO
    / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts"
)
if str(DIAGNOSTIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTIC_SCRIPTS))

from experiment_core import ACTION_TYPES, SCENE_DIM, ACTION_DIM, deterministic_split  # noqa: E402
from run_experiment import _model, _teacher_masks  # noqa: E402

from oof_core import (  # noqa: E402
    DualHeadRouter,
    dual_head_objective,
    strip_cost_inputs,
    value_objective,
)


def _subset(
    data: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _to_device(
    data: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


def _default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _normalization(data: Mapping[str, torch.Tensor]) -> tuple[dict, float, float]:
    valid_cloud = data["action_mask"] & (
        data["action_type"] != ACTION_TYPES["edge"]
    )
    rewards = data["target"][valid_cloud]
    mean = float(rewards.mean())
    std = max(1e-6, float(rewards.std(unbiased=False)))
    normalized = dict(data)
    normalized["target"] = (data["target"] - mean) / std
    return normalized, mean, std


def _value_epoch(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    score = model(
        boxes=data["boxes"],
        classes=data["classes"],
        box_mask=data["box_mask"],
        scene=data["scene"],
        actions=data["actions"],
        action_type=data["action_type"],
        action_mask=data["action_mask"],
        budget=torch.zeros(
            len(data["boxes"]), dtype=torch.float32, device=data["boxes"].device
        ),
    )
    config = protocol["model"]
    result = value_objective(
        score,
        data["target"],
        data["action_mask"],
        reward_weight=float(config["reward_loss_weight"]),
        within_weight=float(config["within_action_weight"]),
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return {key: float(value.detach()) for key, value in result.items()}


def _split_indices(image_ids: Sequence[int], seed: int, fraction: float) -> tuple[list[int], list[int]]:
    validation_count = max(1, int(round(len(image_ids) * float(fraction))))
    split = deterministic_split(
        image_ids, train_count=len(image_ids) - validation_count, seed=int(seed)
    )
    index_by_id = {int(image_id): index for index, image_id in enumerate(image_ids)}
    fit = [index_by_id[int(value)] for value in split["train_image_ids"]]
    validation = [index_by_id[int(value)] for value in split["holdout_image_ids"]]
    return fit, validation


def fit_value_router(
    image_ids: Sequence[int],
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    *,
    category_count: int,
    seed: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Fit a budget-invariant value model with direct cost features removed."""
    device = _default_device() if device is None else device
    cost_free, reward_mean, reward_std = _normalization(strip_cost_inputs(data))
    fit_indices, validation_indices = _split_indices(
        image_ids, int(seed), float(protocol["model"]["validation_fraction"])
    )
    fit_data = _to_device(_subset(cost_free, fit_indices), device)
    validation_data = _to_device(_subset(cost_free, validation_indices), device)

    model = _model(protocol, int(category_count), int(seed)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["model"]["learning_rate"]),
        weight_decay=float(protocol["model"]["weight_decay"]),
    )
    best_epoch, best_loss, stale = 1, math.inf, 0
    history = []
    for epoch in range(1, int(protocol["model"]["epochs"]) + 1):
        train = _value_epoch(model, fit_data, protocol, optimizer)
        with torch.no_grad():
            validation = _value_epoch(model, validation_data, protocol, None)
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(
            f"[SetValue] seed={seed} epoch={epoch} "
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
    final_model = _model(protocol, int(category_count), final_seed).to(device)
    all_data = _to_device(cost_free, device)
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(protocol["model"]["learning_rate"]),
        weight_decay=float(protocol["model"]["weight_decay"]),
    )
    for _ in range(best_epoch):
        _value_epoch(final_model, all_data, protocol, optimizer)
    return {
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "seed": final_seed,
        "selected_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "history": history,
        "device": str(device),
    }


def predict_value_router(
    checkpoint: Mapping[str, Any],
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    *,
    category_count: int,
) -> np.ndarray:
    model = _model(protocol, int(category_count), int(checkpoint["seed"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    cost_free = strip_cost_inputs(data)
    with torch.no_grad():
        score = model(
            boxes=cost_free["boxes"],
            classes=cost_free["classes"],
            box_mask=cost_free["box_mask"],
            scene=cost_free["scene"],
            actions=cost_free["actions"],
            action_type=cost_free["action_type"],
            action_mask=cost_free["action_mask"],
            budget=torch.zeros(len(cost_free["boxes"]), dtype=torch.float32),
        ).numpy()
    score = score * float(checkpoint["reward_std"]) + float(
        checkpoint["reward_mean"]
    )
    score[cost_free["action_type"].numpy() == ACTION_TYPES["edge"]] = 0.0
    score[~cost_free["action_mask"].numpy()] = -1e9
    return np.repeat(score[None, :, :], len(protocol["budgets"]), axis=0)


def _new_dual_model(
    protocol: Mapping[str, Any], category_count: int, seed: int
) -> DualHeadRouter:
    torch.manual_seed(int(seed))
    config = protocol["model"]
    return DualHeadRouter(
        category_count=int(category_count),
        scene_dim=SCENE_DIM,
        action_dim=ACTION_DIM,
        class_embedding_dim=int(config["class_embedding_dim"]),
        box_hidden_dim=int(config["box_hidden_dim"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        joint_hidden_dim=int(config["joint_hidden_dim"]),
        dropout=float(config["dropout"]),
    )


def _payload_statistics(records: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    values = [
        math.log1p(
            float(
                next(
                    action
                    for action in record["actions"]
                    if str(action["action_type"]) == "full"
                )["payload_bytes"]
            )
            / 1024.0
        )
        for record in records
    ]
    return float(np.mean(values)), max(1e-6, float(np.std(values)))


def _payload_tensor(
    records: Sequence[Mapping[str, Any]], *, mean: float, std: float, max_actions: int
) -> torch.Tensor:
    output = torch.zeros((len(records), int(max_actions)), dtype=torch.float32)
    for image_index, record in enumerate(records):
        for action_index, action in enumerate(record["actions"]):
            if str(action["action_type"]) != "edge":
                value = math.log1p(float(action.get("payload_bytes", 0.0)) / 1024.0)
                output[image_index, action_index] = (value - float(mean)) / float(std)
    return output


def _dual_epoch(
    model: DualHeadRouter,
    data: Mapping[str, torch.Tensor],
    payload: torch.Tensor,
    teacher_masks: Mapping[float, np.ndarray],
    protocol: Mapping[str, Any],
    *,
    seed: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    config = protocol["model"]
    results = []
    for offset, budget in enumerate(map(float, protocol["budgets"])):
        output = model(
            boxes=data["boxes"],
            classes=data["classes"],
            box_mask=data["box_mask"],
            scene=data["scene"],
            actions=data["actions"],
            action_type=data["action_type"],
            action_mask=data["action_mask"],
            log_payload=payload,
            budget=torch.full(
                (len(data["boxes"]),),
                budget,
                dtype=torch.float32,
                device=data["boxes"].device,
            ),
        )
        results.append(
            dual_head_objective(
                output["value"],
                output["utility"],
                data["target"],
                data["action_mask"],
                boundary_selected=torch.as_tensor(
                    teacher_masks[budget], dtype=torch.bool, device=data["boxes"].device
                ),
                boundary_pairs_count=int(config["pairs_per_budget"]),
                seed=int(seed) + offset * 1_000_003,
                reward_weight=float(config["reward_loss_weight"]),
                within_weight=float(config["within_action_weight"]),
                boundary_weight=float(config["boundary_weight"]),
            )
        )
    loss = torch.stack([result["loss"] for result in results]).mean()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    metrics = {"loss": float(loss.detach())}
    for key in ("value_loss", "reward_loss", "within_loss", "boundary_loss"):
        metrics[key] = float(
            torch.stack([result[key].detach() for result in results]).mean()
        )
    return metrics


def fit_dual_router(
    records: Sequence[Mapping[str, Any]],
    data: Mapping[str, torch.Tensor],
    raw_rewards: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    category_count: int,
    seed: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Fit separated value and cost-aware utility heads."""
    device = _default_device() if device is None else device
    cost_free, reward_mean, reward_std = _normalization(strip_cost_inputs(data))
    image_ids = [int(record["image_id"]) for record in records]
    fit_indices, validation_indices = _split_indices(
        image_ids, int(seed), float(protocol["model"]["validation_fraction"])
    )
    fit_records = [records[index] for index in fit_indices]
    validation_records = [records[index] for index in validation_indices]
    fit_masks = _teacher_masks(
        fit_records, raw_rewards[fit_indices], protocol["budgets"], protocol
    )
    validation_masks = _teacher_masks(
        validation_records,
        raw_rewards[validation_indices],
        protocol["budgets"],
        protocol,
    )
    payload_mean, payload_std = _payload_statistics(records)
    payload = _payload_tensor(
        records,
        mean=payload_mean,
        std=payload_std,
        max_actions=int(protocol["max_actions"]),
    )
    fit_data = _to_device(_subset(cost_free, fit_indices), device)
    validation_data = _to_device(_subset(cost_free, validation_indices), device)
    fit_payload = payload[fit_indices].to(device)
    validation_payload = payload[validation_indices].to(device)

    model = _new_dual_model(protocol, int(category_count), int(seed)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["model"]["learning_rate"]),
        weight_decay=float(protocol["model"]["weight_decay"]),
    )
    best_epoch, best_loss, stale = 1, math.inf, 0
    history = []
    for epoch in range(1, int(protocol["model"]["epochs"]) + 1):
        train = _dual_epoch(
            model,
            fit_data,
            fit_payload,
            fit_masks,
            protocol,
            seed=int(seed) + epoch * 10_007,
            optimizer=optimizer,
        )
        with torch.no_grad():
            validation = _dual_epoch(
                model,
                validation_data,
                validation_payload,
                validation_masks,
                protocol,
                seed=int(seed) + epoch * 10_007,
                optimizer=None,
            )
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(
            f"[DualHead] seed={seed} epoch={epoch} "
            f"train={train['loss']:.6f} value-val={validation['value_loss']:.6f}",
            flush=True,
        )
        if validation["value_loss"] < best_loss - 1e-6:
            best_epoch, best_loss, stale = epoch, validation["value_loss"], 0
        else:
            stale += 1
        if stale >= int(protocol["model"]["patience"]):
            break

    final_seed = int(seed) + 1
    final_model = _new_dual_model(protocol, int(category_count), final_seed).to(device)
    all_data = _to_device(cost_free, device)
    all_payload = payload.to(device)
    all_masks = _teacher_masks(records, raw_rewards, protocol["budgets"], protocol)
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(protocol["model"]["learning_rate"]),
        weight_decay=float(protocol["model"]["weight_decay"]),
    )
    for epoch in range(1, best_epoch + 1):
        _dual_epoch(
            final_model,
            all_data,
            all_payload,
            all_masks,
            protocol,
            seed=final_seed + epoch * 10_007,
            optimizer=optimizer,
        )
    return {
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "seed": final_seed,
        "selected_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "payload_mean": payload_mean,
        "payload_std": payload_std,
        "history": history,
        "device": str(device),
    }


def predict_dual_value(
    checkpoint: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    *,
    category_count: int,
) -> np.ndarray:
    model = _new_dual_model(protocol, int(category_count), int(checkpoint["seed"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    cost_free = strip_cost_inputs(data)
    payload = _payload_tensor(
        records,
        mean=float(checkpoint["payload_mean"]),
        std=float(checkpoint["payload_std"]),
        max_actions=int(protocol["max_actions"]),
    )
    with torch.no_grad():
        output = model(
            boxes=cost_free["boxes"],
            classes=cost_free["classes"],
            box_mask=cost_free["box_mask"],
            scene=cost_free["scene"],
            actions=cost_free["actions"],
            action_type=cost_free["action_type"],
            action_mask=cost_free["action_mask"],
            log_payload=payload,
            budget=torch.zeros(len(records), dtype=torch.float32),
        )
    score = output["value"].numpy() * float(checkpoint["reward_std"]) + float(
        checkpoint["reward_mean"]
    )
    score[cost_free["action_type"].numpy() == ACTION_TYPES["edge"]] = 0.0
    score[~cost_free["action_mask"].numpy()] = -1e9
    return np.repeat(score[None, :, :], len(protocol["budgets"]), axis=0)
