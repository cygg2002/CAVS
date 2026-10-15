"""Same-backbone CostInValue training used by the final completion experiment."""

from __future__ import annotations

from collections import Counter
import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def cost_in_value_data(
    adapter: Any,
    records: Sequence[Mapping[str, Any]],
    rewards: np.ndarray,
    protocol: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Tensorize without invoking either historical cost-stripping helper."""
    return adapter.tensorize(records, rewards, protocol)


def _inner_split(count: int, seed: int, fraction: float) -> tuple[list[int], list[int]]:
    if count < 4:
        raise ValueError("at least four samples are required")
    order = np.random.default_rng(int(seed)).permutation(count)
    validation_count = min(count - 2, max(1, int(round(count * float(fraction)))))
    return order[validation_count:].tolist(), order[:validation_count].tolist()


def _model_inputs(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def _normalize(
    data: Mapping[str, torch.Tensor], train_indices: Sequence[int]
) -> tuple[dict[str, torch.Tensor], float, float]:
    index = torch.as_tensor(train_indices, dtype=torch.long)
    cloud = data["action_mask"][index] & (data["action_type"][index] != 0)
    values = data["target"][index][cloud]
    mean = float(values.mean())
    std = max(1e-6, float(values.std(unbiased=False)))
    output = dict(data)
    output["target"] = (data["target"] - mean) / std
    return output, mean, std


def fit_cost_in_value(
    adapter: Any,
    records: Sequence[Mapping[str, Any]],
    rewards: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    category_count: int,
    seed: int,
) -> dict[str, Any]:
    """Fit CurrentActionValue without stripping payload fields from action input."""
    torch.set_num_threads(1)
    source = adapter._sources()
    data = cost_in_value_data(adapter, records, rewards, protocol)
    # Deliberately keep the raw action-feature payload fields.  Calling the
    # historical `_frontier_data` helper here would silently strip them and turn
    # this control back into ordinary CAVS, invalidating the one-variable ablation.
    config = protocol["model"]
    train, validation = _inner_split(
        len(data["boxes"]), int(seed), float(config["validation_fraction"])
    )
    normalized, reward_mean, reward_std = _normalize(data, train)
    torch.manual_seed(int(seed))
    model = source["afpd"].build_model(
        category_count=int(category_count),
        scene_dim=int(source["set_core"].SCENE_DIM),
        action_dim=int(source["set_core"].ACTION_DIM),
        config=config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_loss, stale = 1, math.inf, 0
    history: list[dict[str, float]] = []

    def run_epoch(indices: Sequence[int], training: bool, epoch_seed: int) -> float:
        model.train(training)
        order = torch.as_tensor(indices, dtype=torch.long)
        if training:
            generator = torch.Generator().manual_seed(int(epoch_seed))
            order = order[torch.randperm(len(order), generator=generator)]
        totals: Counter[str] = Counter()
        count = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for start in range(0, len(order), int(config["batch_size"])):
                index = order[start : start + int(config["batch_size"])]
                batch = {key: value[index] for key, value in normalized.items()}
                result = source["afpd"].training_objective(
                    model,
                    "CurrentActionValue",
                    batch,
                    reward_mean=reward_mean,
                    reward_std=reward_std,
                    config=config,
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    result["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["gradient_clip"])
                    )
                    optimizer.step()
                size = len(index)
                count += size
                for key, value in result.items():
                    totals[key] += float(value.detach()) * size
        return float(totals["loss"] / max(1, count))

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = run_epoch(train, True, int(seed) + epoch * 1_000_003)
        validation_loss = run_epoch(validation, False, int(seed))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, best_loss, stale = epoch, validation_loss, 0
        else:
            stale += 1
        if stale >= int(config["patience"]):
            break
    model.load_state_dict(best_state)
    return {
        "kind": "cost_in_value",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "category_count": int(category_count),
        "metadata": {
            "selected_epoch": int(best_epoch),
            "best_validation_loss": float(best_loss),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "history": history,
            "payload_visible_to_value": True,
        },
    }


def predict_cost_in_value(
    adapter: Any,
    artifact: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> np.ndarray:
    source = adapter._sources()
    data = adapter.tensorize(records, None, protocol)
    model = source["afpd"].build_model(
        category_count=int(artifact["category_count"]),
        scene_dim=int(source["set_core"].SCENE_DIM),
        action_dim=int(source["set_core"].ACTION_DIM),
        config=protocol["model"],
    )
    model.load_state_dict(artifact["state_dict"])
    values = adapter._predict_static(
        model,
        data,
        reward_mean=float(artifact["metadata"]["reward_mean"]),
        reward_std=float(artifact["metadata"]["reward_std"]),
        batch_size=int(protocol["model"]["batch_size"]),
    )
    return np.repeat(values[None, :, :], len(protocol["budgets"]), axis=0)
