"""Action-conditioned ORIC/MORIC models for the frozen COCO paper protocol."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SCENE_PAYLOAD_INDEX = 1
ACTION_COST_INDICES = (6, 7, 8, 9, 10, 11, 12)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("oric_moric_ac_method_core", ROOT / "scripts/core.py")


class ActionConditionedORICRouter(nn.Module):
    """ORIC Top-25 image representation with the CAVS action-value head."""

    def __init__(
        self,
        *,
        image_feature_dim: int,
        action_dim: int,
        action_type_count: int,
        image_hidden_dims: Sequence[int],
        action_hidden_dim: int,
        joint_hidden_dim: int,
        image_dropout: float,
        joint_dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(image_feature_dim)
        for width in map(int, image_hidden_dims):
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(float(image_dropout)),
                ]
            )
            previous = width
        layers.extend(
            [
                nn.Linear(previous, int(joint_hidden_dim)),
                nn.LayerNorm(int(joint_hidden_dim)),
                nn.SiLU(),
            ]
        )
        self.image_encoder = nn.Sequential(*layers)
        self.action_type_embedding = nn.Embedding(int(action_type_count), 8)
        self.action_encoder = nn.Sequential(
            nn.Linear(int(action_dim) + 8 + 1, int(action_hidden_dim)),
            nn.LayerNorm(int(action_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(action_hidden_dim), int(action_hidden_dim)),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(int(joint_hidden_dim) + int(action_hidden_dim), int(joint_hidden_dim)),
            nn.LayerNorm(int(joint_hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(joint_dropout)),
            nn.Linear(int(joint_hidden_dim), 1),
        )

    def forward(
        self,
        *,
        image_features: torch.Tensor,
        actions: torch.Tensor,
        action_type: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        image = self.image_encoder(image_features)
        action_count = int(actions.shape[1])
        zero_budget = torch.zeros(
            (len(actions), action_count, 1), dtype=actions.dtype, device=actions.device
        )
        action = self.action_encoder(
            torch.cat(
                (
                    actions,
                    self.action_type_embedding(
                        action_type.clamp(0, self.action_type_embedding.num_embeddings - 1)
                    ),
                    zero_budget,
                ),
                dim=-1,
            )
        )
        image = image[:, None, :].expand(-1, action_count, -1)
        scores = self.head(torch.cat((image, action), dim=-1)).squeeze(-1)
        return scores.masked_fill(~action_mask.to(torch.bool), -1e9)


def strip_value_cost_inputs(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove payload/resource inputs without mutating caller-owned tensors."""

    output = dict(data)
    scene = data["scene"].clone()
    actions = data["actions"].clone()
    scene[:, SCENE_PAYLOAD_INDEX] = 0.0
    actions[:, :, list(ACTION_COST_INDICES)] = 0.0
    output["scene"] = scene
    output["actions"] = actions
    return output


def _inner_split(count: int, seed: int, fraction: float) -> tuple[list[int], list[int]]:
    if count < 4:
        raise ValueError("at least four samples are required")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(count))
    validation_count = min(
        int(count) - 2, max(1, int(round(int(count) * float(fraction))))
    )
    return order[validation_count:].tolist(), order[:validation_count].tolist()


def _value_objective(
    scores: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    reward_weight: float,
    within_weight: float,
) -> dict[str, torch.Tensor]:
    valid = action_mask.to(torch.bool)
    reward = F.smooth_l1_loss(scores[valid], targets[valid])
    target_difference = targets.unsqueeze(2) - targets.unsqueeze(1)
    score_difference = scores.unsqueeze(2) - scores.unsqueeze(1)
    action_count = int(scores.shape[1])
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


def _model(
    image_feature_dim: int,
    action_dim: int,
    action_type_count: int,
    protocol: Mapping[str, Any],
) -> ActionConditionedORICRouter:
    return ActionConditionedORICRouter(
        image_feature_dim=int(image_feature_dim),
        action_dim=int(action_dim),
        action_type_count=int(action_type_count),
        image_hidden_dims=protocol["external_mlp"]["hidden_dims"],
        action_hidden_dim=int(protocol["model"]["action_hidden_dim"]),
        joint_hidden_dim=int(protocol["model"]["joint_hidden_dim"]),
        image_dropout=float(protocol["external_mlp"]["dropout"]),
        joint_dropout=float(protocol["model"]["dropout"]),
    )


def _batch_indices(indices: Sequence[int], batch_size: int) -> list[torch.Tensor]:
    ordered = torch.as_tensor(indices, dtype=torch.long)
    batches = [ordered[start : start + int(batch_size)] for start in range(0, len(ordered), int(batch_size))]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = torch.cat((batches[-2], batches[-1]))
        batches.pop()
    return batches


def fit_action_conditioned(
    method: str,
    image_features: np.ndarray,
    tensors: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit one outer-fold model using only the supplied training population."""

    if method not in CORE.METHODS:
        raise ValueError(f"unknown method: {method}")
    torch.set_num_threads(1)
    config = protocol["model"]
    features = torch.from_numpy(np.asarray(image_features, dtype=np.float32))
    if len(features) != len(tensors["actions"]):
        raise ValueError("feature and action populations differ")
    work = strip_value_cost_inputs(tensors)
    train, validation = _inner_split(
        len(features), int(seed), float(config["validation_fraction"])
    )
    raw_target = work["target"].detach().cpu().numpy().astype(np.float64)
    cdf_reference = None
    if method == "MORIC-AC-OQR":
        transformed, cdf_reference = CORE.unified_cloud_cdf_targets(
            raw_target,
            work["action_type"].detach().cpu().numpy(),
            work["action_mask"].detach().cpu().numpy(),
            reference_rows=train,
        )
        target = torch.from_numpy(transformed.astype(np.float32))
    else:
        target = work["target"].clone().to(torch.float32)
    train_index = torch.as_tensor(train, dtype=torch.long)
    cloud = work["action_mask"][train_index] & (work["action_type"][train_index] != 0)
    cloud_values = target[train_index][cloud]
    reward_mean = float(cloud_values.mean())
    reward_std = max(1e-6, float(cloud_values.std(unbiased=False)))
    normalized_target = (target - reward_mean) / reward_std
    torch.manual_seed(int(seed))
    model = _model(
        int(features.shape[1]),
        int(work["actions"].shape[2]),
        int(work["action_type"].max().item()) + 1,
        protocol,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_loss, stale = 1, math.inf, 0
    history = []

    def run_epoch(indices: Sequence[int], training: bool, epoch_seed: int) -> dict[str, float]:
        model.train(training)
        ordered = list(map(int, indices))
        if training:
            generator = torch.Generator().manual_seed(int(epoch_seed))
            order = torch.randperm(len(ordered), generator=generator).tolist()
            ordered = [ordered[index] for index in order]
        totals = {"loss": 0.0, "reward_loss": 0.0, "within_loss": 0.0}
        samples = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for index in _batch_indices(ordered, int(config["batch_size"])):
                scores = model(
                    image_features=features[index],
                    actions=work["actions"][index],
                    action_type=work["action_type"][index],
                    action_mask=work["action_mask"][index],
                )
                result = _value_objective(
                    scores,
                    normalized_target[index],
                    work["action_mask"][index],
                    reward_weight=float(config["reward_loss_weight"]),
                    within_weight=float(config["within_action_weight"]),
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    result["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["gradient_clip"])
                    )
                    optimizer.step()
                count = len(index)
                samples += count
                for key in totals:
                    totals[key] += float(result[key].detach()) * count
        return {key: value / max(1, samples) for key, value in totals.items()}

    for epoch in range(1, int(config["epochs"]) + 1):
        fit = run_epoch(train, True, int(seed) + epoch * 1_000_003)
        valid = run_epoch(validation, False, int(seed))
        value = float(valid["loss"])
        history.append(
            {"epoch": epoch, "train_loss": fit["loss"], "validation_loss": value}
        )
        if value < best_loss - 1e-6:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, best_loss, stale = epoch, value, 0
        else:
            stale += 1
        if stale >= int(config["patience"]):
            break
    model.load_state_dict(best_state)
    return {
        "kind": method,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "image_feature_dim": int(features.shape[1]),
        "action_dim": int(work["actions"].shape[2]),
        "action_type_count": int(work["action_type"].max().item()) + 1,
        "metadata": {
            "selected_epoch": int(best_epoch),
            "best_validation_loss": float(best_loss),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "cdf_reference_count": 0 if cdf_reference is None else int(len(cdf_reference)),
            "cdf_reference_min": None if cdf_reference is None else float(cdf_reference[0]),
            "cdf_reference_max": None if cdf_reference is None else float(cdf_reference[-1]),
            "history": history,
        },
    }


def predict_action_conditioned(
    artifact: Mapping[str, Any],
    image_features: np.ndarray,
    tensors: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
) -> np.ndarray:
    features = torch.from_numpy(np.asarray(image_features, dtype=np.float32))
    work = strip_value_cost_inputs(tensors)
    model = _model(
        int(artifact["image_feature_dim"]),
        int(artifact["action_dim"]),
        int(artifact["action_type_count"]),
        protocol,
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    output = []
    batch_size = int(protocol["model"]["batch_size"])
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(len(features), start + batch_size)
            output.append(
                model(
                    image_features=features[start:stop],
                    actions=work["actions"][start:stop],
                    action_type=work["action_type"][start:stop],
                    action_mask=work["action_mask"][start:stop],
                ).cpu()
            )
    values = torch.cat(output).numpy().astype(np.float64)
    values = values * float(artifact["metadata"]["reward_std"]) + float(
        artifact["metadata"]["reward_mean"]
    )
    types = work["action_type"].numpy()
    mask = work["action_mask"].numpy()
    values[types == 0] = 0.0
    values[~mask] = -1e9
    return values
