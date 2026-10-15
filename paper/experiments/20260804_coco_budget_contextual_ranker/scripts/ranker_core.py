"""Pure contracts for the budget-conditioned contextual ranker diagnostic."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("This diagnostic is frozen to COCO")
    if list(map(float, protocol.get("budgets", []))) != [0.1, 0.25, 0.4]:
        raise ValueError("Fixed-call budgets must remain 10%, 25%, and 40%")
    if protocol.get("fit_only_training") is not True:
        raise ValueError("Ranker training must remain fit-only")
    if protocol.get("calibration_used_for_model_training") is not False:
        raise ValueError("Calibration cannot train the ranker")
    if protocol.get("final_used_for_selection") is not False:
        raise ValueError("Final cannot select the ranker")
    if protocol.get("final_requires_passed_calibration_gate") is not True:
        raise ValueError("Final must require a passed calibration gate")
    seeds = list(map(int, protocol.get("seeds", [])))
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("At least two unique pre-registered seeds are required")


def _top_count(total: int, budget: float) -> int:
    count = int(math.floor(int(total) * float(budget) + 0.5))
    if not 0 < count < int(total):
        raise ValueError("Budget must select a non-empty proper subset")
    return count


def budget_boundary_pairs(
    rewards: np.ndarray,
    *,
    budget: float,
    pair_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample deterministic positive/negative pairs across one Top-K boundary."""
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rewards must be one finite vector")
    if int(pair_count) <= 0:
        raise ValueError("pair_count must be positive")
    count = _top_count(len(values), budget)
    order = np.argsort(-values, kind="mergesort")
    positive_pool, negative_pool = order[:count], order[count:]
    rng = np.random.default_rng(int(seed))
    return (
        rng.choice(positive_pool, size=int(pair_count), replace=True),
        rng.choice(negative_pool, size=int(pair_count), replace=True),
    )


def pairwise_logistic_loss(positive_scores: Any, negative_scores: Any) -> Any:
    import torch

    if positive_scores.shape != negative_scores.shape:
        raise ValueError("Positive and negative score shapes must match")
    return torch.nn.functional.softplus(-(positive_scores - negative_scores)).mean()


def _empirical_cdf(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def _build_model(input_dim: int, config: dict[str, Any], seed: int) -> Any:
    import torch

    torch.manual_seed(int(seed))
    layers: list[Any] = []
    previous = int(input_dim) + 1
    for width in map(int, config["hidden_dims"]):
        linear = torch.nn.Linear(previous, width)
        torch.nn.init.kaiming_uniform_(linear.weight)
        layers.extend(
            [
                linear,
                torch.nn.BatchNorm1d(width),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(config["dropout"])),
            ]
        )
        previous = width
    output = torch.nn.Linear(previous, 1)
    torch.nn.init.kaiming_uniform_(output.weight)
    layers.append(output)
    return torch.nn.Sequential(*layers)


def _with_budget(features: Any, budget: float) -> Any:
    import torch

    budget_column = torch.full(
        (len(features), 1),
        float(budget),
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, budget_column], dim=1)


def _training_pairs(
    rewards: np.ndarray,
    budgets: Sequence[float],
    pair_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives, negatives, pair_budgets = [], [], []
    for offset, budget in enumerate(map(float, budgets)):
        positive, negative = budget_boundary_pairs(
            rewards,
            budget=budget,
            pair_count=pair_count,
            seed=int(seed) + offset * 1_000_003,
        )
        positives.append(positive)
        negatives.append(negative)
        pair_budgets.append(np.full(pair_count, budget, dtype=np.float32))
    positive = np.concatenate(positives)
    negative = np.concatenate(negatives)
    budget_values = np.concatenate(pair_budgets)
    rng = np.random.default_rng(int(seed) + 97)
    order = rng.permutation(len(positive))
    return positive[order], negative[order], budget_values[order]


def _train_one_epoch(
    model: Any,
    features: Any,
    cdf: Any,
    rewards: np.ndarray,
    budgets: Sequence[float],
    config: dict[str, Any],
    *,
    seed: int,
    optimizer: Any,
) -> None:
    import torch

    batch_size = int(config["batch_size"])
    pair_count = int(config["pairs_per_budget"])
    pointwise_weight = float(config["pointwise_weight"])
    positive, negative, pair_budgets = _training_pairs(
        rewards, budgets, pair_count, int(seed)
    )
    model.train()
    for start in range(0, len(positive), batch_size):
        stop = min(start + batch_size, len(positive))
        if stop - start < 2:
            continue
        positive_index = torch.from_numpy(positive[start:stop])
        negative_index = torch.from_numpy(negative[start:stop])
        budget = torch.from_numpy(pair_budgets[start:stop])
        positive_x = torch.cat([features[positive_index], budget[:, None]], dim=1)
        negative_x = torch.cat([features[negative_index], budget[:, None]], dim=1)
        positive_score = model(positive_x).squeeze(1)
        negative_score = model(negative_x).squeeze(1)
        pair_loss = pairwise_logistic_loss(positive_score, negative_score)
        point_loss = 0.5 * (
            (positive_score - cdf[positive_index]).square().mean()
            + (negative_score - cdf[negative_index]).square().mean()
        )
        loss = pair_loss + pointwise_weight * point_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _train_epochs(
    model: Any,
    features: np.ndarray,
    rewards: np.ndarray,
    budgets: Sequence[float],
    config: dict[str, Any],
    *,
    epochs: int,
    seed: int,
) -> None:
    import torch

    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    cdf = torch.from_numpy(_empirical_cdf(rewards).astype(np.float32))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(map(int, config["milestones"])),
        gamma=float(config["gamma"]),
    )
    for epoch in range(int(epochs)):
        _train_one_epoch(
            model,
            x,
            cdf,
            rewards,
            budgets,
            config,
            seed=int(seed) + epoch * 10_007,
            optimizer=optimizer,
        )
        scheduler.step()


def _ndcg_at_budgets(
    scores: np.ndarray, rewards: np.ndarray, budgets: Sequence[float]
) -> float:
    relevance = _empirical_cdf(np.asarray(rewards, dtype=np.float64))
    values = []
    for row, budget in zip(np.asarray(scores), budgets):
        count = _top_count(len(relevance), float(budget))
        discount = 1.0 / np.log2(np.arange(count, dtype=np.float64) + 2.0)
        predicted = np.argsort(-row, kind="mergesort")[:count]
        ideal = np.argsort(-relevance, kind="mergesort")[:count]
        dcg = float(np.sum((np.exp2(relevance[predicted]) - 1.0) * discount))
        idcg = float(np.sum((np.exp2(relevance[ideal]) - 1.0) * discount))
        values.append(dcg / idcg if idcg > 0.0 else 0.0)
    return float(np.mean(values))


def _model_scores(model: Any, features: np.ndarray, budgets: Sequence[float]) -> np.ndarray:
    import torch

    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    model.eval()
    rows = []
    with torch.no_grad():
        for budget in map(float, budgets):
            rows.append(model(_with_budget(x, budget)).squeeze(1).numpy())
    return np.asarray(rows, dtype=np.float64)


def fit_budget_ranker(
    features: np.ndarray,
    rewards: np.ndarray,
    budgets: Sequence[float],
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Select an epoch on fit-internal NDCG, then retrain on all fit samples."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(rewards, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 10:
        raise ValueError("Feature and reward shapes are invalid")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Feature and reward values must be finite")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(x))
    validation_count = max(
        4, int(round(len(x) * float(config["fit_validation_fraction"])))
    )
    validation_count = min(validation_count, len(x) - 6)
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]

    import torch

    model = _build_model(x.shape[1], config, int(seed))
    training_x = torch.from_numpy(x[training_indices])
    training_cdf = torch.from_numpy(
        _empirical_cdf(y[training_indices]).astype(np.float32)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(map(int, config["milestones"])),
        gamma=float(config["gamma"]),
    )
    best_epoch, best_ndcg = 1, -math.inf
    for epoch in range(1, int(config["epochs"]) + 1):
        _train_one_epoch(
            model,
            training_x,
            training_cdf,
            y[training_indices],
            budgets,
            config,
            seed=int(seed) + epoch * 10_007,
            optimizer=optimizer,
        )
        scheduler.step()
        score = _ndcg_at_budgets(
            _model_scores(model, x[validation_indices], budgets),
            y[validation_indices],
            budgets,
        )
        if score > best_ndcg:
            best_epoch, best_ndcg = epoch, score

    final_model = _build_model(x.shape[1], config, int(seed) + 1)
    _train_epochs(
        final_model,
        x,
        y,
        budgets,
        config,
        epochs=best_epoch,
        seed=int(seed) + 1,
    )
    return {
        "state_dict": final_model.state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dims": list(map(int, config["hidden_dims"])),
        "dropout": float(config["dropout"]),
        "budgets": list(map(float, budgets)),
        "selected_epoch": int(best_epoch),
        "fit_validation_ndcg": float(best_ndcg),
        "seed": int(seed),
    }


def predict_budget_scores(
    checkpoint: dict[str, Any],
    features: np.ndarray,
    budgets: Sequence[float],
) -> np.ndarray:
    config = {
        "hidden_dims": checkpoint["hidden_dims"],
        "dropout": checkpoint["dropout"],
    }
    model = _build_model(int(checkpoint["input_dim"]), config, int(checkpoint["seed"]))
    model.load_state_dict(checkpoint["state_dict"])
    return _model_scores(model, features, budgets)


def calibration_gate(
    candidate_ap: dict[float, float],
    baseline_ap: dict[float, float],
    *,
    minimum_mean_ap_gain: float,
    maximum_budget_ap_loss: float,
) -> dict[str, Any]:
    budgets = sorted(set(map(float, candidate_ap)) | set(map(float, baseline_ap)))
    if set(map(float, candidate_ap)) != set(map(float, baseline_ap)):
        raise ValueError("Candidate and baseline budgets differ")
    gains = {
        budget: float(candidate_ap[budget]) - float(baseline_ap[budget])
        for budget in budgets
    }
    failed = [
        str(budget)
        for budget, gain in gains.items()
        if gain < -float(maximum_budget_ap_loss) - 1e-12
    ]
    mean_gain = float(np.mean(list(gains.values())))
    return {
        "passed": mean_gain >= float(minimum_mean_ap_gain) and not failed,
        "mean_ap_gain": mean_gain,
        "gain_by_budget": {str(key): value for key, value in gains.items()},
        "failed_budgets": failed,
    }
