"""Pure contracts for the three-dataset TriAction evaluation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import numpy as np


CONTROLLED_RANKER = "Current-Controlled-32D"
LEGACY_STYLE_RANKER = "Legacy-Style-Rebuilt-32D"
POLICIES = ("SharedAction", "CostEligible-AllROI", "GBDT-Hierarchical")


def apply_extra_32d_evaluation_metadata(
    row: dict[str, Any], *, method: str, action_params: int
) -> dict[str, Any]:
    """Restore the requested 32D Ranker name after SemanticRank evaluation reuse."""
    if method not in (CONTROLLED_RANKER, LEGACY_STYLE_RANKER):
        raise ValueError(f"Unsupported extra 32D Ranker: {method}")
    row["method"] = method
    row["selector_ms_per_image"] = 0.08
    row["selector_params"] = int(action_params) + 32
    return row


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject protocol drift before any expensive evaluation starts."""
    rankers = tuple(map(str, protocol["rankers"]))
    constrained = tuple(map(str, protocol["constrained_rankers"]))
    policies = tuple(map(str, protocol["policies"]))
    if len(rankers) != 10 or len(set(rankers)) != 10:
        raise ValueError("The primary protocol requires ten unique Rankers")
    if len(constrained) != 9 or set(constrained) != set(rankers) - {"Random"}:
        raise ValueError("Exactly the nine non-random Rankers must be constrained")
    if policies != POLICIES:
        raise ValueError(f"Action policies changed: {policies}")
    if (
        CONTROLLED_RANKER not in rankers
        or LEGACY_STYLE_RANKER not in rankers
        or "Legacy-Frozen-32D" in rankers
    ):
        raise ValueError("The primary matrix must include controlled/rebuilt 32D, not frozen Legacy")
    if protocol.get("final_used_for_selection") is not False:
        raise ValueError("Final data must never participate in selection")
    expected = protocol.get("expected_rows_per_dataset", {})
    observed = expected_matrix_counts(protocol)
    if expected != observed:
        raise ValueError(f"Frozen matrix counts changed: {observed} != {expected}")


def expected_matrix_counts(protocol: dict[str, Any]) -> dict[str, int]:
    constrained = len(protocol["constrained_rankers"])
    policies = len(protocol["policies"])
    fixed_call = len(protocol["fixed_call_budgets"])
    online = len(protocol["online_targets"])
    fixed_byte = len(protocol["fixed_byte_budgets"])
    return {
        "main": constrained * policies * (fixed_call + online + fixed_byte)
        + policies * fixed_call,
        "per_ranker_ablation": constrained * (fixed_call + online + fixed_byte),
        "random_seed_rows": int(protocol["random_seeds"]) * policies * fixed_call,
        "bootstrap_comparisons": constrained
        * len(protocol["primary_call_budgets"]),
    }


def pca32_matrix(samples: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.asarray([sample["pca32"] for sample in samples], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 32:
        raise ValueError("Controlled SemanticRank requires exactly 32 dimensions")
    if not np.isfinite(matrix).all():
        raise ValueError("Controlled SemanticRank features must be finite")
    return matrix


def _legacy_style_budget_pairs(
    targets: np.ndarray,
    budgets: tuple[float, ...],
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Legacy-style targets must be finite one-dimensional values")
    if not budgets or any(not 0.0 < float(budget) < 1.0 for budget in budgets):
        raise ValueError("Legacy-style pair budgets must be between zero and one")
    rng = np.random.default_rng(int(seed))
    order = np.argsort(-values, kind="mergesort")
    per_budget = max(1, int(max_pairs) // len(budgets))
    high_parts, low_parts, weight_parts = [], [], []
    for budget in budgets:
        count = min(len(values) - 1, max(1, int(math.floor(len(values) * float(budget) + 0.5))))
        high = np.repeat(order[:count], len(values) - count)
        low = np.tile(order[count:], count)
        valid = values[high] > values[low] + 1e-12
        high, low = high[valid], low[valid]
        if not len(high):
            continue
        if len(high) > per_budget:
            chosen = rng.choice(len(high), size=per_budget, replace=False)
            high, low = high[chosen], low[chosen]
        gap = values[high] - values[low]
        scale = max(1e-9, float(np.median(gap)))
        high_parts.append(high)
        low_parts.append(low)
        weight_parts.append(np.clip(gap / scale, 0.25, 4.0) / np.sqrt(float(budget)))
    if not high_parts:
        raise ValueError("Legacy-style ranker found no valid budget-crossing pairs")
    return (
        np.concatenate(high_parts).astype(np.int64),
        np.concatenate(low_parts).astype(np.int64),
        np.concatenate(weight_parts).astype(np.float64),
    )


def fit_legacy_style_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    anchors: np.ndarray,
    *,
    budgets: tuple[float, ...],
    epochs: int,
    learning_rate: float,
    ridge: float,
    max_pairs: int,
    seed: int,
    score_scale: float,
) -> dict[str, Any]:
    """Fit the historical anchored 32D residual recipe on current fit-only data."""
    matrix = np.asarray(features, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    anchor = np.asarray(anchors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 32 or not np.isfinite(matrix).all():
        raise ValueError("Legacy-style ranker requires finite 32D features")
    if values.shape != (len(matrix),) or anchor.shape != (len(matrix),):
        raise ValueError("Legacy-style features, targets, and anchors must align")
    if not np.isfinite(values).all() or not np.isfinite(anchor).all():
        raise ValueError("Legacy-style targets and anchors must be finite")
    if not 0.0 < float(score_scale) <= 1.0:
        raise ValueError("Legacy-style score scale must be in (0, 1]")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (matrix - mean) / std
    high, low, pair_weight = _legacy_style_budget_pairs(
        values, budgets, int(max_pairs), int(seed)
    )
    differences = normalized[high] - normalized[low]
    anchor_delta = anchor[high] - anchor[low]
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    for _ in range(int(epochs)):
        margin = np.clip(anchor_delta + differences @ weights, -40.0, 40.0)
        miss_probability = 1.0 / (1.0 + np.exp(margin))
        gradient = -(differences.T @ (pair_weight * miss_probability)) / pair_weight.sum()
        gradient += float(ridge) * weights
        weights -= float(learning_rate) * gradient
    final_margin = np.clip(anchor_delta + differences @ weights, -40.0, 40.0)
    final_loss = float(
        np.sum(pair_weight * np.logaddexp(0.0, -final_margin)) / pair_weight.sum()
    )
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "score_scale": float(score_scale),
        "pairs": int(len(high)),
        "final_loss": final_loss,
        "weight_norm": float(np.linalg.norm(weights)),
    }


def predict_legacy_style_ranker(
    model: dict[str, Any], features: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    anchor = np.asarray(anchors, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(anchor), 32):
        raise ValueError("Legacy-style prediction features and anchors must align at 32D")
    if mean.shape != (32,) or std.shape != (32,) or weights.shape != (32,):
        raise ValueError("Legacy-style model must contain 32D normalization and weights")
    scores = anchor + float(model["score_scale"]) * (((matrix - mean) / std) @ weights)
    if not np.isfinite(scores).all():
        raise ValueError("Legacy-style prediction scores must be finite")
    return scores


def assert_compatible_controlled_features(samples: list[dict[str, Any]]) -> None:
    pca32_matrix(samples)
    for sample in samples:
        structured = np.asarray(sample["structured"], dtype=np.float64)
        pca = np.asarray(sample["pca32"], dtype=np.float64)
        current = np.asarray(sample["features"], dtype=np.float64)
        if structured.shape != (17,) or current.shape != (49,):
            raise RuntimeError("Current feature dimensions are not frozen at 17+32")
        if not np.array_equal(current, np.concatenate((structured, pca))):
            raise RuntimeError("Current feature ordering is not structured17+pca32")


def assert_canonical_rank_targets(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        target = float(sample["target"])
        expected = max(map(float, sample["gains"].values()))
        if abs(target - expected) > 1e-12:
            raise RuntimeError("Current rank target is not max frozen-action gain")


def overlay_roi_actions(
    base_actions: dict[int, str],
    roi_scores: dict[int, float],
    eligible_ids: set[int],
    threshold: float,
) -> dict[int, str]:
    """Overlay ROI at a frozen threshold without changing image membership."""
    return {
        int(image_id): (
            "roi"
            if int(image_id) in eligible_ids
            and float(roi_scores[int(image_id)]) >= float(threshold)
            else str(action)
        )
        for image_id, action in base_actions.items()
    }


def eligible_roi_ids(
    data: dict[str, Any], base_actions: dict[int, str]
) -> set[int]:
    """Require ROI to be cheaper than the actual frozen SharedAction base."""
    eligible = set()
    for sample in data["samples"]:
        image_id = int(sample["image_id"])
        base = str(base_actions[image_id])
        roi_bytes = sample["payload_bytes"].get("roi")
        base_bytes = sample["payload_bytes"].get(base)
        if (
            sample["available"].get("roi", False)
            and roi_bytes is not None
            and base_bytes is not None
            and float(roi_bytes) < float(base_bytes)
        ):
            eligible.add(image_id)
    return eligible


def assert_fixed_byte_gates_complete(
    points: Iterable[dict[str, Any]],
    rankers: Iterable[str],
    budgets: Iterable[float],
) -> None:
    keys = [
        (str(point["ranker"]), float(point["budget"]))
        for point in points
        if "lambda_value" in point
    ]
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    observed = set(keys)
    expected = {
        (str(ranker), float(budget)) for ranker in rankers for budget in budgets
    }
    if observed != expected or duplicates:
        raise RuntimeError(
            "Fixed-byte gate set is incomplete or duplicated: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)} "
            f"duplicate={duplicates}"
        )


def verify_reproduction_rows(
    actual_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    *,
    canonical_rankers: Iterable[str],
    metrics: Iterable[str],
    tolerance: float,
) -> dict[str, Any]:
    """Compare only canonical Rankers while allowing the new controlled row."""
    canonical = set(map(str, canonical_rankers))
    actual_items = [
        ((str(row["method"]), str(row["mode"]), float(row["nominal_budget"])), row)
        for row in actual_rows
        if row.get("action_policy") == "SharedAction"
        and str(row["method"]) in canonical
    ]
    expected_items = [
        ((str(row["method"]), str(row["mode"]), float(row["nominal_budget"])), row)
        for row in expected_rows
        if str(row.get("role")) == "final"
        and str(row["method"]) in canonical
        and str(row["mode"]) in {"fixed_call", "online", "fixed_byte"}
    ]
    actual_counts = Counter(key for key, _ in actual_items)
    expected_counts = Counter(key for key, _ in expected_items)
    duplicate_actual = sorted(key for key, count in actual_counts.items() if count != 1)
    duplicate_expected = sorted(key for key, count in expected_counts.items() if count != 1)
    if duplicate_actual or duplicate_expected:
        raise RuntimeError(
            "Canonical reproduction contains duplicate keys: "
            f"actual={duplicate_actual} expected={duplicate_expected}"
        )
    actual = dict(actual_items)
    expected = dict(expected_items)
    if set(actual) != set(expected):
        raise RuntimeError(
            "Canonical reproduction keys differ: "
            f"missing={sorted(set(expected) - set(actual))} "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    worst = 0.0
    for key in sorted(actual):
        for metric in metrics:
            difference = abs(float(actual[key][metric]) - float(expected[key][metric]))
            worst = max(worst, difference)
            if difference > float(tolerance):
                raise RuntimeError(
                    f"Canonical reproduction drift at {key} for {metric}: {difference}"
                )
    return {"passed": True, "rows": len(actual), "worst_difference": worst}


def assert_bootstrap_rows_complete(
    rows: Iterable[dict[str, Any]],
    *,
    rankers: Iterable[str],
    budgets: Iterable[float],
    comparison: str,
    replicates: int,
) -> None:
    values = list(rows)
    expected = {
        (str(ranker), float(budget)) for ranker in rankers for budget in budgets
    }
    keys = [(str(row["method"]), float(row["nominal_budget"])) for row in values]
    counts = Counter(keys)
    observed = set(keys)
    bad_metadata = [
        (row.get("comparison"), row.get("replicates"))
        for row in values
        if str(row.get("comparison")) != str(comparison)
        or int(row.get("replicates", -1)) != int(replicates)
    ]
    if (
        observed != expected
        or len(values) != len(expected)
        or any(count != 1 for count in counts.values())
        or bad_metadata
    ):
        raise RuntimeError(
            "Bootstrap grid is incomplete, duplicated, or mislabeled: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)} "
            f"rows={len(values)} expected_rows={len(expected)} bad_metadata={bad_metadata[:3]}"
        )


def serializable_threshold(value: float) -> str | float:
    if math.isinf(float(value)):
        return "+Infinity" if float(value) > 0 else "-Infinity"
    return float(value)
