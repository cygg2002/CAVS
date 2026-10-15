"""Pure contracts for the three-dataset TriAction evaluation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import numpy as np


CONTROLLED_RANKER = "Current-Controlled-32D"
POLICIES = ("SharedAction", "CostEligible-AllROI", "GBDT-Hierarchical")


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject protocol drift before any expensive evaluation starts."""
    rankers = tuple(map(str, protocol["rankers"]))
    constrained = tuple(map(str, protocol["constrained_rankers"]))
    policies = tuple(map(str, protocol["policies"]))
    if len(rankers) != 9 or len(set(rankers)) != 9:
        raise ValueError("The primary protocol requires nine unique Rankers")
    if len(constrained) != 8 or set(constrained) != set(rankers) - {"Random"}:
        raise ValueError("Exactly the eight non-random Rankers must be constrained")
    if policies != POLICIES:
        raise ValueError(f"Action policies changed: {policies}")
    if CONTROLLED_RANKER not in rankers or "Legacy-Frozen-32D" in rankers:
        raise ValueError("The primary matrix must use controlled 32D, not Legacy")
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
