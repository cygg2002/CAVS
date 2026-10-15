"""Pure contracts for set-valued frontier consensus."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def majority_mask(
    seed_membership: np.ndarray,
    availability: np.ndarray,
    *,
    minimum_votes: int,
    mandatory_columns: np.ndarray,
) -> np.ndarray:
    """Aggregate seed frontier sets while retaining mandatory fallbacks."""

    observed = np.asarray(seed_membership, dtype=bool)
    available = np.asarray(availability, dtype=bool)
    mandatory = np.asarray(mandatory_columns, dtype=bool)
    if observed.ndim != 3 or observed.shape[1:] != available.shape:
        raise ValueError("Seed membership must be [seed, image, action]")
    if mandatory.shape != available.shape:
        raise ValueError("Mandatory-action mask differs from availability")
    if not 1 <= int(minimum_votes) <= observed.shape[0]:
        raise ValueError("Vote threshold is outside the seed count")
    if np.any(observed & ~available[None, :, :]):
        raise ValueError("An unavailable action cannot be on a predicted frontier")
    keep = observed.sum(axis=0) >= int(minimum_votes)
    keep = (keep | mandatory) & available
    if not np.all(np.any(keep, axis=1)):
        raise RuntimeError("Consensus removed every action from an image")
    return keep


def prune_score_map(
    records: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    keep_mask: np.ndarray,
    *,
    roi_only: bool,
) -> dict[int, dict[str, float]]:
    matrix = np.asarray(scores, dtype=np.float64)
    keep = np.asarray(keep_mask, dtype=bool)
    if matrix.shape != keep.shape or matrix.shape[0] != len(records):
        raise ValueError("Records, values, and keep mask differ")
    output: dict[int, dict[str, float]] = {}
    for row, record in enumerate(records):
        values: dict[str, float] = {}
        for column, action in enumerate(record["actions"]):
            name = str(action["action_id"])
            if name == "edge":
                continue
            retained = bool(keep[row, column]) if (not roi_only or name == "roi") else True
            if retained:
                values[name] = float(matrix[row, column])
        output[int(record["image_id"])] = values
    return output


def screen_gate(
    baseline: Mapping[tuple[str, float], float],
    candidate: Mapping[tuple[str, float], float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate work points differ")
    deltas = {key: float(candidate[key]) - float(baseline[key]) for key in baseline}
    byte = [value for (mode, _), value in deltas.items() if mode == "fixed_byte"]
    call = [value for (mode, _), value in deltas.items() if mode == "fixed_call"]
    gain_points = sum(value >= float(config["minimum_fixed_byte_gain"]) for value in byte)
    passed = (
        gain_points >= int(config["required_fixed_byte_gain_points"])
        and all(value >= -float(config["maximum_fixed_byte_loss"]) for value in byte)
        and all(value >= -float(config["maximum_fixed_call_loss"]) for value in call)
    )
    return {
        "passed": bool(passed),
        "fixed_byte_gain_points": int(gain_points),
        "deltas": {f"{mode}@{budget:g}": value for (mode, budget), value in deltas.items()},
    }
