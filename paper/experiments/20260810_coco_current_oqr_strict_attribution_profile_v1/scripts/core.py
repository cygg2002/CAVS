"""Pure contracts for the Current-OQR attribution and CPU profile."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_SEEDS = (20260809, 20260810, 20260811)
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
EXPECTED_METHODS = ("Current-OQ", "Current-OQR")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "posthoc_final_reuse_attribution_and_cpu_profile":
        raise ValueError("Result status changed")
    if protocol.get("dataset") != "coco":
        raise ValueError("Only canonical COCO is allowed")
    if protocol.get("roles") != {"fit": 9678, "calibration": 571, "final": 5000}:
        raise ValueError("Canonical role counts changed")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("Budget grid changed")
    if tuple(protocol.get("current_methods", ())) != EXPECTED_METHODS:
        raise ValueError("Current method matrix changed")
    if tuple(map(int, protocol.get("seeds", ()))) != EXPECTED_SEEDS:
        raise ValueError("Frozen seed set changed")
    if tuple(protocol.get("action_spaces", {}).get("Current-OQ", ())) != (
        "edge",
        "original",
        "q90",
    ):
        raise ValueError("Current-OQ action space changed")
    if tuple(protocol.get("action_spaces", {}).get("Current-OQR", ())) != (
        "edge",
        "original",
        "q90",
        "roi",
    ):
        raise ValueError("Current-OQR action space changed")
    if protocol.get("fit_only_training") is not True:
        raise ValueError("Training provenance changed")
    for key in ("retraining_allowed", "tuning_allowed", "final_is_hidden", "final_used_for_selection"):
        if protocol.get(key) is not False:
            raise ValueError(f"{key} must remain false")
    profile = protocol.get("profile", {})
    if int(profile.get("torch_threads", 0)) != 1:
        raise ValueError("CPU profile must use one PyTorch thread")
    if min(
        int(profile.get("warmup_iterations", 0)),
        int(profile.get("batch1_iterations", 0)),
        int(profile.get("batched_iterations", 0)),
        int(profile.get("scheduler_iterations", 0)),
    ) <= 0:
        raise ValueError("Profile iteration counts must be positive")


def ensemble_scores(
    values: Mapping[int, Any], seeds: Sequence[int] = EXPECTED_SEEDS
) -> np.ndarray:
    expected = tuple(map(int, seeds))
    if tuple(sorted(map(int, values))) != tuple(sorted(expected)):
        raise ValueError("Ensemble requires exactly the frozen seeds")
    arrays = [np.asarray(values[seed], dtype=np.float64) for seed in expected]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Seed score shapes differ")
    output = np.mean(arrays, axis=0)
    if not np.isfinite(output).all():
        raise ValueError("Ensemble scores contain non-finite values")
    return output


def original_only_selection(
    image_ids: Sequence[int], scores: Mapping[int, float], *, budget: float
) -> dict[int, str]:
    ids = list(map(int, image_ids))
    if len(ids) != len(set(ids)) or set(ids) != set(map(int, scores)):
        raise ValueError("Image IDs and Original scores must align one-to-one")
    count = int(math.floor(len(ids) * float(budget) + 0.5))
    if not 0 <= count <= len(ids):
        raise ValueError("Fixed-call budget is outside [0, 1]")
    ordered = sorted(ids, key=lambda image_id: (-float(scores[image_id]), image_id))
    return {image_id: "original" for image_id in ordered[:count]}


def latency_summary_ns(values: Sequence[int]) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Latency samples must be a finite non-empty vector")
    if np.any(samples < 0.0):
        raise ValueError("Latency cannot be negative")
    return {
        "iterations": int(len(samples)),
        "mean_ms": float(samples.mean() / 1e6),
        "median_ms": float(np.median(samples) / 1e6),
        "p95_ms": float(np.percentile(samples, 95) / 1e6),
        "minimum_ms": float(samples.min() / 1e6),
        "maximum_ms": float(samples.max() / 1e6),
    }


def current_linear_flops(
    *,
    max_boxes: int,
    action_count: int,
    class_embedding_dim: int,
    box_hidden_dim: int,
    action_hidden_dim: int,
    joint_hidden_dim: int,
    scene_dim: int,
    action_dim: int,
    multiply_add_flops: int,
) -> dict[str, int | float]:
    values = (
        max_boxes,
        action_count,
        class_embedding_dim,
        box_hidden_dim,
        action_hidden_dim,
        joint_hidden_dim,
        scene_dim,
        action_dim,
        multiply_add_flops,
    )
    if min(map(int, values)) <= 0:
        raise ValueError("FLOP dimensions must be positive")
    box_macs = int(max_boxes) * (
        (6 + int(class_embedding_dim)) * int(box_hidden_dim)
        + int(box_hidden_dim) * int(box_hidden_dim)
    )
    scene_macs = int(scene_dim) * int(box_hidden_dim)
    image_macs = 3 * int(box_hidden_dim) * int(joint_hidden_dim)
    action_macs = int(action_count) * (
        (int(action_dim) + 8 + 1) * int(action_hidden_dim)
        + int(action_hidden_dim) * int(action_hidden_dim)
    )
    head_macs = int(action_count) * (
        (int(joint_hidden_dim) + int(action_hidden_dim)) * int(joint_hidden_dim)
        + int(joint_hidden_dim)
    )
    total_macs = box_macs + scene_macs + image_macs + action_macs + head_macs
    total_flops = total_macs * int(multiply_add_flops)
    return {
        "box_encoder_macs": box_macs,
        "scene_encoder_macs": scene_macs,
        "image_encoder_macs": image_macs,
        "action_encoder_macs": action_macs,
        "head_macs": head_macs,
        "linear_macs": total_macs,
        "linear_flops": total_flops,
        "linear_gflops": total_flops / 1e9,
    }


def parameter_count(state: Mapping[str, Any]) -> int:
    total = 0
    for value in state.values():
        if not hasattr(value, "numel"):
            raise ValueError("State mapping contains a non-tensor value")
        total += int(value.numel())
    if total <= 0:
        raise ValueError("Model has no parameters")
    return total
