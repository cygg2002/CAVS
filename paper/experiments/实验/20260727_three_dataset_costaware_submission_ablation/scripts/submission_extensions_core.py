"""Pure helpers for the submission-only bootstrap and Random diagnostics."""

from __future__ import annotations

import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable, Mapping, Sequence


def validated_parallel_workers(workers: int) -> int:
    value = int(workers)
    if value < 1:
        raise ValueError("Parallel worker count must be positive")
    return value


def raw_checkpoint_was_written(
    *, checkpoint_exists: bool, identical_action_shortcut: bool
) -> bool:
    if checkpoint_exists:
        return True
    if identical_action_shortcut:
        return False
    raise FileNotFoundError("Completed bootstrap raw checkpoint is missing")


def select_unique_workpoint(
    comparisons: Sequence[Mapping[str, Any]], *, mode: str, budget: float
) -> dict[str, Any]:
    matches = [
        comparison
        for comparison in comparisons
        if str(comparison["mode"]) == str(mode)
        and math.isclose(
            float(comparison["budget"]), float(budget), rel_tol=0.0, abs_tol=1e-12
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one bootstrap work point for {mode}-{budget:.2f}; "
            f"found {len(matches)}"
        )
    return dict(matches[0])


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("Quantile requires at least one value")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def gate_passes(
    *,
    mode: str,
    ap_delta_ci_low: float,
    kb_delta_ci_high: float,
    ap_tolerance: float,
) -> bool:
    if mode not in {"fixed_call", "online", "fixed_byte"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if math.isnan(float(ap_delta_ci_low)):
        return False
    ap_passed = float(ap_delta_ci_low) >= -float(ap_tolerance)
    if mode == "fixed_byte":
        return ap_passed
    return ap_passed and not math.isnan(float(kb_delta_ci_high)) and float(
        kb_delta_ci_high
    ) <= 0.0


def summarize_bootstrap_prefixes(
    *,
    ap_deltas: Sequence[float],
    kb_deltas: Sequence[float],
    checkpoints: Iterable[int],
    mode: str,
    ap_tolerance: float,
) -> list[dict[str, Any]]:
    if len(ap_deltas) != len(kb_deltas):
        raise ValueError("AP and request-cost bootstrap arrays differ in length")
    rows = []
    for replicates in sorted(set(map(int, checkpoints))):
        if replicates <= 0:
            raise ValueError("Bootstrap checkpoints must be positive")
        if replicates > len(ap_deltas):
            raise ValueError(
                f"Checkpoint {replicates} exceeds available replicates {len(ap_deltas)}"
            )
        ap = list(map(float, ap_deltas[:replicates]))
        kb = list(map(float, kb_deltas[:replicates]))
        ap_low = _quantile(ap, 0.025)
        ap_high = _quantile(ap, 0.975)
        kb_low = _quantile(kb, 0.025)
        kb_high = _quantile(kb, 0.975)
        rows.append(
            {
                "replicates": replicates,
                "official_map50_95_delta_mean": sum(ap) / replicates,
                "official_map50_95_delta_ci_low": ap_low,
                "official_map50_95_delta_ci_high": ap_high,
                "official_map50_95_probability_gt_zero": sum(
                    value > 0.0 for value in ap
                )
                / replicates,
                "request_kb_per_image_delta_mean": sum(kb) / replicates,
                "request_kb_per_image_delta_ci_low": kb_low,
                "request_kb_per_image_delta_ci_high": kb_high,
                "request_cost_gate_applied": mode in {"fixed_call", "online"},
                "gate_passed": gate_passes(
                    mode=mode,
                    ap_delta_ci_low=ap_low,
                    kb_delta_ci_high=kb_high,
                    ap_tolerance=ap_tolerance,
                ),
            }
        )
    return rows


def calibrated_online_threshold(
    calibration_scores: Sequence[float], target: float
) -> float:
    scores = sorted(map(float, calibration_scores), reverse=True)
    if not scores:
        raise ValueError("Online calibration requires scores")
    count = int(round(len(scores) * float(target)))
    if count <= 0:
        return math.inf
    if count >= len(scores):
        return -math.inf
    return (scores[count - 1] + scores[count]) / 2.0


def select_online_actions(
    samples: Sequence[Mapping[str, Any]],
    *,
    scores: Sequence[float],
    action_map: Mapping[int, str],
    threshold: float,
) -> dict[int, str]:
    if len(samples) != len(scores):
        raise ValueError("Online samples and scores differ in length")
    return {
        int(sample["image_id"]): str(action_map[int(sample["image_id"])])
        for sample, score in zip(samples, scores)
        if float(score) >= float(threshold)
    }


def random_byte_actions(
    samples: Sequence[Mapping[str, Any]],
    *,
    scores: Sequence[float],
    action_map: Mapping[int, str],
    budget: float,
) -> dict[int, str]:
    if len(samples) != len(scores):
        raise ValueError("Fixed-byte samples and scores differ in length")
    capacity = float(budget) * sum(
        float(sample["payload_bytes"]["original"]) for sample in samples
    )
    selected: dict[int, str] = {}
    used = 0.0
    for index in sorted(range(len(scores)), key=lambda value: (-float(scores[value]), value)):
        sample = samples[index]
        image_id = int(sample["image_id"])
        action = str(action_map[image_id])
        payload = float(sample["payload_bytes"][action])
        if used + payload <= capacity + 1e-9:
            selected[image_id] = action
            used += payload
    return selected


def summarize_random_band(
    rows: Sequence[Mapping[str, Any]], *, metrics: Iterable[str]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Random band requires at least one seed row")
    summary: dict[str, Any] = {"seeds": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = sum(values) / len(values)
        summary[f"{metric}_ci_low"] = _quantile(values, 0.025)
        summary[f"{metric}_ci_high"] = _quantile(values, 0.975)
        summary[f"{metric}_min"] = min(values)
        summary[f"{metric}_max"] = max(values)
    return summary


def runtime_manifest() -> dict[str, Any]:
    packages = {}
    for distribution in ("numpy", "scikit-learn", "faster-coco-eval", "Pillow"):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "packages": packages,
    }
