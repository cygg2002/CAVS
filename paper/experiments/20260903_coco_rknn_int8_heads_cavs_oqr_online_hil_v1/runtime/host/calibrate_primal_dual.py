#!/usr/bin/env python3
"""Freeze a projected primal-dual byte controller on calibration only."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from calibrate_controls import _validated_bundle, file_sha256
from online_control import PrimalDualByteController


CONTROL_SCHEMA = "cavs_online_primal_dual_control_v1"
CONTROLLER_ID = "projected_primal_dual"


def _finite_grid(values: Sequence[float]) -> list[float]:
    grid = sorted(set(float(value) for value in values))
    if not grid or any(not math.isfinite(value) or value < 0 for value in grid):
        raise ValueError("step_size_grid must contain finite non-negative values")
    return grid


def _digest(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


def _replay(
    records: Sequence[Mapping[str, Any]],
    *,
    actions: Sequence[str],
    budget: Fraction,
    initial_price: float,
    step_size: float,
    minimum_price: float,
    maximum_price: float,
    median_original_bytes: float,
) -> dict[str, Any]:
    controller = PrimalDualByteController(
        budget=budget,
        initial_price=initial_price,
        step_size=step_size,
        minimum_price=minimum_price,
        maximum_price=maximum_price,
        median_original_bytes=median_original_bytes,
        action_ids=actions,
    )
    objective_value = 0.0
    maximum_violation = 0
    counts: Counter[str] = Counter()
    for record in records:
        decision = controller.decide(
            int(record["image_id"]),
            record["payload_bytes"],
            record["scores"],
        )
        objective_value += float(record["objective_values"][decision.action])
        counts[decision.action] += 1
        earned_floor = (
            decision.cumulative_original_bytes * controller.budget_numerator
            // controller.budget_denominator
        )
        maximum_violation = max(maximum_violation, decision.spent_bytes - earned_floor)
    earned = float(budget) * controller.cumulative_original_bytes
    return {
        "step_size": float(step_size),
        "objective_value": float(objective_value),
        "spent_bytes": int(controller.spent_bytes),
        "earned_bytes": float(earned),
        "capacity_utilization": float(controller.spent_bytes / earned) if earned else 0.0,
        "final_price": float(controller.current_price),
        "action_counts": {name: int(counts[name]) for name in sorted(counts)},
        "maximum_prefix_violation_bytes": int(maximum_violation),
    }


def calibrate_primal_dual_control(
    bundle: Mapping[str, Any],
    *,
    base_control: Mapping[str, Any],
    step_size_grid: Sequence[float],
    minimum_price: float,
    maximum_price: float,
    base_control_sha256: str,
) -> dict[str, Any]:
    """Select a step size without reading final records or labels."""

    if bundle.get("source_split") != "calibration" or bundle.get("final_used_for_tuning") is not False:
        raise ValueError("adaptive control may use calibration only")
    protocol_id, method, actions, records = _validated_bundle(bundle)
    if (
        base_control.get("schema") != "cavs_online_control_v1"
        or base_control.get("formal") is not True
        or base_control.get("source_split") != "calibration"
        or base_control.get("final_used_for_tuning") is not False
        or str(base_control.get("protocol_id")) != protocol_id
        or str(base_control.get("method")) != method
    ):
        raise ValueError("base control is not the matching frozen calibration control")
    try:
        budget = Fraction(
            int(base_control["budget_numerator"]),
            int(base_control["budget_denominator"]),
        )
        initial_price = float(base_control["lambda"])
        median = float(base_control["median_original_bytes"])
        lower, upper = float(minimum_price), float(maximum_price)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid base or adaptive control parameters") from error
    if budget <= 0 or budget > 1 or not all(map(math.isfinite, (initial_price, median, lower, upper))):
        raise ValueError("invalid base or adaptive control parameters")
    if median <= 0 or lower < 0 or upper < lower or not lower <= initial_price <= upper:
        raise ValueError("invalid adaptive price bounds")

    grid = _finite_grid(step_size_grid)
    trials = [
        _replay(
            records,
            actions=actions,
            budget=budget,
            initial_price=initial_price,
            step_size=step_size,
            minimum_price=lower,
            maximum_price=upper,
            median_original_bytes=median,
        )
        for step_size in grid
    ]
    chosen = max(
        trials,
        key=lambda row: (
            row["objective_value"],
            -row["spent_bytes"],
            -row["step_size"],
        ),
    )
    source_hashes = dict(base_control.get("source_hashes", {}))
    if source_hashes.get("policy") != bundle.get("source_hashes", {}).get("policy"):
        raise ValueError("policy hash differs between records and base control")
    source_hashes["base_control"] = _digest(
        base_control_sha256, field="base_control_sha256"
    )
    objective = bundle["calibration_objective"]
    return {
        **dict(base_control),
        "schema": CONTROL_SCHEMA,
        "controller": CONTROLLER_ID,
        "lambda_initial": initial_price,
        "lambda_minimum": lower,
        "lambda_maximum": upper,
        "step_size": float(chosen["step_size"]),
        "step_size_grid": grid,
        "dual_update": "projected_additive_per_frame_normalized",
        "buffer_frames": 0,
        "zero_lookahead": True,
        "global_plan": False,
        "calibration_objective": {
            "name": str(objective["name"]),
            "direction": "maximize",
            "value": float(chosen["objective_value"]),
        },
        "source_hashes": source_hashes,
        "diagnostics": {
            key: value
            for key, value in chosen.items()
            if key not in {"step_size", "objective_value"}
        },
        "calibration_trials": trials,
        "final_used_for_tuning": False,
    }


def _parse_grid(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--step-size-grid must be comma-separated numbers") from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--base-control", type=Path, required=True)
    parser.add_argument("--step-size-grid", required=True)
    parser.add_argument("--minimum-price", type=float, default=0.0)
    parser.add_argument("--maximum-price", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = json.loads(args.records.read_text(encoding="utf-8"))
    base = json.loads(args.base_control.read_text(encoding="utf-8"))
    control = calibrate_primal_dual_control(
        bundle,
        base_control=base,
        step_size_grid=_parse_grid(args.step_size_grid),
        minimum_price=args.minimum_price,
        maximum_price=args.maximum_price,
        base_control_sha256=file_sha256(args.base_control),
    )
    _write_json(args.output, control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
