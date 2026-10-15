"""Calibration and causal replay for the state-price × envelope factorial."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math
import time
from typing import Any, Mapping, Sequence

from factorial_control import (
    BudgetEnvelope,
    EmpiricalDemandCurve,
    ImmediateController,
)


CELL_NAMES = (
    "fixed__strict_prefix",
    "fixed__paced_burst",
    "state_aware__strict_prefix",
    "state_aware__paced_burst",
)


def _fraction_from_control(control: Mapping[str, Any]) -> Fraction:
    try:
        budget = Fraction(
            int(control["budget_numerator"]),
            int(control["budget_denominator"]),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("control budget is invalid") from error
    if budget <= 0 or budget > 1:
        raise ValueError("control budget must lie in (0, 1]")
    return budget


def _total_original_bytes(records: Sequence[Mapping[str, Any]]) -> int:
    try:
        total = sum(int(record["payload_bytes"]["original"]) for record in records)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("records lack valid Original byte costs") from error
    if not records or total <= 0:
        raise ValueError("non-empty records with positive Original bytes are required")
    return total


def _controller(
    *,
    price_mode: str,
    action_ids: Sequence[str],
    median_original_bytes: float,
    fixed_price: float,
    demand_curve: EmpiricalDemandCurve,
    envelope: BudgetEnvelope,
) -> ImmediateController:
    common = {
        "action_ids": action_ids,
        "median_original_bytes": median_original_bytes,
        "envelope": envelope,
    }
    if price_mode == "fixed":
        return ImmediateController.fixed_price(price=fixed_price, **common)
    if price_mode == "state_aware":
        return ImmediateController.state_aware(demand_curve=demand_curve, **common)
    raise ValueError(f"unsupported price mode: {price_mode}")


def _run_cell(
    records: Sequence[Mapping[str, Any]],
    *,
    controller: ImmediateController,
    budget: Fraction,
    price_mode: str,
    burst_fraction: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    objective = 0.0
    maximum_violation = 0
    maximum_strict_prefix_excess = 0
    counts: Counter[str] = Counter()
    selections: dict[str, str] = {}
    prices = []
    started = time.perf_counter_ns()
    for record in records:
        decision = controller.decide(record)
        values = record.get("objective_values", record["scores"])
        objective += float(values[decision.action])
        counts[decision.action] += 1
        selections[str(decision.frame_id)] = decision.action
        prices.append(decision.price)
        maximum_violation = max(
            maximum_violation,
            decision.spent_bytes - decision.capacity_limit_bytes,
        )
        strict_limit = (
            decision.cumulative_original_bytes
            * budget.numerator
            // budget.denominator
        )
        maximum_strict_prefix_excess = max(
            maximum_strict_prefix_excess,
            decision.spent_bytes - strict_limit,
        )
    elapsed_ns = time.perf_counter_ns() - started
    capacity = controller.envelope.total_capacity
    return {
        "frames": len(records),
        "price_mode": price_mode,
        "envelope_mode": controller.envelope.mode,
        "burst_fraction": float(burst_fraction),
        "initial_burst_bytes": int(controller.envelope.initial_burst_bytes),
        "predicted_action_value_sum": float(objective),
        "spent_bytes": int(controller.spent_bytes),
        "total_capacity_bytes": int(capacity),
        "capacity_utilization": float(controller.spent_bytes / capacity),
        "action_counts": {name: int(counts[name]) for name in sorted(counts)},
        "maximum_envelope_violation_bytes": int(maximum_violation),
        "maximum_strict_prefix_excess_bytes": int(maximum_strict_prefix_excess),
        "price_minimum": float(min(prices)),
        "price_maximum": float(max(prices)),
        "price_mean": float(sum(prices) / len(prices)),
        "controller_elapsed_ns": int(elapsed_ns),
        "controller_ns_per_frame": float(elapsed_ns / len(records)),
    }, selections


def _run_configuration(
    records: Sequence[Mapping[str, Any]],
    *,
    action_ids: Sequence[str],
    budget: Fraction,
    fixed_price: float,
    median_original_bytes: float,
    demand_curve: EmpiricalDemandCurve,
    price_mode: str,
    envelope_mode: str,
    burst_fraction: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    total_original = _total_original_bytes(records)
    total_capacity = total_original * budget.numerator // budget.denominator
    if envelope_mode == "strict_prefix":
        envelope = BudgetEnvelope.strict_prefix(
            budget=budget,
            total_capacity=total_capacity,
            total_frames=len(records),
        )
        burst_fraction = 0.0
    elif envelope_mode == "paced_burst":
        burst = math.floor(total_capacity * float(burst_fraction))
        envelope = BudgetEnvelope.paced_burst(
            total_capacity=total_capacity,
            total_frames=len(records),
            initial_burst_bytes=burst,
        )
    else:
        raise ValueError(f"unsupported envelope mode: {envelope_mode}")
    controller = _controller(
        price_mode=price_mode,
        action_ids=action_ids,
        median_original_bytes=median_original_bytes,
        fixed_price=fixed_price,
        demand_curve=demand_curve,
        envelope=envelope,
    )
    return _run_cell(
        records,
        controller=controller,
        budget=budget,
        price_mode=price_mode,
        burst_fraction=burst_fraction,
    )


def _validated_burst_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(sorted(set(float(value) for value in values)))
    if not grid or any(not math.isfinite(value) or value < 0 or value > 1 for value in grid):
        raise ValueError("burst fractions must be finite values in [0, 1]")
    return grid


def calibrate_budget(
    records: Sequence[Mapping[str, Any]],
    *,
    action_ids: Sequence[str],
    base_control: Mapping[str, Any],
    demand_curve: EmpiricalDemandCurve,
    burst_fraction_grid: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        base_control.get("schema") != "cavs_online_control_v1"
        or base_control.get("formal") is not True
        or base_control.get("source_split") != "calibration"
        or base_control.get("final_used_for_tuning") is not False
    ):
        raise ValueError("base control is not calibration-frozen")
    budget = _fraction_from_control(base_control)
    try:
        fixed_price = float(base_control["lambda"])
        median = float(base_control["median_original_bytes"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("base price parameters are invalid") from error
    if not all(math.isfinite(value) for value in (fixed_price, median)) or fixed_price < 0 or median <= 0:
        raise ValueError("base price parameters are invalid")
    grid = _validated_burst_grid(burst_fraction_grid)
    cells: dict[str, dict[str, Any]] = {}
    trials: dict[str, list[dict[str, Any]]] = {}
    chosen_burst: dict[str, float] = {}
    for price_mode in ("fixed", "state_aware"):
        strict, _ = _run_configuration(
            records,
            action_ids=action_ids,
            budget=budget,
            fixed_price=fixed_price,
            median_original_bytes=median,
            demand_curve=demand_curve,
            price_mode=price_mode,
            envelope_mode="strict_prefix",
            burst_fraction=0.0,
        )
        cells[f"{price_mode}__strict_prefix"] = strict
        mode_trials = []
        for burst_fraction in grid:
            trial, _ = _run_configuration(
                records,
                action_ids=action_ids,
                budget=budget,
                fixed_price=fixed_price,
                median_original_bytes=median,
                demand_curve=demand_curve,
                price_mode=price_mode,
                envelope_mode="paced_burst",
                burst_fraction=burst_fraction,
            )
            mode_trials.append(trial)
        chosen = max(
            mode_trials,
            key=lambda row: (
                row["predicted_action_value_sum"],
                -row["burst_fraction"],
                -row["spent_bytes"],
            ),
        )
        cells[f"{price_mode}__paced_burst"] = chosen
        trials[price_mode] = mode_trials
        chosen_burst[price_mode] = float(chosen["burst_fraction"])
    control = {
        "schema": "cavs_online_stateprice_burst_control_v1",
        "source_split": "calibration",
        "formal": True,
        "final_used_for_tuning": False,
        "zero_lookahead": True,
        "global_plan": False,
        "buffer_frames": 0,
        "budget_numerator": budget.numerator,
        "budget_denominator": budget.denominator,
        "lambda_fixed": fixed_price,
        "median_original_bytes": median,
        "burst_fraction_grid": list(grid),
        "burst_fraction_by_price": chosen_burst,
        "state_price": "empirical_inverse_demand_remaining_capacity_per_frame",
        "calibration_frames": len(records),
    }
    return control, {
        "schema": "cavs_online_stateprice_burst_calibration_v1",
        "source_split": "calibration",
        "final_used_for_tuning": False,
        "cells": cells,
        "burst_trials": trials,
    }


def replay_factorial(
    records: Sequence[Mapping[str, Any]],
    *,
    action_ids: Sequence[str],
    frozen_control: Mapping[str, Any],
    demand_curve: EmpiricalDemandCurve,
    expected_fixed_prefix: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    valid = (
        frozen_control.get("schema") == "cavs_online_stateprice_burst_control_v1"
        and frozen_control.get("formal") is True
        and frozen_control.get("source_split") == "calibration"
        and frozen_control.get("final_used_for_tuning") is False
        and frozen_control.get("zero_lookahead") is True
        and frozen_control.get("global_plan") is False
        and frozen_control.get("buffer_frames") == 0
    )
    if not valid:
        raise ValueError("factorial control is not a frozen zero-buffer control")
    budget = _fraction_from_control(frozen_control)
    fixed_price = float(frozen_control["lambda_fixed"])
    median = float(frozen_control["median_original_bytes"])
    burst_by_price = frozen_control.get("burst_fraction_by_price")
    if not isinstance(burst_by_price, Mapping):
        raise ValueError("factorial control lacks selected burst fractions")
    cells: dict[str, dict[str, Any]] = {}
    selections: dict[str, dict[str, str]] = {}
    for price_mode in ("fixed", "state_aware"):
        for envelope_mode in ("strict_prefix", "paced_burst"):
            name = f"{price_mode}__{envelope_mode}"
            burst = 0.0 if envelope_mode == "strict_prefix" else float(burst_by_price[price_mode])
            cells[name], selections[name] = _run_configuration(
                records,
                action_ids=action_ids,
                budget=budget,
                fixed_price=fixed_price,
                median_original_bytes=median,
                demand_curve=demand_curve,
                price_mode=price_mode,
                envelope_mode=envelope_mode,
                burst_fraction=burst,
            )
    baseline = selections["fixed__strict_prefix"]
    if expected_fixed_prefix is not None and dict(expected_fixed_prefix) != baseline:
        differences = sum(
            expected_fixed_prefix.get(key) != action for key, action in baseline.items()
        )
        raise RuntimeError(
            f"fixed+strict replay differs from frozen baseline on {differences} frames"
        )
    for name, cell in cells.items():
        cell["changed_frames_vs_fixed_strict"] = sum(
            selections[name][key] != action for key, action in baseline.items()
        )
        if cell["maximum_envelope_violation_bytes"] != 0:
            raise RuntimeError(f"envelope violation in {name}")
        if cell["spent_bytes"] > cell["total_capacity_bytes"]:
            raise RuntimeError(f"final capacity exceeded in {name}")
    return {
        "schema": "cavs_online_stateprice_burst_final_replay_v1",
        "source_split": "final",
        "complete": True,
        "final_used_for_tuning": False,
        "zero_lookahead": True,
        "global_plan": False,
        "buffer_frames": 0,
        "frames": len(records),
        "cells": cells,
    }, selections
