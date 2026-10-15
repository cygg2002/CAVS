"""Pure protocol contracts for the frozen AFPD full-val diagnostic."""

from __future__ import annotations

from typing import Any, Mapping


ACTION_METHODS = ("Current-OQ", "Current-OQR", "AFPD-OQ", "AFPD-OQR")
OCCUPANCY_METHODS = ("OccupancyFree", "OccupancyAware")
SEEDS = (20260809, 20260810, 20260811)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "posthoc_full_val_paper_format_independent_retrain":
        raise ValueError("Post-hoc status changed")
    required_false = (
        "confirmatory",
        "final_is_hidden",
        "tuning_allowed",
        "final_used_for_selection",
        "gpu_required",
    )
    for key in required_false:
        if protocol.get(key) is not False:
            raise ValueError(f"Forbidden protocol flag: {key}")
    if protocol.get("calibration_gate_failed") is not True:
        raise ValueError("The failed calibration gate must be disclosed")
    if protocol.get("training_allowed") is not True:
        raise ValueError("Independent retraining must remain enabled")
    if protocol.get("independent_retrain") is not True:
        raise ValueError("Independent retraining changed")
    if protocol.get("source_checkpoint_reuse") is not False:
        raise ValueError("Source checkpoints must not be reused")
    if protocol.get("full_val_evaluation_unconditional") is not True:
        raise ValueError("Full-val evaluation must remain predeclared")
    if dict(protocol.get("roles", {})) != {
        "fit": 9678,
        "calibration": 571,
        "final": 5000,
    }:
        raise ValueError("Canonical role counts changed")
    if tuple(protocol.get("methods", ())) != (
        *ACTION_METHODS,
        "AFPD-OQR+OccupancyFree",
        "AFPD-OQR+OccupancyAware",
    ):
        raise ValueError("Reported methods changed")
    if tuple(protocol.get("modes", ())) != ("fixed_byte", "fixed_call"):
        raise ValueError("Evaluation modes changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("Budgets changed")
    if int(protocol.get("bootstrap_replicates", -1)) != 1000:
        raise ValueError("Bootstrap replicate count changed")
    if tuple(map(int, protocol.get("seeds", ()))) != SEEDS:
        raise ValueError("Seeds changed")
    if int(protocol.get("fold_count", -1)) != 5:
        raise ValueError("Five folds are required")


def checkpoint_relative_paths(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    validate_protocol(protocol)
    action = tuple(
        f"runs/calibration/action/jobs/{method}/seed_{seed}/checkpoint.pt"
        for method in ACTION_METHODS
        for seed in SEEDS
    )
    occupancy = tuple(
        f"runs/calibration/occupancy/jobs/{method}/seed_{seed}/checkpoint.pt"
        for method in OCCUPANCY_METHODS
        for seed in SEEDS
    )
    return action + occupancy


def expected_oof_action_jobs(protocol: Mapping[str, Any]) -> int:
    validate_protocol(protocol)
    return len(ACTION_METHODS) * len(SEEDS) * int(protocol["fold_count"])


def expected_oof_occupancy_jobs(protocol: Mapping[str, Any]) -> int:
    validate_protocol(protocol)
    return len(OCCUPANCY_METHODS) * len(SEEDS) * int(protocol["fold_count"])


def expected_fullfit_jobs(protocol: Mapping[str, Any]) -> int:
    return len(checkpoint_relative_paths(protocol))


def expected_result_rows(protocol: Mapping[str, Any]) -> int:
    validate_protocol(protocol)
    return len(protocol["methods"]) * len(protocol["modes"]) * len(
        protocol["budgets"]
    )


def expected_bootstrap_rows(protocol: Mapping[str, Any]) -> int:
    validate_protocol(protocol)
    return len(protocol["bootstrap_comparisons"]) * len(protocol["budgets"])


def assert_capacity(
    used_bytes: float, capacity_bytes: float, *, tolerance: float = 1e-6
) -> None:
    if float(used_bytes) > float(capacity_bytes) + float(tolerance):
        raise ValueError(f"Capacity exceeded: {used_bytes} > {capacity_bytes}")
