"""Pure contracts for the frozen COCO AFPD paper extension."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


EXPECTED_ROLES = {"fit": 9678, "calibration": 571, "final": 5000}
EXPECTED_ACTION_SPACES = {
    "OQ": ("edge", "original", "q90"),
    "OQR": ("edge", "original", "q90", "roi"),
}
EXPECTED_ACTION_METHODS = (
    "Current-OQ",
    "Current-OQR",
    "AFPD-OQ",
    "AFPD-OQR",
)
EXPECTED_REPORTED_METHODS = (
    *EXPECTED_ACTION_METHODS,
    "AFPD-OQR+OccupancyFree",
    "AFPD-OQR+OccupancyAware",
)
EXPECTED_OCCUPANCY_METHODS = ("OccupancyFree", "OccupancyAware")
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
EXPECTED_MODES = ("fixed_byte", "fixed_call")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco_canonical_train9678_cal571_val2017_5000":
        raise ValueError("Unexpected COCO dataset contract")
    if dict(protocol.get("roles", {})) != EXPECTED_ROLES:
        raise ValueError("Canonical role counts changed")
    spaces = {
        str(name): tuple(actions)
        for name, actions in dict(protocol.get("action_spaces", {})).items()
    }
    if spaces != EXPECTED_ACTION_SPACES:
        raise ValueError("OQ/OQR action spaces changed")
    if tuple(protocol.get("action_methods", ())) != EXPECTED_ACTION_METHODS:
        raise ValueError("2x2 action-method matrix changed")
    if tuple(protocol.get("reported_methods", ())) != EXPECTED_REPORTED_METHODS:
        raise ValueError("Reported method matrix changed")
    if tuple(protocol.get("occupancy_methods", ())) != EXPECTED_OCCUPANCY_METHODS:
        raise ValueError("Occupancy ablations changed")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("Canonical budgets changed")
    if tuple(protocol.get("modes", ())) != EXPECTED_MODES:
        raise ValueError("Evaluation modes changed")
    if tuple(map(int, protocol.get("seeds", ()))) != (
        20260809,
        20260810,
        20260811,
    ):
        raise ValueError("Frozen seeds changed")
    if int(protocol.get("fold_count", -1)) != 5:
        raise ValueError("Five OOF folds are required")
    if protocol.get("primary_method") != "AFPD-OQR":
        raise ValueError("AFPD-OQR must remain primary")
    if protocol.get("primary_same_space_baseline") != "Current-OQR":
        raise ValueError("The primary comparison must be same-action-space")
    if protocol.get("occupancy_can_replace_primary") is not False:
        raise ValueError("Occupancy cannot replace the frozen primary")
    if protocol.get("fit_only_training") is not True:
        raise ValueError("Training must remain fit-only")
    for key in (
        "calibration_used_for_training_or_selection",
        "final_used_for_training_or_selection",
        "final_is_hidden",
        "detector_inference_required",
        "gpu_required",
    ):
        if protocol.get(key) is not False:
            raise ValueError(f"Forbidden protocol flag: {key}")
    if int(protocol.get("bootstrap", {}).get("replicates", -1)) != 1000:
        raise ValueError("Bootstrap must use exactly 1,000 replicates")


def expected_oof_action_rows(protocol: Mapping[str, Any]) -> int:
    return (
        len(protocol["action_methods"])
        * (len(protocol["seeds"]) + 1)
        * len(protocol["modes"])
        * len(protocol["budgets"])
    )


def expected_oof_occupancy_rows(protocol: Mapping[str, Any]) -> int:
    policies = 1 + len(protocol["occupancy_methods"])
    return (
        policies
        * (len(protocol["seeds"]) + 1)
        * len(protocol["modes"])
        * len(protocol["budgets"])
    )


def expected_role_rows(protocol: Mapping[str, Any]) -> int:
    return (
        len(protocol["reported_methods"])
        * len(protocol["modes"])
        * len(protocol["budgets"])
    )


def preflight_roles(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    roles = tuple(role for role in protocol["roles"] if role != "final")
    if roles != ("fit", "calibration"):
        raise ValueError("Preflight may inspect only fit and calibration")
    return roles


def assert_fixed_byte_capacity(
    used_bytes: float, capacity_bytes: float, *, tolerance: float = 1e-6
) -> None:
    if float(used_bytes) > float(capacity_bytes) + float(tolerance):
        raise ValueError(
            f"Fixed-byte capacity exceeded: {used_bytes} > {capacity_bytes}"
        )


def evaluate_delta_gate(
    work_points: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    fixed_byte = [row for row in work_points if row["mode"] == "fixed_byte"]
    fixed_call = [row for row in work_points if row["mode"] == "fixed_call"]
    if len(fixed_byte) != 3 or len(fixed_call) != 3:
        raise ValueError("A gate requires three Fixed-byte and three Fixed-call rows")
    gain = float(gate["minimum_fixed_byte_gain"])
    maximum_loss = float(gate["maximum_other_loss"])
    maximum_call_loss = float(gate["maximum_fixed_call_loss"])
    minimum_capacity = float(gate["minimum_capacity_utilization"])
    gain_points = sum(float(row["delta_ap"]) >= gain for row in fixed_byte)
    byte_noninferior = all(
        float(row["delta_ap"]) >= -maximum_loss for row in fixed_byte
    )
    call_noninferior = all(
        float(row["delta_ap"]) >= -maximum_call_loss for row in fixed_call
    )
    capacity_sufficient = all(
        float(row["capacity_utilization"]) >= minimum_capacity
        for row in fixed_byte
    )
    passed = (
        gain_points >= int(gate["required_fixed_byte_gain_points"])
        and byte_noninferior
        and call_noninferior
        and capacity_sufficient
    )
    return {
        "passed": passed,
        "fixed_byte_gain_points": gain_points,
        "fixed_byte_noninferior": byte_noninferior,
        "fixed_call_noninferior": call_noninferior,
        "capacity_sufficient": capacity_sufficient,
    }


def should_run_final(
    *,
    primary_calibration_gate: Mapping[str, Any],
    occupancy_results: Mapping[str, Mapping[str, Any]],
) -> bool:
    del occupancy_results
    return bool(primary_calibration_gate.get("passed", False))


def two_by_two_attribution(values: Mapping[str, float]) -> dict[str, float]:
    required = set(EXPECTED_ACTION_METHODS)
    if set(values) != required:
        raise ValueError(f"Attribution requires exactly {sorted(required)}")
    current_oq = float(values["Current-OQ"])
    current_oqr = float(values["Current-OQR"])
    afpd_oq = float(values["AFPD-OQ"])
    afpd_oqr = float(values["AFPD-OQR"])
    objective_oq = afpd_oq - current_oq
    objective_oqr = afpd_oqr - current_oqr
    return {
        "roi_effect_current": current_oqr - current_oq,
        "roi_effect_afpd": afpd_oqr - afpd_oq,
        "objective_effect_oq": objective_oq,
        "objective_effect_oqr": objective_oqr,
        "interaction": objective_oqr - objective_oq,
        "total_effect": afpd_oqr - current_oq,
    }
