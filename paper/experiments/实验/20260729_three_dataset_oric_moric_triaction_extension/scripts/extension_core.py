"""Pure contracts for the isolated ORIC/MORIC TriAction extension."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable


RANKERS = ("ORIC-MLP", "MORIC-MLP")
POLICIES = ("CostEligible-AllROI", "GBDT-Hierarchical")
MODES = ("fixed_call", "online", "fixed_byte")


def expected_result_rows(protocol: dict[str, Any]) -> int:
    work_points = sum(len(protocol["work_points"][mode]) for mode in MODES)
    return len(protocol["rankers"]) * len(protocol["payload_policies"]) * work_points


def validate_protocol(protocol: dict[str, Any]) -> None:
    if tuple(map(str, protocol["datasets"])) != ("coco", "ena24", "oregon"):
        raise ValueError("The extension requires the three frozen datasets")
    if tuple(map(str, protocol["rankers"])) != RANKERS:
        raise ValueError("Only frozen ORIC-MLP and MORIC-MLP are in scope")
    if tuple(map(str, protocol["payload_policies"])) != POLICIES:
        raise ValueError("The two requested TriAction policies changed")
    if tuple(protocol["work_points"]) != MODES:
        raise ValueError("Budget modes or their ordering changed")
    if tuple(map(str, protocol["control_payload_policies"])) != (
        "Original",
        "SharedAction",
    ):
        raise ValueError("Control reproduction must use Original and SharedAction")
    if protocol.get("external_rankers_retrained") is not False:
        raise ValueError("External Rankers must remain frozen")
    if protocol.get("gbdt_thresholds_reselected") is not False:
        raise ValueError("The shared GBDT thresholds must remain frozen")
    if protocol.get("gbdt_threshold_source") != "frozen_shared_8ranker_calibration":
        raise ValueError("Unexpected GBDT threshold source")
    if protocol.get("fixed_byte_lambda_source") != (
        "calibration_only_per_ranker_policy_budget"
    ):
        raise ValueError("Fixed-byte lambda calibration semantics changed")
    if protocol.get("final_used_for_selection") is not False:
        raise ValueError("Final data must remain evaluation-only")
    expected = expected_result_rows(protocol)
    if int(protocol["expected_rows_per_dataset"]) != expected:
        raise ValueError(f"Expected result count drifted: {expected}")


def result_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["method"]),
        str(row["payload_policy"]),
        str(row["mode"]),
        float(row["nominal_budget"]),
    )


def expected_result_keys(
    protocol: dict[str, Any],
) -> set[tuple[str, str, str, float]]:
    return {
        (str(method), str(policy), str(mode), float(budget))
        for method in protocol["rankers"]
        for policy in protocol["payload_policies"]
        for mode, budgets in protocol["work_points"].items()
        for budget in budgets
    }


def assert_result_matrix_complete(
    rows: Iterable[dict[str, Any]], protocol: dict[str, Any]
) -> None:
    observed = [result_key(row) for row in rows]
    counts = Counter(observed)
    expected = expected_result_keys(protocol)
    unique = set(observed)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if unique != expected or duplicates or len(observed) != len(expected):
        raise RuntimeError(
            "Extension result matrix is incomplete or duplicated: "
            f"missing={sorted(expected - unique)} extra={sorted(unique - expected)} "
            f"duplicates={duplicates}"
        )


def assert_same_membership(selections: dict[str, dict[int, str]]) -> None:
    expected: set[int] | None = None
    for policy, actions in selections.items():
        observed = set(map(int, actions))
        if expected is None:
            expected = observed
        elif observed != expected:
            raise RuntimeError(
                f"Payload policy {policy} changed Fixed-call/Online membership"
            )


def frozen_policy_thresholds(
    choices: dict[str, Any],
) -> dict[str, dict[str, float]]:
    shared = choices.get("shared")
    if not isinstance(shared, dict):
        raise ValueError("Shared frozen threshold choices are missing")
    thresholds: dict[str, dict[str, float]] = {}
    for mode in MODES:
        value = float(shared[mode]["threshold"])
        if math.isnan(value):
            raise ValueError(f"Frozen {mode} threshold is NaN")
        thresholds[mode] = {
            "CostEligible-AllROI": float("-inf"),
            "GBDT-Hierarchical": value,
        }
    return thresholds


def assert_control_reproduction(
    actual_rows: Iterable[dict[str, Any]],
    expected_rows: Iterable[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    metrics = (
        "official_map50_95",
        "request_kb_per_image",
        "upload_ratio",
        "full_ratio",
        "q90_ratio",
        "roi_ratio",
        "selected_images",
    )
    expected = {
        result_key(row): row
        for row in expected_rows
        if str(row["method"]) in RANKERS
        and str(row["payload_policy"]) in {"Original", "SharedAction"}
        and str(row["mode"]) in MODES
    }
    actual_list = list(actual_rows)
    actual = {result_key(row): row for row in actual_list}
    if len(actual) != len(actual_list) or set(actual) != set(expected):
        raise RuntimeError(
            "Control reproduction keys changed: "
            f"missing={sorted(set(expected) - set(actual))} "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    worst = 0.0
    for key, expected_row in expected.items():
        for metric in metrics:
            difference = abs(float(actual[key][metric]) - float(expected_row[metric]))
            worst = max(worst, difference)
            if difference > float(tolerance):
                raise RuntimeError(
                    f"External control drift at {key} for {metric}: {difference}"
                )
    return {"passed": True, "rows": len(actual), "worst_difference": worst}
