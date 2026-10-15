"""Pure protocol contracts for the canonical COCO TriAction extension."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


METHODS = (
    "ORIC-AP50",
    "MORIC-AP50",
    "ORIC-AP5095",
    "MORIC-AP5095",
    "Current-BudgetSet",
    "SetValue",
    "DualHead",
)
EXTERNAL_METHODS = METHODS[:4]
POLICIES = ("SharedAction", "CostEligible-AllROI", "GBDT-Hierarchical")
MODES = ("fixed_call", "fixed_byte")
BUDGETS = (0.1, 0.25, 0.4)
ROLES = {"fit": 9678, "calibration": 571, "registered_final": 5000}


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("This extension is COCO-only")
    if protocol.get("result_status") != "posthoc_canonical_val2017_action_extension":
        raise ValueError("The already-accessed val2017 result status changed")
    if dict(protocol.get("roles", {})) != ROLES:
        raise ValueError("Canonical role counts changed")
    if tuple(map(str, protocol.get("methods", ()))) != METHODS:
        raise ValueError("The seven-ranker matrix changed")
    if tuple(map(str, protocol.get("external_methods", ()))) != EXTERNAL_METHODS:
        raise ValueError("External comparator set changed")
    if protocol.get("primary_candidate") != "DualHead":
        raise ValueError("DualHead must remain the primary candidate")
    if tuple(map(str, protocol.get("payload_policies", ()))) != POLICIES:
        raise ValueError("Frozen action-policy matrix changed")
    if tuple(map(str, protocol.get("actions", ()))) != (
        "edge",
        "original",
        "q90",
        "roi",
    ):
        raise ValueError("Original/Q90/ROI action set changed")
    work_points = protocol.get("work_points", {})
    if tuple(work_points) != MODES:
        raise ValueError("Evaluation modes or ordering changed")
    if any(tuple(map(float, work_points[mode])) != BUDGETS for mode in MODES):
        raise ValueError("Canonical 10/25/40 budget grid changed")
    for key in (
        "ranker_checkpoint_retrained",
        "action_models_retrained",
        "gbdt_thresholds_reselected",
        "final_used_for_selection",
    ):
        if protocol.get(key) is not False:
            raise ValueError(f"Frozen protocol flag changed: {key}")
    if protocol.get("fixed_byte_lambda_source") != (
        "calibration_only_per_ranker_policy_budget"
    ):
        raise ValueError("Fixed-byte calibration semantics changed")
    if int(protocol.get("expected_final_rows", -1)) != len(
        expected_result_keys(protocol)
    ):
        raise ValueError("Expected result count drifted")
    if protocol.get("source_readiness") != "frozen_checkpoint_and_point_selections":
        raise ValueError("Source point-artifact readiness contract changed")
    if dict(protocol.get("cpu_workers", {})) != {"point": 16, "bootstrap": 24}:
        raise ValueError("Frozen CPU worker allocation changed")
    runtime = protocol.get("runtime", {})
    if runtime.get("sklearn_version") != "1.7.1":
        raise ValueError("Frozen scikit-learn runtime version changed")
    if not str(runtime.get("sklearn_site_packages", "")).endswith(
        "/cameratrapai/lib/python3.10/site-packages"
    ):
        raise ValueError("Frozen scikit-learn runtime path changed")
    bootstrap = protocol.get("bootstrap", {})
    if int(bootstrap.get("replicates", -1)) != 1000:
        raise ValueError("Bootstrap replicate count changed")
    if int(bootstrap.get("comparisons", -1)) != 9:
        raise ValueError("Bootstrap comparison count changed")


def result_key(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["method"]),
        str(row["payload_policy"]),
        str(row["mode"]),
        float(row["nominal_budget"]),
    )


def expected_result_keys(
    protocol: Mapping[str, Any],
) -> set[tuple[str, str, str, float]]:
    return {
        (str(method), str(policy), str(mode), float(budget))
        for method in protocol["methods"]
        for policy in protocol["payload_policies"]
        for mode, budgets in protocol["work_points"].items()
        for budget in budgets
    }


def assert_result_matrix_complete(
    rows: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    observed = [result_key(row) for row in rows]
    counts = Counter(observed)
    expected = expected_result_keys(protocol)
    unique = set(observed)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if unique != expected or duplicates or len(observed) != len(expected):
        raise RuntimeError(
            "Result matrix is incomplete or duplicated: "
            f"missing={sorted(expected - unique)} extra={sorted(unique - expected)} "
            f"duplicates={duplicates}"
        )


def assert_fixed_call_membership(
    selections: Mapping[str, Mapping[int, str]],
) -> None:
    expected: set[int] | None = None
    for policy, actions in selections.items():
        observed = set(map(int, actions))
        if expected is None:
            expected = observed
        elif observed != expected:
            raise RuntimeError(
                f"Payload policy {policy} changed Fixed-call membership"
            )


def freeze_external_comparators(
    rows: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, str]:
    """Choose one external comparator per policy/fixed-byte budget."""
    external = set(map(str, protocol["external_methods"]))
    choices: dict[str, str] = {}
    for policy in protocol["payload_policies"]:
        for budget in map(float, protocol["work_points"]["fixed_byte"]):
            candidates = [
                row
                for row in rows
                if str(row["method"]) in external
                and str(row["payload_policy"]) == str(policy)
                and str(row["mode"]) == "fixed_byte"
                and abs(float(row["nominal_budget"]) - budget) < 1e-12
            ]
            if len(candidates) != len(external):
                raise RuntimeError(
                    f"Incomplete external comparator candidates for {policy}/{budget:g}"
                )
            selected = max(
                candidates,
                key=lambda row: (
                    float(row["official_map50_95"]),
                    -float(row["request_kb_per_image"]),
                    str(row["method"]),
                ),
            )
            choices[f"{policy}|{budget:g}"] = str(selected["method"])
    return choices


def validate_checkpoint_fingerprint(
    stored: Mapping[str, str], current: Mapping[str, str]
) -> dict[str, dict[str, str]]:
    """Allow only the documented post-training bootstrap-wrapper patch."""
    if set(stored) != set(current):
        raise RuntimeError("Checkpoint fingerprint keys changed")
    differences = {
        key: {"stored": str(stored[key]), "current": str(current[key])}
        for key in stored
        if str(stored[key]) != str(current[key])
    }
    allowed = {"canonical_core", "canonical_runner"}
    if not set(differences).issubset(allowed):
        raise RuntimeError(
            "Checkpoint scoring/data fingerprint changed: "
            f"{sorted(set(differences) - allowed)}"
        )
    return differences


def validate_source_artifact_state(state: Mapping[str, bool]) -> None:
    """Require frozen point artifacts, independently of source bootstrap state."""
    required = (
        "train_complete",
        "checkpoint_hash_matches",
        "point_report_exists",
        "selections_exist",
    )
    missing = [key for key in required if state.get(key) is not True]
    if missing:
        raise RuntimeError(f"Frozen source point artifacts are incomplete: {missing}")
