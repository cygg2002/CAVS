"""Pure contracts for the ENA24/Oregon DualHead cross-dataset experiment.

The unique module name avoids shadowing the canonical experiment's own
``protocol_core`` dependency when both source trees are imported together.
"""

from __future__ import annotations

from collections import Counter
import copy
import math
from typing import Any, Iterable, Mapping, Sequence


DATASET_ROLES = {
    "ena24": {"fit": 1400, "calibration": 400, "final": 1758},
    "oregon": {"fit": 9923, "calibration": 4961, "final": 4962},
}
CANONICAL_RUNS = {
    "ena24": "canonical_v1",
    "oregon": "canonical_oregon_event_v1",
}
GBDT_THRESHOLDS = {
    "ena24": 0.9028747070995273,
    "oregon": 0.9724118058752602,
}
TRACK_A_METHODS = (
    "Current-BudgetSet-SharedAction",
    "SetValue-SharedAction",
    "DualHead-SharedAction",
    "ActionValue-OQ-Joint",
)
TRACK_A_ROLES = ("fit_oof", "calibration")
TRACK_A_MODES = ("fixed_byte", "fixed_call")
TRACK_B_POLICIES = (
    "SharedAction",
    "CostEligible-AllROI",
    "GBDT-Hierarchical",
)
BUDGETS = (0.1, 0.25, 0.4)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != "ena24_oregon_dualhead_crossdataset_v1":
        raise ValueError("Protocol ID changed")
    if protocol.get("result_status") != "posthoc_crossdataset_extension":
        raise ValueError("Result status changed")
    datasets = protocol.get("datasets", {})
    if set(datasets) != set(DATASET_ROLES):
        raise ValueError("Dataset set changed")
    for dataset, roles in DATASET_ROLES.items():
        config = datasets[dataset]
        if config.get("canonical_run") != CANONICAL_RUNS[dataset]:
            raise ValueError(f"Canonical run changed for {dataset}")
        if dict(config.get("roles", {})) != roles:
            raise ValueError(f"Role population changed for {dataset}")
        if not math.isclose(
            float(config.get("gbdt_fixed_call_threshold", math.nan)),
            GBDT_THRESHOLDS[dataset],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Frozen GBDT threshold changed for {dataset}")
    track_a = protocol.get("track_a", {})
    if tuple(map(str, track_a.get("methods", ()))) != TRACK_A_METHODS:
        raise ValueError("Track A method matrix changed")
    if tuple(map(str, track_a.get("roles", ()))) != TRACK_A_ROLES:
        raise ValueError("Track A roles changed")
    if tuple(map(str, track_a.get("modes", ()))) != TRACK_A_MODES:
        raise ValueError("Track A modes changed")
    if tuple(map(float, track_a.get("budgets", ()))) != BUDGETS:
        raise ValueError("Track A budgets changed")
    if tuple(map(str, track_a.get("actions", ()))) != (
        "edge",
        "original",
        "q90",
    ):
        raise ValueError("Track A action set changed")
    if track_a.get("final_accessed") is not False:
        raise ValueError("Track A cannot access final")
    track_b = protocol.get("track_b", {})
    if track_b.get("method") != "DualHead-Value":
        raise ValueError("Track B ranker changed")
    if tuple(map(str, track_b.get("policies", ()))) != TRACK_B_POLICIES:
        raise ValueError("Track B frozen policy matrix changed")
    if track_b.get("mode") != "fixed_call":
        raise ValueError("Track B mode changed")
    if tuple(map(float, track_b.get("budgets", ()))) != BUDGETS:
        raise ValueError("Track B budgets changed")
    if track_b.get("final_used_for_training_or_selection") is not False:
        raise ValueError("Track B final cannot train or select")
    source_text = "\n".join(map(str, protocol.get("source_experiments", {}).values()))
    for forbidden in protocol.get("forbidden_source_substrings", ()):
        if str(forbidden).lower() in source_text.lower():
            raise ValueError(f"Cloud-V2/forbidden source configured: {forbidden}")


def expected_track_a_keys(
    protocol: Mapping[str, Any],
) -> set[tuple[str, str, str, str, float]]:
    return {
        (dataset, str(role), str(method), str(mode), float(budget))
        for dataset in protocol["datasets"]
        for role in protocol["track_a"]["roles"]
        for method in protocol["track_a"]["methods"]
        for mode in protocol["track_a"]["modes"]
        for budget in protocol["track_a"]["budgets"]
    }


def expected_track_b_keys(
    protocol: Mapping[str, Any],
) -> set[tuple[str, str, str, float]]:
    return {
        (dataset, str(protocol["track_b"]["method"]), str(policy), float(budget))
        for dataset in protocol["datasets"]
        for policy in protocol["track_b"]["policies"]
        for budget in protocol["track_b"]["budgets"]
    }


def _assert_exact_matrix(observed: Sequence[tuple], expected: set[tuple]) -> None:
    counts = Counter(observed)
    unique = set(observed)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if unique != expected or duplicates or len(observed) != len(expected):
        raise RuntimeError(
            "Result matrix differs from protocol: "
            f"missing={sorted(expected - unique)} extra={sorted(unique - expected)} "
            f"duplicates={duplicates}"
        )


def assert_track_a_matrix(
    rows: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    observed = [
        (
            str(row["dataset"]),
            str(row["role"]),
            str(row["method"]),
            str(row["mode"]),
            float(row["nominal_budget"]),
        )
        for row in rows
    ]
    _assert_exact_matrix(observed, expected_track_a_keys(protocol))


def assert_track_b_matrix(
    rows: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    observed = [
        (
            str(row["dataset"]),
            str(row["method"]),
            str(row["action_policy"]),
            float(row["nominal_budget"]),
        )
        for row in rows
    ]
    _assert_exact_matrix(observed, expected_track_b_keys(protocol))


def track_b_bootstrap_metadata(
    key: str, core_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Build trustworthy labels for one Track B paired bootstrap result.

    The historical bootstrap helper returns numerical statistics but also
    carries stale hard-coded labels.  Track B owns the comparison metadata, so
    derive it from the frozen ``results.csv`` row instead of trusting the
    helper's labels.
    """
    parts = str(key).split("|")
    if len(parts) != 2:
        raise ValueError(f"Invalid Track B bootstrap key: {key!r}")
    policy, budget_text = parts
    if policy not in TRACK_B_POLICIES:
        raise ValueError(f"Unknown Track B policy in key: {policy!r}")
    try:
        budget = float(budget_text)
    except ValueError as exc:
        raise ValueError(f"Invalid Track B budget in key: {key!r}") from exc
    if not any(math.isclose(budget, expected, rel_tol=0.0, abs_tol=1e-12) for expected in BUDGETS):
        raise ValueError(f"Unexpected Track B budget in key: {key!r}")

    row_policy = str(core_row.get("action_policy", ""))
    row_budget = float(core_row.get("nominal_budget", math.nan))
    if row_policy != policy or not math.isclose(
        row_budget, budget, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Track B bootstrap key does not match its core result row: "
            f"key={key!r}, row_policy={row_policy!r}, row_budget={row_budget!r}"
        )
    comparator = str(core_row.get("comparator", "")).strip()
    if not comparator:
        raise ValueError(f"Missing Track B comparator for key: {key!r}")
    candidate = str(core_row.get("method", "DualHead-Value")).strip()
    mode = str(core_row.get("mode", "fixed_call")).strip()
    if not candidate or not mode:
        raise ValueError(f"Missing Track B candidate/mode for key: {key!r}")
    return {
        "action_policy": policy,
        "candidate_method": candidate,
        "baseline_method": comparator,
        "comparison": f"{candidate} - {comparator}",
        "mode": mode,
        "nominal_budget": budget,
        "comparator": comparator,
    }


def align_scores(
    samples: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[float]:
    """Align scores by role-local identity, validating image ID as a guard."""
    by_identity: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = str(row["identity"])
        if identity in by_identity:
            raise ValueError(f"Duplicate score identity: {identity}")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score for {identity}")
        by_identity[identity] = row
    expected = {str(sample["identity"]) for sample in samples}
    if set(by_identity) != expected:
        raise ValueError(
            f"Score population mismatch: missing={sorted(expected - set(by_identity))[:5]} "
            f"extra={sorted(set(by_identity) - expected)[:5]}"
        )
    output = []
    for sample in samples:
        identity = str(sample["identity"])
        row = by_identity[identity]
        if int(row["image_id"]) != int(sample["image_id"]):
            raise ValueError(f"Image ID differs for identity {identity}")
        output.append(float(row["score"]))
    return output


def filter_actions(
    record: Mapping[str, Any], allowed: Sequence[str]
) -> dict[str, Any]:
    allowed_set = set(map(str, allowed))
    output = copy.deepcopy(dict(record))
    output["actions"] = [
        action
        for action in output.get("actions", ())
        if str(action["action_id"]) in allowed_set
    ]
    observed = tuple(str(action["action_id"]) for action in output["actions"])
    required = tuple(action for action in map(str, allowed) if action in observed)
    if observed != required or not observed or observed[0] != "edge":
        raise ValueError(f"Action order/availability is invalid: {observed}")
    return output


def assert_capacity_safe(
    used_bytes: float, capacity_bytes: float, tolerance: float = 1e-6
) -> None:
    if float(used_bytes) > float(capacity_bytes) + float(tolerance):
        raise RuntimeError(
            f"Fixed-byte capacity exceeded: {used_bytes} > {capacity_bytes}"
        )


def assert_same_membership(
    selections: Mapping[str, Iterable[int]],
) -> None:
    expected: set[int] | None = None
    for policy, values in selections.items():
        observed = set(map(int, values))
        if expected is None:
            expected = observed
        elif observed != expected:
            raise RuntimeError(f"Track B membership changed for policy {policy}")
