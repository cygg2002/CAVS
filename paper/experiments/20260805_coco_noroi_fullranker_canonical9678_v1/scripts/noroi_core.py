"""Pure contracts for the canonical COCO No-ROI comparison."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HISTORICAL_METHODS = (
    "Random",
    "Confidence",
    "ClassGainPrior",
    "M2E",
    "Confidence-Residual M2E",
    "SemanticRank",
    "MORIC-style",
    "GEC-style",
    "Current-Controlled-32D",
)
INTERNAL_METHODS = ("Current-BudgetSet", "SetValue", "DualHead")
EXTERNAL_METHODS = (
    "ORIC-AP50",
    "MORIC-AP50",
    "ORIC-AP5095",
    "MORIC-AP5095",
)
METHODS = HISTORICAL_METHODS + INTERNAL_METHODS + EXTERNAL_METHODS
POLICIES = ("Original-only", "SharedAction-NoROI")
MODES = ("fixed_call", "fixed_byte")
BUDGETS = (0.1, 0.25, 0.4)
ROLES = {"fit": 9678, "calibration": 571, "registered_final": 5000}
ALLOWED_ACTIONS = {"edge", "original", "q90"}


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("This experiment is COCO-only")
    if protocol.get("result_status") != "posthoc_canonical_val2017_noroi_extension":
        raise ValueError("Post-hoc result status changed")
    if dict(protocol.get("roles", {})) != ROLES:
        raise ValueError("Canonical fit/calibration/val2017 populations changed")
    if tuple(map(str, protocol.get("historical_methods", ()))) != HISTORICAL_METHODS:
        raise ValueError("Historical nine-Ranker matrix changed")
    if tuple(map(str, protocol.get("internal_methods", ()))) != INTERNAL_METHODS:
        raise ValueError("Internal method matrix changed")
    if tuple(map(str, protocol.get("external_methods", ()))) != EXTERNAL_METHODS:
        raise ValueError("External method matrix changed")
    if tuple(map(str, protocol.get("payload_policies", ()))) != POLICIES:
        raise ValueError("No-ROI payload policies changed")
    if set(map(str, protocol.get("allowed_actions", ()))) != ALLOWED_ACTIONS:
        raise ValueError("No-ROI action set changed")
    work_points = protocol.get("work_points", {})
    if tuple(work_points) != MODES:
        raise ValueError("Evaluation modes or ordering changed")
    if any(tuple(map(float, work_points[mode])) != BUDGETS for mode in MODES):
        raise ValueError("Canonical budget grid changed")
    if protocol.get("primary_candidate") != "DualHead":
        raise ValueError("DualHead must remain the primary candidate")
    if protocol.get("primary_mode") != "fixed_byte":
        raise ValueError("Fixed-byte must remain primary")
    if protocol.get("fixed_byte_lambda_source") != (
        "calibration_only_per_method_policy_budget"
    ):
        raise ValueError("Fixed-byte calibration semantics changed")
    if protocol.get("ranker_checkpoints_retrained") is not False:
        raise ValueError("Frozen rankers cannot be retrained")
    if protocol.get("action_models_retrained") is not False:
        raise ValueError("Frozen action heads cannot be retrained")
    if protocol.get("registered_final_used_for_selection") is not False:
        raise ValueError("val2017 cannot be used for model or control selection")
    if int(protocol.get("expected_rows_per_role", -1)) != len(
        expected_result_keys(protocol)
    ):
        raise ValueError("Expected result-row count changed")
    bootstrap = protocol.get("bootstrap", {})
    expected_comparisons = (
        len(POLICIES)
        * len(BUDGETS)
        * len(tuple(bootstrap.get("comparison_types", ())))
    )
    if int(bootstrap.get("replicates", -1)) != 1000:
        raise ValueError("Bootstrap replicate count changed")
    if int(bootstrap.get("expected_comparisons", -1)) != expected_comparisons:
        raise ValueError("Bootstrap comparison count changed")


def expected_result_keys(
    protocol: Mapping[str, Any],
) -> set[tuple[str, str, str, float]]:
    methods = (
        list(protocol["historical_methods"])
        + list(protocol["internal_methods"])
        + list(protocol["external_methods"])
    )
    return {
        (str(method), str(policy), str(mode), float(budget))
        for method in methods
        for policy in protocol["payload_policies"]
        for mode, budgets in protocol["work_points"].items()
        for budget in budgets
    }


def result_key(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["method"]),
        str(row["payload_policy"]),
        str(row["mode"]),
        float(row["nominal_budget"]),
    )


def assert_result_matrix_complete(
    rows: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    observed = [result_key(row) for row in rows]
    counts = Counter(observed)
    expected = expected_result_keys(protocol)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if set(observed) != expected or duplicates or len(observed) != len(expected):
        raise RuntimeError(
            "Result matrix is incomplete or duplicated: "
            f"missing={sorted(expected - set(observed))} "
            f"extra={sorted(set(observed) - expected)} duplicates={duplicates}"
        )


def build_no_roi_action_maps(
    samples: Sequence[Mapping[str, Any]],
    q90_probability: np.ndarray,
    *,
    q90_threshold: float,
) -> dict[str, dict[int, str]]:
    probabilities = np.asarray(q90_probability, dtype=np.float64)
    if probabilities.shape != (len(samples),) or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Invalid Q90 probability vector")
    original: dict[int, str] = {}
    shared: dict[int, str] = {}
    for sample, score in zip(samples, probabilities):
        image_id = int(sample["image_id"])
        original[image_id] = "original"
        shared[image_id] = (
            "q90"
            if bool(sample.get("available", {}).get("q90", False))
            and float(score) >= float(q90_threshold)
            else "original"
        )
    return {"Original-only": original, "SharedAction-NoROI": shared}


def assert_no_roi_actions(actions: Mapping[int, str]) -> None:
    invalid = sorted(
        (int(image_id), str(action))
        for image_id, action in actions.items()
        if str(action) not in {"original", "q90"}
    )
    if invalid:
        raise RuntimeError(f"ROI or invalid action found in No-ROI selection: {invalid[:3]}")


def assert_result_rows_no_roi(rows: Iterable[Mapping[str, Any]]) -> None:
    bad = []
    for row in rows:
        roi_ratio = float(row.get("roi_ratio", 0.0) or 0.0)
        if abs(roi_ratio) > 1e-12:
            bad.append((result_key(row), roi_ratio))
    if bad:
        raise RuntimeError(f"ROI ratio is nonzero in No-ROI result rows: {bad[:3]}")


def selected_bytes(
    samples: Sequence[Mapping[str, Any]], actions: Mapping[int, str]
) -> float:
    by_id = {int(sample["image_id"]): sample for sample in samples}
    total = 0.0
    for image_id, action in actions.items():
        assert_no_roi_actions({int(image_id): str(action)})
        total += float(by_id[int(image_id)]["payload_bytes"][str(action)])
    return total


def assert_fixed_byte_capacity(
    samples: Sequence[Mapping[str, Any]],
    actions: Mapping[int, str],
    *,
    capacity: float,
) -> None:
    used = selected_bytes(samples, actions)
    if used > float(capacity) + 1e-6:
        raise RuntimeError(f"Fixed-byte capacity exceeded: {used} > {capacity}")


def assert_score_alignment(
    score_ids: Sequence[int],
    action_ids: Sequence[int],
    scores: np.ndarray,
) -> None:
    left = np.asarray(score_ids, dtype=np.int64)
    right = np.asarray(action_ids, dtype=np.int64)
    matrix = np.asarray(scores, dtype=np.float64)
    if not np.array_equal(left, right):
        raise RuntimeError("Frozen score/action image ordering differs")
    if matrix.ndim != 3 or matrix.shape[1:] != (len(BUDGETS), len(left)):
        raise RuntimeError(f"Frozen score matrix has invalid shape: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Frozen score matrix contains non-finite values")


def freeze_strongest(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidates: Sequence[str],
    policies: Sequence[str],
    budgets: Sequence[float],
) -> dict[str, str]:
    values = list(rows)
    if any(str(row.get("role")) != "calibration" for row in values):
        raise RuntimeError("Strongest comparators must be frozen on calibration only")
    allowed = set(map(str, candidates))
    output: dict[str, str] = {}
    for policy in policies:
        for budget in map(float, budgets):
            matches = [
                row
                for row in values
                if str(row["method"]) in allowed
                and str(row["payload_policy"]) == str(policy)
                and str(row["mode"]) == "fixed_byte"
                and abs(float(row["nominal_budget"]) - budget) < 1e-12
            ]
            if len(matches) != len(allowed):
                raise RuntimeError(
                    f"Incomplete calibration comparator group for {policy}/{budget:g}"
                )
            winner = max(
                matches,
                key=lambda row: (
                    float(row["official_map50_95"]),
                    -float(row["request_kb_per_image"]),
                    str(row["method"]),
                ),
            )
            output[f"{policy}|{budget:g}"] = str(winner["method"])
    return output

