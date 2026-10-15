"""Pure contracts and monotone calibration for the cross-action diagnostic."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import numpy as np


DATASETS = {
    "coco": {"canonical_run": "canonical_v1", "fit": 9678, "calibration": 571},
    "ena24": {"canonical_run": "canonical_v1", "fit": 1400, "calibration": 400},
    "oregon": {
        "canonical_run": "canonical_oregon_event_v1",
        "fit": 9923,
        "calibration": 4961,
    },
}
ROLES = ("fit_oof", "calibration")
MODES = ("fixed_byte", "fixed_call")
BUDGETS = (0.1, 0.25, 0.4)
ACTIONS = ("edge", "original", "q90")
METHODS = (
    "Raw-ActionValue-OQ",
    "PerAction-Isotonic-OQ",
    "CalibratedMembership-OracleAction",
    "OracleMembership-CalibratedAction",
    "FullOracle-OQ",
)
DEPLOYABLE_METHODS = METHODS[:2]


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != "three_dataset_cross_action_calibration_v1":
        raise ValueError("Protocol ID changed")
    if protocol.get("result_status") != "development_oof_and_frozen_calibration":
        raise ValueError("Result status changed")
    if set(protocol.get("datasets", {})) != set(DATASETS):
        raise ValueError("Dataset set changed")
    for dataset, expected in DATASETS.items():
        observed = protocol["datasets"][dataset]
        if observed.get("canonical_run") != expected["canonical_run"]:
            raise ValueError(f"Canonical run changed for {dataset}")
        roles = observed.get("roles", {})
        if int(roles.get("fit", -1)) != expected["fit"] or int(
            roles.get("calibration", -1)
        ) != expected["calibration"]:
            raise ValueError(f"Role population changed for {dataset}")
    if tuple(map(str, protocol.get("roles", ()))) != ROLES:
        raise ValueError("Role matrix changed")
    if tuple(map(str, protocol.get("modes", ()))) != MODES:
        raise ValueError("Mode matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != BUDGETS:
        raise ValueError("Budget matrix changed")
    if tuple(map(str, protocol.get("actions", ()))) != ACTIONS:
        raise ValueError("Action set changed")
    if tuple(map(str, protocol.get("methods", ()))) != METHODS:
        raise ValueError("Method matrix changed")
    if tuple(map(str, protocol.get("deployable_methods", ()))) != DEPLOYABLE_METHODS:
        raise ValueError("Deployable method set changed")
    calibration = protocol.get("calibration", {})
    if calibration.get("source") != "fit_oof_only":
        raise ValueError("Calibration must be fit-OOF only")
    if int(calibration.get("crossfit_folds", -1)) != 5:
        raise ValueError("Cross-fitting fold count changed")
    if int(calibration.get("bins", -1)) != 64:
        raise ValueError("Isotonic bin count changed")
    if tuple(map(str, calibration.get("actions", ()))) != ACTIONS[1:]:
        raise ValueError("Calibrated action set changed")
    if calibration.get("calibration_used_for_fit_or_selection") is not False:
        raise ValueError("Calibration cannot fit or select models")
    for flag in ("final_accessed", "detector_inference", "gpu_inference"):
        if protocol.get(flag) is not False:
            raise ValueError(f"{flag} must remain false")
    sources = "\n".join(map(str, protocol.get("source_experiments", {}).values()))
    for forbidden in protocol.get("forbidden_source_substrings", ()):
        if str(forbidden).lower() in sources.lower():
            raise ValueError(f"Forbidden source configured: {forbidden}")


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks = [
        [index, index + 1, float(weight), float(value)]
        for index, (value, weight) in enumerate(zip(values, weights))
    ]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][3] <= blocks[index + 1][3]:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        weight = left[2] + right[2]
        value = (left[3] * left[2] + right[3] * right[2]) / max(weight, 1e-12)
        blocks[index : index + 2] = [[left[0], right[1], weight, value]]
        index = max(0, index - 1)
    output = np.empty(len(values), dtype=np.float64)
    for start, end, _, value in blocks:
        output[int(start) : int(end)] = float(value)
    return output


def fit_isotonic(scores: np.ndarray, targets: np.ndarray, *, bins: int) -> dict[str, Any]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("Isotonic inputs must be matching non-empty vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Isotonic inputs must be finite")
    if int(bins) < 2:
        raise ValueError("Isotonic calibration requires at least two bins")
    order = np.argsort(x, kind="mergesort")
    partitions = [
        part for part in np.array_split(order, min(int(bins), len(order))) if len(part)
    ]
    bin_x = np.asarray([float(x[part].mean()) for part in partitions])
    bin_y = np.asarray([float(y[part].mean()) for part in partitions])
    weights = np.asarray([float(len(part)) for part in partitions])
    return {"x": bin_x.tolist(), "y": _pava(bin_y, weights).tolist(), "bins": len(partitions)}


def apply_isotonic(model: Mapping[str, Any], scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    x = np.asarray(model["x"], dtype=np.float64)
    y = np.asarray(model["y"], dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("Invalid isotonic model")
    return np.interp(values, x, y, left=float(y[0]), right=float(y[-1]))


def crossfit_isotonic(
    scores: np.ndarray,
    targets: np.ndarray,
    folds: Sequence[Sequence[int]],
    *,
    bins: int,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Cross-fit inputs must be matching vectors")
    valid = (
        np.ones(len(x), dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != x.shape or not valid.any():
        raise ValueError("Cross-fit valid mask must select at least one target")
    if not np.isfinite(x).all() or not np.isfinite(y[valid]).all():
        raise ValueError("Cross-fit scores and selected targets must be finite")
    output = np.full(len(x), np.nan, dtype=np.float64)
    reports: list[dict[str, Any]] = []
    population = set(range(len(x)))
    heldout_union: set[int] = set()
    for fold_index, fold in enumerate(folds):
        heldout = list(map(int, fold))
        heldout_set = set(heldout)
        if not heldout or heldout_union & heldout_set or not heldout_set <= population:
            raise ValueError("Cross-fit folds must be non-empty, disjoint, and in range")
        train = sorted((population - heldout_set) & set(map(int, np.flatnonzero(valid))))
        if not train:
            raise ValueError("Cross-fit training set is empty")
        model = fit_isotonic(x[train], y[train], bins=bins)
        output[heldout] = apply_isotonic(model, x[heldout])
        heldout_union.update(heldout_set)
        reports.append(
            {
                "fold": int(fold_index),
                "train_indices": train,
                "heldout_indices": heldout,
                "model": model,
            }
        )
    if heldout_union != population or not np.isfinite(output).all():
        raise ValueError("Cross-fit folds do not cover the population exactly")
    return output, reports


def assert_fixed_membership(
    selection: Mapping[int, str], expected_image_ids: set[int]
) -> None:
    observed = set(map(int, selection))
    expected = set(map(int, expected_image_ids))
    if observed != expected:
        raise RuntimeError(
            f"Fixed membership changed: missing={sorted(expected-observed)[:5]} "
            f"extra={sorted(observed-expected)[:5]}"
        )
    invalid = sorted(set(map(str, selection.values())) - set(ACTIONS[1:]))
    if invalid:
        raise RuntimeError(f"Fixed membership contains invalid actions: {invalid}")


def build_fixed_membership_groups(
    samples: Sequence[Mapping[str, Any]],
    image_ids: set[int],
    values: Mapping[int, Mapping[str, float]],
) -> list[list[dict[str, Any]]]:
    """Build Original/Q90 groups with no Edge escape for fixed membership."""
    wanted = set(map(int, image_ids))
    groups: list[list[dict[str, Any]]] = []
    observed: set[int] = set()
    for sample in samples:
        image_id = int(sample["image_id"])
        if image_id not in wanted:
            continue
        group = []
        for action in ACTIONS[1:]:
            available = bool(sample.get("available", {}).get(action, False))
            payload = sample.get("payload_bytes", {}).get(action)
            value = values.get(image_id, {}).get(action)
            if available and payload is not None and value is not None and math.isfinite(float(value)):
                group.append(
                    {
                        "action_id": action,
                        "image_id": image_id,
                        "value": float(value),
                        "cost": float(payload),
                    }
                )
        if not group:
            raise ValueError(f"No Original/Q90 action for fixed image {image_id}")
        groups.append(group)
        observed.add(image_id)
    if observed != wanted:
        raise ValueError(
            f"Fixed-membership sample population differs: missing={sorted(wanted-observed)[:5]}"
        )
    return groups


def restrict_to_oq_values(
    values: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Remove Edge/ROI values so an O/Q oracle cannot schedule another action."""
    return {
        int(image_id): {
            action: float(actions[action])
            for action in ACTIONS[1:]
            if action in actions and math.isfinite(float(actions[action]))
        }
        for image_id, actions in values.items()
    }


def validate_action_availability(samples: Sequence[Mapping[str, Any]]) -> None:
    """Require at least one valid O/Q action, without inventing missing actions."""
    for sample in samples:
        valid = [
            action
            for action in ACTIONS[1:]
            if bool(sample.get("available", {}).get(action, False))
            and sample.get("payload_bytes", {}).get(action) is not None
        ]
        if not valid:
            raise ValueError(
                f"No available Original/Q90 action for image {sample.get('image_id')}"
            )


def expected_result_keys(
    protocol: Mapping[str, Any], dataset: str
) -> set[tuple[str, str, str, float]]:
    if dataset not in protocol.get("datasets", {}):
        raise ValueError(dataset)
    return {
        (str(role), str(method), str(mode), float(budget))
        for role in protocol["roles"]
        for method in protocol["methods"]
        for mode in protocol["modes"]
        for budget in protocol["budgets"]
    }


def validate_result_rows(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], dataset: str
) -> None:
    expected = expected_result_keys(protocol, dataset)
    observed = [
        (
            str(row["role"]),
            str(row["method"]),
            str(row["mode"]),
            float(row["budget"]),
        )
        for row in rows
    ]
    counts = Counter(observed)
    if set(observed) != expected or len(observed) != len(expected) or any(
        value != 1 for value in counts.values()
    ):
        raise ValueError(f"Result matrix incomplete for {dataset}: {len(rows)} rows")
    for row in rows:
        if str(row["mode"]) == "fixed_byte" and not bool(row["capacity_ok"]):
            raise ValueError(f"Fixed-byte capacity exceeded: {row}")
        if str(row["mode"]) == "fixed_call":
            population = DATASETS[dataset][
                "fit" if str(row["role"]) == "fit_oof" else "calibration"
            ]
            expected_count = int(math.floor(population * float(row["budget"]) + 0.5))
            if int(row["selected_images"]) != expected_count:
                raise ValueError(f"Fixed-call count changed: {row}")


def evaluate_gate(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    lookup = {
        (
            str(row["dataset"]),
            str(row["method"]),
            str(row["mode"]),
            float(row["budget"]),
        ): float(row["official_ap50_95"])
        for row in rows
        if str(row.get("role")) == "calibration"
        and str(row.get("method")) in DEPLOYABLE_METHODS
    }
    gate = protocol["success_gate"]
    details: dict[str, Any] = {}
    supporting = 0
    all_byte_safe = True
    all_call_safe = True
    for dataset in DATASETS:
        byte_deltas = []
        call_deltas = []
        for budget in BUDGETS:
            raw_byte = lookup[(dataset, DEPLOYABLE_METHODS[0], "fixed_byte", budget)]
            calibrated_byte = lookup[(dataset, DEPLOYABLE_METHODS[1], "fixed_byte", budget)]
            raw_call = lookup[(dataset, DEPLOYABLE_METHODS[0], "fixed_call", budget)]
            calibrated_call = lookup[(dataset, DEPLOYABLE_METHODS[1], "fixed_call", budget)]
            byte_deltas.append(calibrated_byte - raw_byte)
            call_deltas.append(calibrated_call - raw_call)
        gains = sum(
            delta >= float(gate["dataset_gain_threshold"]) for delta in byte_deltas
        )
        if gains >= int(gate["required_gain_points_per_supporting_dataset"]):
            supporting += 1
        all_byte_safe &= all(
            delta >= float(gate["minimum_fixed_byte_delta"]) for delta in byte_deltas
        )
        all_call_safe &= all(
            delta >= float(gate["minimum_fixed_call_delta"]) for delta in call_deltas
        )
        details[dataset] = {
            "fixed_byte_deltas": byte_deltas,
            "fixed_call_deltas": call_deltas,
            "fixed_byte_gain_points": gains,
        }
    ena40 = lookup[("ena24", DEPLOYABLE_METHODS[1], "fixed_byte", 0.4)]
    oregon_safe = all(
        value >= float(gate["minimum_oregon_fixed_byte_delta"])
        for value in details["oregon"]["fixed_byte_deltas"]
    )
    passed = (
        all_byte_safe
        and all_call_safe
        and ena40 >= float(gate["minimum_ena24_40_ap"])
        and oregon_safe
        and supporting >= int(gate["required_supporting_datasets"])
    )
    return {
        "passed": bool(passed),
        "supporting_datasets": int(supporting),
        "all_fixed_byte_noninferior": bool(all_byte_safe),
        "all_fixed_call_noninferior": bool(all_call_safe),
        "ena24_40_fixed_byte_ap": float(ena40),
        "oregon_fixed_byte_noninferior": bool(oregon_safe),
        "details": details,
    }
