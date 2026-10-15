"""Pure contracts for the unified COCO/Oregon CAVS-OQR benchmark."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_ROLES = {
    "coco": {"fit": 9678, "calibration": 571, "final": 5000},
    "oregon": {"fit": 9923, "calibration": 4961, "final": 4962},
}
EXPECTED_ACTION_SPACES = {
    "O": ["edge", "original"],
    "OQ": ["edge", "original", "q90"],
    "OQR": ["edge", "original", "q90", "current@640"],
}
PRIMARY_EXTERNAL = {"ORIC-AP5095", "MORIC-AP5095"}


def reference_action_maps(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, str]]:
    output = {
        "Edge-only": {},
        "Cloud-only Original": {},
        "Cloud-only Q90": {},
    }
    for sample in samples:
        image_id = int(sample["image_id"])
        available = sample.get("available", {})
        if not bool(available.get("original", False)):
            raise ValueError(f"original action unavailable: {image_id}")
        output["Cloud-only Original"][image_id] = "original"
        output["Cloud-only Q90"][image_id] = (
            "q90" if bool(available.get("q90", False)) else "original"
        )
    return output


def json_threshold(value: float) -> float | None:
    observed = float(value)
    return observed if math.isfinite(observed) else None


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if set(protocol.get("datasets", ())) != set(EXPECTED_ROLES):
        raise ValueError("dataset matrix changed")
    for dataset, roles in EXPECTED_ROLES.items():
        if protocol["datasets"][dataset].get("roles") != roles:
            raise ValueError(f"canonical role population changed: {dataset}")
        seeds = list(map(int, protocol["datasets"][dataset].get("model_seeds", ())))
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError(f"three distinct model seeds required: {dataset}")
    if protocol.get("action_spaces") != EXPECTED_ACTION_SPACES:
        raise ValueError("O/OQ/OQR action definition changed")
    if protocol.get("primary_roi_action") != "current@640":
        raise ValueError("primary ROI must be current@640")
    if protocol.get("primary_roi_fusion") != "Reversible":
        raise ValueError("primary ROI fusion must be Reversible")
    if protocol.get("reversible_fusion") != {
        "match_iou": 0.3,
        "score_margin": 0.0,
        "add_threshold": 0.1,
    }:
        raise ValueError("Reversible Fusion parameters changed")
    if list(map(float, protocol.get("budgets", ()))) != [0.1, 0.25, 0.4]:
        raise ValueError("budget grid changed")
    if list(protocol.get("modes", ())) != ["fixed_byte", "fixed_call"]:
        raise ValueError("evaluation modes changed")
    if int(protocol.get("fold_count", 0)) != 5:
        raise ValueError("five folds required")
    if protocol.get("scalar_policy") != "SharedAction-OQR":
        raise ValueError("scalar methods must share one OQR action policy")
    if set(protocol.get("primary_external_methods", ())) != PRIMARY_EXTERNAL:
        raise ValueError("primary external comparator set changed")
    if bool(protocol.get("final_is_hidden")):
        raise ValueError("reused final cannot be hidden")
    if bool(protocol.get("final_used_for_training_or_selection")):
        raise ValueError("final cannot train or select methods")


def expected_cavs_rows(protocol: Mapping[str, Any], *, roles: int) -> int:
    validate_protocol(protocol)
    return (
        int(roles)
        * len(protocol["cavs_methods"])
        * len(protocol["modes"])
        * len(protocol["budgets"])
    )


def expected_scalar_rows(protocol: Mapping[str, Any], *, roles: int) -> int:
    validate_protocol(protocol)
    return (
        int(roles)
        * len(protocol["scalar_methods"])
        * len(protocol["modes"])
        * len(protocol["budgets"])
    )


def expected_bootstrap_jobs(protocol: Mapping[str, Any], dataset: str) -> int:
    validate_protocol(protocol)
    units = protocol["datasets"][dataset]["bootstrap_units"]
    # OQR has seven named comparators; OQ additionally uses Strongest-External.
    comparisons_per_budget = len(protocol["bootstrap"]["comparators"]) + 1
    return len(protocol["budgets"]) * len(units) * comparisons_per_budget


def expected_bootstrap_job_ids(
    protocol: Mapping[str, Any], dataset: str
) -> set[str]:
    validate_protocol(protocol)
    output: set[str] = set()
    for budget in map(float, protocol["budgets"]):
        for unit in map(str, protocol["datasets"][dataset]["bootstrap_units"]):
            for comparator in map(str, protocol["bootstrap"]["comparators"]):
                output.add(
                    f"CAVS-OQR|vs|{comparator}|budget={budget:g}|{unit}"
                )
            output.add(
                f"CAVS-OQ|vs|Strongest-External|budget={budget:g}|{unit}"
            )
    return output


def validate_grouped_folds(
    expected_ids: Sequence[int],
    folds: Sequence[Sequence[int]],
    group_by_id: Mapping[int, str],
) -> dict[str, Any]:
    expected = list(map(int, expected_ids))
    flattened = [int(image_id) for fold in folds for image_id in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(expected):
        raise ValueError("fold population is incomplete or duplicated")
    if set(group_by_id) != set(expected):
        raise ValueError("group mapping population changed")
    group_folds: dict[str, set[int]] = {}
    groups_per_fold: list[int] = []
    for fold_index, fold in enumerate(folds):
        observed = {str(group_by_id[int(image_id)]) for image_id in fold}
        groups_per_fold.append(len(observed))
        for group in observed:
            group_folds.setdefault(group, set()).add(fold_index)
    leaked = sorted(group for group, locations in group_folds.items() if len(locations) > 1)
    if leaked:
        raise ValueError(f"group leakage across folds: {leaked[:10]}")
    return {
        "group_count": len(group_folds),
        "groups_per_fold": groups_per_fold,
        "group_leakage_count": 0,
    }


def mask_scores(
    scores: Mapping[int, Mapping[str, float]], allowed: Sequence[str]
) -> dict[int, dict[str, float]]:
    actions = set(map(str, allowed)) - {"edge"}
    output: dict[int, dict[str, float]] = {}
    for image_id, values in scores.items():
        filtered = {
            str(action): float(value)
            for action, value in values.items()
            if str(action) in actions and math.isfinite(float(value))
        }
        if not filtered:
            raise ValueError(f"no allowed cloud action for image {image_id}")
        output[int(image_id)] = filtered
    return output


def shared_action_map(
    samples: Sequence[Mapping[str, Any]],
    *,
    q90_probabilities: Sequence[float],
    roi_probabilities: Mapping[int, float],
    q90_threshold: float,
    roi_threshold: float,
) -> dict[int, str]:
    if len(samples) != len(q90_probabilities):
        raise ValueError("Q90 probability alignment changed")
    output: dict[int, str] = {}
    for sample, q90_probability in zip(samples, q90_probabilities):
        image_id = int(sample["image_id"])
        available = sample.get("available", {})
        base = (
            "q90"
            if bool(available.get("q90", False))
            and float(q90_probability) >= float(q90_threshold)
            else "original"
        )
        output[image_id] = (
            "current@640"
            if bool(available.get("current@640", False))
            and float(roi_probabilities.get(image_id, -math.inf))
            >= float(roi_threshold)
            else base
        )
    return output


def empirical_percentile(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    observed = np.asarray(values, dtype=np.float64)
    if ordered.ndim != 1 or observed.ndim != 1 or not len(ordered):
        raise ValueError("rank scores must be non-empty vectors")
    if not np.isfinite(ordered).all() or not np.isfinite(observed).all():
        raise ValueError("rank scores must be finite")
    return np.searchsorted(ordered, observed, side="right") / float(len(ordered))


def scalar_byte_selection(
    samples: Sequence[Mapping[str, Any]],
    *,
    scores: Sequence[float],
    action_map: Mapping[int, str],
    budget: float,
    lambda_value: float,
    reference_scores: Sequence[float],
    median_original_bytes: float,
) -> dict[int, str]:
    if len(samples) != len(scores):
        raise ValueError("scalar score alignment changed")
    percentile = empirical_percentile(reference_scores, scores)
    priorities = []
    for index, sample in enumerate(samples):
        image_id = int(sample["image_id"])
        if image_id not in action_map:
            raise ValueError(f"shared action missing: {image_id}")
        action = str(action_map[image_id])
        cost = float(sample["payload_bytes"][action])
        priority = float(percentile[index]) - float(lambda_value) * cost / max(
            1.0, float(median_original_bytes)
        )
        priorities.append((priority, -cost, -image_id, index))
    capacity = float(budget) * sum(
        float(sample["payload_bytes"]["original"]) for sample in samples
    )
    used = 0.0
    selected: dict[int, str] = {}
    for _, _, _, index in sorted(priorities, reverse=True):
        sample = samples[index]
        image_id = int(sample["image_id"])
        action = str(action_map[image_id])
        cost = float(sample["payload_bytes"][action])
        if used + cost <= capacity + max(1e-6, capacity * 1e-12):
            selected[image_id] = action
            used += cost
    validate_selection(samples, selected, mode="fixed_byte", budget=float(budget))
    return selected


def assemble_oof_scores(
    expected_ids: Sequence[int],
    folds: Sequence[tuple[Sequence[int], Sequence[Sequence[Sequence[float]]]]],
) -> np.ndarray:
    positions = {int(image_id): index for index, image_id in enumerate(expected_ids)}
    if len(positions) != len(expected_ids):
        raise ValueError("expected OOF IDs are not unique")
    output = None
    seen: set[int] = set()
    for fold_ids, fold_values in folds:
        values = np.asarray(fold_values, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] != len(fold_ids):
            raise ValueError("OOF score shape changed")
        if output is None:
            output = np.empty(
                (values.shape[0], len(expected_ids), values.shape[2]),
                dtype=np.float64,
            )
        if values.shape[0] != output.shape[0] or values.shape[2] != output.shape[2]:
            raise ValueError("OOF fold score dimensions differ")
        for column, image_id in enumerate(map(int, fold_ids)):
            if image_id not in positions or image_id in seen:
                raise ValueError(f"invalid OOF image ID: {image_id}")
            output[:, positions[image_id], :] = values[:, column, :]
            seen.add(image_id)
    if output is None or seen != set(positions) or not np.isfinite(output).all():
        raise ValueError("OOF coverage is incomplete")
    return output


def action_score_map(
    records: Sequence[Mapping[str, Any]], values: Sequence[Sequence[float]]
) -> dict[int, dict[str, float]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("action score matrix shape changed")
    output: dict[int, dict[str, float]] = {}
    for row, record in enumerate(records):
        actions = list(record["actions"])
        if len(actions) > matrix.shape[1]:
            raise ValueError("score matrix has fewer columns than actions")
        image_id = int(record["image_id"])
        output[image_id] = {
            str(action["action_id"]): float(matrix[row, column])
            for column, action in enumerate(actions)
            if str(action["action_id"]) != "edge"
            and math.isfinite(float(matrix[row, column]))
        }
        if not output[image_id]:
            raise ValueError(f"no cloud action score: {image_id}")
    return output


def fixed_call_joint(
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    *,
    budget: float,
) -> dict[int, str]:
    best = []
    for sample in samples:
        image_id = int(sample["image_id"])
        candidates = [
            (float(value), -float(sample["payload_bytes"][action]), str(action))
            for action, value in scores[image_id].items()
            if action in sample["payload_bytes"]
        ]
        if not candidates:
            raise ValueError(f"no available cloud action: {image_id}")
        value, negative_cost, action = max(candidates)
        best.append((value, negative_cost, -image_id, image_id, action))
    count = int(math.floor(len(samples) * float(budget) + 0.5))
    selected = {
        image_id: action
        for _, _, _, image_id, action in sorted(best, reverse=True)[:count]
    }
    validate_selection(samples, selected, mode="fixed_call", budget=float(budget))
    return selected


def validate_selection(
    samples: Sequence[Mapping[str, Any]],
    selection: Mapping[int, str],
    *,
    mode: str,
    budget: float,
) -> dict[str, float]:
    sample_by_id = {int(sample["image_id"]): sample for sample in samples}
    if not set(map(int, selection)) <= set(sample_by_id):
        raise ValueError("selection contains unknown images")
    requested = 0.0
    for image_id, action in selection.items():
        payload = sample_by_id[int(image_id)].get("payload_bytes", {})
        if str(action) not in payload:
            raise ValueError(f"missing payload: {image_id}/{action}")
        requested += float(payload[str(action)])
    original_total = sum(
        float(sample["payload_bytes"]["original"]) for sample in samples
    )
    if mode == "fixed_byte":
        capacity = float(budget) * original_total
        if requested > capacity + max(1e-6, capacity * 1e-12):
            raise ValueError("fixed-byte capacity exceeded")
        utilization = requested / capacity if capacity else 0.0
    elif mode == "fixed_call":
        capacity = float("nan")
        expected = int(math.floor(len(samples) * float(budget) + 0.5))
        if len(selection) != expected:
            raise ValueError("fixed-call population changed")
        utilization = len(selection) / max(1, expected)
    else:
        raise ValueError(mode)
    return {
        "request_bytes": float(requested),
        "original_total_bytes": float(original_total),
        "capacity_bytes": float(capacity),
        "utilization": float(utilization),
    }


def result_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, float, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("role", "")),
        str(row.get("method", "")),
        str(row.get("policy", "")),
        float(row.get("budget", -1.0)),
        str(row.get("mode", "")),
    )


def validate_unique_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [result_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result rows")


def strongest_external(
    rows: Sequence[Mapping[str, Any]], budget: float, protocol: Mapping[str, Any]
) -> str:
    validate_protocol(protocol)
    candidates = [
        row
        for row in rows
        if str(row.get("role")) == "calibration"
        and str(row.get("method")) in PRIMARY_EXTERNAL
        and str(row.get("mode")) == "fixed_byte"
        and math.isclose(float(row.get("budget", -1.0)), float(budget))
    ]
    if not candidates:
        raise ValueError(f"missing calibration external rows: {budget:g}")
    return str(
        max(
            candidates,
            key=lambda row: (
                float(row["official_ap50_95"]),
                -float(row.get("kb_per_image", float("inf"))),
                str(row["method"]),
            ),
        )["method"]
    )
