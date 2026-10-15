"""Pure contracts for the registered COCO BudgetSet paper experiment."""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXTERNAL_METHODS = {
    "ORIC-AP50",
    "MORIC-AP50",
    "ORIC-AP5095",
    "MORIC-AP5095",
}


def batches(values: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("Batch size must be positive")
    rows = list(values)
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def canonical_hash(value: Any) -> str:
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def chunk_signature(
    *,
    protocol_hash: str,
    role: str,
    detector: str,
    model_hash: str,
    image_ids: Sequence[int],
    paths: Sequence[str],
) -> str:
    return canonical_hash(
        {
            "protocol_hash": str(protocol_hash),
            "role": str(role),
            "detector": str(detector),
            "model_hash": str(model_hash),
            "image_ids": list(map(int, image_ids)),
            "paths": list(map(str, paths)),
        }
    )


def category_index_to_id(
    model_names: Mapping[int, str] | Sequence[str],
    categories: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    if isinstance(model_names, Mapping):
        names = {int(index): str(name) for index, name in model_names.items()}
    else:
        names = {index: str(name) for index, name in enumerate(model_names)}
    category_by_name = {str(row["name"]): int(row["id"]) for row in categories}
    missing = sorted(set(names.values()) - set(category_by_name))
    if missing:
        raise RuntimeError(f"Detector categories are absent from COCO schema: {missing}")
    return {index: category_by_name[name] for index, name in names.items()}


def validate_prediction_population(
    predictions: Sequence[Mapping[str, Any]], *, expected_ids: set[int]
) -> None:
    observed = {int(row["image_id"]) for row in predictions}
    unknown = observed - set(map(int, expected_ids))
    if unknown:
        raise RuntimeError(f"Predictions contain unknown image IDs: {sorted(unknown)[:10]}")


def resolve_result_image_id(
    result_path: str,
    *,
    path_to_id: Mapping[str, int],
    ordered_image_ids: Sequence[int],
    result_index: int,
) -> int:
    """Map detector results back to inputs, including generated imageN.jpg list paths."""
    resolved = str(Path(result_path).resolve())
    if resolved in path_to_id:
        return int(path_to_id[resolved])
    match = re.fullmatch(r"image(\d+)\.[A-Za-z0-9]+", Path(result_path).name)
    index = int(result_index)
    if match and int(match.group(1)) == index and 0 <= index < len(ordered_image_ids):
        return int(ordered_image_ids[index])
    raise RuntimeError(
        f"Cannot map detector result path {result_path!r} at result index {result_index}"
    )


def validate_reward_isolation(
    rows: Sequence[Mapping[str, Any]], *, development_ids: set[int]
) -> None:
    allowed = set(map(int, development_ids))
    for row in rows:
        target = int(row["target_image_id"])
        context = set(map(int, row["context_image_ids"]))
        if target not in allowed or not context <= allowed or target in context:
            raise RuntimeError(f"Reward row crosses the development boundary: target={target}")


def original_e2e_ms(
    payload_bytes: int | float,
    response_bytes: int | float,
    network: Mapping[str, Any],
) -> float:
    upload_ms = float(payload_bytes) * 8.0 / (float(network["uplink_mbps"]) * 1000.0)
    download_ms = float(response_bytes) * 8.0 / (
        float(network["downlink_mbps"]) * 1000.0
    )
    return (
        float(network["edge_ms"])
        + float(network["router_ms"])
        + float(network["codec_ms"])
        + float(network["rtt_ms"])
        + upload_ms
        + float(network["cloud_ms"])
        + download_ms
        + float(network["fusion_ms"])
    )


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco_train2017_new_disjoint":
        raise ValueError("Unexpected dataset protocol")
    roles = protocol.get("roles", {})
    if roles != {"development": 5000, "audit": 1500, "registered_final": 3000}:
        raise ValueError("Registered role counts changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("Registered budgets changed")
    if tuple(protocol.get("modes", ())) != ("fixed_call", "fixed_byte"):
        raise ValueError("Registered modes changed")
    if protocol.get("development_only_training") is not True:
        raise ValueError("Training must be development-only")
    if protocol.get("audit_used_for_training") is not False:
        raise ValueError("Audit cannot train models")
    if protocol.get("registered_final_used_for_selection") is not False:
        raise ValueError("Final cannot select models or comparators")
    if protocol.get("registered_final_access_once") is not True:
        raise ValueError("Final must be one-time")


def deterministic_roles(
    population_ids: Iterable[int],
    excluded_ids: Iterable[int],
    role_counts: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, list[int]]:
    excluded = set(map(int, excluded_ids))
    population = sorted(set(map(int, population_ids)) - excluded)
    required = sum(map(int, role_counts.values()))
    if len(population) < required:
        raise ValueError(f"Insufficient population: {len(population)} < {required}")
    ordered = sorted(
        population,
        key=lambda image_id: (
            hashlib.sha256(f"{int(seed)}:{image_id}".encode()).digest(),
            image_id,
        ),
    )
    output: dict[str, list[int]] = {}
    offset = 0
    for role, count in role_counts.items():
        output[str(role)] = sorted(ordered[offset : offset + int(count)])
        offset += int(count)
    observed = [image_id for values in output.values() for image_id in values]
    if len(observed) != len(set(observed)) or set(observed) & excluded:
        raise RuntimeError("Role split is not disjoint from itself or exclusions")
    return output


def context_indices(
    development_ids: Sequence[int],
    target_id: int,
    *,
    context_size: int,
    seed: int,
) -> list[int]:
    ids = list(map(int, development_ids))
    target = int(target_id)
    if target not in ids:
        raise ValueError("Target is not in the development population")
    candidates = [image_id for image_id in ids if image_id != target]
    count = min(max(0, int(context_size)), len(candidates))
    rng = random.Random(int(seed) + target * 1_000_003)
    return sorted(rng.sample(candidates, count))


def _predictions(records: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    return [copy.deepcopy(item) for record in records for item in record.get(field, ())]


def dual_metric_context_reward(
    metric_backend: Any,
    context_records: Sequence[Mapping[str, Any]],
    target_record: Mapping[str, Any],
) -> dict[str, Any]:
    target_id = int(target_record["image_id"])
    context_ids = [int(record["image_id"]) for record in context_records]
    if target_id in context_ids:
        raise ValueError("Reward context contains the target image")
    image_ids = [*context_ids, target_id]
    base = _predictions([*context_records, target_record], "edge_predictions")
    action = _predictions(context_records, "edge_predictions")
    action.extend(_predictions([target_record], "cloud_predictions"))
    base_metric = metric_backend.evaluate(base, image_ids)
    action_metric = metric_backend.evaluate(action, image_ids)
    scale = len(image_ids)
    delta_ap = float(action_metric["map50_95"]) - float(base_metric["map50_95"])
    delta_ap50 = float(action_metric["map50"]) - float(base_metric["map50"])
    return {
        "target_image_id": target_id,
        "context_image_ids": context_ids,
        "scale": scale,
        "base_ap50_95": float(base_metric["map50_95"]),
        "cloud_ap50_95": float(action_metric["map50_95"]),
        "base_ap50": float(base_metric["map50"]),
        "cloud_ap50": float(action_metric["map50"]),
        "reward_ap50_95": scale * delta_ap,
        "reward_ap50": scale * delta_ap50,
    }


def select_fixed_call(
    image_ids: Sequence[int], scores: Sequence[float], *, budget: float
) -> set[int]:
    ids = np.asarray(image_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if ids.shape != values.shape:
        raise ValueError("Image IDs and scores must have matching shapes")
    count = int(round(len(ids) * float(budget)))
    if count < 0 or count > len(ids):
        raise ValueError("Budget selects an invalid image count")
    order = np.lexsort((ids, -values))
    return set(map(int, ids[order[:count]]))


def freeze_comparators(
    rows: Sequence[Mapping[str, Any]], *, external_methods: set[str] | None = None
) -> dict[str, str]:
    allowed = EXTERNAL_METHODS if external_methods is None else set(external_methods)
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["method"]) in allowed:
            groups[(str(row["mode"]), float(row["budget"]))].append(row)
    if not groups:
        raise ValueError("No external comparator rows were provided")
    frozen = {}
    for (mode, budget), candidates in sorted(groups.items()):
        winner = max(
            candidates,
            key=lambda row: (float(row["map50_95"]), -float(row.get("upload_kb_per_image", 0.0)), str(row["method"])),
        )
        frozen[f"{mode}:{budget:g}"] = str(winner["method"])
    return frozen


def audit_gate(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    minimum_gain: float,
    required_gain_points: int,
    maximum_other_loss: float,
) -> dict[str, Any]:
    gains = [float(row["ap_gain"]) for row in comparisons]
    if len(gains) != 3:
        raise ValueError("Audit gate requires three fixed-call work points")
    gain_points = sum(value >= float(minimum_gain) for value in gains)
    passed = gain_points >= int(required_gain_points) and min(gains) >= -float(
        maximum_other_loss
    )
    return {"passed": bool(passed), "gain_points": gain_points, "minimum_gain": min(gains)}


def summarize_bootstrap(values: np.ndarray, *, confidence: float) -> dict[str, float]:
    observed = np.asarray(values, dtype=np.float64)
    if observed.ndim != 1 or not len(observed) or not np.isfinite(observed).all():
        raise ValueError("Bootstrap values must be one finite vector")
    alpha = (1.0 - float(confidence)) / 2.0
    if not 0.0 <= alpha < 0.5:
        raise ValueError("Bootstrap confidence must be in (0, 1]")
    return {
        "mean": float(observed.mean()),
        "ci_low": float(np.quantile(observed, alpha)),
        "ci_high": float(np.quantile(observed, 1.0 - alpha)),
        "probability_gt_zero": float(np.mean(observed > 0.0)),
    }


def registered_final_gate(
    intervals: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
    capacities: Sequence[Mapping[str, Any]],
    *,
    required_superior_points: int,
    noninferiority_margin: float,
    maximum_kb_increase_fraction: float,
) -> dict[str, Any]:
    if len(intervals) != 3 or len(costs) != 3:
        raise ValueError("Registered final requires three fixed-call points")
    superior_points = sum(float(row["ci_low"]) > 0.0 for row in intervals)
    noninferior = all(
        float(row["ci_low"]) >= -float(noninferiority_margin) for row in intervals
    )
    cost_safe = all(
        float(row["candidate_kb"])
        <= float(row["comparator_kb"]) * (1.0 + float(maximum_kb_increase_fraction))
        for row in costs
    )
    capacity_safe = all(float(row["used"]) <= float(row["capacity"]) + 1e-6 for row in capacities)
    passed = (
        superior_points >= int(required_superior_points)
        and noninferior
        and cost_safe
        and capacity_safe
    )
    return {
        "passed": bool(passed),
        "superior_points": int(superior_points),
        "noninferior_all": bool(noninferior),
        "fixed_call_cost_safe": bool(cost_safe),
        "fixed_byte_capacity_safe": bool(capacity_safe),
    }


def assert_final_allowed(audit_completion: Mapping[str, Any], *, final_exists: bool) -> None:
    if audit_completion.get("complete") is not True:
        raise RuntimeError("Audit must complete before registered final")
    if audit_completion.get("passed") is not True:
        raise RuntimeError("Audit gate failed; registered final remains sealed")
    if final_exists:
        raise RuntimeError("Registered final has already been evaluated")


def bootstrap_remap(
    ground_truth: Mapping[str, Any],
    left_predictions: Sequence[Mapping[str, Any]],
    right_predictions: Sequence[Mapping[str, Any]],
    *,
    sampled_image_ids: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    images = {int(row["id"]): row for row in ground_truth["images"]}
    annotations: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ground_truth["annotations"]:
        annotations[int(row["image_id"])].append(row)
    prediction_groups = []
    for predictions in (left_predictions, right_predictions):
        groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in predictions:
            groups[int(row["image_id"])].append(row)
        prediction_groups.append(groups)

    remapped_gt = {"images": [], "annotations": [], "categories": copy.deepcopy(ground_truth["categories"])}
    remapped_predictions = [[], []]
    annotation_id = 1
    for new_id, source_id in enumerate(map(int, sampled_image_ids), start=1):
        image = copy.deepcopy(images[source_id])
        image["id"] = new_id
        remapped_gt["images"].append(image)
        for source in annotations[source_id]:
            row = copy.deepcopy(source)
            row["id"] = annotation_id
            row["image_id"] = new_id
            annotation_id += 1
            remapped_gt["annotations"].append(row)
        for target, groups in zip(remapped_predictions, prediction_groups):
            for source in groups[source_id]:
                row = copy.deepcopy(source)
                row["image_id"] = new_id
                target.append(row)
    return remapped_gt, remapped_predictions[0], remapped_predictions[1]
