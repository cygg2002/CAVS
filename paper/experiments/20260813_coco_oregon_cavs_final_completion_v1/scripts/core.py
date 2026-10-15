"""Pure contracts for the final CAVS paper-completion experiment."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


ROW_KEYS = ("dataset", "role", "method", "policy", "mode", "budget", "seed")
MAX_RUNTIME_WORKERS = 32


def runtime_worker_override(raw: str | None, *, cpu_count: int) -> int | None:
    """Validate an execution-only worker override without changing experiment semantics."""
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"invalid CAVS_RUNTIME_WORKERS: {raw}") from error
    maximum = min(int(cpu_count), MAX_RUNTIME_WORKERS)
    if value < 1 or value > maximum:
        raise ValueError(f"CAVS_RUNTIME_WORKERS must be in [1, {maximum}]: {value}")
    return value


def category_ids_from_gt(
    ground_truth: Mapping[str, Any], *, expected_count: int
) -> list[int]:
    categories = ground_truth.get("categories")
    if not isinstance(categories, list):
        raise ValueError("frozen GT categories are missing")
    values = [int(category["id"]) for category in categories]
    if len(values) != int(expected_count) or len(values) != len(set(values)):
        raise ValueError(
            f"category schema changed: {len(values)}/{len(set(values))}/{expected_count}"
        )
    return values


def maximum_action_width(records: Sequence[Mapping[str, Any]]) -> int:
    """Return the widest per-image action set in a frozen role."""
    if not records:
        raise ValueError("action records are empty")
    widths: list[int] = []
    for record in records:
        actions = record.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"image has no action set: {record.get('image_id')}")
        if str(actions[0].get("action_id")) != "edge":
            raise ValueError(f"edge must be the first action: {record.get('image_id')}")
        widths.append(len(actions))
    return max(widths)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "posthoc_paper_completion_final_reuse":
        raise ValueError("this experiment must remain a post-hoc final-reuse extension")
    if protocol.get("primary_method") != "CAVS-OQ-Core":
        raise ValueError("the independent OQ core method changed")
    if protocol.get("roi_extension") != "CAVS-OQR-MAUR":
        raise ValueError("the frozen ROI extension changed")
    if protocol.get("final_is_hidden") is not False:
        raise ValueError("COCO/Oregon final is not a new hidden split")
    if protocol.get("final_used_for_training_or_selection") is not False:
        raise ValueError("final cannot be used for training or selection")
    if list(map(float, protocol.get("budgets", []))) != [0.1, 0.25, 0.4]:
        raise ValueError("paper budget grid changed")
    if protocol.get("modes") != ["fixed_byte", "fixed_call"]:
        raise ValueError("evaluation modes changed")
    if protocol.get("action_spaces", {}).get("OQ") != ["edge", "original", "q90"]:
        raise ValueError("core action space changed")
    if protocol.get("action_spaces", {}).get("OQR") != [
        "edge", "original", "q90", "current@640"
    ]:
        raise ValueError("ROI action space changed")
    expected = {
        "coco": ({"fit": 9678, "calibration": 571, "final": 5000}, 80),
        "oregon": ({"fit": 9923, "calibration": 4961, "final": 4962}, 45),
    }
    if set(protocol.get("datasets", {})) != set(expected):
        raise ValueError("dataset set changed")
    for dataset, (roles, classes) in expected.items():
        config = protocol["datasets"][dataset]
        if config.get("roles") != roles or int(config.get("category_count", -1)) != classes:
            raise ValueError(f"canonical contract changed: {dataset}")
        if len(config.get("model_seeds", [])) != 3:
            raise ValueError(f"three model seeds are required: {dataset}")
    if int(protocol.get("fold_count", 0)) != 5:
        raise ValueError("five OOF folds are required")
    if int(protocol.get("bootstrap", {}).get("random_seeds", 0)) != 50:
        raise ValueError("Random-50 contract changed")
    if protocol.get("bootstrap", {}).get("nested_random") is not True:
        raise ValueError("Random bootstrap must be nested")


def validate_method_name(actual: str, claimed: str) -> None:
    if actual != claimed:
        raise ValueError(f"method semantic alias is forbidden: {actual} != {claimed}")


def validate_training_roles(roles: Sequence[str]) -> None:
    if "final" in set(map(str, roles)):
        raise ValueError("final is forbidden in training, reward and calibration")
    if set(map(str, roles)) - {"fit", "fit_oof", "calibration"}:
        raise ValueError(f"unknown training role: {roles}")


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    jobs: list[dict[str, Any]] = []

    def add(dataset: str, method: str, *, folds: bool) -> None:
        for seed in map(int, protocol["datasets"][dataset]["model_seeds"]):
            tokens: list[int | str] = [*range(int(protocol["fold_count"])), "full"] if folds else ["full"]
            for fold in tokens:
                predict_roles = ["fit_oof"] if fold != "full" else ["calibration", "final"]
                jobs.append(
                    {
                        "id": f"{dataset}|{method}|seed={seed}|fold={fold}",
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "fold": fold,
                        "train_roles": ["fit"],
                        "predict_roles": predict_roles,
                    }
                )

    for dataset in protocol["training"]["cost_in_value_datasets"]:
        add(str(dataset), "CostInValue-OQ", folds=True)
    for dataset in protocol["training"]["maur_retrain_datasets"]:
        add(str(dataset), "CAVS-OQR-MAUR", folds=True)
    for dataset in protocol["training"]["external_retrain_datasets"]:
        add(str(dataset), "ORIC-AP5095", folds=False)
        add(str(dataset), "MORIC-AP5095", folds=False)
    if len(jobs) != 60:
        raise ValueError(f"final training matrix changed: {len(jobs)}")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate training job")
    for job in jobs:
        validate_training_roles(job["train_roles"])
    return jobs


def two_stage_projection(
    action_scores: Mapping[int, Mapping[str, float]],
) -> tuple[dict[int, float], dict[int, str]]:
    image_scores: dict[int, float] = {}
    action_map: dict[int, str] = {}
    for raw_image_id, scores in action_scores.items():
        image_id = int(raw_image_id)
        cloud = [
            (str(action), float(value))
            for action, value in scores.items()
            if action != "edge" and np.isfinite(value)
        ]
        if not cloud:
            raise ValueError(f"image has no available cloud action: {image_id}")
        action, value = min(cloud, key=lambda item: (-item[1], item[0]))
        action_map[image_id] = action
        image_scores[image_id] = value
    return image_scores, action_map


def _payload(sample: Mapping[str, Any], action: str) -> int:
    if action == "edge":
        return 0
    values = sample.get("payload_bytes", {})
    if action in values:
        return int(values[action])
    if action == "original":
        return int(sample["full_payload_bytes"])
    raise ValueError(f"missing payload: {sample.get('image_id')}/{action}")


def validate_selection(
    samples: Sequence[Mapping[str, Any]],
    selection: Mapping[int, str],
    *,
    mode: str,
    budget: float,
) -> dict[str, Any]:
    by_id = {int(sample["image_id"]): sample for sample in samples}
    if not set(map(int, selection)) <= set(by_id):
        raise ValueError("selection contains unknown image ids")
    request = sum(_payload(by_id[int(image_id)], str(action)) for image_id, action in selection.items())
    if mode == "fixed_byte":
        capacity = int(
            math.floor(
                sum(_payload(sample, "original") for sample in samples)
                * float(budget)
            )
        )
        if request > capacity:
            raise ValueError(f"fixed-byte capacity exceeded: {request}>{capacity}")
    elif mode == "fixed_call":
        expected = int(math.floor(len(samples) * float(budget) + 0.5))
        if len(selection) != expected:
            raise ValueError(f"fixed-call count changed: {len(selection)} != {expected}")
        capacity = expected
    else:
        raise ValueError(mode)
    return {
        "request_bytes": int(request),
        "capacity_bytes": int(capacity),
        "utilization": float(request / capacity) if mode == "fixed_byte" and capacity else 1.0,
        "selected_images": len(selection),
    }


def nested_random_delta(
    candidate_metrics: np.ndarray, random_metrics: np.ndarray
) -> np.ndarray:
    candidate = np.asarray(candidate_metrics, dtype=np.float64)
    random = np.asarray(random_metrics, dtype=np.float64)
    if random.ndim != 2 or random.shape[1] != candidate.shape[0]:
        raise ValueError("random metrics must have shape [seed, bootstrap]")
    return candidate - random.mean(axis=0)


def expected_bootstrap_jobs(protocol: Mapping[str, Any], dataset: str) -> int:
    units = len(protocol["datasets"][dataset]["bootstrap_units"])
    # Two candidates, each with five standard comparators plus Random-Mean50.
    return len(protocol["budgets"]) * units * 2 * 6


def validate_unique_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [tuple(row.get(key) for key in ROW_KEYS) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result rows")
