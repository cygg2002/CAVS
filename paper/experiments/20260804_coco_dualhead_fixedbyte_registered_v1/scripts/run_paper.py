#!/usr/bin/env python3
"""Run registered stages for the COCO DualHead fixed-byte experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
DIAGNOSTIC = REPO / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1"
OOF = REPO / "paper/experiments/20260804_coco_budgetset_value_cost_oof_v1"
EXTERNAL = REPO / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines"
for module_root in (
    REPO,
    ROOT / "scripts",
    DIAGNOSTIC / "scripts",
    OOF / "scripts",
    EXTERNAL / "scripts",
    REPO / "paper/experiments/CA_AORIC",
):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from baseline_core import (  # noqa: E402
    build_output_feature,
    empirical_cdf,
    fit_reward_mlp,
    predict_reward_mlp,
)
from experiment_core import select_costeligible  # noqa: E402
from oof_train import (  # noqa: E402
    fit_dual_router,
    fit_value_router,
    predict_dual_value,
    predict_value_router,
)
from paper_core import (  # noqa: E402
    assert_final_allowed,
    bootstrap_remap,
    canonical_hash,
    deterministic_evaluation_roles,
    fixed_byte_audit_gate,
    fixed_byte_final_gate,
    freeze_primary_comparators,
    original_e2e_ms,
    select_fixed_call,
    summarize_bootstrap,
    validate_protocol,
)
from run_experiment import _fit_one_router, _predict_router, _record_tensors  # noqa: E402


_BOOTSTRAP_STATE: dict[str, Any] = {}
EXTERNAL_METHODS = (
    "ORIC-AP50",
    "MORIC-AP50",
    "ORIC-AP5095",
    "MORIC-AP5095",
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def source_fingerprint(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = protocol["paths"]
    source = Path(paths["development_source"])
    observed = {
        "protocol": file_sha256(PROTOCOL_PATH),
        "annotations": file_sha256(Path(paths["annotations"])),
        "canonical_excluded_roles": file_sha256(Path(paths["canonical_excluded_roles"])),
        "previous_registered_manifest": file_sha256(Path(paths["previous_registered_manifest"])),
        "edge_model": file_sha256(Path(paths["edge_model"])),
        "cloud_model": file_sha256(Path(paths["cloud_model"])),
        "development_manifest": file_sha256(source / "runs/prepare/split_manifest.json"),
        "development_edge": file_sha256(source / "runs/inference/development/edge/predictions.json"),
        "development_cloud": file_sha256(source / "runs/inference/development/cloud/predictions.json"),
        "development_rewards": file_sha256(source / "runs/rewards/contextual_rewards.jsonl"),
        "dualhead_core": file_sha256(OOF / "scripts/oof_core.py"),
        "dualhead_train": file_sha256(OOF / "scripts/oof_train.py"),
        "budgetset_core": file_sha256(DIAGNOSTIC / "scripts/experiment_core.py"),
        "budgetset_train": file_sha256(DIAGNOSTIC / "scripts/run_experiment.py"),
        "external_core": file_sha256(EXTERNAL / "scripts/baseline_core.py"),
        "registered_core": file_sha256(ROOT / "scripts/paper_core.py"),
        "registered_runner": file_sha256(Path(__file__)),
        "registered_inference": file_sha256(ROOT / "scripts/infer_original.py"),
    }
    if observed["edge_model"] != protocol["hashes"]["edge_model"]:
        raise RuntimeError("Edge model hash mismatch")
    if observed["cloud_model"] != protocol["hashes"]["cloud_model"]:
        raise RuntimeError("Cloud model hash mismatch")
    return observed


def _role_values(path: Path) -> list[dict]:
    payload = read_json(path)
    roles = payload.get("roles", payload)
    return [dict(row) for values in roles.values() for row in values]


def run_prepare() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    completion_path = RUNS / "prepare/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "prepare/report.json")
    fingerprint = source_fingerprint(protocol)
    paths = protocol["paths"]
    canonical_rows = _role_values(Path(paths["canonical_excluded_roles"]))
    previous_rows = _role_values(Path(paths["previous_registered_manifest"]))
    canonical_ids = {int(row["image_id"]) for row in canonical_rows}
    previous_ids = {int(row["image_id"]) for row in previous_rows}
    excluded = canonical_ids | previous_ids
    annotations = read_json(Path(paths["annotations"]))
    images = {int(row["id"]): row for row in annotations["images"]}
    roles = deterministic_evaluation_roles(
        images,
        excluded,
        protocol["roles"],
        seed=int(protocol["split_seed"]),
    )
    image_root = Path(paths["image_root"])
    role_records = {}
    for role, image_ids in roles.items():
        rows = []
        for image_id in image_ids:
            image = images[int(image_id)]
            path = (image_root / str(image["file_name"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "image_id": int(image_id),
                    "file_name": str(image["file_name"]),
                    "path": str(path),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "payload_bytes": int(path.stat().st_size),
                }
            )
        role_records[role] = rows
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "roles": role_records,
        "role_counts": {role: len(rows) for role, rows in role_records.items()},
        "canonical_excluded_count": len(canonical_ids),
        "previous_excluded_count": len(previous_ids),
        "combined_excluded_count": len(excluded),
        "canonical_previous_overlap": len(canonical_ids & previous_ids),
        "excluded_ids_sha256": canonical_hash(sorted(excluded)),
        "new_ids_sha256": canonical_hash(
            sorted(int(row["image_id"]) for rows in role_records.values() for row in rows)
        ),
        "source_fingerprint": fingerprint,
        "audit_metrics_accessed": False,
        "registered_final_accessed": False,
    }
    manifest_path = RUNS / "prepare/split_manifest.json"
    atomic_json(manifest_path, manifest)
    report = {
        key: manifest[key]
        for key in (
            "protocol_id",
            "role_counts",
            "canonical_excluded_count",
            "previous_excluded_count",
            "combined_excluded_count",
            "canonical_previous_overlap",
            "new_ids_sha256",
            "audit_metrics_accessed",
            "registered_final_accessed",
        )
    }
    atomic_json(RUNS / "prepare/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "manifest_sha256": file_sha256(manifest_path),
            "source_fingerprint": fingerprint,
            "audit_metrics_accessed": False,
            "registered_final_accessed": False,
        },
    )
    return report


def _group_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict]]:
    output: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        output[int(row["image_id"])].append(dict(row))
    return output


def _records_from_sources(
    sources: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    cloud_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict]:
    edge_by_id = _group_predictions(edge_rows)
    cloud_by_id = _group_predictions(cloud_rows)
    annotations = read_json(Path(protocol["paths"]["annotations"]))
    category_ids = [int(row["id"]) for row in annotations["categories"]]
    category_index = {value: index for index, value in enumerate(category_ids)}
    network = protocol["network"]
    output = []
    for source in sources:
        image_id = int(source["image_id"])
        edge_predictions = edge_by_id.get(image_id, [])
        cloud_predictions = cloud_by_id.get(image_id, [])
        edge_boxes = []
        for prediction in edge_predictions:
            x, y, width, height = map(float, prediction["bbox"])
            edge_boxes.append(
                {
                    "xyxy": [x, y, x + width, y + height],
                    "conf": float(prediction["score"]),
                    "cls": category_index[int(prediction["category_id"])],
                }
            )
        payload = int(source["payload_bytes"])
        response = len(cloud_predictions) * int(network["response_bytes_per_detection"])
        full_e2e = original_e2e_ms(payload, response, network)
        output.append(
            {
                **source,
                "edge_boxes": edge_boxes,
                "edge_predictions": edge_predictions,
                "cloud_predictions": cloud_predictions,
                "full_payload_bytes": payload,
                "image_stats": {},
                "regions": [],
                "actions": [
                    {
                        "action_id": "edge",
                        "action_type": "edge",
                        "payload_bytes": 0,
                        "response_bytes": len(edge_predictions)
                        * int(network["response_bytes_per_detection"]),
                        "cloud_calls": 0,
                        "cloud_gflops": 0.0,
                        "e2e_ms": float(network["edge_ms"]),
                        "resource_costs": {"upload": 0.0, "compute": 0.0, "e2e": 0.0},
                        "resolutions": [],
                    },
                    {
                        "action_id": "full-original",
                        "action_type": "full",
                        "payload_bytes": payload,
                        "response_bytes": response,
                        "cloud_calls": 1,
                        "cloud_gflops": float(protocol["detector"]["cloud_gflops"]),
                        "e2e_ms": full_e2e,
                        "resource_costs": {
                            "upload": float(payload),
                            "compute": float(protocol["detector"]["cloud_gflops"]),
                            "e2e": full_e2e - float(network["edge_ms"]),
                        },
                        "resolutions": [640],
                    },
                ],
            }
        )
    return output


def _inference_root(role: str, detector: str) -> Path:
    return RUNS / "inference" / role / detector


def _role_records(role: str, protocol: Mapping[str, Any]) -> list[dict]:
    split = read_json(RUNS / "prepare/split_manifest.json")
    if role not in split["roles"]:
        raise ValueError(f"Unknown evaluation role: {role}")
    rows = {}
    for detector in ("edge", "cloud"):
        root = _inference_root(role, detector)
        completion = read_json(root / "completion.json")
        if completion.get("complete") is not True:
            raise RuntimeError(f"Incomplete inference: {role}/{detector}")
        predictions = root / "predictions.json"
        if completion.get("predictions_sha256") != file_sha256(predictions):
            raise RuntimeError(f"Inference hash changed: {role}/{detector}")
        rows[detector] = read_json(predictions)
    return _records_from_sources(
        split["roles"][role], rows["edge"], rows["cloud"], protocol
    )


def _development_records(protocol: Mapping[str, Any]) -> list[dict]:
    source = Path(protocol["paths"]["development_source"])
    manifest = read_json(source / "runs/prepare/split_manifest.json")
    rows = {}
    for detector in ("edge", "cloud"):
        root = source / f"runs/inference/development/{detector}"
        completion = read_json(root / "completion.json")
        predictions = root / "predictions.json"
        if completion.get("complete") is not True or completion.get(
            "predictions_sha256"
        ) != file_sha256(predictions):
            raise RuntimeError(f"Frozen development inference changed: {detector}")
        rows[detector] = read_json(predictions)
    records = _records_from_sources(
        manifest["roles"]["development"], rows["edge"], rows["cloud"], protocol
    )
    if len(records) != 5000:
        raise RuntimeError("Frozen development population changed")
    return records


def _reward_vectors(records: Sequence[Mapping[str, Any]], path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {int(row["target_image_id"]): row for row in rows}
    image_ids = {int(record["image_id"]) for record in records}
    if set(by_id) != image_ids:
        raise RuntimeError("Frozen rewards and development records differ")
    ap50 = np.asarray([float(by_id[int(record["image_id"])]["reward_ap50"]) for record in records])
    ap5095 = np.asarray([float(by_id[int(record["image_id"])]["reward_ap50_95"]) for record in records])
    return ap50, ap5095


def _category_ids(protocol: Mapping[str, Any]) -> list[int]:
    return [
        int(row["id"])
        for row in read_json(Path(protocol["paths"]["annotations"]))["categories"]
    ]


def _set_protocol(protocol: Mapping[str, Any]) -> dict:
    return {
        "budgets": list(map(float, protocol["budgets"])),
        "max_boxes": int(protocol["set_model"]["max_boxes"]),
        "max_actions": 2,
        "deploy_confidence": float(protocol["set_model"]["deploy_confidence"]),
        "model": dict(protocol["set_model"]),
        "resource_profile": "upload",
    }


def run_train() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    completion_path = RUNS / "train/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "train/report.json")
    prepared = read_json(RUNS / "prepare/completion.json")
    if prepared.get("complete") is not True:
        raise RuntimeError("Prepare must complete before training")
    fingerprint = source_fingerprint(protocol)
    if prepared.get("source_fingerprint") != fingerprint:
        raise RuntimeError("Frozen source fingerprint changed")
    records = _development_records(protocol)
    rewards_path = Path(protocol["paths"]["development_source"]) / "runs/rewards/contextual_rewards.jsonl"
    ap50, ap5095 = _reward_vectors(records, rewards_path)
    raw_rewards = np.stack((np.zeros_like(ap5095), ap5095), axis=1).astype(np.float32)
    set_protocol = _set_protocol(protocol)
    data = _record_tensors(records, raw_rewards, set_protocol)
    category_ids = _category_ids(protocol)
    seeds = list(map(int, protocol["set_model"]["seeds"]))
    current, value, dual = [], [], []
    image_ids = [int(record["image_id"]) for record in records]
    for seed in seeds:
        current.append(
            _fit_one_router(
                records,
                data,
                raw_rewards,
                set_protocol,
                category_count=len(category_ids),
                seed=seed,
            )
        )
        value.append(
            fit_value_router(
                image_ids,
                data,
                set_protocol,
                category_count=len(category_ids),
                seed=seed,
            )
        )
        dual.append(
            fit_dual_router(
                records,
                data,
                raw_rewards,
                set_protocol,
                category_count=len(category_ids),
                seed=seed,
            )
        )
    features = np.asarray(
        [
            build_output_feature(
                record["edge_predictions"],
                width=int(record["width"]),
                height=int(record["height"]),
                category_ids=category_ids,
                top_k=int(protocol["baseline_mlp"]["top_k"]),
            )
            for record in records
        ],
        dtype=np.float64,
    )
    baseline_config = dict(protocol["baseline_mlp"])
    baseline_config.pop("top_k")
    cdf50 = empirical_cdf(ap50, ap50)
    cdf5095 = empirical_cdf(ap5095, ap5095)
    external = {
        "ORIC-AP50": fit_reward_mlp(features, ap50, np.ones_like(ap50), baseline_config),
        "MORIC-AP50": fit_reward_mlp(features, cdf50, cdf50, baseline_config),
        "ORIC-AP5095": fit_reward_mlp(features, ap5095, np.ones_like(ap5095), baseline_config),
        "MORIC-AP5095": fit_reward_mlp(features, cdf5095, cdf5095, baseline_config),
    }
    artifact = {
        "current": current,
        "value": value,
        "dual": dual,
        "external": external,
        "category_ids": category_ids,
        "set_protocol": set_protocol,
        "source_fingerprint": fingerprint,
        "development_ids_sha256": canonical_hash(image_ids),
    }
    checkpoint_path = RUNS / "train/checkpoints.pkl"
    atomic_pickle(checkpoint_path, artifact)
    report = {
        "protocol_id": protocol["protocol_id"],
        "development_images": len(records),
        "selected_epochs": {
            "Current-BudgetSet": [int(row["selected_epoch"]) for row in current],
            "SetValue": [int(row["selected_epoch"]) for row in value],
            "DualHead": [int(row["selected_epoch"]) for row in dual],
            **{name: int(row["selected_epoch"]) for name, row in external.items()},
        },
        "audit_used": False,
        "registered_final_used": False,
        "source_fingerprint": fingerprint,
    }
    atomic_json(RUNS / "train/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "report_sha256": file_sha256(RUNS / "train/report.json"),
            "audit_used": False,
            "registered_final_used": False,
        },
    )
    return report


def _load_checkpoints() -> dict:
    completion = read_json(RUNS / "train/completion.json")
    path = RUNS / "train/checkpoints.pkl"
    if completion.get("complete") is not True or completion.get(
        "checkpoint_sha256"
    ) != file_sha256(path):
        raise RuntimeError("Frozen training checkpoint is incomplete or changed")
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    current = source_fingerprint(read_json(PROTOCOL_PATH))
    if artifact.get("source_fingerprint") != current:
        raise RuntimeError("Frozen training implementation or source changed")
    return artifact


def _method_scores(records: list[dict], protocol: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    category_ids = list(map(int, artifact["category_ids"]))
    set_protocol = artifact["set_protocol"]
    data = _record_tensors(records, None, set_protocol)
    current = np.mean(
        [_predict_router(row, data, set_protocol, len(category_ids)) for row in artifact["current"]],
        axis=0,
    )[:, :, 1]
    value = np.mean(
        [predict_value_router(row, data, set_protocol, category_count=len(category_ids)) for row in artifact["value"]],
        axis=0,
    )[:, :, 1]
    dual = np.mean(
        [predict_dual_value(row, records, data, set_protocol, category_count=len(category_ids)) for row in artifact["dual"]],
        axis=0,
    )[:, :, 1]
    features = np.asarray(
        [
            build_output_feature(
                record["edge_predictions"],
                width=int(record["width"]),
                height=int(record["height"]),
                category_ids=category_ids,
                top_k=int(protocol["baseline_mlp"]["top_k"]),
            )
            for record in records
        ],
        dtype=np.float64,
    )
    budget_count = len(protocol["budgets"])
    output = {
        "Current-BudgetSet": current,
        "SetValue": value,
        "DualHead": dual,
    }
    for method, checkpoint in artifact["external"].items():
        values = predict_reward_mlp(checkpoint, features)
        output[method] = np.repeat(values[None, :], budget_count, axis=0)
    return output


def _score_matrix(full_scores: np.ndarray) -> np.ndarray:
    values = np.asarray(full_scores, dtype=np.float64)
    return np.stack((np.zeros_like(values), values), axis=1)


def _selected_predictions(records: Sequence[Mapping[str, Any]], selected_ids: set[int]) -> list[dict]:
    return [
        copy.deepcopy(prediction)
        for record in records
        for prediction in (
            record["cloud_predictions"]
            if int(record["image_id"]) in selected_ids
            else record["edge_predictions"]
        )
    ]


def _selection_metrics(records: Sequence[Mapping[str, Any]], selected_ids: set[int], protocol: Mapping[str, Any]) -> dict:
    selected = [record for record in records if int(record["image_id"]) in selected_ids]
    count = max(1, len(records))
    payload = sum(int(record["payload_bytes"]) for record in selected)
    response = sum(len(record["cloud_predictions"]) for record in selected) * int(
        protocol["network"]["response_bytes_per_detection"]
    )
    e2e = sum(
        float(record["actions"][1]["e2e_ms"])
        if int(record["image_id"]) in selected_ids
        else float(protocol["network"]["edge_ms"])
        for record in records
    ) / count
    return {
        "upload_kb_per_image": payload / count / 1024.0,
        "response_kb_per_image": response / count / 1024.0,
        "upload_image_ratio": len(selected) / count,
        "cloud_calls_per_image": len(selected) / count,
        "gflops_per_image": float(protocol["detector"]["edge_gflops"])
        + len(selected) / count * float(protocol["detector"]["cloud_gflops"]),
        "mean_e2e_ms": e2e,
    }


def _evaluate_role(role: str, protocol: Mapping[str, Any], artifact: Mapping[str, Any]) -> tuple[list[dict], dict[str, list[int]]]:
    from ca_aoric import CocoMetricBackend

    records = _role_records(role, protocol)
    image_ids = [int(record["image_id"]) for record in records]
    scores = _method_scores(records, protocol, artifact)
    metric = CocoMetricBackend(Path(protocol["paths"]["annotations"]))
    full_capacity = sum(int(record["payload_bytes"]) for record in records)
    rows, selections = [], {}
    for budget_index, budget in enumerate(map(float, protocol["budgets"])):
        for method, method_scores in scores.items():
            values = method_scores[budget_index]
            fixed_ids = select_fixed_call(image_ids, values, budget=budget)
            byte_result = select_costeligible(
                records,
                _score_matrix(values),
                resource_profile="upload",
                capacity=budget * full_capacity,
                full_only=True,
            )
            if float(byte_result["total_cost"]) > budget * full_capacity + 1e-6:
                raise RuntimeError(f"Fixed-byte capacity exceeded for {method}")
            byte_ids = {
                int(item["image_id"])
                for item in byte_result["selected"]
                if str(item["action_type"]) == "full"
            }
            for mode, selected_ids, extra in (
                ("fixed_call", fixed_ids, {}),
                (
                    "fixed_byte",
                    byte_ids,
                    {
                        "capacity_bytes": budget * full_capacity,
                        "used_capacity_bytes": float(byte_result["total_cost"]),
                        "lambda": float(byte_result["lambda"]),
                    },
                ),
            ):
                key = f"{method}|{mode}|{budget:g}"
                selections[key] = sorted(map(int, selected_ids))
                rows.append(
                    {
                        "method": method,
                        "mode": mode,
                        "budget": budget,
                        **metric.evaluate(_selected_predictions(records, selected_ids), image_ids),
                        **_selection_metrics(records, selected_ids, protocol),
                        **extra,
                    }
                )
    expected = len(protocol["methods"]) * len(protocol["modes"]) * len(protocol["budgets"])
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete result matrix: {len(rows)} != {expected}")
    return rows, selections


def _result_row(rows: Sequence[Mapping[str, Any]], method: str, mode: str, budget: float) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if str(row["method"]) == method
        and str(row["mode"]) == mode
        and abs(float(row["budget"]) - float(budget)) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {method}/{mode}/{budget:g}")
    return matches[0]


def run_audit() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    completion_path = RUNS / "audit/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "audit/report.json")
    artifact = _load_checkpoints()
    rows, selections = _evaluate_role("audit", protocol, artifact)
    frozen = freeze_primary_comparators(rows, mode="fixed_byte")
    comparisons, capacities = [], []
    for budget in map(float, protocol["budgets"]):
        comparator = frozen[f"fixed_byte:{budget:g}"]
        candidate = _result_row(rows, "DualHead", "fixed_byte", budget)
        baseline = _result_row(rows, comparator, "fixed_byte", budget)
        comparisons.append(
            {
                "budget": budget,
                "candidate": "DualHead",
                "comparator": comparator,
                "candidate_ap": float(candidate["map50_95"]),
                "comparator_ap": float(baseline["map50_95"]),
                "ap_gain": float(candidate["map50_95"]) - float(baseline["map50_95"]),
            }
        )
        capacities.append(
            {
                "budget": budget,
                "used": float(candidate["used_capacity_bytes"]),
                "capacity": float(candidate["capacity_bytes"]),
            }
        )
    gate_config = protocol["audit_gate"]
    gate = fixed_byte_audit_gate(
        comparisons,
        capacities,
        minimum_gain=float(gate_config["minimum_gain"]),
        required_gain_points=int(gate_config["required_gain_points"]),
        maximum_other_loss=float(gate_config["maximum_other_loss"]),
    )
    output = RUNS / "audit"
    atomic_csv(output / "results.csv", rows)
    atomic_json(output / "selections.json", selections)
    atomic_json(output / "frozen_comparators.json", frozen)
    report = {
        "protocol_id": protocol["protocol_id"],
        "role": "audit",
        "primary_mode": "fixed_byte",
        "rows": rows,
        "comparisons": comparisons,
        "capacities": capacities,
        "frozen_comparators": frozen,
        "gate": gate,
        "registered_final_accessed": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "passed": bool(gate["passed"]),
            "report_sha256": file_sha256(output / "report.json"),
            "results_sha256": file_sha256(output / "results.csv"),
            "selections_sha256": file_sha256(output / "selections.json"),
            "comparators_sha256": file_sha256(output / "frozen_comparators.json"),
            "registered_final_accessed": False,
        },
    )
    return report


def run_open_final() -> dict:
    audit = read_json(RUNS / "audit/completion.json")
    completion = RUNS / "registered_final/completion.json"
    assert_final_allowed(audit, final_exists=completion.is_file())
    access = RUNS / "registered_final/accessed.json"
    if access.is_file():
        return read_json(access)
    split = read_json(RUNS / "prepare/split_manifest.json")
    frozen = RUNS / "audit/frozen_comparators.json"
    if audit.get("comparators_sha256") != file_sha256(frozen):
        raise RuntimeError("Frozen comparator hash changed")
    value = {
        "accessed": True,
        "role": "registered_final",
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "audit_completion_sha256": file_sha256(RUNS / "audit/completion.json"),
        "comparators_sha256": file_sha256(frozen),
        "image_ids_sha256": canonical_hash(
            [int(row["image_id"]) for row in split["roles"]["registered_final"]]
        ),
        "purpose": "audit-gated inference, registered point estimates, and paired bootstrap",
    }
    atomic_json(access, value)
    return value


def _registered_ground_truth(protocol: Mapping[str, Any], image_ids: Sequence[int]) -> dict:
    source = read_json(Path(protocol["paths"]["annotations"]))
    allowed = set(map(int, image_ids))
    output = {key: value for key, value in source.items() if key not in {"images", "annotations"}}
    output["images"] = [row for row in source["images"] if int(row["id"]) in allowed]
    output["annotations"] = [row for row in source["annotations"] if int(row["image_id"]) in allowed]
    if {int(row["id"]) for row in output["images"]} != allowed:
        raise RuntimeError("Registered-final ground-truth population mismatch")
    return output


def _in_memory_ap(ground_truth: dict, predictions: list[dict]) -> float:
    from faster_coco_eval import COCO, COCOeval_faster

    annotation = COCO()
    annotation.dataset = ground_truth
    annotation.createIndex()
    result = annotation.loadRes(predictions)
    evaluator = COCOeval_faster(annotation, result, iouType="bbox", print_function=lambda *args, **kwargs: None)
    evaluator.params.imgIds = list(range(1, len(ground_truth["images"]) + 1))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats_as_dict["AP_all"])


def _bootstrap_worker(item: tuple[int, np.ndarray]) -> tuple[int, float]:
    replicate, indices = item
    source_ids = [_BOOTSTRAP_STATE["image_ids"][int(index)] for index in indices]
    ground_truth, candidate, comparator = bootstrap_remap(
        _BOOTSTRAP_STATE["ground_truth"],
        _BOOTSTRAP_STATE["candidate_predictions"],
        _BOOTSTRAP_STATE["comparator_predictions"],
        sampled_image_ids=source_ids,
    )
    return int(replicate), float(
        _in_memory_ap(ground_truth, candidate) - _in_memory_ap(ground_truth, comparator)
    )


def _paired_bootstrap(
    *,
    ground_truth: dict,
    image_ids: list[int],
    candidate_predictions: list[dict],
    comparator_predictions: list[dict],
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    global _BOOTSTRAP_STATE
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(image_ids), size=(int(replicates), len(image_ids)), dtype=np.int32)
    indices_sha256 = hashlib.sha256(indices.tobytes()).hexdigest()
    _BOOTSTRAP_STATE = {
        "ground_truth": ground_truth,
        "image_ids": list(map(int, image_ids)),
        "candidate_predictions": candidate_predictions,
        "comparator_predictions": comparator_predictions,
    }
    values = np.empty(int(replicates), dtype=np.float64)
    context = mp.get_context("fork")
    workers = min(24, max(1, (os.cpu_count() or 1) - 2))
    with context.Pool(processes=workers) as pool:
        tasks = ((index, indices[index]) for index in range(int(replicates)))
        for completed, (replicate, delta) in enumerate(
            pool.imap_unordered(_bootstrap_worker, tasks, chunksize=1), start=1
        ):
            values[replicate] = delta
            if completed % 25 == 0 or completed == int(replicates):
                print(f"[bootstrap] {completed}/{replicates}", flush=True)
    _BOOTSTRAP_STATE = {}
    return values, indices_sha256


def run_final() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(RUNS / "audit/completion.json")
    completion_path = RUNS / "registered_final/completion.json"
    assert_final_allowed(audit, final_exists=completion_path.is_file())
    access = RUNS / "registered_final/accessed.json"
    if not access.is_file():
        raise RuntimeError("Registered final must be opened before inference")
    frozen_path = RUNS / "audit/frozen_comparators.json"
    if audit.get("comparators_sha256") != file_sha256(frozen_path):
        raise RuntimeError("Frozen comparator hash changed")
    output = RUNS / "registered_final"
    point_path = output / "point_report.json"
    selections_path = output / "selections.json"
    if point_path.is_file() and selections_path.is_file():
        rows = read_json(point_path)["rows"]
        selections = read_json(selections_path)
    else:
        rows, selections = _evaluate_role("registered_final", protocol, _load_checkpoints())
        atomic_csv(output / "results.csv", rows)
        atomic_json(selections_path, selections)
        atomic_json(
            point_path,
            {
                "protocol_id": protocol["protocol_id"],
                "role": "registered_final",
                "rows": rows,
                "comparators_selected_on": "audit",
                "model_or_threshold_selection_on_final": False,
            },
        )
    frozen = read_json(frozen_path)
    records = _role_records("registered_final", protocol)
    image_ids = [int(record["image_id"]) for record in records]
    ground_truth = _registered_ground_truth(protocol, image_ids)
    config = protocol["final_gate"]
    intervals, bootstrap_reports, capacities, comparisons = [], [], [], []
    for budget in map(float, protocol["budgets"]):
        comparator = frozen[f"fixed_byte:{budget:g}"]
        candidate_key = f"DualHead|fixed_byte|{budget:g}"
        comparator_key = f"{comparator}|fixed_byte|{budget:g}"
        candidate_ids = set(map(int, selections[candidate_key]))
        comparator_ids = set(map(int, selections[comparator_key]))
        bootstrap_path = output / f"bootstrap_{budget:g}.json"
        if bootstrap_path.is_file():
            bootstrap = read_json(bootstrap_path)
            values = np.asarray(bootstrap["values"], dtype=np.float64)
            if len(values) != int(config["bootstrap_replicates"]):
                raise RuntimeError("Bootstrap replicate count changed")
        else:
            values, indices_sha256 = _paired_bootstrap(
                ground_truth=ground_truth,
                image_ids=image_ids,
                candidate_predictions=_selected_predictions(records, candidate_ids),
                comparator_predictions=_selected_predictions(records, comparator_ids),
                replicates=int(config["bootstrap_replicates"]),
                seed=int(config["bootstrap_seed"]),
            )
            bootstrap = {
                "budget": budget,
                "candidate": "DualHead",
                "comparator": comparator,
                "replicates": len(values),
                "seed": int(config["bootstrap_seed"]),
                "indices_sha256": indices_sha256,
                "values": values.tolist(),
            }
            atomic_json(bootstrap_path, bootstrap)
        summary = summarize_bootstrap(values, confidence=float(config["confidence"]))
        interval = {"budget": budget, "candidate": "DualHead", "comparator": comparator, **summary}
        intervals.append(interval)
        bootstrap_reports.append({**interval, "replicates": len(values)})
        candidate = _result_row(rows, "DualHead", "fixed_byte", budget)
        baseline = _result_row(rows, comparator, "fixed_byte", budget)
        comparisons.append(
            {
                "budget": budget,
                "comparator": comparator,
                "candidate_ap": float(candidate["map50_95"]),
                "comparator_ap": float(baseline["map50_95"]),
                "ap_gain": float(candidate["map50_95"]) - float(baseline["map50_95"]),
            }
        )
        for method, row in (("DualHead", candidate), (str(comparator), baseline)):
            capacities.append(
                {
                    "budget": budget,
                    "method": method,
                    "used": float(row["used_capacity_bytes"]),
                    "capacity": float(row["capacity_bytes"]),
                }
            )
    gate = fixed_byte_final_gate(
        intervals,
        capacities,
        required_superior_points=int(config["required_superior_points"]),
        noninferiority_margin=float(config["remaining_noninferiority_margin"]),
    )
    report = {
        "protocol_id": protocol["protocol_id"],
        "role": "registered_final",
        "primary_mode": "fixed_byte",
        "registered_final_accessed": True,
        "comparators_frozen_on_audit": frozen,
        "comparisons": comparisons,
        "bootstrap": bootstrap_reports,
        "capacities": capacities,
        "gate": gate,
        "final_used_for_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "passed": bool(gate["passed"]),
            "access_sha256": file_sha256(access),
            "results_sha256": file_sha256(output / "results.csv"),
            "selections_sha256": file_sha256(selections_path),
            "report_sha256": file_sha256(output / "report.json"),
            "registered_final_accessed": True,
            "final_used_for_selection": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "audit", "open_final", "final"),
        required=True,
    )
    args = parser.parse_args()
    report = globals()[f"run_{args.stage}"]()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
