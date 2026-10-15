#!/usr/bin/env python3
"""Run gated stages for the registered COCO BudgetSet paper experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import tempfile
from collections import defaultdict
import csv

import numpy as np

from paper_core import (
    assert_final_allowed,
    audit_gate,
    bootstrap_remap,
    canonical_hash,
    deterministic_roles,
    context_indices,
    dual_metric_context_reward,
    freeze_comparators,
    original_e2e_ms,
    registered_final_gate,
    select_fixed_call,
    summarize_bootstrap,
    validate_reward_isolation,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
for module_root in (REPO, REPO / "paper/experiments/CA_AORIC"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

EXTERNAL_SCRIPTS = REPO / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines/scripts"
DIAGNOSTIC_SCRIPTS = REPO / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts"
for module_root in (EXTERNAL_SCRIPTS, DIAGNOSTIC_SCRIPTS):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))


_BOOTSTRAP_STATE: dict = {}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
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


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def source_fingerprint(protocol: dict) -> dict:
    paths = protocol["paths"]
    observed = {
        "protocol": file_sha256(PROTOCOL_PATH),
        "annotations": file_sha256(Path(paths["annotations"])),
        "excluded_roles": file_sha256(Path(paths["excluded_roles"])),
        "edge_model": file_sha256(Path(paths["edge_model"])),
        "cloud_model": file_sha256(Path(paths["cloud_model"])),
    }
    if observed["edge_model"] != protocol["hashes"]["edge_model"]:
        raise RuntimeError("Edge-model hash mismatch")
    if observed["cloud_model"] != protocol["hashes"]["cloud_model"]:
        raise RuntimeError("Cloud-model hash mismatch")
    return observed


def _group_predictions(rows: list[dict]) -> dict[int, list[dict]]:
    output: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        output[int(row["image_id"])].append(dict(row))
    return output


def _inference_path(role: str, detector: str, name: str) -> Path:
    return RUNS / "inference" / role / detector / name


def _role_records(role: str, protocol: dict) -> list[dict]:
    split = read_json(RUNS / "prepare/split_manifest.json")
    if role not in split["roles"]:
        raise ValueError(f"Unknown registered role: {role}")
    for detector in ("edge", "cloud"):
        completion = read_json(_inference_path(role, detector, "completion.json"))
        if completion.get("complete") is not True:
            raise RuntimeError(f"Incomplete inference: {role}/{detector}")
    edge_rows = read_json(_inference_path(role, "edge", "predictions.json"))
    cloud_rows = read_json(_inference_path(role, "cloud", "predictions.json"))
    edge_by_id = _group_predictions(edge_rows)
    cloud_by_id = _group_predictions(cloud_rows)
    annotations = read_json(Path(protocol["paths"]["annotations"]))
    category_ids = [int(row["id"]) for row in annotations["categories"]]
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    network = protocol["network"]
    output = []
    for source in split["roles"][role]:
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
        edge_action = {
            "action_id": "edge",
            "action_type": "edge",
            "payload_bytes": 0,
            "response_bytes": len(edge_predictions) * int(network["response_bytes_per_detection"]),
            "cloud_calls": 0,
            "cloud_gflops": 0.0,
            "e2e_ms": float(network["edge_ms"]),
            "resource_costs": {"upload": 0.0, "compute": 0.0, "e2e": 0.0},
            "resolutions": [],
        }
        full_action = {
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
        }
        output.append(
            {
                **source,
                "edge_boxes": edge_boxes,
                "edge_predictions": edge_predictions,
                "cloud_predictions": cloud_predictions,
                "full_payload_bytes": payload,
                "image_stats": {},
                "regions": [],
                "actions": [edge_action, full_action],
            }
        )
    return output


def _load_reward_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid reward checkpoint line {line_number}") from error
    return rows


def _set_protocol(protocol: dict) -> dict:
    return {
        "budgets": list(map(float, protocol["budgets"])),
        "max_boxes": int(protocol["set_model"]["max_boxes"]),
        "max_actions": 2,
        "deploy_confidence": float(protocol["set_model"]["deploy_confidence"]),
        "model": dict(protocol["set_model"]),
        "resource_profile": "upload",
    }


def _category_ids(protocol: dict) -> list[int]:
    annotations = read_json(Path(protocol["paths"]["annotations"]))
    return [int(row["id"]) for row in annotations["categories"]]


def _reward_vectors(records: list[dict], reward_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    by_id = {int(row["target_image_id"]): row for row in reward_rows}
    image_ids = {int(record["image_id"]) for record in records}
    if set(by_id) != image_ids:
        raise RuntimeError("Reward and development populations differ")
    ap50 = np.asarray([float(by_id[int(record["image_id"])]["reward_ap50"]) for record in records])
    ap5095 = np.asarray(
        [float(by_id[int(record["image_id"])]["reward_ap50_95"]) for record in records]
    )
    return ap50, ap5095


def _score_matrix(full_scores: np.ndarray) -> np.ndarray:
    values = np.asarray(full_scores, dtype=np.float64)
    return np.stack((np.zeros_like(values), values), axis=1)


def run_prepare() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    completion_path = RUNS / "prepare/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "prepare/report.json")
    fingerprint = source_fingerprint(protocol)
    annotations = read_json(Path(protocol["paths"]["annotations"]))
    excluded_source = read_json(Path(protocol["paths"]["excluded_roles"]))
    excluded = {
        int(row["image_id"])
        for values in excluded_source["roles"].values()
        for row in values
    }
    images = {int(row["id"]): row for row in annotations["images"]}
    roles = deterministic_roles(
        images,
        excluded,
        protocol["roles"],
        seed=int(protocol["split_seed"]),
    )
    image_root = Path(protocol["paths"]["image_root"])
    role_records = {}
    for role, image_ids in roles.items():
        records = []
        for image_id in image_ids:
            image = images[image_id]
            path = (image_root / str(image["file_name"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(
                {
                    "image_id": image_id,
                    "file_name": str(image["file_name"]),
                    "path": str(path),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "payload_bytes": path.stat().st_size,
                }
            )
        role_records[role] = records
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "roles": role_records,
        "role_counts": {role: len(records) for role, records in role_records.items()},
        "excluded_count": len(excluded),
        "excluded_ids_sha256": canonical_hash(sorted(excluded)),
        "new_ids_sha256": canonical_hash(
            sorted(record["image_id"] for records in role_records.values() for record in records)
        ),
        "source_fingerprint": fingerprint,
        "audit_metrics_accessed": False,
        "registered_final_metrics_accessed": False,
    }
    manifest_path = RUNS / "prepare/split_manifest.json"
    atomic_json(manifest_path, manifest)
    report = {
        "protocol_id": protocol["protocol_id"],
        "role_counts": manifest["role_counts"],
        "excluded_count": len(excluded),
        "new_ids_sha256": manifest["new_ids_sha256"],
        "audit_metrics_accessed": False,
        "registered_final_metrics_accessed": False,
        "source_fingerprint": fingerprint,
    }
    atomic_json(RUNS / "prepare/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "manifest_sha256": file_sha256(manifest_path),
            "source_fingerprint": fingerprint,
            "audit_metrics_accessed": False,
            "registered_final_metrics_accessed": False,
        },
    )
    return report


def run_rewards() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    completion_path = RUNS / "rewards/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "rewards/report.json")
    prepared = read_json(RUNS / "prepare/completion.json")
    if prepared.get("complete") is not True:
        raise RuntimeError("Prepare must complete before rewards")
    records = _role_records("development", protocol)
    development_ids = {int(record["image_id"]) for record in records}
    by_id = {int(record["image_id"]): record for record in records}
    reward_path = RUNS / "rewards/contextual_rewards.jsonl"
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_reward_rows(reward_path)
    validate_reward_isolation(rows, development_ids=development_ids)
    completed_ids = [int(row["target_image_id"]) for row in rows]
    completed_ids_set = set(completed_ids)
    if len(completed_ids) != len(completed_ids_set):
        raise RuntimeError("Reward checkpoint contains duplicate target IDs")

    from ca_aoric import CocoMetricBackend

    metric = CocoMetricBackend(Path(protocol["paths"]["annotations"]))
    ordered_ids = [int(record["image_id"]) for record in records]
    progress_every = int(protocol["reward"]["progress_every"])
    with reward_path.open("a", encoding="utf-8") as handle:
        for index, target_id in enumerate(ordered_ids):
            if target_id in completed_ids_set:
                continue
            context_ids = context_indices(
                ordered_ids,
                target_id,
                context_size=int(protocol["reward"]["context_size"]),
                seed=int(protocol["reward"]["seed"]),
            )
            row = dual_metric_context_reward(
                metric,
                [by_id[image_id] for image_id in context_ids],
                by_id[target_id],
            )
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            if (index + 1) % progress_every == 0:
                os.fsync(handle.fileno())
                print(f"[reward] {index + 1}/{len(ordered_ids)}", flush=True)
        os.fsync(handle.fileno())

    rows = _load_reward_rows(reward_path)
    validate_reward_isolation(rows, development_ids=development_ids)
    if {int(row["target_image_id"]) for row in rows} != development_ids:
        raise RuntimeError("Development reward population is incomplete")
    completion_inputs = {
        "protocol": file_sha256(PROTOCOL_PATH),
        "annotations": file_sha256(Path(protocol["paths"]["annotations"])),
        "edge_predictions": file_sha256(_inference_path("development", "edge", "predictions.json")),
        "cloud_predictions": file_sha256(_inference_path("development", "cloud", "predictions.json")),
        "reward_implementation": file_sha256(ROOT / "scripts/paper_core.py"),
    }
    report = {
        "protocol_id": protocol["protocol_id"],
        "role": "development",
        "images": len(rows),
        "context_size": int(protocol["reward"]["context_size"]),
        "reward_ap50_min": min(float(row["reward_ap50"]) for row in rows),
        "reward_ap50_max": max(float(row["reward_ap50"]) for row in rows),
        "reward_ap50_95_min": min(float(row["reward_ap50_95"]) for row in rows),
        "reward_ap50_95_max": max(float(row["reward_ap50_95"]) for row in rows),
        "audit_ids_used": False,
        "registered_final_ids_used": False,
        "inputs": completion_inputs,
    }
    atomic_json(RUNS / "rewards/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "rewards_sha256": file_sha256(reward_path),
            "report_sha256": file_sha256(RUNS / "rewards/report.json"),
            "audit_ids_used": False,
            "registered_final_ids_used": False,
            "inputs": completion_inputs,
        },
    )
    return report


def run_train() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    completion_path = RUNS / "train/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "train/report.json")
    reward_completion = read_json(RUNS / "rewards/completion.json")
    reward_path = RUNS / "rewards/contextual_rewards.jsonl"
    if reward_completion.get("complete") is not True or reward_completion.get("rewards_sha256") != file_sha256(reward_path):
        raise RuntimeError("Frozen development rewards are incomplete or changed")
    records = _role_records("development", protocol)
    reward_rows = _load_reward_rows(reward_path)
    ap50, ap5095 = _reward_vectors(records, reward_rows)
    set_protocol = _set_protocol(protocol)

    from baseline_core import build_output_feature, empirical_cdf, fit_reward_mlp
    from run_experiment import _fit_one_router, _record_tensors

    raw_rewards = np.stack((np.zeros_like(ap5095), ap5095), axis=1).astype(np.float32)
    data = _record_tensors(records, raw_rewards, set_protocol)
    category_ids = _category_ids(protocol)
    routers = [
        _fit_one_router(
            records,
            data,
            raw_rewards,
            set_protocol,
            category_count=len(category_ids),
            seed=int(seed),
        )
        for seed in protocol["set_model"]["seeds"]
    ]
    features = []
    for record in records:
        features.append(
            build_output_feature(
                record["edge_predictions"],
                width=int(record["width"]),
                height=int(record["height"]),
                category_ids=category_ids,
                top_k=int(protocol["baseline_mlp"]["top_k"]),
            )
        )
    features = np.asarray(features, dtype=np.float64)
    baseline_config = dict(protocol["baseline_mlp"])
    baseline_config.pop("top_k")
    checkpoints = {
        "ORIC-AP50": fit_reward_mlp(features, ap50, np.ones_like(ap50), baseline_config),
        "MORIC-AP50": None,
        "ORIC-AP5095": fit_reward_mlp(features, ap5095, np.ones_like(ap5095), baseline_config),
        "MORIC-AP5095": None,
    }
    moric50 = empirical_cdf(ap50, ap50)
    moric5095 = empirical_cdf(ap5095, ap5095)
    checkpoints["MORIC-AP50"] = fit_reward_mlp(features, moric50, moric50, baseline_config)
    checkpoints["MORIC-AP5095"] = fit_reward_mlp(
        features, moric5095, moric5095, baseline_config
    )
    artifact = {
        "routers": routers,
        "external": checkpoints,
        "category_ids": category_ids,
        "development_ids": [int(record["image_id"]) for record in records],
        "set_protocol": set_protocol,
        "source_hashes": {
            "rewards": file_sha256(reward_path),
            "diagnostic_core": file_sha256(DIAGNOSTIC_SCRIPTS / "experiment_core.py"),
            "diagnostic_runner": file_sha256(DIAGNOSTIC_SCRIPTS / "run_experiment.py"),
            "external_core": file_sha256(EXTERNAL_SCRIPTS / "baseline_core.py"),
        },
    }
    checkpoint_path = RUNS / "train/checkpoints.pkl"
    atomic_pickle(checkpoint_path, artifact)
    report = {
        "protocol_id": protocol["protocol_id"],
        "development_images": len(records),
        "set_selected_epochs": [int(row["selected_epoch"]) for row in routers],
        "set_validation_losses": [float(row["best_validation_loss"]) for row in routers],
        "external_selected_epochs": {
            name: int(checkpoint["selected_epoch"]) for name, checkpoint in checkpoints.items()
        },
        "reward_ap50_mean": float(ap50.mean()),
        "reward_ap50_95_mean": float(ap5095.mean()),
        "audit_ids_used": False,
        "final_ids_used": False,
        "source_hashes": artifact["source_hashes"],
    }
    atomic_json(RUNS / "train/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "report_sha256": file_sha256(RUNS / "train/report.json"),
            "audit_ids_used": False,
            "final_ids_used": False,
        },
    )
    return report


def _load_checkpoints() -> dict:
    completion = read_json(RUNS / "train/completion.json")
    path = RUNS / "train/checkpoints.pkl"
    if completion.get("complete") is not True or completion.get("checkpoint_sha256") != file_sha256(path):
        raise RuntimeError("Training checkpoint is incomplete or changed")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _method_scores(records: list[dict], protocol: dict, artifact: dict) -> dict[str, np.ndarray]:
    from baseline_core import build_output_feature, predict_reward_mlp
    from run_experiment import _predict_router, _record_tensors

    category_ids = list(map(int, artifact["category_ids"]))
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
    data = _record_tensors(records, None, artifact["set_protocol"])
    candidate = np.mean(
        [
            _predict_router(checkpoint, data, artifact["set_protocol"], len(category_ids))
            for checkpoint in artifact["routers"]
        ],
        axis=0,
    )[:, :, 1]
    output = {"BudgetSet": candidate}
    for method, checkpoint in artifact["external"].items():
        values = predict_reward_mlp(checkpoint, features)
        output[method] = np.repeat(values[None, :], len(protocol["budgets"]), axis=0)
    return output


def _selected_predictions(records: list[dict], selected_ids: set[int]) -> list[dict]:
    return [
        dict(prediction)
        for record in records
        for prediction in (
            record["cloud_predictions"] if int(record["image_id"]) in selected_ids else record["edge_predictions"]
        )
    ]


def _selection_metrics(records: list[dict], selected_ids: set[int], protocol: dict) -> dict:
    divisor = max(1, len(records))
    selected = [record for record in records if int(record["image_id"]) in selected_ids]
    payload = sum(int(record["payload_bytes"]) for record in selected)
    response = sum(len(record["cloud_predictions"]) for record in selected) * int(
        protocol["network"]["response_bytes_per_detection"]
    )
    e2e = sum(
        float(record["actions"][1]["e2e_ms"])
        if int(record["image_id"]) in selected_ids
        else float(protocol["network"]["edge_ms"])
        for record in records
    ) / divisor
    return {
        "upload_kb_per_image": payload / divisor / 1024.0,
        "response_kb_per_image": response / divisor / 1024.0,
        "cloud_calls_per_image": len(selected) / divisor,
        "upload_image_ratio": len(selected) / divisor,
        "gflops_per_image": float(protocol["detector"]["edge_gflops"])
        + len(selected) / divisor * float(protocol["detector"]["cloud_gflops"]),
        "mean_e2e_ms": e2e,
    }


def _evaluate_role(role: str, protocol: dict, artifact: dict) -> tuple[list[dict], dict[str, list[int]]]:
    from ca_aoric import CocoMetricBackend
    from experiment_core import select_costeligible

    records = _role_records(role, protocol)
    image_ids = [int(record["image_id"]) for record in records]
    scores = _method_scores(records, protocol, artifact)
    metric = CocoMetricBackend(Path(protocol["paths"]["annotations"]))
    full_capacity = sum(int(record["payload_bytes"]) for record in records)
    rows = []
    selections = {}
    for budget_index, budget in enumerate(map(float, protocol["budgets"])):
        for method, budget_scores in scores.items():
            fixed_ids = select_fixed_call(
                image_ids, budget_scores[budget_index], budget=budget
            )
            for mode, selected_ids, extra in (
                ("fixed_call", fixed_ids, {}),
                ("fixed_byte", None, {}),
            ):
                if mode == "fixed_byte":
                    result = select_costeligible(
                        records,
                        _score_matrix(budget_scores[budget_index]),
                        resource_profile="upload",
                        capacity=budget * full_capacity,
                        full_only=True,
                    )
                    selected_ids = {
                        int(item["image_id"])
                        for item in result["selected"]
                        if str(item["action_type"]) == "full"
                    }
                    extra = {
                        "capacity_bytes": budget * full_capacity,
                        "used_capacity_bytes": float(result["total_cost"]),
                        "lambda": float(result["lambda"]),
                    }
                predictions = _selected_predictions(records, selected_ids)
                official = metric.evaluate(predictions, image_ids)
                key = f"{method}|{mode}|{budget:g}"
                selections[key] = sorted(selected_ids)
                rows.append(
                    {
                        "method": method,
                        "mode": mode,
                        "budget": budget,
                        **official,
                        **_selection_metrics(records, selected_ids, protocol),
                        **extra,
                    }
                )
    return rows, selections


def run_audit() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    completion_path = RUNS / "audit/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "audit/report.json")
    artifact = _load_checkpoints()
    rows, selections = _evaluate_role("audit", protocol, artifact)
    frozen = freeze_comparators(rows)
    comparisons = []
    for budget in map(float, protocol["budgets"]):
        key = f"fixed_call:{budget:g}"
        comparator = frozen[key]
        candidate_row = next(
            row for row in rows if row["method"] == "BudgetSet" and row["mode"] == "fixed_call" and float(row["budget"]) == budget
        )
        comparator_row = next(
            row for row in rows if row["method"] == comparator and row["mode"] == "fixed_call" and float(row["budget"]) == budget
        )
        comparisons.append(
            {
                "budget": budget,
                "comparator": comparator,
                "candidate_ap": float(candidate_row["map50_95"]),
                "comparator_ap": float(comparator_row["map50_95"]),
                "ap_gain": float(candidate_row["map50_95"]) - float(comparator_row["map50_95"]),
            }
        )
    gate_config = protocol["audit_gate"]
    gate = audit_gate(
        comparisons,
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
        "rows": rows,
        "comparisons": comparisons,
        "gate": gate,
        "registered_final_accessed": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "passed": bool(gate["passed"]),
            "results_sha256": file_sha256(output / "results.csv"),
            "report_sha256": file_sha256(output / "report.json"),
            "comparators_sha256": file_sha256(output / "frozen_comparators.json"),
            "registered_final_accessed": False,
        },
    )
    return report


def _registered_ground_truth(protocol: dict, image_ids: list[int]) -> dict:
    source = read_json(Path(protocol["paths"]["annotations"]))
    allowed = set(map(int, image_ids))
    output = {
        key: value
        for key, value in source.items()
        if key not in {"images", "annotations"}
    }
    output["images"] = [row for row in source["images"] if int(row["id"]) in allowed]
    output["annotations"] = [
        row for row in source["annotations"] if int(row["image_id"]) in allowed
    ]
    observed = {int(row["id"]) for row in output["images"]}
    if observed != allowed:
        raise RuntimeError("Registered-final ground-truth population mismatch")
    return output


def _in_memory_ap(ground_truth: dict, predictions: list[dict]) -> float:
    from faster_coco_eval import COCO, COCOeval_faster

    annotation = COCO()
    annotation.dataset = ground_truth
    annotation.createIndex()
    result = annotation.loadRes(predictions)
    evaluator = COCOeval_faster(
        annotation,
        result,
        iouType="bbox",
        print_function=lambda *args, **kwargs: None,
    )
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
    delta = _in_memory_ap(ground_truth, candidate) - _in_memory_ap(
        ground_truth, comparator
    )
    return int(replicate), float(delta)


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
    indices = generator.integers(
        0, len(image_ids), size=(int(replicates), len(image_ids)), dtype=np.int32
    )
    indices_sha256 = hashlib.sha256(indices.tobytes()).hexdigest()
    _BOOTSTRAP_STATE = {
        "ground_truth": ground_truth,
        "image_ids": list(map(int, image_ids)),
        "candidate_predictions": candidate_predictions,
        "comparator_predictions": comparator_predictions,
    }
    values = np.empty(int(replicates), dtype=np.float64)
    workers = min(24, max(1, (os.cpu_count() or 1) - 2))
    context = mp.get_context("fork")
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


def _result_row(rows: list[dict], method: str, mode: str, budget: float) -> dict:
    matches = [
        row
        for row in rows
        if row["method"] == method
        and row["mode"] == mode
        and abs(float(row["budget"]) - float(budget)) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result row for {method}/{mode}/{budget:g}")
    return matches[0]


def run_final() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    audit_completion = read_json(RUNS / "audit/completion.json")
    final_completion = RUNS / "registered_final/completion.json"
    frozen_comparators = RUNS / "audit/frozen_comparators.json"
    if not frozen_comparators.is_file():
        raise RuntimeError("frozen_comparators.json is required")
    if audit_completion.get("comparators_sha256") != file_sha256(frozen_comparators):
        raise RuntimeError("Frozen audit comparator hash mismatch")
    assert_final_allowed(audit_completion, final_exists=final_completion.is_file())
    output = RUNS / "registered_final"
    output.mkdir(parents=True, exist_ok=True)
    access_path = output / "accessed.json"
    if not access_path.is_file():
        split = read_json(RUNS / "prepare/split_manifest.json")
        atomic_json(
            access_path,
            {
                "accessed": True,
                "role": "registered_final",
                "protocol_sha256": file_sha256(PROTOCOL_PATH),
                "audit_completion_sha256": file_sha256(RUNS / "audit/completion.json"),
                "comparators_sha256": file_sha256(frozen_comparators),
                "image_ids_sha256": canonical_hash(
                    [int(row["image_id"]) for row in split["roles"]["registered_final"]]
                ),
                "purpose": "single registered point evaluation and paired bootstrap",
            },
        )

    point_path = output / "point_report.json"
    selections_path = output / "selections.json"
    if point_path.is_file() and selections_path.is_file():
        point_report = read_json(point_path)
        rows = point_report["rows"]
        selections = read_json(selections_path)
    else:
        artifact = _load_checkpoints()
        rows, selections = _evaluate_role("registered_final", protocol, artifact)
        atomic_csv(output / "results.csv", rows)
        atomic_json(selections_path, selections)
        point_report = {
            "protocol_id": protocol["protocol_id"],
            "role": "registered_final",
            "rows": rows,
            "comparators_selected_on": "audit",
            "model_or_threshold_selection_on_final": False,
        }
        atomic_json(point_path, point_report)

    frozen = read_json(frozen_comparators)
    records = _role_records("registered_final", protocol)
    image_ids = [int(record["image_id"]) for record in records]
    ground_truth = _registered_ground_truth(protocol, image_ids)
    final_config = protocol["final_gate"]
    bootstrap_reports = []
    intervals = []
    costs = []
    for budget_index, budget in enumerate(map(float, protocol["budgets"])):
        comparator = frozen[f"fixed_call:{budget:g}"]
        candidate_key = f"BudgetSet|fixed_call|{budget:g}"
        comparator_key = f"{comparator}|fixed_call|{budget:g}"
        candidate_ids = set(map(int, selections[candidate_key]))
        comparator_ids = set(map(int, selections[comparator_key]))
        bootstrap_path = output / f"bootstrap_{budget:g}.json"
        if bootstrap_path.is_file():
            bootstrap = read_json(bootstrap_path)
            if int(bootstrap["replicates"]) != int(final_config["bootstrap_replicates"]):
                raise RuntimeError("Frozen bootstrap replicate count mismatch")
            values = np.asarray(bootstrap["values"], dtype=np.float64)
        else:
            values, indices_sha256 = _paired_bootstrap(
                ground_truth=ground_truth,
                image_ids=image_ids,
                candidate_predictions=_selected_predictions(records, candidate_ids),
                comparator_predictions=_selected_predictions(records, comparator_ids),
                replicates=int(final_config["bootstrap_replicates"]),
                seed=int(final_config["bootstrap_seed"]),
            )
            bootstrap = {
                "budget": budget,
                "candidate": "BudgetSet",
                "comparator": comparator,
                "replicates": len(values),
                "seed": int(final_config["bootstrap_seed"]),
                "indices_sha256": indices_sha256,
                "values": values.tolist(),
            }
            atomic_json(bootstrap_path, bootstrap)
        summary = summarize_bootstrap(
            values, confidence=float(final_config["confidence"])
        )
        interval = {
            "budget": budget,
            "candidate": "BudgetSet",
            "comparator": comparator,
            **summary,
        }
        intervals.append(interval)
        bootstrap_reports.append({**interval, "replicates": len(values)})
        candidate_row = _result_row(rows, "BudgetSet", "fixed_call", budget)
        comparator_row = _result_row(rows, comparator, "fixed_call", budget)
        costs.append(
            {
                "budget": budget,
                "candidate_kb": float(candidate_row["upload_kb_per_image"]),
                "comparator_kb": float(comparator_row["upload_kb_per_image"]),
            }
        )

    capacities = []
    for budget in map(float, protocol["budgets"]):
        row = _result_row(rows, "BudgetSet", "fixed_byte", budget)
        capacities.append(
            {
                "budget": budget,
                "used": float(row["used_capacity_bytes"]),
                "capacity": float(row["capacity_bytes"]),
            }
        )
    gate = registered_final_gate(
        intervals,
        costs,
        capacities,
        required_superior_points=int(final_config["required_superior_points"]),
        noninferiority_margin=float(final_config["remaining_noninferiority_margin"]),
        maximum_kb_increase_fraction=float(
            final_config["maximum_fixed_call_kb_increase_fraction"]
        ),
    )
    comparisons = []
    for budget in map(float, protocol["budgets"]):
        comparator = frozen[f"fixed_call:{budget:g}"]
        candidate = _result_row(rows, "BudgetSet", "fixed_call", budget)
        baseline = _result_row(rows, comparator, "fixed_call", budget)
        comparisons.append(
            {
                "budget": budget,
                "comparator": comparator,
                "candidate_ap": float(candidate["map50_95"]),
                "comparator_ap": float(baseline["map50_95"]),
                "ap_gain": float(candidate["map50_95"]) - float(baseline["map50_95"]),
            }
        )
    report = {
        "protocol_id": protocol["protocol_id"],
        "role": "registered_final",
        "registered_final_accessed": True,
        "comparators_frozen_on_audit": frozen,
        "comparisons": comparisons,
        "bootstrap": bootstrap_reports,
        "costs": costs,
        "capacities": capacities,
        "gate": gate,
        "final_used_for_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(
        final_completion,
        {
            "complete": True,
            "passed": bool(gate["passed"]),
            "access_sha256": file_sha256(access_path),
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
        "--stage", choices=("prepare", "rewards", "train", "audit", "final"), required=True
    )
    args = parser.parse_args()
    report = globals()[f"run_{args.stage}"]()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
