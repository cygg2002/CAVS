#!/usr/bin/env python3
"""Evaluate smoke or full fit/calibration endpoint gates without touching final."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/protocol.json"
RUNS = ROOT / "runs"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_module("voc_torchvision_eval_gate", ROOT / "scripts/gate_core.py")
RUNTIME = load_module("voc_torchvision_eval_runtime", ROOT / "scripts/runtime_core.py")
PROTOCOL = GATE.read_protocol(PROTOCOL_PATH)
METRIC_PATH = RUNTIME.source_role_paths(ROOT, PROTOCOL, "fit")["metric"]
METRIC = load_module("voc_torchvision_eval_metric", METRIC_PATH)


def _population(data: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "full":
        return [dict(row) for row in data["images"]]
    if stage != "smoke" or data["role"] != "fit":
        raise ValueError("smoke evaluation is restricted to fit")
    return GATE.select_smoke_images(
        data["images"],
        count=int(PROTOCOL["smoke"]["images"]),
        seed=int(PROTOCOL["smoke"]["seed"]),
        category_ids=[int(row["id"]) for row in data["categories"]],
    )


def _action_rows(role: str, stage: str, action: str, expected_ids: set[int]) -> dict[int, dict]:
    root = RUNS / f"actions/{stage}/{role}/{action}"
    completion = root / "completion.json"
    rows_path = root / "rows.jsonl"
    if not completion.is_file():
        raise RuntimeError(f"inference is incomplete: {stage}/{role}/{action}")
    protocol_sha256 = RUNTIME.file_sha256(PROTOCOL_PATH)
    report = RUNTIME.read_protocol_bound_json(completion, protocol_sha256)
    expected_contract = {"stage": stage, "role": role, "action": action}
    if any(str(report.get(key)) != str(value) for key, value in expected_contract.items()):
        raise RuntimeError(f"inference contract changed: {stage}/{role}/{action}")
    if report.get("rows_sha256") != RUNTIME.file_sha256(rows_path):
        raise RuntimeError(f"inference hash changed: {stage}/{role}/{action}")
    rows = RUNTIME.read_checkpoint(
        rows_path,
        action=action,
        expected_ids=expected_ids,
        protocol_sha256=protocol_sha256,
    )
    if set(rows) != expected_ids:
        raise RuntimeError(f"inference population changed: {stage}/{role}/{action}")
    return rows


def _predictions(rows: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(prediction) for row in rows.values() for prediction in row["predictions"]]


def _evaluate_role(role: str, stage: str) -> dict[str, Any]:
    data = RUNTIME.load_source_role(ROOT, PROTOCOL, role)
    population = _population(data, stage)
    image_ids = [int(row["id"]) for row in population]
    expected = set(image_ids)
    evaluator = METRIC.VocMetricBackend(data["paths"]["gt"])
    actions = {}
    for action in ("edge", "original", "q90"):
        rows = _action_rows(role, stage, action, expected)
        predictions = _predictions(rows)
        metrics = evaluator.evaluate(predictions, image_ids, return_details=True)
        sanitization = {
            key: sum(int(row["sanitization"].get(key, 0)) for row in rows.values())
            for key in (
                "below_threshold",
                "filtered_non_voc",
                "nonfinite_boxes",
                "nonfinite_scores",
                "degenerate_boxes",
                "kept",
            )
        }
        actions[action] = {
            "metrics": metrics,
            "predictions": predictions,
            "rows": rows,
            "payload_bytes": sum(int(row["payload_bytes"]) for row in rows.values()),
            "sanitization": sanitization,
        }
    return {
        "role": role,
        "data": data,
        "population": population,
        "image_ids": image_ids,
        "actions": actions,
    }


_BOOT_STATE: dict[str, Any] = {}


def _bootstrap_worker(index: int) -> tuple[int, float]:
    os.nice(19)
    rng = np.random.default_rng(int(PROTOCOL["gate"]["paired_bootstrap_seed"]) + int(index))
    image_ids = _BOOT_STATE["image_ids"]
    sampled = rng.choice(image_ids, size=len(image_ids), replace=True).astype(np.int64).tolist()
    gt, predictions = GATE.bootstrap_remap(
        _BOOT_STATE["ground_truth"],
        [_BOOT_STATE["original"], _BOOT_STATE["edge"]],
        sampled,
    )
    evaluator = METRIC.VocMetricBackend(gt)
    remapped_ids = [int(row["id"]) for row in gt["images"]]
    original_ap = float(evaluator.evaluate(predictions[0], remapped_ids)["map50_95"])
    edge_ap = float(evaluator.evaluate(predictions[1], remapped_ids)["map50_95"])
    return int(index), original_ap - edge_ap


def _paired_bootstrap(calibration: Mapping[str, Any]) -> dict[str, Any]:
    output = RUNS / "gate2/bootstrap_partial.json"
    total = int(PROTOCOL["gate"]["paired_bootstrap_replicates"])
    protocol_sha256 = RUNTIME.file_sha256(PROTOCOL_PATH)
    input_sha256 = RUNTIME.text_sha256(
        "\n".join(
            (
                protocol_sha256,
                RUNTIME.file_sha256(RUNS / "actions/full/calibration/original/rows.jsonl"),
                RUNTIME.file_sha256(RUNS / "actions/full/calibration/edge/rows.jsonl"),
            )
        )
    )
    if output.is_file():
        partial = RUNTIME.read_protocol_bound_json(output, protocol_sha256)
        if str(partial.get("input_sha256")) != input_sha256:
            raise RuntimeError("bootstrap input changed")
        values = {int(key): float(value) for key, value in partial.get("values", {}).items()}
    else:
        values = {}
    pending = [index for index in range(total) if index not in values]
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        ground_truth=calibration["data"]["gt"],
        image_ids=np.asarray(calibration["image_ids"], dtype=np.int64),
        original=calibration["actions"]["original"]["predictions"],
        edge=calibration["actions"]["edge"]["predictions"],
    )
    workers = min(
        int(PROTOCOL["gate"]["paired_bootstrap_workers"]),
        max(1, os.cpu_count() or 1),
        max(1, len(pending)),
    )
    interval = int(PROTOCOL["gate"]["paired_bootstrap_checkpoint_interval"])
    if pending:
        with mp.get_context("fork").Pool(workers) as pool:
            for completed, (index, value) in enumerate(
                pool.imap_unordered(_bootstrap_worker, pending, chunksize=1), 1
            ):
                values[int(index)] = float(value)
                if completed % interval == 0 or completed == len(pending):
                    RUNTIME.atomic_json(
                        output,
                        {
                            "protocol_sha256": protocol_sha256,
                            "input_sha256": input_sha256,
                            "values": {str(key): values[key] for key in sorted(values)},
                            "replicates_complete": len(values),
                            "replicates_expected": total,
                        },
                    )
                    print(f"[calibration bootstrap] {len(values)}/{total}", flush=True)
    ordered = np.asarray([values[index] for index in range(total)], dtype=np.float64)
    lower, upper = np.percentile(ordered, [2.5, 97.5])
    return {
        "replicates": total,
        "delta_mean": float(ordered.mean()),
        "delta_median": float(np.median(ordered)),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "probability_delta_le_zero": float(np.mean(ordered <= 0.0)),
        "resampling_unit": "image",
        "multiplicity_preserved_by_id_remap": True,
        "workers": workers,
    }


def _public_role_result(role: Mapping[str, Any]) -> dict[str, Any]:
    return {
        action: {
            **{key: value for key, value in row["metrics"].items() if key != "per_class"},
            "payload_bytes": int(row["payload_bytes"]),
            "sanitization": dict(row["sanitization"]),
        }
        for action, row in role["actions"].items()
    }


def evaluate_smoke() -> dict[str, Any]:
    fit = _evaluate_role("fit", "smoke")
    invalid = {}
    deterministic = {}
    for action, row in fit["actions"].items():
        invalid[action] = sum(
            int(row["sanitization"][key])
            for key in ("nonfinite_boxes", "nonfinite_scores", "degenerate_boxes")
        )
        completion = RUNTIME.read_protocol_bound_json(
            RUNS / f"actions/smoke/fit/{action}/completion.json",
            RUNTIME.file_sha256(PROTOCOL_PATH),
        )
        deterministic[action] = completion.get("determinism") == {
            "images": 2,
            "exact_match": True,
        }
    passed = len(fit["image_ids"]) == int(PROTOCOL["smoke"]["images"]) and all(
        value == 0 for value in invalid.values()
    ) and all(deterministic.values())
    report = {
        "complete": True,
        "stage": "smoke",
        "images": len(fit["image_ids"]),
        "actions": _public_role_result(fit),
        "invalid_raw_outputs": invalid,
        "determinism_checks": deterministic,
        "passed": passed,
        "proceed_to_full_endpoints": passed,
        "protocol_sha256": RUNTIME.file_sha256(PROTOCOL_PATH),
        "final_accessed": False,
    }
    RUNTIME.atomic_json(RUNS / "gate1/report.json", report)
    RUNTIME.atomic_json(RUNS / "gate1/completion.json", report)
    return report


def evaluate_calibration_preflight() -> dict[str, Any]:
    if not (RUNS / "gate1/completion.json").is_file():
        raise RuntimeError("Gate 1 must complete before calibration preflight")
    gate1 = RUNTIME.read_protocol_bound_json(
        RUNS / "gate1/completion.json", RUNTIME.file_sha256(PROTOCOL_PATH)
    )
    if gate1.get("proceed_to_full_endpoints") is not True:
        raise RuntimeError("Gate 1 rejected calibration preflight")
    calibration = _evaluate_role("calibration", "full")
    public = _public_role_result(calibration)
    decision = GATE.assess_calibration_preflight(
        PROTOCOL,
        metrics={action: row["metrics"] for action, row in calibration["actions"].items()},
        original_bytes=public["original"]["payload_bytes"],
        q90_bytes=public["q90"]["payload_bytes"],
    )
    report = {
        "complete": True,
        "stage": "calibration_preflight",
        "calibration": public,
        **decision,
        "protocol_sha256": RUNTIME.file_sha256(PROTOCOL_PATH),
        "final_accessed": False,
    }
    RUNTIME.atomic_json(RUNS / "gate2a/report.json", report)
    RUNTIME.atomic_json(RUNS / "gate2a/completion.json", report)
    return report


def evaluate_full() -> dict[str, Any]:
    if not (RUNS / "gate2a/completion.json").is_file():
        raise RuntimeError("calibration preflight must complete before fit evaluation")
    gate2a = RUNTIME.read_protocol_bound_json(
        RUNS / "gate2a/completion.json", RUNTIME.file_sha256(PROTOCOL_PATH)
    )
    if gate2a.get("proceed_to_fit") is not True:
        raise RuntimeError("calibration preflight rejected fit evaluation")
    fit = _evaluate_role("fit", "full")
    calibration = _evaluate_role("calibration", "full")
    identity_to_id = {str(row["identity"]): int(row["id"]) for row in fit["population"]}
    folds = GATE.deterministic_folds(
        list(identity_to_id),
        count=int(PROTOCOL["folds"]["count"]),
        seed=int(PROTOCOL["folds"]["seed"]),
    )
    evaluator = METRIC.VocMetricBackend(fit["data"]["paths"]["gt"])
    fit_fold_rows = []
    for index, identities in enumerate(folds):
        image_ids = [identity_to_id[value] for value in identities]
        edge = evaluator.evaluate(fit["actions"]["edge"]["predictions"], image_ids)
        original = evaluator.evaluate(fit["actions"]["original"]["predictions"], image_ids)
        fit_fold_rows.append(
            {
                "fold": index,
                "images": len(image_ids),
                "edge_map50_95": float(edge["map50_95"]),
                "original_map50_95": float(original["map50_95"]),
                "delta": float(original["map50_95"] - edge["map50_95"]),
            }
        )
    bootstrap = _paired_bootstrap(calibration)
    fit_public = _public_role_result(fit)
    calibration_public = _public_role_result(calibration)
    decision = GATE.assess_endpoint_gate(
        PROTOCOL,
        fit={action: row["metrics"] for action, row in fit["actions"].items()},
        calibration={action: row["metrics"] for action, row in calibration["actions"].items()},
        fit_fold_deltas=[row["delta"] for row in fit_fold_rows],
        bootstrap_ci=[bootstrap["ci95_low"], bootstrap["ci95_high"]],
        original_bytes=fit_public["original"]["payload_bytes"]
        + calibration_public["original"]["payload_bytes"],
        q90_bytes=fit_public["q90"]["payload_bytes"]
        + calibration_public["q90"]["payload_bytes"],
    )
    report = {
        "complete": True,
        "stage": "full",
        "fit": fit_public,
        "calibration": calibration_public,
        "fit_folds": fit_fold_rows,
        "calibration_paired_bootstrap": bootstrap,
        **decision,
        "protocol_sha256": RUNTIME.file_sha256(PROTOCOL_PATH),
        "final_accessed": False,
    }
    RUNTIME.atomic_json(RUNS / "gate2/report.json", report)
    RUNTIME.atomic_json(RUNS / "gate2/completion.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "calibration", "full"))
    args = parser.parse_args()
    GATE.validate_protocol(PROTOCOL)
    if RUNTIME.file_sha256(METRIC_PATH) != str(PROTOCOL["source_metric_sha256"]):
        raise RuntimeError("frozen VOC metric source changed")
    if args.stage == "smoke":
        result = evaluate_smoke()
    elif args.stage == "calibration":
        result = evaluate_calibration_preflight()
    else:
        result = evaluate_full()
    print(json.dumps(result, ensure_ascii=False, indent=2))
