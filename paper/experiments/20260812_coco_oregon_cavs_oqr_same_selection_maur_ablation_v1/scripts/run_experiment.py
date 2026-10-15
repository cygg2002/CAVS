#!/usr/bin/env python3
"""Evaluate MAUR versus Reversible on frozen CAVS-OQR selections."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
SOURCE_ROOT = REPO / PROTOCOL["source_unified_experiment"]
sys.dont_write_bytecode = True


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("same_selection_maur_core", ROOT / "scripts/core.py")
CORE.validate_protocol(PROTOCOL)
SOURCE = load_module(
    "same_selection_source_unified", SOURCE_ROOT / "scripts/run_experiment.py"
)
MAUR_CORE = load_module(
    "same_selection_canonical_maur",
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/protocol_core.py",
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def is_complete(path: Path) -> bool:
    return Path(path).is_file() and read_json(path).get("complete") is True


def source_point_root(dataset: str) -> Path:
    return SOURCE_ROOT / f"runs/{dataset}/point"


def wait_for_source_point(dataset: str) -> dict[str, str]:
    root = source_point_root(dataset)
    completion = root / "completion.json"
    wait_seconds = int(PROTOCOL["resource"]["wait_seconds"])
    announced = False
    while True:
        if is_complete(completion):
            marker = read_json(completion)
            results = root / "results.csv"
            selections = root / "selections.json"
            if not results.is_file() or not selections.is_file():
                raise RuntimeError(f"source point marker precedes artifacts: {dataset}")
            if marker.get("results_sha256") != file_sha256(results):
                raise RuntimeError(f"source result hash mismatch: {dataset}")
            if marker.get("selections_sha256") != file_sha256(selections):
                raise RuntimeError(f"source selection hash mismatch: {dataset}")
            return {
                "completion": file_sha256(completion),
                "results": file_sha256(results),
                "selections": file_sha256(selections),
            }
        if not announced:
            print(
                f"[{dataset}] waiting for frozen source point selections; "
                f"polling every {wait_seconds}s",
                flush=True,
            )
            announced = True
        time.sleep(wait_seconds)


def load_adapter(dataset: str) -> Any:
    return SOURCE.DatasetAdapter(dataset)


def manifest_records(path: Path) -> dict[int, dict[str, Any]]:
    payload = read_json(path)
    records = payload.get("records", payload)
    if isinstance(records, list):
        return {int(record["image_id"]): dict(record) for record in records}
    return {int(key): dict(value) for key, value in records.items()}


def load_grouped(path: Path) -> dict[int, list[dict[str, Any]]]:
    return CORE.group_predictions(read_json(path))


def _flat(grouped: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for image_id in sorted(grouped) for row in grouped[image_id]]


def _coco_maur_sources(adapter: Any, role: str) -> tuple[Path, Path]:
    root = adapter.cavs_root / f"runs/actions/{role}/roi/current"
    return (
        root / "geometry/roi_manifest.json",
        root / "imgsz_640/roi_predictions_mapped.json",
    )


def required_action_sources(adapter: Any, role: str) -> list[Path]:
    manifest_path = adapter.common_manifest_path(role)
    manifest = read_json(manifest_path)
    required = [manifest_path, Path(manifest["action_sources"]["current@640"])]
    if adapter.dataset == "oregon":
        required.append(Path(manifest["maur_action_sources"]["current@640"]))
    else:
        required.extend(_coco_maur_sources(adapter, role))
    return required


def build_coco_maur(adapter: Any, role: str) -> tuple[Path, dict[str, Any]]:
    output = RUNS / f"coco/actions/{role}/current__640_maur.json"
    report_path = output.with_name("current__640_maur_report.json")
    geometry_path, raw_path = _coco_maur_sources(adapter, role)
    source_hashes = {
        str(path.resolve()): file_sha256(path) for path in (geometry_path, raw_path)
    }
    if output.is_file() and report_path.is_file():
        report = read_json(report_path)
        if (
            report.get("source_sha256") == source_hashes
            and report.get("output_sha256") == file_sha256(output)
        ):
            return output, report
    data = adapter.load_role(role)
    geometry = manifest_records(geometry_path)
    raw = load_grouped(raw_path)
    eligible = {
        int(sample["image_id"])
        for sample in data["samples"]
        if bool(sample.get("available", {}).get("current@640", False))
    }
    if set(geometry) != eligible:
        raise RuntimeError(
            f"COCO current@640 geometry population changed: {role}/"
            f"missing={len(eligible - set(geometry))}/extra={len(set(geometry) - eligible)}"
        )
    if not set(raw) <= eligible:
        raise RuntimeError(f"COCO ROI predictions include ineligible images: {role}")
    grouped: dict[int, list[dict[str, Any]]] = {}
    edge = data["actions"]["edge"]
    for image_id in sorted(eligible):
        merged = MAUR_CORE.maur_candidates(
            edge.get(image_id, []),
            raw.get(image_id, []),
            tuple(map(int, geometry[image_id]["xyxy"])),
            duplicate_iou=float(PROTOCOL["maur"]["duplicate_iou"]),
        )
        grouped[image_id] = MAUR_CORE.class_aware_nms(
            merged, iou_threshold=float(PROTOCOL["maur"]["nms_iou"])
        )
    atomic_json(output, _flat(grouped))
    report = {
        "complete": True,
        "dataset": "coco",
        "role": role,
        "eligible_images": len(eligible),
        "images_with_predictions": sum(bool(value) for value in grouped.values()),
        "predictions": sum(map(len, grouped.values())),
        "source_sha256": source_hashes,
        "output_sha256": file_sha256(output),
        "detector_inference_performed": False,
        "fusion": "MAUR",
    }
    atomic_json(report_path, report)
    return output, report


def maur_action_path(adapter: Any, role: str) -> tuple[Path, dict[str, Any]]:
    if adapter.dataset == "coco":
        return build_coco_maur(adapter, role)
    manifest = read_json(adapter.common_manifest_path(role))
    path = Path(manifest["maur_action_sources"]["current@640"])
    return path, {
        "complete": True,
        "dataset": "oregon",
        "role": role,
        "source_sha256": {str(path.resolve()): file_sha256(path)},
        "output_sha256": file_sha256(path),
        "detector_inference_performed": False,
        "fusion": "MAUR",
    }


def run_preflight(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "preflight"
    source_hashes = wait_for_source_point(dataset)
    adapter = load_adapter(dataset)
    selections = read_json(source_point_root(dataset) / "selections.json")
    expected_keys = {
        CORE.selection_key(role, mode, budget)
        for role in PROTOCOL["roles"]
        for mode in PROTOCOL["modes"]
        for budget in map(float, PROTOCOL["budgets"])
    }
    missing = expected_keys - set(selections)
    if missing:
        raise RuntimeError(f"missing frozen selections: {dataset}/{sorted(missing)}")
    role_reports = {}
    for role, expected_count in PROTOCOL["datasets"][dataset]["roles"].items():
        data = adapter.load_role(role)
        if len(data["samples"]) != int(expected_count):
            raise RuntimeError(f"role population changed: {dataset}/{role}")
        manifest = read_json(adapter.common_manifest_path(role))
        if manifest.get("primary_roi_fusion") != "Reversible":
            raise RuntimeError(f"source is not Reversible: {dataset}/{role}")
        if manifest.get("cloud_model_sha256") != adapter.config["cloud_model_sha256"]:
            raise RuntimeError(f"cloud model cache changed: {dataset}/{role}")
        paths = required_action_sources(adapter, role)
        missing_paths = [str(path) for path in paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(missing_paths)
        selected_roi: dict[str, int] = {}
        for mode in PROTOCOL["modes"]:
            for budget in map(float, PROTOCOL["budgets"]):
                key = CORE.selection_key(role, mode, budget)
                selection = {
                    int(image_id): str(action)
                    for image_id, action in selections[key].items()
                }
                CORE.validate_selection_actions(selection)
                SOURCE.CORE.validate_selection(
                    data["samples"], selection, mode=mode, budget=budget
                )
                selected_roi[f"{mode}|{budget:g}"] = sum(
                    action == "current@640" for action in selection.values()
                )
        _, maur_report = maur_action_path(adapter, role)
        role_reports[role] = {
            "images": len(data["samples"]),
            "selected_roi": selected_roi,
            "source_paths_sha256": {
                str(path.resolve()): file_sha256(path) for path in paths
            },
            "maur": maur_report,
        }
    report = {
        "complete": True,
        "dataset": dataset,
        "source_point_sha256": source_hashes,
        "source_selection_keys": sorted(expected_keys),
        "roles": role_reports,
        "same_selection": True,
        "training_performed": False,
        "detector_inference_performed": False,
        "source_experiments_modified": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(
        output / "completion.json",
        {"complete": True, "report_sha256": file_sha256(output / "report.json")},
    )
    return report


def load_selection(
    stored: Mapping[str, Any], role: str, mode: str, budget: float
) -> dict[int, str]:
    key = CORE.selection_key(role, mode, budget)
    if key not in stored:
        raise RuntimeError(f"missing frozen selection: {key}")
    selection = {int(image_id): str(action) for image_id, action in stored[key].items()}
    CORE.validate_selection_actions(selection)
    return selection


def data_with_fusion(adapter: Any, role: str, fusion: str) -> dict[str, Any]:
    data = adapter.load_role(role)
    if fusion == "Reversible":
        return data
    if fusion != "MAUR":
        raise ValueError(fusion)
    path, _ = maur_action_path(adapter, role)
    actions = dict(data["actions"])
    actions["current@640"] = load_grouped(path)
    return {**data, "actions": actions}


def mixed_predictions(
    adapter: Any,
    role: str,
    selection: Mapping[int, str],
    fusion: str,
) -> list[dict[str, Any]]:
    data = data_with_fusion(adapter, role, fusion)
    predictions, _ = adapter.cloud.CANONICAL.mixed_predictions(
        data, dict(selection), float(PROTOCOL["detector_confidence"])
    )
    return predictions


def evaluate_selection(
    adapter: Any,
    role: str,
    selection: Mapping[int, str],
    *,
    mode: str,
    budget: float,
    fusion: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = adapter.load_role(role)
    predictions = mixed_predictions(adapter, role, selection, fusion)
    metric = data["evaluator"].evaluate(predictions)
    validation = SOURCE.CORE.validate_selection(
        data["samples"], selection, mode=mode, budget=budget
    )
    counts = Counter(selection.values())
    row = {
        "dataset": adapter.dataset,
        "role": role,
        "method": PROTOCOL["method"],
        "policy": PROTOCOL["policy"],
        "fusion": fusion,
        "mode": mode,
        "budget": float(budget),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": validation["request_bytes"] / 1024.0 / len(data["samples"]),
        "request_bytes": int(validation["request_bytes"]),
        "capacity_utilization": float(validation["utilization"]),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["current@640"]),
        "selection_sha256": canonical_sha256(
            {str(image_id): selection[image_id] for image_id in sorted(selection)}
        ),
        "cloud_model_sha256": adapter.config["cloud_model_sha256"],
        "same_selection": True,
        "training_performed": False,
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }
    return row, predictions


def _source_row(
    rows: Sequence[Mapping[str, Any]], role: str, mode: str, budget: float
) -> Mapping[str, Any]:
    found = [
        row
        for row in rows
        if row["role"] == role
        and row["method"] == PROTOCOL["method"]
        and row["policy"] == PROTOCOL["policy"]
        and row["mode"] == mode
        and math.isclose(float(row["budget"]), budget)
    ]
    if len(found) != 1:
        raise RuntimeError(f"source row is not unique: {role}/{mode}/{budget}")
    return found[0]


def verify_reversible_source(row: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    for field in (
        "official_ap50_95",
        "official_ap50",
        "official_ap75",
        "kb_per_image",
        "request_bytes",
        "selected_images",
        "original_count",
        "q90_count",
        "roi_count",
    ):
        if not math.isclose(
            float(row[field]), float(source[field]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"Reversible source reproduction failed: {field}")


def run_point(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "point"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    preflight = run_preflight(dataset)
    adapter = load_adapter(dataset)
    stored = read_json(source_point_root(dataset) / "selections.json")
    source_rows = read_csv(source_point_root(dataset) / "results.csv")
    rows: list[dict[str, Any]] = []
    for role in PROTOCOL["roles"]:
        for mode in PROTOCOL["modes"]:
            for budget in map(float, PROTOCOL["budgets"]):
                selection = load_selection(stored, role, mode, budget)
                pair = {}
                for fusion in PROTOCOL["fusions"]:
                    row, _ = evaluate_selection(
                        adapter,
                        role,
                        selection,
                        mode=mode,
                        budget=budget,
                        fusion=fusion,
                    )
                    pair[fusion] = row
                    rows.append(row)
                CORE.validate_transport_pair(pair["Reversible"], pair["MAUR"])
                CORE.validate_zero_roi_metrics(pair["Reversible"], pair["MAUR"])
                verify_reversible_source(
                    pair["Reversible"], _source_row(source_rows, role, mode, budget)
                )
    CORE.validate_unique_rows(rows)
    expected = (
        len(PROTOCOL["roles"])
        * len(PROTOCOL["fusions"])
        * len(PROTOCOL["modes"])
        * len(PROTOCOL["budgets"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"point matrix incomplete: {dataset}/{len(rows)} != {expected}")
    atomic_csv(output / "results.csv", rows)
    report = {
        "complete": True,
        "dataset": dataset,
        "rows": len(rows),
        "source_point_sha256": preflight["source_point_sha256"],
        "results_sha256": file_sha256(output / "results.csv"),
        "transport_invariant": True,
        "reversible_source_reproduced": True,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(
        completion,
        {"complete": True, "results_sha256": report["results_sha256"]},
    )
    return report


_BOOT_STATE: dict[str, Any] = {}


def _sample_ids(rng: np.random.Generator, unit: str) -> list[int]:
    if unit == "image":
        ids = _BOOT_STATE["image_ids"]
        return [int(ids[index]) for index in rng.integers(0, len(ids), size=len(ids))]
    if unit == "event":
        groups = _BOOT_STATE["event_groups"]
        return [
            int(image_id)
            for index in rng.integers(0, len(groups), size=len(groups))
            for image_id in groups[int(index)]
        ]
    raise ValueError(unit)


def _bootstrap_job(job: Mapping[str, Any]) -> dict[str, Any]:
    token = hashlib.sha256(str(job["id"]).encode("utf-8")).hexdigest()[:16]
    path = Path(_BOOT_STATE["job_root"]) / f"{token}.json"
    partial = path.with_suffix(".partial.json")
    replicates = int(PROTOCOL["bootstrap"]["replicates"])
    if path.is_file():
        result = read_json(path)
        if int(result["replicates"]) != replicates:
            raise RuntimeError(f"bootstrap cache has wrong size: {path}")
        return result
    values = list(map(float, read_json(partial)["values"])) if partial.is_file() else []
    seed_bytes = hashlib.sha256(
        f"{PROTOCOL['bootstrap']['seed']}:{job['id']}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
    for _ in values:
        _sample_ids(rng, str(job["unit"]))
    maur, reversible = _BOOT_STATE["pairs"][int(job["pair_index"])]
    for replicate in range(len(values), replicates):
        sampled = _sample_ids(rng, str(job["unit"]))
        gt, left, right = _BOOT_STATE["bootstrap_core"]._bootstrap_remap(
            _BOOT_STATE["ground_truth"], maur, reversible, sampled
        )
        values.append(
            float(
                _BOOT_STATE["bootstrap_core"]._in_memory_ap(gt, left)
                - _BOOT_STATE["bootstrap_core"]._in_memory_ap(gt, right)
            )
        )
        if (replicate + 1) % 100 == 0:
            atomic_json(partial, {"values": values})
    array = np.asarray(values, dtype=np.float64)
    result = {
        **{key: value for key, value in job.items() if key != "pair_index"},
        "candidate": "MAUR",
        "comparator": "Reversible",
        "replicates": len(values),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0.0)),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(path, result)
    partial.unlink(missing_ok=True)
    return result


def run_bootstrap(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "bootstrap"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_point(dataset)
    adapter = load_adapter(dataset)
    stored = read_json(source_point_root(dataset) / "selections.json")
    data = adapter.load_role("final")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    events: dict[str, list[int]] = {}
    for sample in data["samples"]:
        event = str(sample.get("event_id", sample["image_id"]))
        events.setdefault(event, []).append(int(sample["image_id"]))
    jobs = []
    pairs = []
    for budget in map(float, PROTOCOL["budgets"]):
        selection = load_selection(stored, "final", "fixed_byte", budget)
        reversible = mixed_predictions(adapter, "final", selection, "Reversible")
        maur = mixed_predictions(adapter, "final", selection, "MAUR")
        pair_index = len(pairs)
        pairs.append((maur, reversible))
        for unit in PROTOCOL["datasets"][dataset]["bootstrap_units"]:
            jobs.append(
                {
                    "id": f"MAUR|vs|Reversible|budget={budget:g}|{unit}",
                    "dataset": dataset,
                    "budget": budget,
                    "unit": unit,
                    "pair_index": pair_index,
                    "selection_key": CORE.selection_key(
                        "final", "fixed_byte", budget
                    ),
                }
            )
    expected = len(PROTOCOL["budgets"]) * len(
        PROTOCOL["datasets"][dataset]["bootstrap_units"]
    )
    if len(jobs) != expected:
        raise RuntimeError(f"bootstrap job matrix incomplete: {dataset}")
    job_root = output / "jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        image_ids=image_ids,
        event_groups=[events[key] for key in sorted(events)],
        pairs=pairs,
        ground_truth=read_json(adapter.gt_path("final")),
        bootstrap_core=adapter.bootstrap_core,
        job_root=str(job_root),
    )
    workers = min(
        int(PROTOCOL["bootstrap"]["workers"]),
        int(PROTOCOL["resource"]["maximum_workers"]),
        len(jobs),
    )
    with mp.get_context("fork").Pool(workers) as pool:
        results = list(pool.imap_unordered(_bootstrap_job, jobs, chunksize=1))
    _BOOT_STATE.clear()
    results.sort(key=lambda row: (str(row["unit"]), float(row["budget"])))
    if len(results) != expected or any(
        int(row["replicates"]) != int(PROTOCOL["bootstrap"]["replicates"])
        for row in results
    ):
        raise RuntimeError(f"bootstrap result matrix incomplete: {dataset}")
    report = {
        "complete": True,
        "dataset": dataset,
        "jobs": len(results),
        "replicates_per_job": int(PROTOCOL["bootstrap"]["replicates"]),
        "comparisons": results,
        "event_bootstrap_primary": dataset == "oregon",
        "image_bootstrap_sensitivity_only": dataset == "oregon",
        "same_selection": True,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(results)})
    return report


def run_report() -> dict[str, Any]:
    reports = {dataset: run_bootstrap(dataset) for dataset in PROTOCOL["datasets"]}
    lines = [
        "# CAVS-OQR 同选择 MAUR / Reversible 融合消融",
        "",
        "本实验冻结当前统一 CAVS-OQR 的图片与动作选择，仅替换 ROI 融合结果。",
        "因此两种融合的上传图片、动作计数、字节和容量利用率完全相同。",
        "COCO 使用 image bootstrap；Oregon 以 event-block bootstrap 为主、image bootstrap 为敏感性分析。",
        "本结果使用已开放的 evaluation/final，仅属于 post-hoc 配对消融。",
    ]
    for dataset in PROTOCOL["datasets"]:
        rows = read_csv(RUNS / dataset / "point/results.csv")
        lines.extend(
            [
                "",
                f"## {dataset.upper()} Fixed-byte final",
                "",
                "| 融合 | 10% | 25% | 40% |",
                "|---|---:|---:|---:|",
            ]
        )
        for fusion in PROTOCOL["fusions"]:
            cells = []
            for budget in map(float, PROTOCOL["budgets"]):
                found = [
                    row
                    for row in rows
                    if row["role"] == "final"
                    and row["fusion"] == fusion
                    and row["mode"] == "fixed_byte"
                    and math.isclose(float(row["budget"]), budget)
                ]
                if len(found) != 1:
                    raise RuntimeError(f"missing report row: {dataset}/{fusion}/{budget}")
                row = found[0]
                cells.append(
                    f"{float(row['official_ap50_95']):.6f} / "
                    f"{float(row['kb_per_image']):.2f} KB / ROI {int(float(row['roi_count']))}"
                )
            lines.append(f"| {fusion} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                f"## {dataset.upper()} 同选择点估计差值（MAUR − Reversible）",
                "",
                "| 预算 | ΔAP50-95 | ΔAP50 | ΔAP75 |",
                "|---:|---:|---:|---:|",
            ]
        )
        for budget in map(float, PROTOCOL["budgets"]):
            pair = {
                row["fusion"]: row
                for row in rows
                if row["role"] == "final"
                and row["mode"] == "fixed_byte"
                and math.isclose(float(row["budget"]), budget)
            }
            lines.append(
                f"| {budget:.0%} | "
                f"{float(pair['MAUR']['official_ap50_95']) - float(pair['Reversible']['official_ap50_95']):+.6f} | "
                f"{float(pair['MAUR']['official_ap50']) - float(pair['Reversible']['official_ap50']):+.6f} | "
                f"{float(pair['MAUR']['official_ap75']) - float(pair['Reversible']['official_ap75']):+.6f} |"
            )
        bootstrap = read_json(RUNS / dataset / "bootstrap/report.json")
        lines.extend(
            [
                "",
                f"## {dataset.upper()} paired bootstrap",
                "",
                "| 单位 | 预算 | ΔAP均值 | 95% CI | P(Δ>0) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in bootstrap["comparisons"]:
            lines.append(
                f"| {row['unit']} | {float(row['budget']):.0%} | "
                f"{float(row['mean_delta']):+.6f} | "
                f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
                f"{float(row['probability_gt_zero']):.3f} |"
            )
    path = ROOT / "RESULTS_ZH.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "complete": True,
        "datasets": reports,
        "point_rows": sum(
            len(read_csv(RUNS / dataset / "point/results.csv"))
            for dataset in PROTOCOL["datasets"]
        ),
        "bootstrap_jobs": sum(report["jobs"] for report in reports.values()),
        "results_sha256": file_sha256(path),
        "final_used_for_training_or_selection": False,
    }
    if report["point_rows"] != CORE.expected_point_rows(PROTOCOL):
        raise RuntimeError("global point matrix incomplete")
    if report["bootstrap_jobs"] != CORE.expected_bootstrap_jobs(PROTOCOL):
        raise RuntimeError("global bootstrap matrix incomplete")
    atomic_json(RUNS / "report.json", report)
    atomic_json(
        RUNS / "COMPLETED.json",
        {"complete": True, "results_sha256": report["results_sha256"]},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "oregon"), default=None)
    parser.add_argument(
        "--stage", choices=("preflight", "point", "bootstrap", "report", "all"), default="all"
    )
    args = parser.parse_args()
    if args.stage == "report":
        result = run_report()
    else:
        if args.dataset is None:
            raise SystemExit("--dataset is required except for --stage report")
        if args.stage == "preflight":
            result = run_preflight(args.dataset)
        elif args.stage == "point":
            result = run_point(args.dataset)
        elif args.stage == "bootstrap":
            result = run_bootstrap(args.dataset)
        else:
            result = {
                "preflight": run_preflight(args.dataset),
                "point": run_point(args.dataset),
                "bootstrap": run_bootstrap(args.dataset),
            }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
