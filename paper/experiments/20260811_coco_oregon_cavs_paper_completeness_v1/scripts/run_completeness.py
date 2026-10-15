#!/usr/bin/env python3
"""Run the four isolated cache-level paper-completeness diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from completeness_core import action_mask, bootstrap_job_count, event_groups, validate_isolated_sources


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
SOURCE = REPO / "paper/experiments/20260809_coco_oregon_full_scheduler_benchmark_v1"
OREGON_V3 = REPO / "paper/experiments/20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1"
EXTERNAL = REPO / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines"
CURRENT_RUN = "20260811_coco_currentoqr12_profile_scale_extension_v1"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SOURCE / "scripts"))
BENCH = load_module("cavs_source_benchmark", SOURCE / "scripts/run_benchmark.py")
BENCH.RUNS = SOURCE / "runs"  # read-only source; all new writes use RUNS below
# This isolated diagnostic never changes the source protocol file.  Keep each
# evaluator process single-threaded and bound the fork pool while the unrelated
# OQR12 experiment is running on the same server.
BENCH.PROTOCOL["workers"]["point_per_dataset"] = 8
BOOT_CORE = load_module(
    "cavs_bootstrap_core",
    REPO / "paper/experiments/20260804_coco_dualhead_canonical9678_val2017_paper_v1/scripts/canonical_core.py",
)
sys.path.insert(0, str(OREGON_V3 / "scripts"))
ORE_EVAL = load_module("cavs_oregon_cloudv3_evaluate", OREGON_V3 / "scripts/evaluate.py")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_context(dataset: str) -> dict[str, Any]:
    if dataset == "oregon":
        layout = BENCH.canonical_layout("oregon")
        context = {
            "loader": ORE_EVAL.CANONICAL,
            "layout": layout,
            "roles": {role: ORE_EVAL.load_role(role) for role in ("calibration", "final")},
            "models": read_json(layout["benchmark"] / "models.json"),
            "gate": read_json(layout["benchmark"] / "action_gate.json"),
            "cloud_model_sha256": ORE_EVAL.PROTOCOL["cloud_model"]["sha256"],
        }
    else:
        context = BENCH.load_context(dataset)
    validate_isolated_sources([str(SOURCE), str(OREGON_V3), str(EXTERNAL)], PROTOCOL)
    if CURRENT_RUN in " ".join(map(str, context["layout"].values())):
        raise RuntimeError("current OQR12 run leaked into source context")
    return context


def source_point(dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    point = (OREGON_V3 / "runs/evaluation/point") if dataset == "oregon" else (SOURCE / "runs" / dataset / "point")
    rows = []
    result_name = "results.csv" if dataset == "oregon" else "final_results.csv"
    with (point / result_name).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("budget", "official_ap50_95", "kb_per_image"):
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
            if row.get("official_ap50_95") in (None, "") and row.get("official_map50_95") not in (None, ""):
                row["official_ap50_95"] = float(row["official_map50_95"])
            if row.get("kb_per_image") in (None, "") and row.get("total_kb_per_image") not in (None, ""):
                row["kb_per_image"] = float(row["total_kb_per_image"])
            rows.append(row)
    controls = read_json(point / "controls.json") if (point / "controls.json").is_file() else {"cloudv3": True}
    return rows, read_json(point / "selections.json"), controls


def _key(method: str, policy: str, mode: str, budget: float) -> str:
    return f"{method}|{policy}|{mode}|{float(budget):g}"


def _frozen_external_key(controls: Mapping[str, Any], budget: float) -> str:
    allowed = ("ORIC", "MORIC")
    candidates = []
    for key, item in controls.get("fixed_byte", {}).items():
        method = str(key).split("|")[0]
        if any(token in method.upper() for token in allowed) and key.endswith(f"fixed_byte|{float(budget):g}"):
            candidates.append((float(item.get("calibration_teacher_gain", -1e9)), key))
    if not candidates:
        raise RuntimeError(f"No frozen ORIC/MORIC comparator for budget {budget:g}")
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def _confidence_key(controls: Mapping[str, Any], budget: float) -> str:
    candidates = [
        (float(item.get("calibration_teacher_gain", -1e9)), key)
        for key, item in controls.get("fixed_byte", {}).items()
        if str(key).startswith("Confidence|") and str(key).endswith(f"fixed_byte|{float(budget):g}")
    ]
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def _oregon_best_key(rows: Sequence[Mapping[str, Any]], budget: float, methods: set[str]) -> str:
    candidates = [
        row for row in rows
        if row.get("role") == "calibration"
        and str(row.get("method")) in methods
        and row.get("mode") == "fixed_byte"
        and abs(float(row.get("budget", -1)) - budget) < 1e-12
    ]
    if not candidates:
        raise RuntimeError(f"missing Oregon calibration comparator: {methods}/{budget:g}")
    chosen = max(candidates, key=lambda row: (float(row["official_ap50_95"]), -float(row["kb_per_image"]), str(row["policy"]), str(row["method"])))
    return f"final|{chosen['method']}|{chosen['policy']}|fixed_byte|{budget:g}"


def _comparator_keys(dataset: str, rows: Sequence[Mapping[str, Any]], controls: Mapping[str, Any], budget: float) -> dict[str, str]:
    if dataset == "coco":
        return {
            "Raw-CAVS": _key("Raw-ActionValue-OQ", "Joint-OQ", "fixed_byte", budget),
            "ORIC-MORIC": _frozen_external_key(controls, budget),
            "Confidence": _confidence_key(controls, budget),
            "Random": _key("Random", "SharedAction-NoROI", "fixed_byte", budget),
            "DualHead": _key("DualHead", "SharedAction-NoROI", "fixed_byte", budget),
        }
    seed = int(ORE_EVAL.PROTOCOL["fold_seed"])
    return {
        "Raw-CAVS": f"final|Raw-ActionValue-OQ|Joint|fixed_byte|{budget:g}",
        "ORIC-MORIC": _oregon_best_key(rows, budget, {"ORIC-AP50", "MORIC-AP50", "ORIC-AP5095", "MORIC-AP5095"}),
        "Confidence": _oregon_best_key(rows, budget, {"Confidence"}),
        "Random": f"final|Random[seed={seed}]|SharedAction-NoROI|fixed_byte|{budget:g}",
        "DualHead": f"final|DualHead|SharedAction-NoROI|fixed_byte|{budget:g}",
    }


def _selection(stored: Mapping[str, Any], key: str) -> dict[int, str]:
    if key not in stored:
        raise RuntimeError(f"missing frozen selection: {key}")
    return {int(image_id): str(action) for image_id, action in stored[key].items()}


def _prediction_pair(loader: Any, data: Mapping[str, Any], candidate: Mapping[int, str], comparator: Mapping[int, str]):
    candidate_predictions, _ = loader.mixed_predictions(data, candidate, 0.001)
    comparator_predictions, _ = loader.mixed_predictions(data, comparator, 0.001)
    return candidate_predictions, comparator_predictions


_BOOT: dict[str, Any] = {}


def _sample_ids(rng: np.random.Generator, unit: str) -> list[int]:
    if unit == "image":
        ids = _BOOT["image_ids"]
        indices = rng.integers(0, len(ids), size=len(ids))
        return [int(ids[int(index)]) for index in indices]
    groups = _BOOT["groups"]
    indices = rng.integers(0, len(groups), size=len(groups))
    return [int(image_id) for index in indices for image_id in groups[int(index)]]


def _bootstrap_job(job: Mapping[str, Any]) -> dict[str, Any]:
    path = RUNS / "bootstrap" / str(job["dataset"]) / f"{job['unit']}__{job['method']}__b{job['budget']:g}.json"
    if path.is_file():
        cached = read_json(path)
        if int(cached.get("replicates", -1)) != int(PROTOCOL["bootstrap_replicates"]):
            raise RuntimeError(f"incomplete bootstrap cache: {path}")
        return cached
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{PROTOCOL['bootstrap_seed']}:{job['id']}".encode()).digest()[:8], "big"))
    values = []
    for _ in range(int(PROTOCOL["bootstrap_replicates"])):
        sampled = _sample_ids(rng, str(job["unit"]))
        gt, cand, comp = BOOT_CORE._bootstrap_remap(_BOOT["ground_truth"], _BOOT["candidate_predictions"], _BOOT["comparator_predictions"], sampled)
        values.append(float(BOOT_CORE._in_memory_ap(gt, cand) - BOOT_CORE._in_memory_ap(gt, comp)))
    array = np.asarray(values, dtype=np.float64)
    result = {
        **{key: value for key, value in job.items() if key != "id"},
        "replicates": len(values), "mean_delta": float(array.mean()), "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)), "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0)), "point_delta": float(job["candidate_point_ap"] - job["comparator_point_ap"]),
        "final_used_for_selection": False, "final_used_for_training": False, "posthoc": True,
    }
    atomic_json(path, result)
    return result


def _bootstrap_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    cp, qp = _BOOT["predictions"][int(job["prediction_index"])]
    _BOOT["candidate_predictions"], _BOOT["comparator_predictions"] = cp, qp
    return _bootstrap_job(job)


def run_bootstrap(dataset: str) -> dict[str, Any]:
    rows, stored, controls = source_point(dataset)
    context = load_context(dataset)
    final = context["roles"]["final"]
    ground_truth = read_json(Path(final["gt_path"]))
    image_ids = [int(sample["image_id"]) for sample in final["samples"]]
    groups = event_groups(final["samples"]) if dataset == "oregon" else []
    jobs = []
    for budget in map(float, PROTOCOL["budgets"]):
        keys = _comparator_keys(dataset, rows, controls, budget)
        candidate = _selection(stored, keys["Raw-CAVS"])
        for comparator_name in ("ORIC-MORIC", "Confidence", "Random", "DualHead"):
            comparator = _selection(stored, keys[comparator_name])
            unit_list = ("image",) if dataset == "coco" else ("event", "image")
            for unit in unit_list:
                jobs.append({
                    "id": f"Raw-CAVS-vs-{comparator_name}|{unit}|{budget:g}", "dataset": dataset,
                    "unit": unit, "budget": budget, "method": f"Raw-CAVS-vs-{comparator_name}", "comparator": comparator_name,
                    "candidate_point_ap": _find_ap(rows, "Raw-ActionValue-OQ", "Joint" if dataset == "oregon" else "Joint-OQ", "fixed_byte", budget),
                    "comparator_point_ap": _find_ap_by_key(rows, keys[comparator_name], budget),
                    "candidate_key": keys["Raw-CAVS"], "comparator_key": keys[comparator_name],
                })
    _BOOT.clear()
    _BOOT.update(ground_truth=ground_truth, image_ids=image_ids, groups=groups)
    # Keep one prediction cache per frozen selection key. Fork workers inherit
    # these read-only objects, avoiding a full copy per comparison.
    _BOOT["predictions"] = {}
    prediction_cache: dict[str, Any] = {}
    for job in jobs:
        for frozen_key in (str(job["candidate_key"]), str(job["comparator_key"])):
            if frozen_key not in prediction_cache:
                selection = _selection(stored, frozen_key)
                prediction_cache[frozen_key], _ = context["loader"].mixed_predictions(final, selection, 0.001)
        cp = prediction_cache[str(job["candidate_key"])]
        qp = prediction_cache[str(job["comparator_key"])]
        job["prediction_index"] = len(_BOOT["predictions"])
        _BOOT["predictions"][job["prediction_index"]] = (cp, qp)
    workers = min(12, len(jobs), max(1, (os.cpu_count() or 1) - 10))
    with mp.get_context("fork").Pool(workers) as pool:
        results = list(pool.imap_unordered(_bootstrap_worker, jobs, chunksize=1))
    results.sort(key=lambda row: (row["unit"], row["budget"], row["comparator"]))
    report = {"dataset": dataset, "jobs": len(results), "expected_jobs": bootstrap_job_count(dataset, PROTOCOL), "comparisons": results, "complete": True, "posthoc": True}
    atomic_json(RUNS / "bootstrap" / dataset / "report.json", report)
    return report


def _find_ap(rows: Sequence[Mapping[str, Any]], method: str, policy: str, mode: str, budget: float) -> float:
    matches = [row for row in rows if row.get("method") == method and row.get("policy") == policy and row.get("mode") == mode and row.get("role", "final") == "final" and abs(float(row.get("budget", -1)) - budget) < 1e-12 and row.get("seed", "aggregate") == "aggregate"]
    if not matches:
        matches = [row for row in rows if row.get("method") == method and row.get("policy") == policy and row.get("mode") == mode and row.get("role", "final") == "final" and abs(float(row.get("budget", -1)) - budget) < 1e-12]
    if len(matches) != 1:
        raise RuntimeError(f"missing point AP: {method}/{policy}/{mode}/{budget:g}")
    return float(matches[0]["official_ap50_95"])


def _find_ap_by_key(rows: Sequence[Mapping[str, Any]], key: str, budget: float) -> float:
    parts = str(key).split("|")
    method, policy, mode = parts[-4], parts[-3], parts[-2]
    if method.startswith("Random["):
        method = "Random"
    return _find_ap(rows, method, policy, mode, budget)


def _raw_cavs_scores(dataset: str, context: Mapping[str, Any]) -> np.ndarray:
    if dataset == "oregon":
        with np.load(RUNS / "preflight/oregon_raw_cavs_scores.npz", allow_pickle=False) as artifact:
            values = np.asarray(artifact["scores"], dtype=np.float64)
        if values.shape[:2] != (len(PROTOCOL["budgets"]), len(context["roles"]["final"]["samples"])):
            raise RuntimeError(f"Oregon Cloud-V3 Raw CAVS score alignment changed: {values.shape}")
        return values
    return np.asarray(BENCH.raw_actionvalue_scores(dataset, context)["raw"]["final"], dtype=np.float64)


def run_action_masks(dataset: str) -> dict[str, Any]:
    report_path = RUNS / "action_masks" / dataset / "report.json"
    if report_path.is_file() and read_json(report_path).get("complete") is True:
        return read_json(report_path)
    context = load_context(dataset)
    final = context["roles"]["final"]
    raw = _raw_cavs_scores(dataset, context)
    cross = BENCH.modules()["cross"]
    tasks = []
    for name in PROTOCOL["action_masks"]:
        allowed = action_mask(name, PROTOCOL)
        for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
            values = BENCH.actionvalue_map(final, raw[budget_index])
            for image_id in values:
                for action in ("original", "q90"):
                    if action not in allowed:
                        values[image_id][action] = -1e9
            for mode in ("fixed_byte", "fixed_call"):
                if mode == "fixed_call":
                    scheduled = cross.schedule_fixed_call(final["samples"], values, budget=budget)
                    task = {"method": f"Raw-CAVS-{name}", "policy": name, "mode": mode, "budget": budget, "actions": scheduled["selection"]}
                else:
                    capacity = budget * sum(float(sample["payload_bytes"]["original"]) for sample in final["samples"])
                    scheduled = cross.schedule_fixed_byte(final["samples"], values, capacity_bytes=capacity)
                    task = {"method": f"Raw-CAVS-{name}", "policy": name, "mode": mode, "budget": budget, "actions": scheduled["selection"], "capacity": capacity}
                tasks.append(task)
    rows = BENCH._evaluate_tasks(dataset, context, tasks)
    for row in rows:
        row["final_used_for_selection"] = False
        row["posthoc"] = True
        if dataset == "oregon":
            row["detector_protocol"] = "cloud_v3_epoch28"
            row["cloud_model_sha256"] = ORE_EVAL.PROTOCOL["cloud_model"]["sha256"]
    atomic_csv(RUNS / "action_masks" / dataset / "results.csv", rows)
    atomic_json(RUNS / "action_masks" / dataset / "report.json", {"dataset": dataset, "rows": len(rows), "masks": list(PROTOCOL["action_masks"]), "complete": True, "final_used_for_selection": False})
    return {"dataset": dataset, "rows": len(rows)}


def run_dcsb() -> dict[str, Any]:
    """Fit DCSB from Oregon fit only, then evaluate its frozen policy."""
    report_path = RUNS / "dcsb_oregon/report.json"
    if report_path.is_file() and read_json(report_path).get("complete") is True:
        return read_json(report_path)
    sys.path.insert(0, str(EXTERNAL / "scripts"))
    ext = load_module("cavs_external_dataset", EXTERNAL / "scripts/run_dataset.py")
    ext_protocol = read_json(EXTERNAL / "configs/protocol.json")
    output_root = RUNS / "dcsb_oregon"
    fit = ORE_EVAL.load_role("fit")
    image_ids = list(map(int, fit["image_ids"]))
    sizes = {int(sample["image_id"]): (int(sample["width"]), int(sample["height"])) for sample in fit["samples"]}
    ground_truth_counts = {image_id: len(fit["gt_map"].get(image_id, [])) for image_id in image_ids}
    dcsb = ext.fit_dcsb(
        image_ids=image_ids,
        edge_predictions=fit["actions"]["edge"],
        cloud_predictions=fit["actions"]["original"],
        image_sizes=sizes,
        ground_truth_counts=ground_truth_counts,
        detector_confidence=float(ext_protocol["dcsb"]["detector_confidence"]),
        count_thresholds=ext_protocol["dcsb"]["count_thresholds"],
        area_thresholds=ext.area_threshold_grid(
            ext_protocol["dcsb"]["area_threshold_start"],
            ext_protocol["dcsb"]["area_threshold_stop"],
            ext_protocol["dcsb"]["area_threshold_step"],
        ),
    )
    atomic_json(output_root / "model.json", dcsb)
    atomic_json(output_root / "fit_manifest.json", {"fit_images": len(image_ids), "fit_only": True, "calibration_images": 0, "final_images": 0, "cloud_model_sha256": ORE_EVAL.PROTOCOL["cloud_model"]["sha256"], "complete": True})
    context = load_context("oregon")
    external = BENCH.modules()["external"]
    roles = context["roles"]
    frozen_controls = read_json(RUNS / "preflight/oregon_noroi_maps.json")
    maps = {
        role: {"SharedAction-NoROI": {int(image_id): str(action) for image_id, action in frozen_controls[role].items()}}
        for role in ("calibration", "final")
    }
    scores = {}
    decisions = {}
    for role, data in roles.items():
        values, selected = [], []
        for sample in data["samples"]:
            image_id = int(sample["image_id"])
            decision, score = external.dcsb_decision_and_score(data["actions"]["edge"].get(image_id, []), int(sample["width"]), int(sample["height"]), dcsb)
            selected.append(int(decision)); values.append(float(score))
        scores[role] = np.asarray(values, dtype=np.float64); decisions[role] = np.asarray(selected, dtype=np.int64)
    final = roles["final"]
    tasks = []
    native = {int(sample["image_id"]): maps["final"]["SharedAction-NoROI"][int(sample["image_id"])] for sample, decision in zip(final["samples"], decisions["final"]) if int(decision) == 1}
    tasks.append({"method": "DCSB-Native-FitOnly", "policy": "SharedAction-NoROI", "mode": "native_rule", "budget": -1.0, "actions": native})
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        for mode in ("fixed_call", "fixed_byte"):
            if mode == "fixed_call":
                selected = context["loader"].top_k_actions(final, scores["final"], maps["final"]["SharedAction-NoROI"], budget)
                task = {"method": "DCSB-Score-Adapted-FitOnly", "policy": "SharedAction-NoROI", "mode": mode, "budget": budget, "actions": selected}
            else:
                frozen = context["loader"].calibrate_lambda(roles["calibration"], scores["calibration"], maps["calibration"]["SharedAction-NoROI"], budget, read_json(EXTERNAL / "configs/protocol.json"))
                priorities = context["loader"].audit_priority(scores["calibration"], scores["final"], final, maps["final"]["SharedAction-NoROI"], float(frozen["lambda"]), float(frozen["median_original_bytes"]))
                selected = context["loader"].byte_selection(final, priorities, maps["final"]["SharedAction-NoROI"], budget)
                capacity = budget * sum(float(sample["payload_bytes"]["original"]) for sample in final["samples"])
                task = {"method": "DCSB-Score-Adapted-FitOnly", "policy": "SharedAction-NoROI", "mode": mode, "budget": budget, "actions": selected, "capacity": capacity}
            tasks.append(task)
    rows = BENCH._evaluate_tasks("oregon", context, tasks)
    for row in rows: row.update(final_used_for_selection=False, dcsb_fit_only=True, posthoc=True, detector_protocol="cloud_v3_epoch28", cloud_model_sha256=ORE_EVAL.PROTOCOL["cloud_model"]["sha256"])
    atomic_csv(RUNS / "dcsb_oregon/results.csv", rows)
    atomic_json(RUNS / "dcsb_oregon/report.json", {"rows": len(rows), "dcsb_model": dcsb, "fit_only": True, "complete": True, "final_used_for_selection": False})
    return {"rows": len(rows), "fit_only": True}


def _group_label(sample: Mapping[str, Any], annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]]) -> tuple[str, ...]:
    image_id = int(sample["image_id"])
    anns = annotations_by_image.get(image_id, ())
    classes = {str(a.get("category_id", a.get("cls", "unknown"))) for a in anns}
    sizes = []
    for ann in anns:
        area = float(ann.get("area", float(ann.get("bbox", [0, 0, 0, 0])[2]) * float(ann.get("bbox", [0, 0, 0, 0])[3])))
        ratio = area / max(1.0, float(sample["width"]) * float(sample["height"]))
        sizes.append("small" if ratio < 0.01 else ("medium" if ratio < 0.05 else "large"))
    labels = [f"class:{value}" for value in sorted(classes)] or ["class:none"]
    labels.extend(f"size:{value}" for value in sorted(set(sizes)) or ["none"])
    labels.extend((f"camera:{sample.get('camera_id', 'unknown')}", f"event:{sample.get('event_id', 'unknown')}"))
    return tuple(labels)


def run_group_audit(dataset: str) -> dict[str, Any]:
    report_path = RUNS / "group_audit" / dataset / "report.json"
    if report_path.is_file() and read_json(report_path).get("complete") is True:
        return read_json(report_path)
    context = load_context(dataset); final = context["roles"]["final"]
    rows, stored, controls = source_point(dataset)
    budget = 0.25
    comparator_keys = _comparator_keys(dataset, rows, controls, budget)
    raw_key = comparator_keys["Raw-CAVS"]
    moric_key = comparator_keys["ORIC-MORIC"]
    raw = _selection(stored, raw_key); moric = _selection(stored, moric_key)
    ground_truth = read_json(Path(final["gt_path"]))
    raw_predictions, moric_predictions = _prediction_pair(context["loader"], final, raw, moric)
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = {}
    for annotation in ground_truth.get("annotations", ()):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    labels = [_group_label(sample, annotations_by_image) for sample in final["samples"]]
    records = []
    for group_type in ("class", "size", "camera", "event") if dataset == "oregon" else ("class", "size"):
        groups = sorted({value for row in labels for value in row if value.startswith(group_type + ":")})
        for group in groups:
            members = [i for i, row in enumerate(labels) if group in row]
            member_ids = [int(final["samples"][i]["image_id"]) for i in members]
            record = {"dataset": dataset, "group": group, "images": len(members), "raw_action_rate": float(np.mean([raw.get(image_id, "edge") != "edge" for image_id in member_ids])), "moric_action_rate": float(np.mean([moric.get(image_id, "edge") != "edge" for image_id in member_ids]))}
            if group_type != "event" and member_ids:
                gt, raw_group, moric_group = BOOT_CORE._bootstrap_remap(ground_truth, raw_predictions, moric_predictions, member_ids)
                raw_ap = float(BOOT_CORE._in_memory_ap(gt, raw_group)); moric_ap = float(BOOT_CORE._in_memory_ap(gt, moric_group))
                record.update(raw_official_ap50_95=raw_ap, moric_official_ap50_95=moric_ap, cavs_minus_moric_ap=raw_ap - moric_ap)
            else:
                record.update(raw_official_ap50_95="", moric_official_ap50_95="", cavs_minus_moric_ap="", note="event-level action decomposition; event-block bootstrap is the AP inference")
            records.append(record)
    atomic_csv(RUNS / "group_audit" / dataset / "groups.csv", records)
    # Exact objective audit on a deterministic 12-image prefix per budget.
    value_payload = _raw_cavs_scores(dataset, context)
    cross = BENCH.modules()["cross"]; exact_rows = []
    for bi, budget in enumerate(map(float, PROTOCOL["budgets"])):
        values = BENCH.actionvalue_map(final, value_payload[bi])
        subset = final["samples"][:12]
        capacity = budget * sum(float(s["payload_bytes"]["original"]) for s in subset)
        costs = {int(s["image_id"]): {"original": float(s["payload_bytes"]["original"]), "q90": float(s["payload_bytes"]["q90"]), "edge": 0.0} for s in subset}
        small_values = {int(s["image_id"]): values[int(s["image_id"])] for s in subset}
        exact = __import__("completeness_core").exact_oq_objective(small_values, costs, capacity=capacity)
        scheduled = cross.schedule_fixed_byte(final["samples"], values, capacity_bytes=budget * sum(float(s["payload_bytes"]["original"]) for s in final["samples"]))
        heuristic_actions = {int(s["image_id"]): str(scheduled["selection"].get(int(s["image_id"]), "edge")) for s in subset}
        heuristic = sum(float(small_values[i].get(a, 0.0)) for i, a in heuristic_actions.items())
        exact_rows.append({"dataset": dataset, "budget": budget, "subset_images": len(subset), "heuristic_objective": heuristic, "exact_objective": float(exact["objective"]), "objective_gap": float(exact["objective"] - heuristic), "exact_used_bytes": float(exact["used_bytes"]), "capacity": float(capacity)})
    atomic_csv(RUNS / "group_audit" / dataset / "solver.csv", exact_rows)
    atomic_json(RUNS / "group_audit" / dataset / "report.json", {"dataset": dataset, "groups": len(records), "solver_rows": len(exact_rows), "complete": True, "posthoc": True})
    return {"dataset": dataset, "groups": len(records), "solver_rows": len(exact_rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("bootstrap", "masks", "dcsb", "groups", "all"), required=True)
    parser.add_argument("--dataset", choices=("coco", "oregon"), default=None)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.stage == "bootstrap":
        print(json.dumps(run_bootstrap(args.dataset), indent=2)); return
    if args.stage == "masks":
        print(json.dumps(run_action_masks(args.dataset), indent=2)); return
    if args.stage == "dcsb":
        print(json.dumps(run_dcsb(), indent=2)); return
    if args.stage == "groups":
        print(json.dumps(run_group_audit(args.dataset), indent=2)); return
    outputs = {"bootstrap": {ds: run_bootstrap(ds) for ds in ("coco", "oregon")}, "masks": {ds: run_action_masks(ds) for ds in ("coco", "oregon")}, "dcsb": run_dcsb(), "groups": {ds: run_group_audit(ds) for ds in ("coco", "oregon")}}
    atomic_json(RUNS / "report.json", {"complete": True, "outputs": outputs, "posthoc": True})
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
