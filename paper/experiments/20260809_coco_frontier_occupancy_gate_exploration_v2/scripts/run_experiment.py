#!/usr/bin/env python3
"""Run the COCO fit-OOF frontier-occupancy gate diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
V1_ROOT = REPO / PROTOCOL["source_v1"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("occupancy_runner_core", ROOT / "scripts/core.py")
TRAINING = _load("occupancy_runner_training", ROOT / "scripts/training.py")
V1 = _load("occupancy_v1_runner", V1_ROOT / "scripts/run_experiment.py")
CORE.validate_protocol(PROTOCOL)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run_preflight() -> dict[str, Any]:
    completion = RUNS / "preflight/completion.json"
    output = RUNS / "preflight/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    required = [
        V1_ROOT / "runs/confirm/evaluation_completion.json",
        V1_ROOT / "runs/confirm/scores.pkl",
        V1_ROOT / "runs/confirm/roi_gate_ceiling.json",
        V1_ROOT / "runs/preflight/report.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete v1 source: {missing}")
    v1_report = json.loads(
        (V1_ROOT / "runs/confirm/evaluation_report.json").read_text(encoding="utf-8")
    )
    if not bool(v1_report["gate"]["passed"]):
        raise RuntimeError("The frozen v1 candidate did not pass")
    data = V1.REL.load_role("fit")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    if len(image_ids) != 9678 or len(image_ids) != len(set(image_ids)):
        raise RuntimeError("Canonical fit population changed")
    folds = json.loads(
        (V1.REL_ROOT / "runs/preflight/folds.json").read_text(encoding="utf-8")
    )["folds"]
    if len(folds) != 5 or {int(x) for fold in folds for x in fold} != set(image_ids):
        raise RuntimeError("Frozen OOF folds changed")
    sources = [
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/core.py",
        ROOT / "scripts/training.py",
        *required,
    ]
    hashes = {str(path): file_sha256(path) for path in sources}
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "fit_images": len(image_ids),
        "fold_sizes": list(map(len, folds)),
        "source_hashes": hashes,
        "detector_inference_required": False,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output)})
    return report


def _build_training_data() -> tuple[
    list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], torch.Tensor
]:
    records = V1._records_for_space("fit", "OQR")
    rewards = V1._reward_map_for_space("fit", "OQR")
    scale = float(np.mean([record["full_payload_bytes"] for record in records]))
    raw_reward, cost_ratio = V1.TRAINING.reward_and_cost_tensors(
        records,
        rewards,
        max_actions=int(PROTOCOL["model"]["max_actions"]),
        cost_scale_bytes=scale,
    )
    tensor_protocol = {
        "max_boxes": int(PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(PROTOCOL["model"]["max_actions"]),
        "deploy_confidence": float(PROTOCOL["model"]["deploy_confidence"]),
    }
    raw = V1.REL._record_tensors(records, None, tensor_protocol)
    prices, price_mask = V1.CORE.frontier_probe_prices(
        raw_reward.to(torch.float32),
        cost_ratio.to(torch.float32),
        raw["action_mask"],
        relative_offset=float(PROTOCOL["model"]["frontier_relative_offset"]),
        maximum_price=float(PROTOCOL["model"]["frontier_maximum_price"]),
    )
    targets = V1.CORE.frontier_target_actions(
        raw_reward.to(torch.float32),
        cost_ratio.to(torch.float32),
        raw["action_mask"],
        prices,
    )
    occupancy = CORE.frontier_occupancy_labels(
        targets, price_mask, raw["action_mask"]
    )
    aware = dict(raw)
    aware["occupancy_target"] = occupancy
    free = TRAINING.V1_TRAINING.strip_cost_inputs(raw)
    free["occupancy_target"] = occupancy
    return records, {
        "CostFree-FrontierGate": free,
        "CostAware-FrontierGate": aware,
    }, occupancy


def _index_data(
    data: Mapping[str, torch.Tensor], indices: Sequence[int] | torch.Tensor
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _epoch(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    *,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> dict[str, float]:
    model.train(optimizer is not None)
    ordered = torch.as_tensor(indices, dtype=torch.long)
    if optimizer is not None:
        generator = torch.Generator().manual_seed(int(seed))
        ordered = ordered[torch.randperm(len(ordered), generator=generator)]
    totals: Counter[str] = Counter()
    total = 0
    batch_size = int(PROTOCOL["model"]["batch_size"])
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(ordered), batch_size):
            batch_indices = ordered[start : start + batch_size]
            result = TRAINING.occupancy_objective(
                model, _index_data(data, batch_indices)
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(PROTOCOL["model"]["gradient_clip"])
                )
                optimizer.step()
            count = len(batch_indices)
            total += count
            for key, value in result.items():
                totals[key] += float(value.detach()) * count
    return {key: value / max(1, total) for key, value in totals.items()}


def _fit(
    data: Mapping[str, torch.Tensor],
    image_ids: Sequence[int],
    train_indices: Sequence[int],
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    roi_available = data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
    train_indices = [index for index in train_indices if bool(roi_available[index].any())]
    train_ids = [int(image_ids[index]) for index in train_indices]
    inner_ids, validation_ids = V1.REL.deterministic_validation_split(
        train_ids,
        seed=int(seed),
        fraction=float(PROTOCOL["model"]["validation_fraction"]),
    )
    by_id = {int(image_id): index for index, image_id in enumerate(image_ids)}
    inner = [by_id[image_id] for image_id in inner_ids]
    validation = [by_id[image_id] for image_id in validation_ids]
    torch.manual_seed(int(seed))
    model = TRAINING.build_model(
        category_count=len(V1.REL.category_ids("fit")),
        scene_dim=int(V1.REL.SCENE_DIM),
        action_dim=int(V1.REL.ACTION_DIM),
        config=PROTOCOL["model"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(PROTOCOL["model"]["learning_rate"]),
        weight_decay=float(PROTOCOL["model"]["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_loss, stale = 0, math.inf, 0
    history = []
    for epoch in range(1, int(PROTOCOL["model"]["epochs"]) + 1):
        train = _epoch(
            model,
            data,
            inner,
            optimizer=optimizer,
            seed=int(seed) + epoch * 1_000_003,
        )
        validation_metrics = _epoch(
            model, data, validation, optimizer=None, seed=int(seed)
        )
        loss = float(validation_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train["loss"]),
                "validation_loss": loss,
                "validation_accuracy": float(validation_metrics["accuracy"]),
            }
        )
        if loss < best_loss - 1e-6:
            best_epoch, best_loss, stale = epoch, loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(PROTOCOL["model"]["patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "inner_train_images": len(inner),
        "validation_images": len(validation),
        "history": history,
    }


def _predict_roi(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
) -> np.ndarray:
    model.eval()
    outputs = []
    batch_size = int(PROTOCOL["model"]["batch_size"])
    ordered = torch.as_tensor(indices, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(ordered), batch_size):
            batch = _index_data(data, ordered[start : start + batch_size])
            logits = model(**TRAINING.model_inputs(batch))
            roi = batch["action_mask"].to(torch.bool) & (batch["action_type"] == 2)
            if not bool((roi.sum(dim=1) == 1).all()):
                raise RuntimeError("ROI prediction rows require exactly one ROI")
            outputs.append(torch.sigmoid(logits[roi]).cpu().numpy())
    values = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("ROI probability contains non-finite values")
    return values


_TRAIN_STATE: dict[str, Any] = {}


def _train_job(task: tuple[str, int]) -> dict[str, Any]:
    method, fold_index = str(task[0]), int(task[1])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    root = RUNS / f"oof/jobs/{method}/fold_{fold_index}"
    completion = root / "completion.json"
    if completion.is_file():
        return json.loads((root / "report.json").read_text(encoding="utf-8"))
    data = _TRAIN_STATE["data"][method]
    image_ids = _TRAIN_STATE["image_ids"]
    heldout_ids = set(map(int, _TRAIN_STATE["folds"][fold_index]))
    heldout = [
        index
        for index, image_id in enumerate(image_ids)
        if int(image_id) in heldout_ids
        and bool(
            (data["action_mask"][index] & (data["action_type"][index] == 2)).any()
        )
    ]
    train = [
        index for index, image_id in enumerate(image_ids) if int(image_id) not in heldout_ids
    ]
    seed = int(PROTOCOL["seed"]) + fold_index * 10_007
    started = time.time()
    model, fit = _fit(data, image_ids, train, seed=seed)
    probabilities = _predict_roi(model, data, heldout)
    score_path = root / "probabilities.npz"
    checkpoint_path = root / "checkpoint.pt"
    atomic_npz(
        score_path,
        image_ids=np.asarray([image_ids[index] for index in heldout], dtype=np.int64),
        probabilities=probabilities,
    )
    atomic_torch(
        checkpoint_path,
        {"method": method, "seed": seed, "model_state": model.state_dict(), "fit": fit},
    )
    report = {
        "method": method,
        "fold": fold_index,
        "seed": seed,
        "train_roi_images": int(fit["inner_train_images"] + fit["validation_images"]),
        "heldout_roi_images": len(heldout),
        "selected_epoch": int(fit["selected_epoch"]),
        "best_validation_loss": float(fit["best_validation_loss"]),
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(root / "report.json", report)
    atomic_json(completion, {"complete": True, "score_sha256": report["score_sha256"]})
    return report


def run_oof() -> dict[str, Any]:
    completion = RUNS / "oof/completion.json"
    output = RUNS / "oof/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    run_preflight()
    records, data, occupancy = _build_training_data()
    image_ids = [int(record["image_id"]) for record in records]
    folds = json.loads(
        (V1.REL_ROOT / "runs/preflight/folds.json").read_text(encoding="utf-8")
    )["folds"]
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update({"data": data, "image_ids": image_ids, "folds": folds})
    tasks = [
        (method, fold_index)
        for method in PROTOCOL["methods"]
        for fold_index in range(int(PROTOCOL["fold_count"]))
    ]
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(_train_job, tasks, chunksize=1))
    roi_available = {
        image_ids[index]
        for index in range(len(image_ids))
        if bool((data[PROTOCOL["methods"][0]]["action_type"][index] == 2).any())
    }
    probabilities = {}
    for method in PROTOCOL["methods"]:
        values = {}
        for fold_index in range(int(PROTOCOL["fold_count"])):
            path = RUNS / f"oof/jobs/{method}/fold_{fold_index}/probabilities.npz"
            with np.load(path, allow_pickle=False) as artifact:
                values.update(
                    {
                        int(image_id): float(probability)
                        for image_id, probability in zip(
                            artifact["image_ids"], artifact["probabilities"]
                        )
                    }
                )
        if set(values) != roi_available:
            raise RuntimeError(f"Incomplete ROI OOF probabilities: {method}")
        probabilities[method] = values
    occupancy_map = {}
    for row, image_id in enumerate(image_ids):
        roi_columns = torch.nonzero(
            data[PROTOCOL["methods"][0]]["action_type"][row] == 2
        ).flatten()
        if len(roi_columns):
            occupancy_map[image_id] = bool(occupancy[row, int(roi_columns[0])])
    artifact_path = RUNS / "oof/probabilities.pkl"
    atomic_pickle(
        artifact_path,
        {
            "probabilities": probabilities,
            "occupancy_target": occupancy_map,
            "threshold": float(PROTOCOL["probability_threshold"]),
        },
    )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "jobs": len(tasks),
        "completed_jobs": len(reports),
        "workers": workers,
        "roi_images": len(roi_available),
        "positive_frontier_roi_images": sum(occupancy_map.values()),
        "elapsed_seconds": time.time() - started,
        "probability_sha256": file_sha256(artifact_path),
        "fit_oof_only": True,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "probability_sha256": report["probability_sha256"]})
    _TRAIN_STATE.clear()
    return report


_EVAL_STATE: dict[str, Any] = {}


def _eval_task(task: Mapping[str, Any]) -> dict[str, Any]:
    data = _EVAL_STATE["data"]
    selection = {int(key): str(value) for key, value in task["selection"].items()}
    predictions, _ = V1.REL.canonical.mixed_predictions(data, selection, 0.001)
    metric = data["evaluator"].evaluate(predictions)
    counts = Counter(selection.values())
    request_bytes = sum(
        int(data["sample_by_id"][image_id]["payload_bytes"][action])
        for image_id, action in selection.items()
    )
    teacher_roi = set(map(int, task["teacher_roi"]))
    learned_roi = {image_id for image_id, action in selection.items() if action == "roi"}
    common = teacher_roi & learned_roi
    mode = str(task["mode"])
    expected_calls = int(
        math.floor(len(data["samples"]) * float(task["budget"]) + 0.5)
    )
    capacity = float(task["capacity_bytes"])
    utilization = (
        len(selection) / max(1, expected_calls)
        if mode == "fixed_call"
        else request_bytes / max(1.0, capacity)
    )
    return {
        "policy": str(task["policy"]),
        "mode": mode,
        "budget": float(task["budget"]),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": request_bytes,
        "capacity_bytes": capacity,
        "capacity_utilization": utilization,
        "capacity_ok": bool(request_bytes <= capacity + 1e-6)
        if mode == "fixed_byte"
        else len(selection) == expected_calls,
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "exact_roi_precision": len(common) / max(1, len(learned_roi)),
        "exact_roi_recall": len(common) / max(1, len(teacher_roi)),
    }


def _classification_metrics(
    probabilities: Mapping[int, float], targets: Mapping[int, bool]
) -> dict[str, float]:
    threshold = float(PROTOCOL["probability_threshold"])
    prediction = {image_id for image_id, value in probabilities.items() if value >= threshold}
    positive = {image_id for image_id, value in targets.items() if value}
    common = prediction & positive
    correct = sum((image_id in prediction) == bool(targets[image_id]) for image_id in targets)
    return {
        "predicted_positive": len(prediction),
        "true_positive": len(positive),
        "precision": len(common) / max(1, len(prediction)),
        "recall": len(common) / max(1, len(positive)),
        "accuracy": correct / max(1, len(targets)),
    }


def run_evaluate() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    output = RUNS / "evaluation/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    oof = run_oof()
    probability_path = RUNS / "oof/probabilities.pkl"
    if file_sha256(probability_path) != oof["probability_sha256"]:
        raise RuntimeError("OOF probability artifact changed")
    with probability_path.open("rb") as handle:
        gate_artifact = pickle.load(handle)
    with (V1_ROOT / "runs/confirm/scores.pkl").open("rb") as handle:
        score_artifact = pickle.load(handle)
    label = "OQR/BoundaryPathDistill"
    ensemble = np.mean(list(score_artifact["scores"][label].values()), axis=0)
    records = V1._records_for_space("fit", "OQR")
    ungated = V1._score_map(records, ensemble)
    policies = {"Ungated-AFPD": ungated}
    for method in PROTOCOL["methods"]:
        policies[method] = CORE.gate_roi_scores(
            ungated,
            gate_artifact["probabilities"][method],
            threshold=float(PROTOCOL["probability_threshold"]),
        )
    policies["GT-AnyFrontier-Gate"] = CORE.gate_roi_scores(
        ungated,
        {
            image_id: float(value)
            for image_id, value in gate_artifact["occupancy_target"].items()
        },
        threshold=float(PROTOCOL["probability_threshold"]),
    )
    data = V1.REL.load_role("fit")
    samples = data["samples"]
    rewards = V1._reward_map_for_space("fit", "OQR")
    total_original = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    tasks = []
    for budget in PROTOCOL["budgets"]:
        capacity = float(budget) * total_original
        for mode in PROTOCOL["modes"]:
            teacher = (
                V1.ACTION.schedule_fixed_byte(samples, rewards, capacity_bytes=capacity)
                if mode == "fixed_byte"
                else V1.ACTION.schedule_fixed_call(samples, rewards, budget=float(budget))
            )
            teacher_roi = [
                int(image_id)
                for image_id, action in teacher["selection"].items()
                if action == "roi"
            ]
            for policy, score_map in policies.items():
                scheduled = (
                    V1.ACTION.schedule_fixed_byte(
                        samples, score_map, capacity_bytes=capacity
                    )
                    if mode == "fixed_byte"
                    else V1.ACTION.schedule_fixed_call(
                        samples, score_map, budget=float(budget)
                    )
                )
                tasks.append(
                    {
                        "policy": policy,
                        "mode": mode,
                        "budget": float(budget),
                        "capacity_bytes": capacity,
                        "selection": scheduled["selection"],
                        "teacher_roi": teacher_roi,
                    }
                )
    _EVAL_STATE.clear()
    _EVAL_STATE.update({"data": data})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(_eval_task, tasks, chunksize=1))
    if len(rows) != 24 or not all(row["capacity_ok"] for row in rows):
        raise RuntimeError("Evaluation matrix is invalid")
    order = ["Ungated-AFPD", *PROTOCOL["methods"], "GT-AnyFrontier-Gate"]
    rows.sort(
        key=lambda row: (
            order.index(str(row["policy"])),
            PROTOCOL["modes"].index(str(row["mode"])),
            float(row["budget"]),
        )
    )
    comparisons = {}
    for method in PROTOCOL["methods"]:
        work_points = []
        for mode in PROTOCOL["modes"]:
            for budget in PROTOCOL["budgets"]:
                baseline = next(
                    row
                    for row in rows
                    if row["policy"] == "Ungated-AFPD"
                    and row["mode"] == mode
                    and math.isclose(float(row["budget"]), float(budget))
                )
                candidate = next(
                    row
                    for row in rows
                    if row["policy"] == method
                    and row["mode"] == mode
                    and math.isclose(float(row["budget"]), float(budget))
                )
                work_points.append(
                    {
                        "mode": mode,
                        "budget": float(budget),
                        "delta_ap": float(candidate["official_ap50_95"])
                        - float(baseline["official_ap50_95"]),
                        "capacity_utilization": float(candidate["capacity_utilization"]),
                    }
                )
        comparisons[method] = {
            "work_points": work_points,
            "gate": V1.CORE.evaluate_screen_gate(work_points, PROTOCOL["gate"]),
            "occupancy_classification": _classification_metrics(
                gate_artifact["probabilities"][method],
                gate_artifact["occupancy_target"],
            ),
        }
    passed = [method for method, value in comparisons.items() if value["gate"]["passed"]]
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "comparisons": comparisons,
        "passed_methods": passed,
        "promoted_method": max(
            passed,
            key=lambda method: np.mean(
                [
                    row["delta_ap"]
                    for row in comparisons[method]["work_points"]
                    if row["mode"] == "fixed_byte"
                ]
            ),
        )
        if passed
        else None,
        "threshold": float(PROTOCOL["probability_threshold"]),
        "threshold_selected_on_oof": False,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    result_path = RUNS / "evaluation/results.csv"
    atomic_csv(result_path, rows)
    atomic_json(output, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "results_sha256": file_sha256(result_path),
            "report_sha256": file_sha256(output),
        },
    )
    _EVAL_STATE.clear()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("preflight", "oof", "evaluate", "all"), default="all"
    )
    args = parser.parse_args()
    if args.stage in {"preflight", "all"}:
        print(json.dumps(run_preflight(), indent=2), flush=True)
    if args.stage in {"oof", "all"}:
        print(json.dumps(run_oof(), indent=2), flush=True)
    if args.stage in {"evaluate", "all"}:
        print(json.dumps(run_evaluate(), indent=2), flush=True)


if __name__ == "__main__":
    main()
