#!/usr/bin/env python3
"""Run the frozen COCO AFPD 2x2 attribution and paper extension."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import gc
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


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("afpd_paper_core", ROOT / "scripts/core.py")
SOURCES = {
    name: REPO / path for name, path in PROTOCOL["source_experiments"].items()
}
V1 = _load("afpd_paper_source_v1", SOURCES["afpd_v1"] / "scripts/run_experiment.py")
V2 = _load(
    "afpd_paper_source_v2", SOURCES["occupancy_v2"] / "scripts/run_experiment.py"
)
BOOTSTRAP_CORE = _load("afpd_paper_bootstrap", SOURCES["bootstrap_core"])
COMMON = SOURCES["canonical"] / "runs/canonical_v1/common/coco"

ACTION_METHOD_SPECS = {
    "Current-OQ": ("OQ", "CurrentActionValue"),
    "Current-OQR": ("OQR", "CurrentActionValue"),
    "AFPD-OQ": ("OQ", "BoundaryPathDistill"),
    "AFPD-OQR": ("OQR", "BoundaryPathDistill"),
}
OCCUPANCY_SPECS = {
    "OccupancyFree": "CostFree-FrontierGate",
    "OccupancyAware": "CostAware-FrontierGate",
}
CORE.validate_protocol(PROTOCOL)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def ensemble_arrays(values: Mapping[int, Any]) -> np.ndarray:
    expected = tuple(map(int, PROTOCOL["seeds"]))
    if tuple(sorted(map(int, values))) != tuple(sorted(expected)):
        raise ValueError("Ensemble requires every frozen seed exactly once")
    arrays = [np.asarray(values[seed], dtype=np.float64) for seed in expected]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Ensemble score shapes changed")
    output = np.mean(arrays, axis=0)
    if not np.isfinite(output).all():
        raise ValueError("Ensemble contains non-finite values")
    return output


_ROLE_CACHE: dict[str, dict[str, Any]] = {}


def load_role(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(f"Forbidden role: {role}")
    if role not in _ROLE_CACHE:
        if role in {"fit", "calibration"}:
            data = V1.REL.load_role(role)
        else:
            V1.REL._configure_canonical()
            data = V1.REL.canonical.load_role("coco", "final")
        if len(data["samples"]) != int(PROTOCOL["roles"][role]):
            raise RuntimeError(f"Canonical {role} population changed")
        _ROLE_CACHE[role] = data
    return _ROLE_CACHE[role]


def category_ids(role: str) -> list[int]:
    values = [
        int(row["id"])
        for row in read_json(COMMON / role / "gt.json")["categories"]
    ]
    if len(values) != 80 or len(values) != len(set(values)):
        raise RuntimeError(f"COCO category schema changed for {role}")
    return values


def records_for_role(role: str, action_space: str) -> list[dict[str, Any]]:
    data = load_role(role)
    records = V1.ACTION.build_action_records(
        data["samples"], data["actions"]["edge"], category_ids=category_ids(role)
    )
    output = V1.CORE.filter_action_space_records(
        records, PROTOCOL["action_spaces"][action_space]
    )
    expected = [int(sample["image_id"]) for sample in data["samples"]]
    if [int(row["image_id"]) for row in output] != expected:
        raise RuntimeError(f"{role}/{action_space} record order changed")
    return output


def prediction_tensors(
    role: str, action_space: str
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    records = records_for_role(role, action_space)
    tensor_protocol = {
        "max_boxes": int(V1.PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(V1.PROTOCOL["model"]["max_actions"]),
        "deploy_confidence": float(V1.PROTOCOL["model"]["deploy_confidence"]),
    }
    data = V1.TRAINING.strip_cost_inputs(
        V1.REL._record_tensors(records, None, tensor_protocol)
    )
    return records, data


def _sample_identities(role: str) -> list[str]:
    return [str(row["identity"]) for row in read_json(COMMON / role / "samples.json")]


def run_preflight() -> dict[str, Any]:
    completion = RUNS / "preflight/completion.json"
    output = RUNS / "preflight/report.json"
    if completion.is_file():
        return read_json(output)
    required = [
        SOURCES["afpd_v1"] / "runs/confirm/evaluation_completion.json",
        SOURCES["afpd_v1"] / "runs/confirm/scores.pkl",
        SOURCES["afpd_v1"] / "runs/rewards/fit/roi_reward_map.pkl",
        SOURCES["occupancy_v2"] / "runs/evaluation/completion.json",
        V1.REL_ROOT / "runs/preflight/folds.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete source experiments: {missing}")
    role_ids: dict[str, list[int]] = {}
    identities: dict[str, set[str]] = {}
    for role in CORE.preflight_roles(PROTOCOL):
        expected = PROTOCOL["roles"][role]
        samples = read_json(COMMON / role / "samples.json")
        ids = [int(row["image_id"]) for row in samples]
        if len(ids) != int(expected) or len(ids) != len(set(ids)):
            raise RuntimeError(f"Canonical {role} population changed")
        role_ids[role] = ids
        identities[role] = set(_sample_identities(role))
    if identities["fit"] & identities["calibration"]:
        raise RuntimeError("Fit/calibration identity overlap")
    folds = read_json(V1.REL_ROOT / "runs/preflight/folds.json")["folds"]
    flat_folds = [int(image_id) for fold in folds for image_id in fold]
    if len(folds) != int(PROTOCOL["fold_count"]) or set(flat_folds) != set(
        role_ids["fit"]
    ):
        raise RuntimeError("Frozen OOF folds changed")
    source_paths = [
        PROTOCOL_PATH,
        ROOT / "scripts/core.py",
        SCRIPT,
        SOURCES["afpd_v1"] / "configs/protocol.json",
        SOURCES["afpd_v1"] / "scripts/core.py",
        SOURCES["afpd_v1"] / "scripts/training.py",
        SOURCES["afpd_v1"] / "scripts/run_experiment.py",
        SOURCES["occupancy_v2"] / "configs/protocol.json",
        SOURCES["occupancy_v2"] / "scripts/core.py",
        SOURCES["occupancy_v2"] / "scripts/training.py",
        SOURCES["occupancy_v2"] / "scripts/run_experiment.py",
        *required,
    ]
    if any("cloudv2" in str(path).lower() for path in source_paths):
        raise RuntimeError("Cloud-V2 sources are forbidden")
    hashes = {str(path): file_sha256(path) for path in source_paths}
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "role_counts": {role: len(ids) for role, ids in role_ids.items()},
        "final_expected_count_unread": int(PROTOCOL["roles"]["final"]),
        "fit_calibration_identity_overlap": 0,
        "fold_sizes": list(map(len, folds)),
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "calibration_accessed": False,
        "final_accessed": False,
        "final_is_hidden": False,
        "detector_inference_used": False,
    }
    atomic_json(output, report)
    atomic_json(
        completion,
        {"complete": True, "source_digest": report["source_digest"]},
    )
    return report


def frozen_folds() -> list[list[int]]:
    return [
        list(map(int, fold))
        for fold in read_json(V1.REL_ROOT / "runs/preflight/folds.json")["folds"]
    ]


_ACTION_OOF_STATE: dict[str, Any] = {}


def _train_action_oof_job(task: tuple[str, int, int]) -> dict[str, Any]:
    label, base_seed, fold_index = str(task[0]), int(task[1]), int(task[2])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    action_space, training_method = ACTION_METHOD_SPECS[label]
    root = RUNS / f"oof/action/jobs/{label}/seed_{base_seed}/fold_{fold_index}"
    completion = root / "completion.json"
    if completion.is_file():
        return read_json(root / "report.json")
    data = _ACTION_OOF_STATE["data"][action_space]
    image_ids = _ACTION_OOF_STATE["image_ids"][action_space]
    heldout_ids = set(_ACTION_OOF_STATE["folds"][fold_index])
    heldout = [
        index for index, image_id in enumerate(image_ids) if image_id in heldout_ids
    ]
    heldout_indices = set(heldout)
    train = [index for index in range(len(image_ids)) if index not in heldout_indices]
    if len(heldout) != len(heldout_ids):
        raise RuntimeError(f"Fold {fold_index} population changed")
    job_seed = base_seed + fold_index * 10_007
    started = time.time()
    model, fit = V1._fit_screen_model(
        training_method, data, image_ids, train, seed=job_seed
    )
    scores = V1._predict_screen_model(
        model,
        data,
        heldout,
        reward_mean=float(fit["reward_mean"]),
        reward_std=float(fit["reward_std"]),
    )
    score_path = root / "scores.npz"
    checkpoint_path = root / "checkpoint.pt"
    atomic_npz(
        score_path,
        image_ids=np.asarray([image_ids[index] for index in heldout], dtype=np.int64),
        scores=scores,
    )
    atomic_torch(
        checkpoint_path,
        {
            "label": label,
            "action_space": action_space,
            "training_method": training_method,
            "base_seed": base_seed,
            "job_seed": job_seed,
            "model_state": model.state_dict(),
            "fit": fit,
        },
    )
    report = {
        "label": label,
        "action_space": action_space,
        "training_method": training_method,
        "base_seed": base_seed,
        "job_seed": job_seed,
        "fold": fold_index,
        "train_images": len(train),
        "heldout_images": len(heldout),
        "selected_epoch": int(fit["selected_epoch"]),
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(root / "report.json", report)
    atomic_json(
        completion,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
    )
    return report


def _assemble_action_oof(
    label: str, seed: int, image_ids: Sequence[int]
) -> np.ndarray:
    position = {int(image_id): index for index, image_id in enumerate(image_ids)}
    matrix = np.full(
        (len(image_ids), int(V1.PROTOCOL["model"]["max_actions"])),
        np.nan,
        dtype=np.float64,
    )
    for fold_index in range(int(PROTOCOL["fold_count"])):
        path = RUNS / f"oof/action/jobs/{label}/seed_{seed}/fold_{fold_index}/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            for row, image_id in enumerate(map(int, artifact["image_ids"])):
                matrix[position[image_id]] = artifact["scores"][row]
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"Incomplete action OOF matrix: {label}/{seed}")
    return matrix


def run_action_oof_training() -> dict[str, Any]:
    completion = RUNS / "oof/action/completion.json"
    report_path = RUNS / "oof/action/report.json"
    if completion.is_file():
        return read_json(report_path)
    run_preflight()
    data_by_space: dict[str, dict[str, torch.Tensor]] = {}
    ids_by_space: dict[str, list[int]] = {}
    for action_space in PROTOCOL["action_spaces"]:
        records, data = V1._screen_tensor_data("fit", action_space)
        data_by_space[action_space] = data
        ids_by_space[action_space] = [int(row["image_id"]) for row in records]
    folds = frozen_folds()
    _ACTION_OOF_STATE.clear()
    _ACTION_OOF_STATE.update(
        {"data": data_by_space, "image_ids": ids_by_space, "folds": folds}
    )
    tasks = [
        (label, int(seed), fold_index)
        for label in PROTOCOL["action_methods"]
        for seed in PROTOCOL["seeds"]
        for fold_index in range(int(PROTOCOL["fold_count"]))
    ]
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        reports = list(
            pool.imap_unordered(_train_action_oof_job, tasks, chunksize=1)
        )
    scores: dict[str, dict[int, np.ndarray]] = {}
    for label, (action_space, _) in ACTION_METHOD_SPECS.items():
        scores[label] = {
            int(seed): _assemble_action_oof(
                label, int(seed), ids_by_space[action_space]
            )
            for seed in PROTOCOL["seeds"]
        }
    artifact_path = RUNS / "oof/action/scores.pkl"
    atomic_pickle(
        artifact_path,
        {
            "image_ids": ids_by_space,
            "scores": scores,
            "seeds": list(map(int, PROTOCOL["seeds"])),
        },
    )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "jobs": len(tasks),
        "completed_jobs": len(reports),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(artifact_path),
        "fit_oof_only": True,
        "calibration_accessed": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {"complete": True, "score_sha256": report["score_sha256"]},
    )
    _ACTION_OOF_STATE.clear()
    gc.collect()
    return report


_EVALUATION_STATE: dict[str, Any] = {}


def _evaluate_selection_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    data = _EVALUATION_STATE["data"]
    selection = {int(key): str(value) for key, value in task["selection"].items()}
    predictions, _ = V1.REL.canonical.mixed_predictions(data, selection, 0.001)
    metric = data["evaluator"].evaluate(predictions)
    request_bytes = sum(
        int(data["sample_by_id"][image_id]["payload_bytes"][action])
        for image_id, action in selection.items()
    )
    mode = str(task["mode"])
    budget = float(task["budget"])
    expected_calls = int(math.floor(len(data["samples"]) * budget + 0.5))
    capacity = float(task["capacity_bytes"])
    if mode == "fixed_byte":
        CORE.assert_fixed_byte_capacity(request_bytes, capacity)
        utilization = request_bytes / max(1.0, capacity)
    else:
        if len(selection) != expected_calls:
            raise RuntimeError("Fixed-call selection count changed")
        utilization = len(selection) / max(1, expected_calls)
    counts = Counter(selection.values())
    return {
        "role": str(task["role"]),
        "method": str(task["method"]),
        "seed": str(task["seed"]),
        "mode": mode,
        "budget": budget,
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": request_bytes,
        "capacity_bytes": capacity,
        "capacity_utilization": utilization,
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "final_used_for_selection": False,
    }


def _schedule(
    samples: Sequence[Mapping[str, Any]],
    score_map: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
    capacity_bytes: float,
) -> dict[int, str]:
    result = (
        V1.ACTION.schedule_fixed_byte(
            samples, score_map, capacity_bytes=capacity_bytes
        )
        if mode == "fixed_byte"
        else V1.ACTION.schedule_fixed_call(samples, score_map, budget=budget)
    )
    return {int(image_id): str(action) for image_id, action in result["selection"].items()}


def _comparison_work_points(
    rows: Sequence[Mapping[str, Any]], candidate: str, baseline: str, seed: str
) -> list[dict[str, Any]]:
    output = []
    for mode in PROTOCOL["modes"]:
        for budget in map(float, PROTOCOL["budgets"]):
            base = next(
                row
                for row in rows
                if row["method"] == baseline
                and str(row["seed"]) == seed
                and row["mode"] == mode
                and math.isclose(float(row["budget"]), budget)
            )
            value = next(
                row
                for row in rows
                if row["method"] == candidate
                and str(row["seed"]) == seed
                and row["mode"] == mode
                and math.isclose(float(row["budget"]), budget)
            )
            output.append(
                {
                    "mode": mode,
                    "budget": budget,
                    "delta_ap": float(value["official_ap50_95"])
                    - float(base["official_ap50_95"]),
                    "capacity_utilization": float(value["capacity_utilization"]),
                }
            )
    return output


def _action_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {
        "roi_effect_current": ("Current-OQR", "Current-OQ"),
        "objective_effect_oq": ("AFPD-OQ", "Current-OQ"),
        "objective_effect_oqr": ("AFPD-OQR", "Current-OQR"),
        "total_effect": ("AFPD-OQR", "Current-OQ"),
    }
    comparisons = {}
    for name, (candidate, baseline) in pairs.items():
        ensemble = _comparison_work_points(rows, candidate, baseline, "ensemble")
        seed_points = {
            str(seed): _comparison_work_points(
                rows, candidate, baseline, str(seed)
            )
            for seed in PROTOCOL["seeds"]
        }
        support = {}
        for budget in map(float, PROTOCOL["budgets"]):
            support[f"{budget:g}"] = sum(
                next(
                    point
                    for point in points
                    if point["mode"] == "fixed_byte"
                    and math.isclose(float(point["budget"]), budget)
                )["delta_ap"]
                >= float(PROTOCOL["gate"]["minimum_fixed_byte_gain"])
                for points in seed_points.values()
            )
        gate = CORE.evaluate_delta_gate(ensemble, PROTOCOL["gate"])
        gate["seed_support"] = support
        gate["seed_support_sufficient"] = all(
            count >= int(PROTOCOL["gate"]["minimum_supporting_seeds"])
            for budget, count in support.items()
            if next(
                point
                for point in ensemble
                if point["mode"] == "fixed_byte"
                and math.isclose(float(point["budget"]), float(budget))
            )["delta_ap"]
            >= float(PROTOCOL["gate"]["minimum_fixed_byte_gain"])
        )
        gate["passed_with_seed_support"] = bool(
            gate["passed"] and gate["seed_support_sufficient"]
        )
        comparisons[name] = {
            "candidate": candidate,
            "baseline": baseline,
            "ensemble_work_points": ensemble,
            "seed_work_points": seed_points,
            "gate": gate,
        }
    attribution = {}
    for mode in PROTOCOL["modes"]:
        attribution[mode] = {}
        for budget in map(float, PROTOCOL["budgets"]):
            values = {
                method: float(
                    next(
                        row
                        for row in rows
                        if row["method"] == method
                        and row["seed"] == "ensemble"
                        and row["mode"] == mode
                        and math.isclose(float(row["budget"]), budget)
                    )["official_ap50_95"]
                )
                for method in ACTION_METHOD_SPECS
            }
            attribution[mode][f"{budget:g}"] = CORE.two_by_two_attribution(values)
    return {"comparisons": comparisons, "attribution": attribution}


def run_action_oof_evaluation() -> dict[str, Any]:
    completion = RUNS / "oof/action_evaluation/completion.json"
    report_path = RUNS / "oof/action_evaluation/report.json"
    if completion.is_file():
        return read_json(report_path)
    training = run_action_oof_training()
    artifact_path = RUNS / "oof/action/scores.pkl"
    if training["score_sha256"] != file_sha256(artifact_path):
        raise RuntimeError("Action OOF score artifact changed")
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    data = load_role("fit")
    samples = data["samples"]
    original_total = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    tasks = []
    selections: dict[str, dict[str, str]] = {}
    for label, (action_space, _) in ACTION_METHOD_SPECS.items():
        records = records_for_role("fit", action_space)
        expected_ids = [int(row["image_id"]) for row in records]
        if expected_ids != list(map(int, artifact["image_ids"][action_space])):
            raise RuntimeError(f"OOF score order changed: {label}")
        matrices = {
            str(seed): np.asarray(artifact["scores"][label][int(seed)])
            for seed in PROTOCOL["seeds"]
        }
        matrices["ensemble"] = ensemble_arrays(
            {int(seed): artifact["scores"][label][int(seed)] for seed in PROTOCOL["seeds"]}
        )
        for seed, matrix in matrices.items():
            score_map = V1._score_map(records, matrix)
            for mode in PROTOCOL["modes"]:
                for budget in map(float, PROTOCOL["budgets"]):
                    capacity = budget * original_total
                    selection = _schedule(
                        samples,
                        score_map,
                        mode=mode,
                        budget=budget,
                        capacity_bytes=capacity,
                    )
                    key = f"{label}|{seed}|{mode}|{budget:g}"
                    selections[key] = {
                        str(image_id): action for image_id, action in selection.items()
                    }
                    tasks.append(
                        {
                            "role": "fit_oof",
                            "method": label,
                            "seed": seed,
                            "mode": mode,
                            "budget": budget,
                            "capacity_bytes": capacity,
                            "selection": selection,
                        }
                    )
    _EVALUATION_STATE.clear()
    _EVALUATION_STATE.update({"data": data})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        rows = list(
            pool.imap_unordered(_evaluate_selection_worker, tasks, chunksize=1)
        )
    expected = CORE.expected_oof_action_rows(PROTOCOL)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} action OOF rows, got {len(rows)}")
    keys = {
        (row["method"], row["seed"], row["mode"], float(row["budget"]))
        for row in rows
    }
    if len(keys) != expected:
        raise RuntimeError("Duplicate action OOF rows")
    order = list(PROTOCOL["action_methods"])
    seeds = [*map(str, PROTOCOL["seeds"]), "ensemble"]
    rows.sort(
        key=lambda row: (
            order.index(str(row["method"])),
            seeds.index(str(row["seed"])),
            PROTOCOL["modes"].index(str(row["mode"])),
            float(row["budget"]),
        )
    )
    analysis = _action_comparisons(rows)
    result_path = RUNS / "oof/action_evaluation/results.csv"
    selection_path = RUNS / "oof/action_evaluation/selections.json"
    atomic_csv(result_path, rows)
    atomic_json(selection_path, selections)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        **analysis,
        "calibration_accessed": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "results_sha256": file_sha256(result_path),
            "selections_sha256": file_sha256(selection_path),
            "report_sha256": file_sha256(report_path),
        },
    )
    _EVALUATION_STATE.clear()
    _ROLE_CACHE.clear()
    gc.collect()
    return report


_OCCUPANCY_OOF_STATE: dict[str, Any] = {}


def _roi_available_indices(data: Mapping[str, torch.Tensor]) -> list[int]:
    available = data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
    return [index for index in range(len(available)) if bool(available[index].any())]


def _train_occupancy_oof_job(task: tuple[str, int, int]) -> dict[str, Any]:
    label, base_seed, fold_index = str(task[0]), int(task[1]), int(task[2])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    source_method = OCCUPANCY_SPECS[label]
    root = RUNS / f"oof/occupancy/jobs/{label}/seed_{base_seed}/fold_{fold_index}"
    completion = root / "completion.json"
    if completion.is_file():
        return read_json(root / "report.json")
    data = _OCCUPANCY_OOF_STATE["data"][source_method]
    image_ids = _OCCUPANCY_OOF_STATE["image_ids"]
    heldout_ids = set(_OCCUPANCY_OOF_STATE["folds"][fold_index])
    roi_available = set(_roi_available_indices(data))
    heldout = [
        index
        for index, image_id in enumerate(image_ids)
        if image_id in heldout_ids and index in roi_available
    ]
    train = [
        index for index, image_id in enumerate(image_ids) if image_id not in heldout_ids
    ]
    job_seed = base_seed + fold_index * 10_007
    started = time.time()
    model, fit = V2._fit(data, image_ids, train, seed=job_seed)
    probabilities = V2._predict_roi(model, data, heldout)
    probability_path = root / "probabilities.npz"
    checkpoint_path = root / "checkpoint.pt"
    atomic_npz(
        probability_path,
        image_ids=np.asarray([image_ids[index] for index in heldout], dtype=np.int64),
        probabilities=probabilities,
    )
    atomic_torch(
        checkpoint_path,
        {
            "label": label,
            "source_method": source_method,
            "base_seed": base_seed,
            "job_seed": job_seed,
            "model_state": model.state_dict(),
            "fit": fit,
        },
    )
    report = {
        "label": label,
        "source_method": source_method,
        "base_seed": base_seed,
        "job_seed": job_seed,
        "fold": fold_index,
        "heldout_roi_images": len(heldout),
        "selected_epoch": int(fit["selected_epoch"]),
        "elapsed_seconds": time.time() - started,
        "probability_sha256": file_sha256(probability_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(root / "report.json", report)
    atomic_json(
        completion,
        {
            "complete": True,
            "probability_sha256": report["probability_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
    )
    return report


def _assemble_occupancy_oof(label: str, seed: int) -> dict[int, float]:
    values: dict[int, float] = {}
    for fold_index in range(int(PROTOCOL["fold_count"])):
        path = RUNS / (
            f"oof/occupancy/jobs/{label}/seed_{seed}/fold_{fold_index}/"
            "probabilities.npz"
        )
        with np.load(path, allow_pickle=False) as artifact:
            for image_id, probability in zip(
                artifact["image_ids"], artifact["probabilities"]
            ):
                image_id = int(image_id)
                if image_id in values:
                    raise RuntimeError(f"Duplicate occupancy OOF ID: {image_id}")
                values[image_id] = float(probability)
    if not values or not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"Invalid occupancy OOF probabilities: {label}/{seed}")
    return values


def run_occupancy_oof_training() -> dict[str, Any]:
    completion = RUNS / "oof/occupancy/completion.json"
    report_path = RUNS / "oof/occupancy/report.json"
    if completion.is_file():
        return read_json(report_path)
    run_action_oof_training()
    records, source_data, occupancy_target = V2._build_training_data()
    image_ids = [int(row["image_id"]) for row in records]
    expected_roi = {
        image_ids[index]
        for index in _roi_available_indices(source_data[OCCUPANCY_SPECS["OccupancyFree"]])
    }
    _OCCUPANCY_OOF_STATE.clear()
    _OCCUPANCY_OOF_STATE.update(
        {
            "data": source_data,
            "image_ids": image_ids,
            "folds": frozen_folds(),
        }
    )
    tasks = [
        (label, int(seed), fold_index)
        for label in PROTOCOL["occupancy_methods"]
        for seed in PROTOCOL["seeds"]
        for fold_index in range(int(PROTOCOL["fold_count"]))
    ]
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        reports = list(
            pool.imap_unordered(_train_occupancy_oof_job, tasks, chunksize=1)
        )
    probabilities: dict[str, dict[int, dict[int, float]]] = {}
    for label in PROTOCOL["occupancy_methods"]:
        probabilities[label] = {
            int(seed): _assemble_occupancy_oof(label, int(seed))
            for seed in PROTOCOL["seeds"]
        }
        for seed, values in probabilities[label].items():
            if set(values) != expected_roi:
                raise RuntimeError(f"Occupancy OOF population changed: {label}/{seed}")
    target_map = {}
    action_type = source_data[OCCUPANCY_SPECS["OccupancyFree"]]["action_type"]
    for row, image_id in enumerate(image_ids):
        columns = torch.nonzero(action_type[row] == 2).flatten()
        if len(columns):
            target_map[image_id] = bool(occupancy_target[row, int(columns[0])])
    artifact_path = RUNS / "oof/occupancy/probabilities.pkl"
    atomic_pickle(
        artifact_path,
        {
            "probabilities": probabilities,
            "occupancy_target": target_map,
            "threshold": float(PROTOCOL["occupancy_threshold"]),
        },
    )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "jobs": len(tasks),
        "completed_jobs": len(reports),
        "workers": workers,
        "roi_images": len(expected_roi),
        "positive_frontier_roi_images": sum(target_map.values()),
        "elapsed_seconds": time.time() - started,
        "probability_sha256": file_sha256(artifact_path),
        "fit_oof_only": True,
        "calibration_accessed": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {"complete": True, "probability_sha256": report["probability_sha256"]},
    )
    _OCCUPANCY_OOF_STATE.clear()
    gc.collect()
    return report


def _mean_probability_maps(
    values: Mapping[int, Mapping[int, float]]
) -> dict[int, float]:
    expected = tuple(map(int, PROTOCOL["seeds"]))
    if tuple(sorted(map(int, values))) != tuple(sorted(expected)):
        raise ValueError("Probability ensemble requires all frozen seeds")
    populations = [set(values[seed]) for seed in expected]
    if any(population != populations[0] for population in populations[1:]):
        raise ValueError("Probability populations changed across seeds")
    return {
        image_id: float(np.mean([values[seed][image_id] for seed in expected]))
        for image_id in populations[0]
    }


def run_occupancy_oof_evaluation() -> dict[str, Any]:
    completion = RUNS / "oof/occupancy_evaluation/completion.json"
    report_path = RUNS / "oof/occupancy_evaluation/report.json"
    if completion.is_file():
        return read_json(report_path)
    gate_training = run_occupancy_oof_training()
    action_training = run_action_oof_training()
    probability_path = RUNS / "oof/occupancy/probabilities.pkl"
    action_path = RUNS / "oof/action/scores.pkl"
    if gate_training["probability_sha256"] != file_sha256(probability_path):
        raise RuntimeError("Occupancy probability artifact changed")
    if action_training["score_sha256"] != file_sha256(action_path):
        raise RuntimeError("Action score artifact changed")
    with probability_path.open("rb") as handle:
        gate_artifact = pickle.load(handle)
    with action_path.open("rb") as handle:
        action_artifact = pickle.load(handle)
    records = records_for_role("fit", "OQR")
    data = load_role("fit")
    samples = data["samples"]
    original_total = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    action_matrices = {
        str(seed): np.asarray(action_artifact["scores"]["AFPD-OQR"][int(seed)])
        for seed in PROTOCOL["seeds"]
    }
    action_matrices["ensemble"] = ensemble_arrays(
        action_artifact["scores"]["AFPD-OQR"]
    )
    probability_maps: dict[str, dict[str, dict[int, float]]] = {}
    for label in PROTOCOL["occupancy_methods"]:
        probability_maps[label] = {
            str(seed): gate_artifact["probabilities"][label][int(seed)]
            for seed in PROTOCOL["seeds"]
        }
        probability_maps[label]["ensemble"] = _mean_probability_maps(
            gate_artifact["probabilities"][label]
        )
    tasks = []
    selections: dict[str, dict[str, str]] = {}
    policy_order = [
        "AFPD-OQR",
        "AFPD-OQR+OccupancyFree",
        "AFPD-OQR+OccupancyAware",
    ]
    for seed, matrix in action_matrices.items():
        base = V1._score_map(records, matrix)
        policies = {"AFPD-OQR": base}
        for label in PROTOCOL["occupancy_methods"]:
            policies[f"AFPD-OQR+{label}"] = V2.CORE.gate_roi_scores(
                base,
                probability_maps[label][seed],
                threshold=float(PROTOCOL["occupancy_threshold"]),
            )
        for policy, score_map in policies.items():
            for mode in PROTOCOL["modes"]:
                for budget in map(float, PROTOCOL["budgets"]):
                    capacity = budget * original_total
                    selection = _schedule(
                        samples,
                        score_map,
                        mode=mode,
                        budget=budget,
                        capacity_bytes=capacity,
                    )
                    key = f"{policy}|{seed}|{mode}|{budget:g}"
                    selections[key] = {
                        str(image_id): action for image_id, action in selection.items()
                    }
                    tasks.append(
                        {
                            "role": "fit_oof",
                            "method": policy,
                            "seed": seed,
                            "mode": mode,
                            "budget": budget,
                            "capacity_bytes": capacity,
                            "selection": selection,
                        }
                    )
    _EVALUATION_STATE.clear()
    _EVALUATION_STATE.update({"data": data})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        rows = list(
            pool.imap_unordered(_evaluate_selection_worker, tasks, chunksize=1)
        )
    expected = CORE.expected_oof_occupancy_rows(PROTOCOL)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} occupancy rows, got {len(rows)}")
    seeds = [*map(str, PROTOCOL["seeds"]), "ensemble"]
    rows.sort(
        key=lambda row: (
            policy_order.index(str(row["method"])),
            seeds.index(str(row["seed"])),
            PROTOCOL["modes"].index(str(row["mode"])),
            float(row["budget"]),
        )
    )
    comparisons = {}
    for label in PROTOCOL["occupancy_methods"]:
        candidate = f"AFPD-OQR+{label}"
        ensemble = _comparison_work_points(
            rows, candidate, "AFPD-OQR", "ensemble"
        )
        comparisons[label] = {
            "candidate": candidate,
            "baseline": "AFPD-OQR",
            "ensemble_work_points": ensemble,
            "gate": CORE.evaluate_delta_gate(ensemble, PROTOCOL["gate"]),
        }
    result_path = RUNS / "oof/occupancy_evaluation/results.csv"
    selection_path = RUNS / "oof/occupancy_evaluation/selections.json"
    atomic_csv(result_path, rows)
    atomic_json(selection_path, selections)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "comparisons": comparisons,
        "calibration_accessed": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "results_sha256": file_sha256(result_path),
            "selections_sha256": file_sha256(selection_path),
            "report_sha256": file_sha256(report_path),
        },
    )
    _EVALUATION_STATE.clear()
    _ROLE_CACHE.clear()
    gc.collect()
    return report


_FULL_ACTION_STATE: dict[str, Any] = {}


def _train_full_action_job(task: tuple[str, int]) -> dict[str, Any]:
    label, seed = str(task[0]), int(task[1])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    action_space, training_method = ACTION_METHOD_SPECS[label]
    root = RUNS / f"calibration/action/jobs/{label}/seed_{seed}"
    completion = root / "completion.json"
    if completion.is_file():
        return read_json(root / "report.json")
    fit_data = _FULL_ACTION_STATE["fit_data"][action_space]
    fit_ids = _FULL_ACTION_STATE["fit_ids"][action_space]
    calibration_data = _FULL_ACTION_STATE["calibration_data"][action_space]
    calibration_ids = _FULL_ACTION_STATE["calibration_ids"][action_space]
    started = time.time()
    model, fit = V1._fit_screen_model(
        training_method,
        fit_data,
        fit_ids,
        list(range(len(fit_ids))),
        seed=seed,
    )
    scores = V1._predict_screen_model(
        model,
        calibration_data,
        list(range(len(calibration_ids))),
        reward_mean=float(fit["reward_mean"]),
        reward_std=float(fit["reward_std"]),
    )
    score_path = root / "calibration_scores.npz"
    checkpoint_path = root / "checkpoint.pt"
    atomic_npz(
        score_path,
        image_ids=np.asarray(calibration_ids, dtype=np.int64),
        scores=scores,
    )
    atomic_torch(
        checkpoint_path,
        {
            "label": label,
            "action_space": action_space,
            "training_method": training_method,
            "seed": seed,
            "model_state": model.state_dict(),
            "fit": fit,
            "fit_images": len(fit_ids),
        },
    )
    report = {
        "label": label,
        "action_space": action_space,
        "training_method": training_method,
        "seed": seed,
        "fit_images": len(fit_ids),
        "calibration_images": len(calibration_ids),
        "selected_epoch": int(fit["selected_epoch"]),
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(root / "report.json", report)
    atomic_json(
        completion,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
    )
    return report


def occupancy_prediction_tensors(
    role: str,
) -> tuple[list[int], dict[str, dict[str, torch.Tensor]], list[int]]:
    records = records_for_role(role, "OQR")
    tensor_protocol = {
        "max_boxes": int(V2.PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(V2.PROTOCOL["model"]["max_actions"]),
        "deploy_confidence": float(V2.PROTOCOL["model"]["deploy_confidence"]),
    }
    raw = V1.REL._record_tensors(records, None, tensor_protocol)
    data = {
        "CostAware-FrontierGate": raw,
        "CostFree-FrontierGate": V1.TRAINING.strip_cost_inputs(raw),
    }
    available = _roi_available_indices(raw)
    ids = [int(row["image_id"]) for row in records]
    return ids, data, available


_FULL_OCCUPANCY_STATE: dict[str, Any] = {}


def _train_full_occupancy_job(task: tuple[str, int]) -> dict[str, Any]:
    label, seed = str(task[0]), int(task[1])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    source_method = OCCUPANCY_SPECS[label]
    root = RUNS / f"calibration/occupancy/jobs/{label}/seed_{seed}"
    completion = root / "completion.json"
    if completion.is_file():
        return read_json(root / "report.json")
    fit_data = _FULL_OCCUPANCY_STATE["fit_data"][source_method]
    fit_ids = _FULL_OCCUPANCY_STATE["fit_ids"]
    calibration_data = _FULL_OCCUPANCY_STATE["calibration_data"][source_method]
    calibration_ids = _FULL_OCCUPANCY_STATE["calibration_ids"]
    calibration_available = _FULL_OCCUPANCY_STATE["calibration_available"]
    started = time.time()
    model, fit = V2._fit(
        fit_data, fit_ids, list(range(len(fit_ids))), seed=seed
    )
    probabilities = V2._predict_roi(
        model, calibration_data, calibration_available
    )
    probability_path = root / "calibration_probabilities.npz"
    checkpoint_path = root / "checkpoint.pt"
    atomic_npz(
        probability_path,
        image_ids=np.asarray(
            [calibration_ids[index] for index in calibration_available],
            dtype=np.int64,
        ),
        probabilities=probabilities,
    )
    atomic_torch(
        checkpoint_path,
        {
            "label": label,
            "source_method": source_method,
            "seed": seed,
            "model_state": model.state_dict(),
            "fit": fit,
            "fit_images": len(fit_ids),
        },
    )
    report = {
        "label": label,
        "source_method": source_method,
        "seed": seed,
        "fit_images": len(fit_ids),
        "calibration_roi_images": len(calibration_available),
        "selected_epoch": int(fit["selected_epoch"]),
        "elapsed_seconds": time.time() - started,
        "probability_sha256": file_sha256(probability_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(root / "report.json", report)
    atomic_json(
        completion,
        {
            "complete": True,
            "probability_sha256": report["probability_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
    )
    return report


def run_calibration_training() -> dict[str, Any]:
    completion = RUNS / "calibration/training_completion.json"
    report_path = RUNS / "calibration/training_report.json"
    if completion.is_file():
        return read_json(report_path)
    run_action_oof_evaluation()
    run_occupancy_oof_evaluation()
    fit_data: dict[str, dict[str, torch.Tensor]] = {}
    fit_ids: dict[str, list[int]] = {}
    calibration_data: dict[str, dict[str, torch.Tensor]] = {}
    calibration_ids: dict[str, list[int]] = {}
    for action_space in PROTOCOL["action_spaces"]:
        fit_records, fit_tensor = V1._screen_tensor_data("fit", action_space)
        calibration_records, calibration_tensor = prediction_tensors(
            "calibration", action_space
        )
        fit_data[action_space] = fit_tensor
        fit_ids[action_space] = [int(row["image_id"]) for row in fit_records]
        calibration_data[action_space] = calibration_tensor
        calibration_ids[action_space] = [
            int(row["image_id"]) for row in calibration_records
        ]
    _FULL_ACTION_STATE.clear()
    _FULL_ACTION_STATE.update(
        {
            "fit_data": fit_data,
            "fit_ids": fit_ids,
            "calibration_data": calibration_data,
            "calibration_ids": calibration_ids,
        }
    )
    action_tasks = [
        (label, int(seed))
        for label in PROTOCOL["action_methods"]
        for seed in PROTOCOL["seeds"]
    ]
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(action_tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        action_reports = list(
            pool.imap_unordered(_train_full_action_job, action_tasks, chunksize=1)
        )
    _FULL_ACTION_STATE.clear()
    gc.collect()

    occupancy_records, occupancy_fit_data, _ = V2._build_training_data()
    occupancy_fit_ids = [int(row["image_id"]) for row in occupancy_records]
    occ_cal_ids, occ_cal_data, occ_cal_available = occupancy_prediction_tensors(
        "calibration"
    )
    _FULL_OCCUPANCY_STATE.clear()
    _FULL_OCCUPANCY_STATE.update(
        {
            "fit_data": occupancy_fit_data,
            "fit_ids": occupancy_fit_ids,
            "calibration_data": occ_cal_data,
            "calibration_ids": occ_cal_ids,
            "calibration_available": occ_cal_available,
        }
    )
    occupancy_tasks = [
        (label, int(seed))
        for label in PROTOCOL["occupancy_methods"]
        for seed in PROTOCOL["seeds"]
    ]
    workers_occ = min(
        int(PROTOCOL["cpu"]["training_workers"]), len(occupancy_tasks)
    )
    with context.Pool(processes=workers_occ) as pool:
        occupancy_reports = list(
            pool.imap_unordered(
                _train_full_occupancy_job, occupancy_tasks, chunksize=1
            )
        )
    _FULL_OCCUPANCY_STATE.clear()
    gc.collect()

    action_scores: dict[str, dict[int, np.ndarray]] = {}
    for label, (action_space, _) in ACTION_METHOD_SPECS.items():
        action_scores[label] = {}
        for seed in map(int, PROTOCOL["seeds"]):
            path = RUNS / f"calibration/action/jobs/{label}/seed_{seed}/calibration_scores.npz"
            with np.load(path, allow_pickle=False) as artifact:
                if list(map(int, artifact["image_ids"])) != calibration_ids[action_space]:
                    raise RuntimeError(f"Calibration score order changed: {label}/{seed}")
                action_scores[label][seed] = np.asarray(
                    artifact["scores"], dtype=np.float64
                )
    occupancy_probabilities: dict[str, dict[int, dict[int, float]]] = {}
    expected_roi_ids = {occ_cal_ids[index] for index in occ_cal_available}
    for label in PROTOCOL["occupancy_methods"]:
        occupancy_probabilities[label] = {}
        for seed in map(int, PROTOCOL["seeds"]):
            path = RUNS / (
                f"calibration/occupancy/jobs/{label}/seed_{seed}/"
                "calibration_probabilities.npz"
            )
            with np.load(path, allow_pickle=False) as artifact:
                values = {
                    int(image_id): float(probability)
                    for image_id, probability in zip(
                        artifact["image_ids"], artifact["probabilities"]
                    )
                }
            if set(values) != expected_roi_ids:
                raise RuntimeError(f"Calibration ROI population changed: {label}")
            occupancy_probabilities[label][seed] = values
    artifact_path = RUNS / "calibration/scores.pkl"
    atomic_pickle(
        artifact_path,
        {
            "image_ids": calibration_ids,
            "action_scores": action_scores,
            "occupancy_probabilities": occupancy_probabilities,
            "occupancy_threshold": float(PROTOCOL["occupancy_threshold"]),
        },
    )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "action_jobs": len(action_tasks),
        "occupancy_jobs": len(occupancy_tasks),
        "completed_action_jobs": len(action_reports),
        "completed_occupancy_jobs": len(occupancy_reports),
        "action_workers": workers,
        "occupancy_workers": workers_occ,
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(artifact_path),
        "fit_images": int(PROTOCOL["roles"]["fit"]),
        "calibration_images": int(PROTOCOL["roles"]["calibration"]),
        "fit_only_training": True,
        "calibration_used_for_training_or_selection": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {"complete": True, "score_sha256": report["score_sha256"]},
    )
    _ROLE_CACHE.clear()
    gc.collect()
    return report


def _load_role_artifact(role: str) -> dict[str, Any]:
    path = RUNS / f"{role}/scores.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def _reported_score_maps(
    role: str, artifact: Mapping[str, Any]
) -> dict[str, dict[int, dict[str, float]]]:
    maps: dict[str, dict[int, dict[str, float]]] = {}
    for label, (action_space, _) in ACTION_METHOD_SPECS.items():
        records = records_for_role(role, action_space)
        expected = list(map(int, artifact["image_ids"][action_space]))
        if [int(row["image_id"]) for row in records] != expected:
            raise RuntimeError(f"{role} score order changed: {label}")
        matrix = ensemble_arrays(artifact["action_scores"][label])
        maps[label] = V1._score_map(records, matrix)
    base = maps["AFPD-OQR"]
    for label in PROTOCOL["occupancy_methods"]:
        probabilities = _mean_probability_maps(
            artifact["occupancy_probabilities"][label]
        )
        maps[f"AFPD-OQR+{label}"] = V2.CORE.gate_roi_scores(
            base,
            probabilities,
            threshold=float(PROTOCOL["occupancy_threshold"]),
        )
    return maps


def _role_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {
        "roi_effect_current": ("Current-OQR", "Current-OQ"),
        "objective_effect_oq": ("AFPD-OQ", "Current-OQ"),
        "objective_effect_oqr": ("AFPD-OQR", "Current-OQR"),
        "total_effect": ("AFPD-OQR", "Current-OQ"),
        "occupancy_free": ("AFPD-OQR+OccupancyFree", "AFPD-OQR"),
        "occupancy_aware": ("AFPD-OQR+OccupancyAware", "AFPD-OQR"),
    }
    comparisons = {}
    for name, (candidate, baseline) in pairs.items():
        points = _comparison_work_points(rows, candidate, baseline, "ensemble")
        comparisons[name] = {
            "candidate": candidate,
            "baseline": baseline,
            "work_points": points,
            "gate": CORE.evaluate_delta_gate(points, PROTOCOL["gate"]),
        }
    attribution = {}
    for mode in PROTOCOL["modes"]:
        attribution[mode] = {}
        for budget in map(float, PROTOCOL["budgets"]):
            values = {
                method: float(
                    next(
                        row
                        for row in rows
                        if row["method"] == method
                        and row["mode"] == mode
                        and math.isclose(float(row["budget"]), budget)
                    )["official_ap50_95"]
                )
                for method in ACTION_METHOD_SPECS
            }
            attribution[mode][f"{budget:g}"] = CORE.two_by_two_attribution(values)
    return {"comparisons": comparisons, "attribution": attribution}


def _frozen_reference_rows(role: str) -> list[dict[str, Any]]:
    if role == "calibration":
        path = SOURCES["noroi_reference"] / "runs/point/calibration_results.csv"
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [
            dict(row) | {"reference_source": "frozen_noroi_calibration"}
            for row in rows
            if row.get("method")
            in {"ORIC-AP50", "MORIC-AP50", "ORIC-AP5095", "MORIC-AP5095", "DualHead", "SetValue"}
            and row.get("mode") in PROTOCOL["modes"]
            and math.isclose(
                float(row.get("nominal_budget", -1)),
                min(
                    PROTOCOL["budgets"],
                    key=lambda value: abs(float(value) - float(row.get("nominal_budget", -1))),
                ),
            )
        ]
    path = SOURCES["full_benchmark"] / "runs/coco/point/final_results.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    allowed = {"Raw-ActionValue-OQ", "PerAction-Isotonic-OQ", "ORIC-MLP", "MORIC-MLP", "Confidence", "Random"}
    return [
        dict(row) | {"reference_source": "completed_full_benchmark"}
        for row in rows
        if row.get("method") in allowed
        and row.get("mode") in PROTOCOL["modes"]
        and float(row.get("budget", -1)) in set(map(float, PROTOCOL["budgets"]))
    ]


def evaluate_role(role: str) -> dict[str, Any]:
    output_root = RUNS / role / "evaluation"
    completion = output_root / "completion.json"
    report_path = output_root / "report.json"
    if completion.is_file():
        return read_json(report_path)
    artifact = _load_role_artifact(role)
    maps = _reported_score_maps(role, artifact)
    data = load_role(role)
    samples = data["samples"]
    original_total = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    tasks = []
    selections: dict[str, dict[str, str]] = {}
    for method in PROTOCOL["reported_methods"]:
        for mode in PROTOCOL["modes"]:
            for budget in map(float, PROTOCOL["budgets"]):
                capacity = budget * original_total
                selection = _schedule(
                    samples,
                    maps[method],
                    mode=mode,
                    budget=budget,
                    capacity_bytes=capacity,
                )
                key = f"{method}|{mode}|{budget:g}"
                selections[key] = {
                    str(image_id): action for image_id, action in selection.items()
                }
                tasks.append(
                    {
                        "role": role,
                        "method": method,
                        "seed": "ensemble",
                        "mode": mode,
                        "budget": budget,
                        "capacity_bytes": capacity,
                        "selection": selection,
                    }
                )
    _EVALUATION_STATE.clear()
    _EVALUATION_STATE.update({"data": data})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        rows = list(
            pool.imap_unordered(_evaluate_selection_worker, tasks, chunksize=1)
        )
    expected = CORE.expected_role_rows(PROTOCOL)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} {role} rows, got {len(rows)}")
    rows.sort(
        key=lambda row: (
            PROTOCOL["reported_methods"].index(str(row["method"])),
            PROTOCOL["modes"].index(str(row["mode"])),
            float(row["budget"]),
        )
    )
    analysis = _role_analysis(rows)
    reference_rows = _frozen_reference_rows(role)
    result_path = output_root / "results.csv"
    selection_path = output_root / "selections.json"
    reference_path = output_root / "frozen_references.csv"
    atomic_csv(result_path, rows)
    atomic_json(selection_path, selections)
    atomic_csv(reference_path, reference_rows)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": role,
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        **analysis,
        "primary_gate": analysis["comparisons"]["objective_effect_oqr"]["gate"],
        "system_gate": analysis["comparisons"]["total_effect"]["gate"],
        "frozen_reference_rows": len(reference_rows),
        "calibration_used_for_training_or_selection": False,
        "final_used_for_training_or_selection": False,
        "final_is_hidden": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "results_sha256": file_sha256(result_path),
            "selections_sha256": file_sha256(selection_path),
            "report_sha256": file_sha256(report_path),
        },
    )
    _EVALUATION_STATE.clear()
    _ROLE_CACHE.clear()
    gc.collect()
    return report


def run_calibration_evaluation() -> dict[str, Any]:
    completion = RUNS / "calibration/completion.json"
    report_path = RUNS / "calibration/report.json"
    if completion.is_file():
        return read_json(report_path)
    training = run_calibration_training()
    source = RUNS / "calibration/scores.pkl"
    if training["score_sha256"] != file_sha256(source):
        raise RuntimeError("Calibration score artifact changed")
    target = RUNS / "calibration/scores.pkl"
    if source != target:
        raise RuntimeError("Calibration score path changed")
    evaluation = evaluate_role("calibration")
    oof_action = read_json(RUNS / "oof/action_evaluation/report.json")
    oof_occupancy = read_json(RUNS / "oof/occupancy_evaluation/report.json")
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "calibration",
        "status": "evaluated_once",
        "evaluated": True,
        "primary_gate": evaluation["primary_gate"],
        "system_gate": evaluation["system_gate"],
        "oof_primary_gate": oof_action["comparisons"]["objective_effect_oqr"]["gate"],
        "oof_occupancy": oof_occupancy["comparisons"],
        "open_posthoc_final": CORE.should_run_final(
            primary_calibration_gate=evaluation["primary_gate"],
            occupancy_results=oof_occupancy["comparisons"],
        ),
        "score_sha256": training["score_sha256"],
        "calibration_used_for_training_or_selection": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "evaluated": True,
            "open_posthoc_final": report["open_posthoc_final"],
            "report_sha256": file_sha256(report_path),
        },
    )
    return report


_FINAL_ACTION_STATE: dict[str, Any] = {}


def _predict_final_action_job(task: tuple[str, int]) -> dict[str, Any]:
    label, seed = str(task[0]), int(task[1])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    action_space, training_method = ACTION_METHOD_SPECS[label]
    root = RUNS / f"final/action/jobs/{label}/seed_{seed}"
    completion = root / "completion.json"
    if completion.is_file():
        return read_json(root / "report.json")
    checkpoint_path = RUNS / f"calibration/action/jobs/{label}/seed_{seed}/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint["label"] != label
        or checkpoint["action_space"] != action_space
        or checkpoint["training_method"] != training_method
        or int(checkpoint["fit_images"]) != int(PROTOCOL["roles"]["fit"])
    ):
        raise RuntimeError(f"Frozen action checkpoint changed: {label}/{seed}")
    model = V1.TRAINING.build_model(
        category_count=len(category_ids("fit")),
        scene_dim=int(V1.REL.SCENE_DIM),
        action_dim=int(V1.REL.ACTION_DIM),
        config=V1.PROTOCOL["model"],
    )
    model.load_state_dict(checkpoint["model_state"])
    data = _FINAL_ACTION_STATE["data"][action_space]
    image_ids = _FINAL_ACTION_STATE["image_ids"][action_space]
    fit = checkpoint["fit"]
    started = time.time()
    scores = V1._predict_screen_model(
        model,
        data,
        list(range(len(image_ids))),
        reward_mean=float(fit["reward_mean"]),
        reward_std=float(fit["reward_std"]),
    )
    score_path = root / "scores.npz"
    atomic_npz(
        score_path,
        image_ids=np.asarray(image_ids, dtype=np.int64),
        scores=scores,
    )
    report = {
        "label": label,
        "seed": seed,
        "final_images": len(image_ids),
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(root / "report.json", report)
    atomic_json(completion, {"complete": True, "score_sha256": report["score_sha256"]})
    return report


_FINAL_OCCUPANCY_STATE: dict[str, Any] = {}


def _predict_final_occupancy_job(task: tuple[str, int]) -> dict[str, Any]:
    label, seed = str(task[0]), int(task[1])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    source_method = OCCUPANCY_SPECS[label]
    root = RUNS / f"final/occupancy/jobs/{label}/seed_{seed}"
    completion = root / "completion.json"
    if completion.is_file():
        return read_json(root / "report.json")
    checkpoint_path = RUNS / f"calibration/occupancy/jobs/{label}/seed_{seed}/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint["label"] != label
        or checkpoint["source_method"] != source_method
        or int(checkpoint["fit_images"]) != int(PROTOCOL["roles"]["fit"])
    ):
        raise RuntimeError(f"Frozen occupancy checkpoint changed: {label}/{seed}")
    model = V2.TRAINING.build_model(
        category_count=len(category_ids("fit")),
        scene_dim=int(V1.REL.SCENE_DIM),
        action_dim=int(V1.REL.ACTION_DIM),
        config=V2.PROTOCOL["model"],
    )
    model.load_state_dict(checkpoint["model_state"])
    data = _FINAL_OCCUPANCY_STATE["data"][source_method]
    image_ids = _FINAL_OCCUPANCY_STATE["image_ids"]
    available = _FINAL_OCCUPANCY_STATE["available"]
    started = time.time()
    probabilities = V2._predict_roi(model, data, available)
    probability_path = root / "probabilities.npz"
    atomic_npz(
        probability_path,
        image_ids=np.asarray([image_ids[index] for index in available], dtype=np.int64),
        probabilities=probabilities,
    )
    report = {
        "label": label,
        "seed": seed,
        "final_roi_images": len(available),
        "elapsed_seconds": time.time() - started,
        "probability_sha256": file_sha256(probability_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(root / "report.json", report)
    atomic_json(
        completion,
        {"complete": True, "probability_sha256": report["probability_sha256"]},
    )
    return report


def run_final_prediction() -> dict[str, Any]:
    completion = RUNS / "final/prediction_completion.json"
    report_path = RUNS / "final/prediction_report.json"
    if completion.is_file():
        return read_json(report_path)
    calibration = run_calibration_evaluation()
    if not bool(calibration["open_posthoc_final"]):
        report = {
            "protocol_id": PROTOCOL["protocol_id"],
            "status": "skipped_primary_calibration_gate_failed",
            "evaluated": False,
            "final_accessed": False,
            "primary_gate": calibration["primary_gate"],
        }
        atomic_json(report_path, report)
        atomic_json(completion, {"complete": True, "evaluated": False})
        return report
    action_data: dict[str, dict[str, torch.Tensor]] = {}
    action_ids: dict[str, list[int]] = {}
    for action_space in PROTOCOL["action_spaces"]:
        records, data = prediction_tensors("final", action_space)
        action_data[action_space] = data
        action_ids[action_space] = [int(row["image_id"]) for row in records]
    _FINAL_ACTION_STATE.clear()
    _FINAL_ACTION_STATE.update({"data": action_data, "image_ids": action_ids})
    action_tasks = [
        (label, int(seed))
        for label in PROTOCOL["action_methods"]
        for seed in PROTOCOL["seeds"]
    ]
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(action_tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        action_reports = list(
            pool.imap_unordered(_predict_final_action_job, action_tasks, chunksize=1)
        )
    _FINAL_ACTION_STATE.clear()
    gc.collect()
    occ_ids, occ_data, occ_available = occupancy_prediction_tensors("final")
    _FINAL_OCCUPANCY_STATE.clear()
    _FINAL_OCCUPANCY_STATE.update(
        {"data": occ_data, "image_ids": occ_ids, "available": occ_available}
    )
    occupancy_tasks = [
        (label, int(seed))
        for label in PROTOCOL["occupancy_methods"]
        for seed in PROTOCOL["seeds"]
    ]
    workers_occ = min(
        int(PROTOCOL["cpu"]["training_workers"]), len(occupancy_tasks)
    )
    with context.Pool(processes=workers_occ) as pool:
        occupancy_reports = list(
            pool.imap_unordered(
                _predict_final_occupancy_job, occupancy_tasks, chunksize=1
            )
        )
    _FINAL_OCCUPANCY_STATE.clear()
    gc.collect()
    action_scores: dict[str, dict[int, np.ndarray]] = {}
    for label, (action_space, _) in ACTION_METHOD_SPECS.items():
        action_scores[label] = {}
        for seed in map(int, PROTOCOL["seeds"]):
            path = RUNS / f"final/action/jobs/{label}/seed_{seed}/scores.npz"
            with np.load(path, allow_pickle=False) as artifact:
                if list(map(int, artifact["image_ids"])) != action_ids[action_space]:
                    raise RuntimeError(f"Final score order changed: {label}/{seed}")
                action_scores[label][seed] = np.asarray(
                    artifact["scores"], dtype=np.float64
                )
    expected_roi_ids = {occ_ids[index] for index in occ_available}
    occupancy_probabilities: dict[str, dict[int, dict[int, float]]] = {}
    for label in PROTOCOL["occupancy_methods"]:
        occupancy_probabilities[label] = {}
        for seed in map(int, PROTOCOL["seeds"]):
            path = RUNS / f"final/occupancy/jobs/{label}/seed_{seed}/probabilities.npz"
            with np.load(path, allow_pickle=False) as artifact:
                values = {
                    int(image_id): float(probability)
                    for image_id, probability in zip(
                        artifact["image_ids"], artifact["probabilities"]
                    )
                }
            if set(values) != expected_roi_ids:
                raise RuntimeError(f"Final ROI population changed: {label}/{seed}")
            occupancy_probabilities[label][seed] = values
    artifact_path = RUNS / "final/scores.pkl"
    atomic_pickle(
        artifact_path,
        {
            "image_ids": action_ids,
            "action_scores": action_scores,
            "occupancy_probabilities": occupancy_probabilities,
            "occupancy_threshold": float(PROTOCOL["occupancy_threshold"]),
        },
    )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "status": "posthoc_val2017_reuse_predicted_once",
        "evaluated": True,
        "action_jobs": len(action_tasks),
        "occupancy_jobs": len(occupancy_tasks),
        "completed_action_jobs": len(action_reports),
        "completed_occupancy_jobs": len(occupancy_reports),
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(artifact_path),
        "final_images": int(PROTOCOL["roles"]["final"]),
        "final_accessed": True,
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {"complete": True, "evaluated": True, "score_sha256": report["score_sha256"]},
    )
    _ROLE_CACHE.clear()
    gc.collect()
    return report


def run_final_evaluation() -> dict[str, Any]:
    completion = RUNS / "final/completion.json"
    report_path = RUNS / "final/report.json"
    if completion.is_file():
        return read_json(report_path)
    prediction = run_final_prediction()
    if not bool(prediction.get("evaluated", False)):
        report = {
            "protocol_id": PROTOCOL["protocol_id"],
            "status": "skipped_primary_calibration_gate_failed",
            "evaluated": False,
            "final_accessed": False,
        }
        atomic_json(report_path, report)
        atomic_json(completion, {"complete": True, "evaluated": False})
        return report
    evaluation = evaluate_role("final")
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "status": "posthoc_val2017_reuse",
        "evaluated": True,
        "primary_gate": evaluation["primary_gate"],
        "system_gate": evaluation["system_gate"],
        "comparisons": evaluation["comparisons"],
        "attribution": evaluation["attribution"],
        "frozen_reference_rows": evaluation["frozen_reference_rows"],
        "final_accessed": True,
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {"complete": True, "evaluated": True, "report_sha256": file_sha256(report_path)},
    )
    return report


def _bootstrap_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_superior": float(np.mean(values > 0.0)),
    }


def _selection_from_json(values: Mapping[str, Mapping[str, str]], key: str) -> dict[int, str]:
    if key not in values:
        raise RuntimeError(f"Missing selection: {key}")
    return {int(image_id): str(action) for image_id, action in values[key].items()}


def run_bootstrap() -> dict[str, Any]:
    completion = RUNS / "bootstrap/completion.json"
    report_path = RUNS / "bootstrap/report.json"
    if completion.is_file():
        return read_json(report_path)
    final = run_final_evaluation()
    if not bool(final.get("evaluated", False)):
        report = {
            "protocol_id": PROTOCOL["protocol_id"],
            "status": "skipped_final_not_evaluated",
            "evaluated": False,
        }
        atomic_json(report_path, report)
        atomic_json(completion, {"complete": True, "evaluated": False})
        return report
    new_selections = read_json(RUNS / "final/evaluation/selections.json")
    benchmark_root = SOURCES["full_benchmark"] / "runs/coco/point"
    frozen_selections = read_json(benchmark_root / "selections.json")
    data = load_role("final")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    ground_truth = read_json(COMMON / "final/gt.json")
    comparison_specs = (
        ("AFPD_vs_RawActionValue", "AFPD-OQR", "Raw-ActionValue-OQ|Joint-OQ"),
        ("AFPD_vs_MORIC", "AFPD-OQR", "MORIC-MLP|SharedAction-NoROI"),
        ("OccupancyFree_vs_AFPD", "AFPD-OQR+OccupancyFree", "AFPD-OQR"),
        ("OccupancyAware_vs_AFPD", "AFPD-OQR+OccupancyAware", "AFPD-OQR"),
    )
    comparisons = []
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        seed = int(PROTOCOL["bootstrap"]["seed"]) + budget_index * 101
        generator = np.random.default_rng(seed)
        indices = generator.integers(
            0,
            len(image_ids),
            size=(int(PROTOCOL["bootstrap"]["replicates"]), len(image_ids)),
            dtype=np.int32,
        )
        for comparison_type, candidate_name, comparator_name in comparison_specs:
            result_path = RUNS / f"bootstrap/{budget:g}_{comparison_type}.json"
            if result_path.is_file():
                item = read_json(result_path)
                values = np.asarray(item["values"], dtype=np.float64)
            else:
                candidate_key = f"{candidate_name}|fixed_byte|{budget:g}"
                candidate = _selection_from_json(new_selections, candidate_key)
                if comparator_name in {"AFPD-OQR"}:
                    comparator_key = f"{comparator_name}|fixed_byte|{budget:g}"
                    comparator = _selection_from_json(new_selections, comparator_key)
                else:
                    comparator_key = f"{comparator_name}|fixed_byte|{budget:g}"
                    comparator = _selection_from_json(
                        frozen_selections, comparator_key
                    )
                candidate_predictions, _ = V1.REL.canonical.mixed_predictions(
                    data, candidate, 0.001
                )
                comparator_predictions, _ = V1.REL.canonical.mixed_predictions(
                    data, comparator, 0.001
                )
                BOOTSTRAP_CORE.configure_bootstrap(
                    ground_truth,
                    image_ids,
                    candidate_predictions,
                    comparator_predictions,
                )
                values = np.empty(len(indices), dtype=np.float64)
                workers = min(
                    int(PROTOCOL["cpu"]["bootstrap_workers"]),
                    max(1, (os.cpu_count() or 1) - 2),
                )
                context = mp.get_context("fork")
                try:
                    with context.Pool(processes=workers) as pool:
                        tasks = (
                            (replicate, indices[replicate])
                            for replicate in range(len(indices))
                        )
                        for count, (replicate, delta) in enumerate(
                            pool.imap_unordered(
                                BOOTSTRAP_CORE.bootstrap_worker, tasks, chunksize=1
                            ),
                            start=1,
                        ):
                            values[int(replicate)] = float(delta)
                            if count % 100 == 0 or count == len(indices):
                                print(
                                    f"[bootstrap:{comparison_type}:{budget:g}] "
                                    f"{count}/{len(indices)}",
                                    flush=True,
                                )
                finally:
                    BOOTSTRAP_CORE.clear_bootstrap()
                item = {
                    "protocol_id": PROTOCOL["protocol_id"],
                    "budget": budget,
                    "candidate": candidate_name,
                    "candidate_key": candidate_key,
                    "comparator": comparator_name,
                    "comparator_key": comparator_key,
                    "comparison_type": comparison_type,
                    "replicates": len(values),
                    "seed": seed,
                    "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    "resampling_unit": "image",
                    "values": values.tolist(),
                }
                atomic_json(result_path, item)
            comparisons.append(
                {key: value for key, value in item.items() if key != "values"}
                | _bootstrap_summary(values)
            )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "status": "posthoc_val2017_reuse",
        "evaluated": True,
        "comparisons": comparisons,
        "replicates_per_comparison": int(PROTOCOL["bootstrap"]["replicates"]),
        "final_used_for_training_or_selection": False,
        "final_is_hidden": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {"complete": True, "evaluated": True, "report_sha256": file_sha256(report_path)},
    )
    _ROLE_CACHE.clear()
    gc.collect()
    return report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _markdown_role_table(rows: Sequence[Mapping[str, Any]], title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |",
        "|---|---:|---:|---:|",
    ]
    for method in PROTOCOL["reported_methods"]:
        cells = []
        for budget in map(float, PROTOCOL["budgets"]):
            row = next(
                value
                for value in rows
                if value["method"] == method
                and value["mode"] == "fixed_byte"
                and math.isclose(float(value["budget"]), budget)
            )
            cells.append(
                f'{float(row["official_ap50_95"]):.6f} / '
                f'{float(row["kb_per_image"]):.2f}'
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return lines


def run_report() -> dict[str, Any]:
    completion = RUNS / "COMPLETED.json"
    report_path = RUNS / "report.json"
    if completion.is_file():
        return read_json(report_path)
    oof_action = run_action_oof_evaluation()
    oof_occupancy = run_occupancy_oof_evaluation()
    calibration = run_calibration_evaluation()
    final = run_final_evaluation()
    bootstrap = run_bootstrap()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_status": PROTOCOL["result_status"],
        "oof_action": oof_action,
        "oof_occupancy": oof_occupancy,
        "calibration": calibration,
        "final": final,
        "bootstrap": bootstrap,
        "primary_method": PROTOCOL["primary_method"],
        "occupancy_can_replace_primary": False,
        "detector_inference_used": False,
        "gpu_used": False,
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    calibration_rows = _read_csv(RUNS / "calibration/evaluation/results.csv")
    lines = [
        "# COCO AFPD + Occupancy 论文协议结果",
        "",
        "本实验使用 canonical train2017 fit=9,678、calibration=571。"
        "val2017 若被访问，仅属于 post-hoc final reuse，不是新的隐藏 final。",
        "",
        *_markdown_role_table(calibration_rows, "一次性 calibration Fixed-byte"),
        "",
        "## 协议结论",
        "",
        f'- calibration primary gate: **{str(bool(calibration["primary_gate"]["passed"])).lower()}**',
        f'- post-hoc final opened: **{str(bool(final.get("evaluated", False))).lower()}**',
        "- Occupancy Gate 始终是次要消融，不能替换 AFPD-OQR 主方法。",
    ]
    if final.get("evaluated", False):
        final_rows = _read_csv(RUNS / "final/evaluation/results.csv")
        lines.extend(["", *_markdown_role_table(final_rows, "Post-hoc val2017 Fixed-byte")])
    notes = ROOT / "notes/RESULTS_ZH.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    temporary = notes.with_name(f".{notes.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, notes)
    atomic_json(
        completion,
        {
            "complete": True,
            "report_sha256": file_sha256(report_path),
            "notes_sha256": file_sha256(notes),
            "final_is_hidden": False,
        },
    )
    return report


def run_all() -> dict[str, Any]:
    outputs = {}
    stages = (
        ("preflight", run_preflight),
        ("action_oof", run_action_oof_training),
        ("action_oof_evaluation", run_action_oof_evaluation),
        ("occupancy_oof", run_occupancy_oof_training),
        ("occupancy_oof_evaluation", run_occupancy_oof_evaluation),
        ("calibration_training", run_calibration_training),
        ("calibration", run_calibration_evaluation),
        ("final_prediction", run_final_prediction),
        ("final", run_final_evaluation),
        ("bootstrap", run_bootstrap),
        ("report", run_report),
    )
    for name, function in stages:
        print(f"[stage] {name}", flush=True)
        outputs[name] = function()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "action_oof",
            "action_oof_evaluation",
            "occupancy_oof",
            "occupancy_oof_evaluation",
            "calibration_training",
            "calibration",
            "final_prediction",
            "final",
            "bootstrap",
            "report",
            "all",
        ),
        default="preflight",
    )
    args = parser.parse_args()
    stages = {
        "preflight": run_preflight,
        "action_oof": run_action_oof_training,
        "action_oof_evaluation": run_action_oof_evaluation,
        "occupancy_oof": run_occupancy_oof_training,
        "occupancy_oof_evaluation": run_occupancy_oof_evaluation,
        "calibration_training": run_calibration_training,
        "calibration": run_calibration_evaluation,
        "final_prediction": run_final_prediction,
        "final": run_final_evaluation,
        "bootstrap": run_bootstrap,
        "report": run_report,
        "all": run_all,
    }
    print(json.dumps(stages[args.stage](), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
