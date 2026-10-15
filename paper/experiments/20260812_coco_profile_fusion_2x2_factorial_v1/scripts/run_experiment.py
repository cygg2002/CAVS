#!/usr/bin/env python3
"""Gate, retrain and evaluate the COCO Profile × Fusion factorial."""

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
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
SOURCE_ROOT = REPO / PROTOCOL["source_current_experiment"]
sys.dont_write_bytecode = True


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("factorial_run_core", ROOT / "scripts/core.py")
CORE.validate_protocol(PROTOCOL)
BASE = load_module(
    "factorial_current_source", SOURCE_ROOT / "scripts/run_experiment.py"
)
BOOTSTRAP_CORE = BASE.BOOTSTRAP_CORE


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


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


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def is_complete(path: Path) -> bool:
    return path.is_file() and read_json(path).get("complete") is True


def action_path(role: str, profile: str, fusion: str) -> Path:
    if fusion == "MAUR":
        return RUNS / f"actions/{role}/{profile}__640_maur.json"
    if fusion != "Reversible":
        raise ValueError(fusion)
    manifest = read_json(
        SOURCE_ROOT / f"runs/common/coco/{role}/source_manifest.json"
    )
    return Path(manifest["action_sources"][f"{profile}@640"])


def data_for_cell(role: str, method: str) -> dict[str, Any]:
    data = BASE.load_role(role)
    cell = PROTOCOL["cells"][method]
    if cell["fusion"] == "Reversible":
        return data
    actions = dict(data["actions"])
    actions[f"{cell['profile']}@640"] = BASE.SOURCE_EVAL.group_predictions(
        action_path(role, str(cell["profile"]), "MAUR")
    )
    return {**data, "actions": actions}


def source_reward_map(role: str) -> dict[int, dict[str, float]]:
    return load_pickle(SOURCE_ROOT / f"runs/rewards/{role}/reward_map.pkl")


def maur_reward_map(role: str) -> dict[int, dict[str, float]]:
    return load_pickle(RUNS / f"rewards/{role}/reward_map.pkl")


def reward_map_for_method(role: str, method: str) -> dict[int, dict[str, float]]:
    source = source_reward_map(role)
    cell = PROTOCOL["cells"][method]
    profile_action = f"{cell['profile']}@640"
    if cell["fusion"] == "Reversible":
        return {
            int(image_id): {
                action: float(values[action])
                for action in ("original", "q90", profile_action)
            }
            for image_id, values in source.items()
        }
    maur = maur_reward_map(role)
    if set(source) != set(maur):
        raise RuntimeError(f"MAUR reward population changed: {role}")
    return {
        int(image_id): {
            "original": float(values["original"]),
            "q90": float(values["q90"]),
            profile_action: float(maur[int(image_id)][profile_action]),
        }
        for image_id, values in source.items()
    }


def run_preflight() -> dict[str, Any]:
    output = RUNS / "preflight"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    source_protocol = read_json(SOURCE_ROOT / "configs/protocol.json")
    if source_protocol["roles"] != PROTOCOL["roles"]:
        raise RuntimeError("canonical population changed")
    if source_protocol["cloud_model"]["sha256"] != PROTOCOL["cloud_model_sha256"]:
        raise RuntimeError("Cloud model changed")
    if source_protocol["model"] != PROTOCOL["model"]:
        raise RuntimeError("learner hyperparameters are not factorially controlled")
    if source_protocol["model_seeds"] != PROTOCOL["model_seeds"]:
        raise RuntimeError("model seeds changed")
    if source_protocol["fold_count"] != PROTOCOL["fold_count"]:
        raise RuntimeError("fold count changed")
    required = [
        ROOT / "configs/protocol.json",
        ROOT / "scripts/core.py",
        ROOT / "scripts/build_actions.py",
        ROOT / "scripts/build_rewards.py",
        SCRIPT,
        SOURCE_ROOT / "runs/common/completion.json",
        SOURCE_ROOT / "runs/rewards/completion.json",
        SOURCE_ROOT / "runs/training/completion.json",
        SOURCE_ROOT / "runs/evaluation/completion.json",
        RUNS / "actions/prefinal_report.json",
        RUNS / "rewards/completion.json",
    ]
    for role in ("fit", "calibration"):
        required.extend(
            [
                SOURCE_ROOT / f"runs/common/coco/{role}/source_manifest.json",
                SOURCE_ROOT / f"runs/rewards/{role}/reward_map.pkl",
                RUNS / f"rewards/{role}/reward_map.pkl",
            ]
        )
        for profile in PROTOCOL["profiles"]:
            required.extend(
                [
                    action_path(role, profile, "Reversible"),
                    action_path(role, profile, "MAUR"),
                    action_path(role, profile, "MAUR").with_suffix(
                        ".completion.json"
                    ),
                ]
            )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[:10])
    role_counts = {}
    eligible = {}
    for role in ("fit", "calibration"):
        data = BASE.load_role(role)
        role_counts[role] = len(data["samples"])
        if role_counts[role] != int(PROTOCOL["roles"][role]):
            raise RuntimeError(f"role population changed: {role}")
        eligible[role] = {}
        for profile in PROTOCOL["profiles"]:
            action = f"{profile}@640"
            expected = sum(
                bool(sample.get("available", {}).get(action, False))
                for sample in data["samples"]
            )
            maur_report = read_json(
                action_path(role, profile, "MAUR").with_suffix(".completion.json")
            )
            if int(maur_report["eligible_images"]) != expected:
                raise RuntimeError(f"MAUR eligibility changed: {role}/{profile}")
            if maur_report.get("same_geometry_as_reversible") is not True or maur_report.get(
                "same_payload_as_reversible"
            ) is not True:
                raise RuntimeError(f"MAUR pairing contract failed: {role}/{profile}")
            eligible[role][profile] = expected
    anchor_method = source_protocol["methods"]["single"]
    anchor_hashes = {}
    for seed in PROTOCOL["model_seeds"]:
        for fold in (*range(PROTOCOL["fold_count"]), "full"):
            token = "full" if fold == "full" else f"fold_{fold}"
            path = (
                SOURCE_ROOT
                / f"runs/training/jobs/{anchor_method}/seed_{seed}/{token}/artifact.pkl"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            anchor_hashes[f"seed={seed}/{token}"] = file_sha256(path)
    hashes = {str(path.resolve()): file_sha256(path) for path in required}
    report = {
        "complete": True,
        "role_counts": role_counts,
        "eligible_roi": eligible,
        "factorial_cells": list(PROTOCOL["cells"]),
        "new_training_jobs": CORE.expected_new_training_jobs(PROTOCOL),
        "anchor_method": anchor_method,
        "anchor_checkpoint_sha256": anchor_hashes,
        "source_sha256": hashes,
        "source_experiments_read_only": True,
        "final_accessed": False,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output / "report.json")})
    return report


def score_map_from_reward(
    records: Sequence[Mapping[str, Any]], reward_map: Mapping[int, Mapping[str, float]]
) -> dict[int, dict[str, float]]:
    output = {}
    for record in records:
        image_id = int(record["image_id"])
        output[image_id] = {
            str(action["action_id"]): float(reward_map[image_id][str(action["action_id"])])
            for action in record["actions"]
            if str(action["action_id"]) != "edge"
        }
    return output


def evaluate_selection(
    role: str,
    method: str,
    mode: str,
    budget: float,
    selection: Mapping[int, str],
    *,
    point_role: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = data_for_cell(role, method)
    predictions, _ = BASE.CANONICAL.mixed_predictions(data, dict(selection), 0.001)
    metric = data["evaluator"].evaluate(predictions)
    by_id = {int(sample["image_id"]): sample for sample in data["samples"]}
    request_bytes = sum(
        int(by_id[int(image_id)]["payload_bytes"][str(action)])
        for image_id, action in selection.items()
    )
    original_total = sum(int(sample["payload_bytes"]["original"]) for sample in data["samples"])
    capacity = float(budget) * original_total
    counts = Counter(selection.values())
    cell = PROTOCOL["cells"][method]
    row = {
        "role": point_role or role,
        "method": method,
        "profile": str(cell["profile"]),
        "fusion": str(cell["fusion"]),
        "mode": mode,
        "budget": float(budget),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": request_bytes,
        "capacity_bytes": capacity,
        "capacity_utilization": request_bytes / max(1.0, capacity) if mode == "fixed_byte" else 1.0,
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts[f"{cell['profile']}@640"]),
        "reused_anchor": bool(cell["reused_anchor"]),
        "cloud_model_sha256": PROTOCOL["cloud_model_sha256"],
        "final_is_hidden": False,
        "final_used_for_training_gate_or_selection": False,
    }
    return row, predictions


def run_gate() -> dict[str, Any]:
    output = RUNS / "gate"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_preflight()
    role = "calibration"
    rows = []
    for method in PROTOCOL["cells"]:
        records = BASE._filtered(BASE.load_role(role)["records"], CORE.action_space(PROTOCOL, method))
        scores = score_map_from_reward(records, reward_map_for_method(role, method))
        for budget in map(float, PROTOCOL["budgets"]):
            selection = BASE._schedule(
                BASE.load_role(role)["samples"], scores, mode="fixed_byte", budget=budget
            )
            row, _ = evaluate_selection(role, method, "fixed_byte", budget, selection)
            rows.append(row)
    anchor = "CAVS-Current-Reversible"
    passed_cells = []
    deltas = {}
    for method in list(PROTOCOL["cells"])[1:]:
        values = []
        for budget in map(float, PROTOCOL["budgets"]):
            candidate = next(row for row in rows if row["method"] == method and row["budget"] == budget)
            comparator = next(row for row in rows if row["method"] == anchor and row["budget"] == budget)
            values.append(float(candidate["official_ap50_95"]) - float(comparator["official_ap50_95"]))
        deltas[method] = values
        if CORE.cell_passes_gate(values, PROTOCOL["teacher_gate"]):
            passed_cells.append(method)
    gate_passed = bool(passed_cells)
    atomic_csv(output / "teacher_results.csv", rows)
    report = {
        "complete": True,
        "gate_passed": gate_passed,
        "passed_cells": passed_cells,
        "deltas_vs_current_reversible": deltas,
        "criterion": PROTOCOL["teacher_gate"],
        "if_open_train_all_new_cells": True,
        "calibration_images": PROTOCOL["roles"]["calibration"],
        "final_accessed": False,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "gate_passed": gate_passed})
    return report


_TRAIN_STATE: dict[str, Any] = {}


def fit_folds() -> list[list[int]]:
    folds = [
        list(map(int, fold))
        for fold in read_json(SOURCE_ROOT / "runs/rewards/fit/folds.json")["folds"]
    ]
    expected = {int(sample["image_id"]) for sample in BASE.load_role("fit")["samples"]}
    observed = [image_id for fold in folds for image_id in fold]
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise RuntimeError("OOF fold population changed")
    return folds


def initialize_training() -> None:
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        fit_records=BASE.load_role("fit")["records"],
        role_records={role: BASE.load_role(role)["records"] for role in ("calibration", "final")},
        rewards={method: reward_map_for_method("fit", method) for method in PROTOCOL["cells"]},
        folds=fit_folds(),
        category_ids=list(map(int, BASE.load_role("fit")["category_ids"])),
    )


def job_root(job: Mapping[str, Any]) -> Path:
    token = "full" if job["fold"] == "full" else f"fold_{int(job['fold'])}"
    return RUNS / f"training/jobs/{job['method']}/seed_{int(job['seed'])}/{token}"


def train_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    root = job_root(job)
    completion = root / "completion.json"
    if is_complete(completion):
        report = read_json(root / "report.json")
        if report["artifact_sha256"] != file_sha256(root / "artifact.pkl"):
            raise RuntimeError(f"training artifact changed: {job['id']}")
        return report
    records = BASE._filtered(_TRAIN_STATE["fit_records"], job["action_space"])
    rewards = BASE.SOURCE_TRAIN.TRAIN_CORE.reward_matrix(
        records, _TRAIN_STATE["rewards"][job["method"]]
    )
    ids = [int(record["image_id"]) for record in records]
    if job["fold"] == "full":
        train_indices = list(range(len(ids)))
        predictions = {
            role: BASE._filtered(_TRAIN_STATE["role_records"][role], job["action_space"])
            for role in ("calibration", "final")
        }
        model_seed = int(job["seed"])
    else:
        heldout_ids = set(_TRAIN_STATE["folds"][int(job["fold"])])
        heldout = [index for index, image_id in enumerate(ids) if image_id in heldout_ids]
        train_indices = [index for index, image_id in enumerate(ids) if image_id not in heldout_ids]
        if len(heldout) != len(heldout_ids):
            raise RuntimeError(f"OOF population changed: {job['id']}")
        predictions = {"fit_oof": [records[index] for index in heldout]}
        model_seed = int(job["seed"]) + int(job["fold"]) * 10_007
    started = time.time()
    artifact = BASE.SOURCE_TRAIN.ADAPTER.fit_model(
        "current_action",
        [records[index] for index in train_indices],
        rewards[train_indices],
        PROTOCOL,
        category_count=len(_TRAIN_STATE["category_ids"]),
        seed=model_seed,
    )
    arrays = {}
    for role, role_records in predictions.items():
        values = BASE.SOURCE_TRAIN.ADAPTER.predict_model(
            artifact,
            role_records,
            PROTOCOL,
            category_count=len(_TRAIN_STATE["category_ids"]),
        )
        arrays[f"scores_{role}"] = np.asarray(values, dtype=np.float64)
        arrays[f"image_ids_{role}"] = np.asarray(
            [int(record["image_id"]) for record in role_records], dtype=np.int64
        )
    artifact_path = root / "artifact.pkl"
    score_path = root / "scores.npz"
    atomic_pickle(
        artifact_path,
        {
            "method": job["method"],
            "profile": job["profile"],
            "fusion": job["fusion"],
            "action_space": job["action_space"],
            "seed": int(job["seed"]),
            "fold": job["fold"],
            "category_ids": _TRAIN_STATE["category_ids"],
            "cloud_model_sha256": PROTOCOL["cloud_model_sha256"],
            "artifact": artifact,
        },
    )
    atomic_npz(score_path, **arrays)
    metadata = artifact.get("metadata", {})
    report = {
        "complete": True,
        "id": job["id"],
        "method": job["method"],
        "profile": job["profile"],
        "fusion": job["fusion"],
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "fit_images": len(train_indices),
        "selected_epoch": int(metadata.get("selected_epoch", -1)),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": file_sha256(artifact_path),
        "scores_sha256": file_sha256(score_path),
        "old_learner_checkpoint_reused": False,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(root / "report.json", report)
    atomic_json(completion, {"complete": True, "artifact_sha256": report["artifact_sha256"]})
    print(f"[train] {job['id']} {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def run_train() -> dict[str, Any]:
    output = RUNS / "training"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    gate = run_gate()
    if not gate["gate_passed"]:
        report = {"complete": True, "skipped": True, "reason": "teacher_gate_failed", "jobs": 0}
        atomic_json(output / "report.json", report)
        atomic_json(completion, report)
        return report
    final_action_report = RUNS / "actions/final_report.json"
    if not final_action_report.is_file():
        raise RuntimeError("final MAUR actions must be built only after gate passes")
    initialize_training()
    jobs = CORE.training_jobs(PROTOCOL)
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(jobs))
    started = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        reports = list(pool.imap_unordered(train_job, jobs, chunksize=1))
    _TRAIN_STATE.clear()
    if len(reports) != CORE.expected_new_training_jobs(PROTOCOL):
        raise RuntimeError("training matrix incomplete")
    report = {
        "complete": True,
        "skipped": False,
        "jobs": len(reports),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "reused_anchor_jobs": 18,
        "new_jobs": 54,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(reports)})
    return report


def score_path(method: str, seed: int, fold: int | str) -> Path:
    token = "full" if fold == "full" else f"fold_{int(fold)}"
    if PROTOCOL["cells"][method]["reused_anchor"]:
        anchor = BASE.PROTOCOL["methods"]["single"]
        return SOURCE_ROOT / f"runs/training/jobs/{anchor}/seed_{seed}/{token}/scores.npz"
    return RUNS / f"training/jobs/{method}/seed_{seed}/{token}/scores.npz"


def ensemble_scores(method: str, point_role: str) -> np.ndarray:
    role = "fit" if point_role == "fit_oof" else point_role
    records = BASE._filtered(BASE.load_role(role)["records"], CORE.action_space(PROTOCOL, method))
    expected = [int(record["image_id"]) for record in records]
    seed_values = []
    for seed in map(int, PROTOCOL["model_seeds"]):
        if point_role == "fit_oof":
            by_id = {}
            for fold in range(int(PROTOCOL["fold_count"])):
                with np.load(score_path(method, seed, fold), allow_pickle=False) as artifact:
                    ids = list(map(int, artifact["image_ids_fit_oof"]))
                    values = CORE.normalize_oof_scores(
                        np.asarray(artifact["scores_fit_oof"], dtype=np.float64),
                        expected_images=len(ids),
                        expected_budgets=len(PROTOCOL["budgets"]),
                    )
                by_id.update({image_id: values[index] for index, image_id in enumerate(ids)})
            if set(by_id) != set(expected) or len(by_id) != len(expected):
                raise RuntimeError(f"OOF score population changed: {method}/{seed}")
            matrix = np.stack([by_id[image_id] for image_id in expected])
        else:
            with np.load(score_path(method, seed, "full"), allow_pickle=False) as artifact:
                ids = list(map(int, artifact[f"image_ids_{role}"]))
                matrix = np.asarray(artifact[f"scores_{role}"], dtype=np.float64)
            if ids != expected:
                raise RuntimeError(f"full score alignment changed: {method}/{seed}/{role}")
        seed_values.append(
            BASE.CORE.collapse_static_budget_scores(
                matrix, expected_budgets=len(PROTOCOL["budgets"])
            )
        )
    result = np.mean(seed_values, axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite scores: {method}/{point_role}")
    return result


def evaluation_job(job: Mapping[str, str]) -> dict[str, Any]:
    point_role = str(job["point_role"])
    role = "fit" if point_role == "fit_oof" else point_role
    method = str(job["method"])
    data = BASE.load_role(role)
    records = BASE._filtered(
        data["records"], CORE.action_space(PROTOCOL, method)
    )
    scores = BASE._score_map(records, ensemble_scores(method, point_role))
    rows = []
    selections = {}
    for budget in map(float, PROTOCOL["budgets"]):
        for mode in PROTOCOL["modes"]:
            selection = BASE._schedule(
                data["samples"], scores, mode=mode, budget=budget
            )
            key = f"{point_role}|{method}|{mode}|{budget:g}"
            selections[key] = {
                str(image_id): action for image_id, action in selection.items()
            }
            row, _ = evaluate_selection(
                role, method, mode, budget, selection, point_role=point_role
            )
            rows.append(row)
    return {"rows": rows, "selections": selections}


def run_evaluate() -> dict[str, Any]:
    output = RUNS / "evaluation"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    training = run_train()
    if training.get("skipped"):
        report = {"complete": True, "skipped": True, "reason": training["reason"], "rows": 0}
        atomic_json(output / "report.json", report)
        atomic_json(completion, report)
        return report
    jobs = CORE.evaluation_jobs(PROTOCOL)
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(jobs))
    with mp.get_context("fork").Pool(workers) as pool:
        evaluated = list(pool.map(evaluation_job, jobs, chunksize=1))
    rows = [row for result in evaluated for row in result["rows"]]
    selections = {
        key: selection
        for result in evaluated
        for key, selection in result["selections"].items()
    }
    if len(rows) != CORE.expected_point_rows(PROTOCOL):
        raise RuntimeError(f"point matrix incomplete: {len(rows)}")
    keys = [
        (row["role"], row["method"], row["mode"], row["budget"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate point rows")
    source_rows = read_csv(SOURCE_ROOT / "runs/evaluation/results.csv")
    for role in ("calibration", "final"):
        for budget in map(float, PROTOCOL["budgets"]):
            for mode in PROTOCOL["modes"]:
                own = next(
                    row
                    for row in rows
                    if row["role"] == role
                    and row["method"] == "CAVS-Current-Reversible"
                    and row["mode"] == mode
                    and row["budget"] == budget
                )
                source = next(
                    row
                    for row in source_rows
                    if row["role"] == role
                    and row["method"] == BASE.PROTOCOL["methods"]["single"]
                    and row["mode"] == mode
                    and math.isclose(float(row["budget"]), budget)
                )
                for field in ("official_ap50_95", "official_ap50", "official_ap75", "request_bytes"):
                    if not math.isclose(float(own[field]), float(source[field]), rel_tol=0.0, abs_tol=1e-12):
                        raise RuntimeError(f"anchor reproduction failed: {role}/{mode}/{budget}/{field}")
    atomic_csv(output / "results.csv", rows)
    atomic_json(output / "selections.json", selections)
    report = {
        "complete": True,
        "skipped": False,
        "rows": len(rows),
        "workers": workers,
        "anchor_reproduced": True,
        "results_sha256": file_sha256(output / "results.csv"),
        "selections_sha256": file_sha256(output / "selections.json"),
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "rows": len(rows)})
    return report


_BOOT_STATE: dict[str, Any] = {}


def bootstrap_job(job: Mapping[str, Any]) -> dict[str, Any]:
    safe = hashlib.sha256(str(job["id"]).encode("utf-8")).hexdigest()[:16]
    path = RUNS / f"bootstrap/jobs/{safe}.json"
    partial = path.with_suffix(".partial.json")
    replicates = int(PROTOCOL["bootstrap"]["replicates"])
    if path.is_file():
        report = read_json(path)
        if int(report["replicates"]) != replicates:
            raise RuntimeError(f"bootstrap cache incomplete: {job['id']}")
        return report
    values = list(map(float, read_json(partial)["values"])) if partial.is_file() else []
    token = hashlib.sha256(f"{PROTOCOL['bootstrap']['seed']}:{job['id']}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
    ids = _BOOT_STATE["image_ids"]
    for _ in values:
        rng.integers(0, len(ids), size=len(ids))
    for replicate in range(len(values), replicates):
        sampled = [ids[int(index)] for index in rng.integers(0, len(ids), size=len(ids))]
        effects = 0.0
        for method, coefficient in job["coefficients"].items():
            gt, prediction, _ = BOOTSTRAP_CORE._bootstrap_remap(
                _BOOT_STATE["ground_truth"],
                _BOOT_STATE["predictions"][job["budget"]][method],
                _BOOT_STATE["predictions"][job["budget"]][method],
                sampled,
            )
            effects += float(coefficient) * float(BOOTSTRAP_CORE._in_memory_ap(gt, prediction))
        values.append(float(effects))
        if (replicate + 1) % 100 == 0:
            atomic_json(partial, {"values": values})
    array = np.asarray(values, dtype=np.float64)
    report = {
        **dict(job),
        "replicates": len(values),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0.0)),
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(path, report)
    partial.unlink(missing_ok=True)
    return report


def run_bootstrap() -> dict[str, Any]:
    output = RUNS / "bootstrap"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    evaluation = run_evaluate()
    if evaluation.get("skipped"):
        report = {"complete": True, "skipped": True, "jobs": 0}
        atomic_json(output / "report.json", report)
        atomic_json(completion, report)
        return report
    data = BASE.load_role("final")
    stored = read_json(RUNS / "evaluation/selections.json")
    predictions: dict[float, dict[str, list[dict[str, Any]]]] = {}
    for budget in map(float, PROTOCOL["budgets"]):
        predictions[budget] = {}
        for method in PROTOCOL["cells"]:
            key = f"final|{method}|fixed_byte|{budget:g}"
            selection = {int(image_id): str(action) for image_id, action in stored[key].items()}
            _, values = evaluate_selection("final", method, "fixed_byte", budget, selection)
            predictions[budget][method] = values
    jobs = [
        {
            "id": f"{contrast['name']}|budget={budget:g}|image",
            "name": contrast["name"],
            "coefficients": contrast["coefficients"],
            "budget": budget,
            "unit": "image",
        }
        for budget in map(float, PROTOCOL["budgets"])
        for contrast in CORE.bootstrap_contrasts()
    ]
    if len(jobs) != CORE.expected_bootstrap_jobs(PROTOCOL):
        raise RuntimeError("bootstrap job matrix incomplete")
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        ground_truth=read_json(SOURCE_ROOT / "runs/common/coco/final/gt.json"),
        image_ids=[int(sample["image_id"]) for sample in data["samples"]],
        predictions=predictions,
    )
    workers = min(int(PROTOCOL["cpu"]["bootstrap_workers"]), len(jobs))
    with mp.get_context("fork").Pool(workers) as pool:
        results = list(pool.imap_unordered(bootstrap_job, jobs, chunksize=1))
    _BOOT_STATE.clear()
    results.sort(key=lambda row: (float(row["budget"]), str(row["name"])))
    report = {
        "complete": True,
        "skipped": False,
        "jobs": len(results),
        "replicates": int(PROTOCOL["bootstrap"]["replicates"]),
        "comparisons": results,
        "workers": workers,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(results)})
    return report


def run_report() -> dict[str, Any]:
    bootstrap = run_bootstrap()
    gate = read_json(RUNS / "gate/report.json")
    lines = [
        "# COCO Current/Historical × Reversible/MAUR 2×2 因子实验",
        "",
        "本实验统一了 Profile、Fusion 之外的划分、Cloud、学习器、fold、seed、reward 和调度器。",
        "Current-Reversible 为严格复用锚点，其余三个单元按各自动作 reward 重新训练。",
        "COCO val2017 已经公开复用，因此本结果属于 post-hoc factorial ablation，不是隐藏确认集。",
        "",
        "## Calibration Teacher gate",
        "",
        f"Gate: {'PASS' if gate['gate_passed'] else 'FAIL'}；通过单元：{', '.join(gate['passed_cells']) or '无'}。",
    ]
    if not gate["gate_passed"]:
        lines.extend(["", "没有候选达到预注册 Teacher 门槛，学习器训练已按协议停止。"])
    else:
        rows = read_csv(RUNS / "evaluation/results.csv")
        lines.extend(["", "## Fixed-byte final", "", "| 单元 | 10% | 25% | 40% |", "|---|---:|---:|---:|"])
        for method in PROTOCOL["cells"]:
            cells = []
            for budget in map(float, PROTOCOL["budgets"]):
                row = next(
                    value
                    for value in rows
                    if value["role"] == "final"
                    and value["method"] == method
                    and value["mode"] == "fixed_byte"
                    and math.isclose(float(value["budget"]), budget)
                )
                cells.append(
                    f"{float(row['official_ap50_95']):.6f} / {float(row['kb_per_image']):.2f} KB / ROI {int(float(row['roi_count']))}"
                )
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.extend(["", "## 点估计因子效应", "", "| 预算 | Fusion@Current | Fusion@Historical | Profile@Rev | Profile@MAUR | Interaction |", "|---:|---:|---:|---:|---:|---:|"])
        for budget in map(float, PROTOCOL["budgets"]):
            values = {
                method: float(
                    next(
                        row["official_ap50_95"]
                        for row in rows
                        if row["role"] == "final"
                        and row["method"] == method
                        and row["mode"] == "fixed_byte"
                        and math.isclose(float(row["budget"]), budget)
                    )
                )
                for method in PROTOCOL["cells"]
            }
            effects = CORE.factorial_effects(values)
            lines.append(
                f"| {budget:.0%} | {effects['fusion_at_current']:+.6f} | "
                f"{effects['fusion_at_historical']:+.6f} | {effects['profile_at_reversible']:+.6f} | "
                f"{effects['profile_at_maur']:+.6f} | {effects['interaction']:+.6f} |"
            )
        lines.extend(["", "## Paired bootstrap", "", "| 预算 | 效应 | 均值 | 95% CI | P(>0) |", "|---:|---|---:|---:|---:|"])
        for row in bootstrap["comparisons"]:
            lines.append(
                f"| {float(row['budget']):.0%} | {row['name']} | {float(row['mean_delta']):+.6f} | "
                f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
                f"{float(row['probability_gt_zero']):.3f} |"
            )
    path = ROOT / "RESULTS_ZH.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "complete": True,
        "gate_passed": bool(gate["gate_passed"]),
        "training_skipped": not bool(gate["gate_passed"]),
        "bootstrap_jobs": int(bootstrap.get("jobs", 0)),
        "results_sha256": file_sha256(path),
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(RUNS / "report.json", report)
    atomic_json(RUNS / "COMPLETED.json", {"complete": True, "results_sha256": report["results_sha256"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "gate", "train", "evaluate", "bootstrap", "report", "all"), default="all"
    )
    args = parser.parse_args()
    functions = {
        "preflight": run_preflight,
        "gate": run_gate,
        "train": run_train,
        "evaluate": run_evaluate,
        "bootstrap": run_bootstrap,
        "report": run_report,
        "all": run_report,
    }
    print(json.dumps(functions[args.stage](), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
