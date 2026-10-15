#!/usr/bin/env python3
"""Run the paper-domain COCO ORIC/MORIC action-conditioned experiment."""

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
import torch


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
FACTORIAL_ROOT = REPO / PROTOCOL["source_factorial_experiment"]
COMPLETION_ROOT = REPO / PROTOCOL["source_completion_experiment"]
sys.dont_write_bytecode = True


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("oric_moric_ac_run_core", ROOT / "scripts/core.py")
METHODS = load_module("oric_moric_ac_run_methods", ROOT / "scripts/methods.py")
CORE.validate_protocol(PROTOCOL)
FACTORIAL = load_module(
    "oric_moric_ac_factorial_source", FACTORIAL_ROOT / "scripts/run_experiment.py"
)
BASE = FACTORIAL.BASE
ADAPTER = BASE.SOURCE_TRAIN.ADAPTER
TRAIN_CORE = BASE.SOURCE_TRAIN.TRAIN_CORE
BOOTSTRAP_CORE = FACTORIAL.BOOTSTRAP_CORE
CAVS_SOURCE_METHOD = str(PROTOCOL["cavs_source_method"])
CAVS_REPORT_NAME = str(PROTOCOL["cavs_report_name"])


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
    return path.is_file() and read_json(path).get("complete") is True


def source_paths() -> dict[str, Path]:
    return {
        "factorial_protocol": FACTORIAL_ROOT / "configs/protocol.json",
        "factorial_results": FACTORIAL_ROOT / "runs/evaluation/results.csv",
        "factorial_selections": FACTORIAL_ROOT / "runs/evaluation/selections.json",
        "fit_rewards": FACTORIAL_ROOT / "runs/rewards/fit/reward_map.pkl",
        "fit_folds": FACTORIAL_ROOT / "runs/rewards/fit/folds.json",
        "completion_protocol": COMPLETION_ROOT / "configs/protocol.json",
        "completion_results": COMPLETION_ROOT / "runs/coco/point/results.csv",
        "completion_selections": COMPLETION_ROOT / "runs/coco/point/selections.json",
    }


def paper_data(role: str) -> dict[str, Any]:
    return FACTORIAL.data_for_cell(role, CAVS_SOURCE_METHOD)


def paper_records(role: str) -> list[dict[str, Any]]:
    return BASE._filtered(BASE.load_role(role)["records"], PROTOCOL["action_space"])


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _walk_strings(child)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for child in value for item in _walk_strings(child)]
    return []


def run_preflight() -> dict[str, Any]:
    output = RUNS / "preflight"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    observed_hashes = {key: file_sha256(path) for key, path in paths.items()}
    if observed_hashes != PROTOCOL["source_hashes"]:
        changed = {
            key: {"expected": PROTOCOL["source_hashes"].get(key), "observed": value}
            for key, value in observed_hashes.items()
            if PROTOCOL["source_hashes"].get(key) != value
        }
        raise RuntimeError(f"paper source hash changed: {changed}")
    source_protocol = read_json(paths["factorial_protocol"])
    if source_protocol["roles"] != PROTOCOL["roles"]:
        raise RuntimeError("paper role population changed")
    if source_protocol["cloud_model_sha256"] != PROTOCOL["cloud_model_sha256"]:
        raise RuntimeError("paper Cloud model changed")
    if FACTORIAL.CORE.action_space(source_protocol, CAVS_SOURCE_METHOD) != PROTOCOL["action_space"]:
        raise RuntimeError("paper action space changed")
    cell = source_protocol["cells"][CAVS_SOURCE_METHOD]
    if (str(cell["profile"]), str(cell["fusion"])) != ("current", "MAUR"):
        raise RuntimeError("CAVS source is not Current@640 + MAUR")
    source_rows = read_csv(paths["factorial_results"])
    observed_points = {
        str(row["budget"]): float(row["official_ap50_95"])
        for row in source_rows
        if row["role"] == "final"
        and row["method"] == CAVS_SOURCE_METHOD
        and row["mode"] == "fixed_byte"
    }
    for budget, expected in PROTOCOL["cavs_expected_fixed_byte"].items():
        if budget not in observed_points or abs(observed_points[budget] - float(expected)) > 1e-12:
            raise RuntimeError(f"CAVS paper point changed: {budget}")
    forbidden = tuple(str(value).lower() for value in PROTOCOL["forbidden_path_tokens"])
    role_counts = {}
    manifest_hashes = {}
    for role, expected in PROTOCOL["roles"].items():
        data = paper_data(role)
        role_counts[role] = len(data["samples"])
        if role_counts[role] != int(expected):
            raise RuntimeError(f"role population changed: {role}")
        manifest_path = FACTORIAL.SOURCE_ROOT / f"runs/common/coco/{role}/source_manifest.json"
        manifest = read_json(manifest_path)
        strings = [value.lower() for value in _walk_strings(manifest)]
        if any(token in value for token in forbidden for value in strings):
            raise RuntimeError(f"deployment-domain artifact found in paper manifest: {role}")
        manifest_hashes[role] = file_sha256(manifest_path)
    scalar_rows = read_csv(paths["completion_results"])
    scalar_selections = read_json(paths["completion_selections"])
    expected_scalar_keys = []
    for role in ("calibration", "final"):
        for control in PROTOCOL["scalar_controls"].values():
            for mode in PROTOCOL["modes"]:
                for budget in map(float, PROTOCOL["budgets"]):
                    expected_scalar_keys.append(
                        f"{role}|{control['source_method']}|{control['source_policy']}|{mode}|{budget:g}|seed=ensemble"
                    )
    absent = [key for key in expected_scalar_keys if key not in scalar_selections]
    if absent:
        raise RuntimeError(f"scalar control selections are incomplete: {absent[:3]}")
    ensemble_rows = [row for row in scalar_rows if row.get("seed") == "ensemble"]
    if len(ensemble_rows) < len(expected_scalar_keys):
        raise RuntimeError("scalar control ensemble rows are incomplete")
    report = {
        "complete": True,
        "runtime_domain": PROTOCOL["runtime_domain"],
        "rknn_artifacts_read": 0,
        "detector_inference_required": False,
        "role_counts": role_counts,
        "training_jobs": len(CORE.training_jobs(PROTOCOL)),
        "bootstrap_jobs": len(CORE.bootstrap_jobs(PROTOCOL)),
        "cavs_fixed_byte": observed_points,
        "source_sha256": observed_hashes,
        "role_manifest_sha256": manifest_hashes,
        "source_experiments_read_only": True,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output / "report.json")})
    return report


_TRAIN_STATE: dict[str, Any] = {}


def fit_folds() -> list[list[int]]:
    folds = FACTORIAL.fit_folds()
    if len(folds) != int(PROTOCOL["fold_count"]):
        raise RuntimeError("fit fold count changed")
    return folds


def initialize_training() -> None:
    fit = BASE.load_role("fit")
    records = paper_records("fit")
    reward_map = FACTORIAL.reward_map_for_method("fit", CAVS_SOURCE_METHOD)
    rewards = TRAIN_CORE.reward_matrix(records, reward_map)
    category_ids = list(map(int, fit["category_ids"]))
    features = ADAPTER.external_features(records, category_ids, PROTOCOL)
    tensors = ADAPTER.tensorize(records, rewards, PROTOCOL)
    role_records = {role: paper_records(role) for role in ("calibration", "final")}
    role_features = {
        role: ADAPTER.external_features(values, category_ids, PROTOCOL)
        for role, values in role_records.items()
    }
    role_tensors = {
        role: ADAPTER.tensorize(values, None, PROTOCOL)
        for role, values in role_records.items()
    }
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        records=records,
        rewards=rewards,
        features=features,
        tensors=tensors,
        folds=fit_folds(),
        category_ids=category_ids,
        role_records=role_records,
        role_features=role_features,
        role_tensors=role_tensors,
    )


def subset_tensors(
    tensors: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in tensors.items()}


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
        if report["scores_sha256"] != file_sha256(root / "scores.npz"):
            raise RuntimeError(f"training scores changed: {job['id']}")
        return report
    ids = [int(record["image_id"]) for record in _TRAIN_STATE["records"]]
    if job["fold"] == "full":
        train_indices = list(range(len(ids)))
        predictions = {
            role: (
                _TRAIN_STATE["role_features"][role],
                _TRAIN_STATE["role_tensors"][role],
                _TRAIN_STATE["role_records"][role],
            )
            for role in ("calibration", "final")
        }
        model_seed = int(job["seed"])
    else:
        heldout_ids = set(_TRAIN_STATE["folds"][int(job["fold"])])
        heldout = [index for index, image_id in enumerate(ids) if image_id in heldout_ids]
        train_indices = [index for index, image_id in enumerate(ids) if image_id not in heldout_ids]
        if len(heldout) != len(heldout_ids):
            raise RuntimeError(f"OOF population changed: {job['id']}")
        predictions = {
            "fit_oof": (
                _TRAIN_STATE["features"][heldout],
                subset_tensors(_TRAIN_STATE["tensors"], heldout),
                [_TRAIN_STATE["records"][index] for index in heldout],
            )
        }
        model_seed = int(job["seed"]) + int(job["fold"]) * 10_007
    started = time.time()
    artifact = METHODS.fit_action_conditioned(
        str(job["method"]),
        _TRAIN_STATE["features"][train_indices],
        subset_tensors(_TRAIN_STATE["tensors"], train_indices),
        PROTOCOL,
        seed=model_seed,
    )
    arrays: dict[str, np.ndarray] = {}
    for role, (features, tensors, records) in predictions.items():
        values = METHODS.predict_action_conditioned(artifact, features, tensors, PROTOCOL)
        arrays[f"scores_{role}"] = np.asarray(values, dtype=np.float64)
        arrays[f"image_ids_{role}"] = np.asarray(
            [int(record["image_id"]) for record in records], dtype=np.int64
        )
    artifact_path = root / "artifact.pkl"
    score_path = root / "scores.npz"
    atomic_pickle(
        artifact_path,
        {
            "method": str(job["method"]),
            "target": str(job["target"]),
            "action_space": list(job["action_space"]),
            "seed": int(job["seed"]),
            "fold": job["fold"],
            "category_ids": _TRAIN_STATE["category_ids"],
            "cloud_model_sha256": PROTOCOL["cloud_model_sha256"],
            "runtime_domain": PROTOCOL["runtime_domain"],
            "artifact": artifact,
        },
    )
    atomic_npz(score_path, **arrays)
    report = {
        "complete": True,
        "id": str(job["id"]),
        "method": str(job["method"]),
        "target": str(job["target"]),
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "fit_images": len(train_indices),
        "selected_epoch": int(artifact["metadata"]["selected_epoch"]),
        "cdf_reference_count": int(artifact["metadata"]["cdf_reference_count"]),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": file_sha256(artifact_path),
        "scores_sha256": file_sha256(score_path),
        "rknn_artifacts_read": 0,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(root / "report.json", report)
    atomic_json(
        completion,
        {
            "complete": True,
            "artifact_sha256": report["artifact_sha256"],
            "scores_sha256": report["scores_sha256"],
        },
    )
    print(f"[train] {job['id']} {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def run_train() -> dict[str, Any]:
    output = RUNS / "training"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_preflight()
    initialize_training()
    jobs = CORE.training_jobs(PROTOCOL)
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(jobs))
    started = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        reports = list(pool.imap_unordered(train_job, jobs, chunksize=1))
    _TRAIN_STATE.clear()
    if len(reports) != 36:
        raise RuntimeError(f"training matrix incomplete: {len(reports)}")
    report = {
        "complete": True,
        "jobs": len(reports),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "methods": list(CORE.METHODS),
        "runtime_domain": PROTOCOL["runtime_domain"],
        "rknn_artifacts_read": 0,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(reports)})
    return report


def score_path(method: str, seed: int, fold: int | str) -> Path:
    token = "full" if fold == "full" else f"fold_{int(fold)}"
    return RUNS / f"training/jobs/{method}/seed_{int(seed)}/{token}/scores.npz"


def scores_for_seed(method: str, point_role: str, seed: int) -> np.ndarray:
    role = "fit" if point_role == "fit_oof" else point_role
    records = paper_records(role)
    expected = [int(record["image_id"]) for record in records]
    if point_role == "fit_oof":
        by_id: dict[int, np.ndarray] = {}
        for fold in range(int(PROTOCOL["fold_count"])):
            with np.load(score_path(method, seed, fold), allow_pickle=False) as artifact:
                ids = list(map(int, artifact["image_ids_fit_oof"]))
                values = np.asarray(artifact["scores_fit_oof"], dtype=np.float64)
            if values.ndim != 2 or values.shape[0] != len(ids):
                raise RuntimeError(f"OOF score shape changed: {method}/{seed}/{fold}")
            overlap = set(by_id).intersection(ids)
            if overlap:
                raise RuntimeError(f"OOF score overlap: {method}/{seed}/{fold}")
            by_id.update({image_id: values[index] for index, image_id in enumerate(ids)})
        if set(by_id) != set(expected) or len(by_id) != len(expected):
            raise RuntimeError(f"OOF population changed: {method}/{seed}")
        return np.stack([by_id[image_id] for image_id in expected])
    with np.load(score_path(method, seed, "full"), allow_pickle=False) as artifact:
        ids = list(map(int, artifact[f"image_ids_{role}"]))
        values = np.asarray(artifact[f"scores_{role}"], dtype=np.float64)
    if ids != expected or values.ndim != 2 or values.shape[0] != len(expected):
        raise RuntimeError(f"full score alignment changed: {method}/{seed}/{role}")
    return values


def ensemble_scores(method: str, point_role: str) -> np.ndarray:
    values = [
        scores_for_seed(method, point_role, int(seed))
        for seed in PROTOCOL["model_seeds"]
    ]
    output = np.mean(values, axis=0)
    if not np.isfinite(output).all():
        raise RuntimeError(f"non-finite ensemble scores: {method}/{point_role}")
    return output


def score_map(
    records: Sequence[Mapping[str, Any]], matrix: np.ndarray
) -> dict[int, dict[str, float]]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(records):
        raise ValueError(f"unexpected score shape: {values.shape}")
    output = {}
    for row, record in enumerate(records):
        output[int(record["image_id"])] = {
            str(action["action_id"]): float(values[row, column])
            for column, action in enumerate(record["actions"])
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
    seed: str = "ensemble",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row, predictions = FACTORIAL.evaluate_selection(
        role,
        CAVS_SOURCE_METHOD,
        mode,
        budget,
        selection,
        point_role=point_role,
    )
    row.update(
        {
            "method": method,
            "policy": "Joint-OQR",
            "seed": seed,
            "target": str(PROTOCOL["methods"].get(method, {}).get("target", "anchor")),
            "runtime_domain": PROTOCOL["runtime_domain"],
            "rknn_artifacts_read": 0,
        }
    )
    return row, predictions


def evaluate_new_job(job: Mapping[str, Any]) -> dict[str, Any]:
    method = str(job["method"])
    point_role = str(job["point_role"])
    role = "fit" if point_role == "fit_oof" else point_role
    seed = job.get("seed", "ensemble")
    matrix = (
        ensemble_scores(method, point_role)
        if seed == "ensemble"
        else scores_for_seed(method, point_role, int(seed))
    )
    records = paper_records(role)
    scores = score_map(records, matrix)
    rows = []
    selections = {}
    modes = PROTOCOL["modes"] if seed == "ensemble" else ["fixed_byte"]
    for mode in modes:
        for budget in map(float, PROTOCOL["budgets"]):
            selection = BASE._schedule(
                BASE.load_role(role)["samples"], scores, mode=mode, budget=budget
            )
            row, _ = evaluate_selection(
                role,
                method,
                mode,
                budget,
                selection,
                point_role=point_role,
                seed=str(seed),
            )
            rows.append(row)
            if seed == "ensemble":
                key = f"{point_role}|{method}|{mode}|{budget:g}"
                selections[key] = {
                    str(image_id): action for image_id, action in selection.items()
                }
    return {"rows": rows, "selections": selections}


def _source_row(
    rows: Sequence[Mapping[str, str]],
    *,
    role: str,
    method: str,
    policy: str | None,
    mode: str,
    budget: float,
    seed: str | None,
) -> dict[str, str]:
    matched = [
        row
        for row in rows
        if row.get("role") == role
        and row.get("method") == method
        and (policy is None or row.get("policy") == policy)
        and row.get("mode") == mode
        and abs(float(row.get("budget", "nan")) - float(budget)) < 1e-12
        and (seed is None or row.get("seed") == seed)
    ]
    if len(matched) != 1:
        raise RuntimeError(
            f"source row is not unique: {role}/{method}/{policy}/{mode}/{budget}/{seed}"
        )
    return dict(matched[0])


def imported_ensemble_anchors() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    selections = {}
    factorial_rows = read_csv(FACTORIAL_ROOT / "runs/evaluation/results.csv")
    factorial_selections = read_json(FACTORIAL_ROOT / "runs/evaluation/selections.json")
    for role in ("fit_oof", "calibration", "final"):
        for mode in PROTOCOL["modes"]:
            for budget in map(float, PROTOCOL["budgets"]):
                source = _source_row(
                    factorial_rows,
                    role=role,
                    method=CAVS_SOURCE_METHOD,
                    policy=None,
                    mode=mode,
                    budget=budget,
                    seed=None,
                )
                source.update(
                    {
                        "method": CAVS_REPORT_NAME,
                        "source_method": CAVS_SOURCE_METHOD,
                        "policy": "Joint-OQR",
                        "seed": "ensemble",
                        "target": "raw_reward",
                        "runtime_domain": PROTOCOL["runtime_domain"],
                        "rknn_artifacts_read": 0,
                    }
                )
                rows.append(source)
                source_key = f"{role}|{CAVS_SOURCE_METHOD}|{mode}|{budget:g}"
                target_key = f"{role}|{CAVS_REPORT_NAME}|{mode}|{budget:g}"
                selections[target_key] = factorial_selections[source_key]
    completion_rows = read_csv(COMPLETION_ROOT / "runs/coco/point/results.csv")
    completion_selections = read_json(COMPLETION_ROOT / "runs/coco/point/selections.json")
    for report_name, control in PROTOCOL["scalar_controls"].items():
        for role in ("calibration", "final"):
            for mode in PROTOCOL["modes"]:
                for budget in map(float, PROTOCOL["budgets"]):
                    source = _source_row(
                        completion_rows,
                        role=role,
                        method=str(control["source_method"]),
                        policy=str(control["source_policy"]),
                        mode=mode,
                        budget=budget,
                        seed="ensemble",
                    )
                    source.update(
                        {
                            "method": report_name,
                            "source_method": str(control["source_method"]),
                            "source_policy": str(control["source_policy"]),
                            "policy": "SharedAction-OQR-MAUR",
                            "target": "scalar_image_reward",
                            "runtime_domain": PROTOCOL["runtime_domain"],
                            "rknn_artifacts_read": 0,
                        }
                    )
                    rows.append(source)
                    source_key = (
                        f"{role}|{control['source_method']}|{control['source_policy']}|"
                        f"{mode}|{budget:g}|seed=ensemble"
                    )
                    target_key = f"{role}|{report_name}|{mode}|{budget:g}"
                    selections[target_key] = completion_selections[source_key]
    return rows, selections


def imported_seed_anchors() -> list[dict[str, Any]]:
    source_rows = read_csv(COMPLETION_ROOT / "runs/coco/point/results.csv")
    anchors = {
        CAVS_REPORT_NAME: PROTOCOL["cavs_seed_source"],
        **PROTOCOL["scalar_controls"],
    }
    rows = []
    for report_name, source in anchors.items():
        for seed in map(int, PROTOCOL["model_seeds"]):
            for budget in map(float, PROTOCOL["budgets"]):
                row = _source_row(
                    source_rows,
                    role="final",
                    method=str(source["source_method"]),
                    policy=str(source["source_policy"]),
                    mode="fixed_byte",
                    budget=budget,
                    seed=str(seed),
                )
                row.update(
                    {
                        "method": report_name,
                        "source_method": str(source["source_method"]),
                        "source_policy": str(source["source_policy"]),
                        "runtime_domain": PROTOCOL["runtime_domain"],
                        "rknn_artifacts_read": 0,
                    }
                )
                rows.append(row)
    return rows


def run_evaluate() -> dict[str, Any]:
    output = RUNS / "evaluation"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_train()
    ensemble_jobs = [
        {"method": method, "point_role": role, "seed": "ensemble"}
        for method in CORE.METHODS
        for role in ("fit_oof", "calibration", "final")
    ]
    seed_jobs = [
        {"method": method, "point_role": "final", "seed": int(seed)}
        for method in CORE.METHODS
        for seed in PROTOCOL["model_seeds"]
    ]
    workers = min(
        int(PROTOCOL["cpu"]["evaluation_workers"]),
        len(ensemble_jobs) + len(seed_jobs),
    )
    with mp.get_context("fork").Pool(workers) as pool:
        evaluated = list(pool.map(evaluate_new_job, [*ensemble_jobs, *seed_jobs], chunksize=1))
    ensemble_new = evaluated[: len(ensemble_jobs)]
    seed_new = evaluated[len(ensemble_jobs) :]
    anchor_rows, anchor_selections = imported_ensemble_anchors()
    rows = [row for result in ensemble_new for row in result["rows"]] + anchor_rows
    selections = {
        key: selection
        for result in ensemble_new
        for key, selection in result["selections"].items()
    }
    selections.update(anchor_selections)
    seed_rows = [row for result in seed_new for row in result["rows"]]
    seed_rows.extend(imported_seed_anchors())
    if len(rows) != 78:
        raise RuntimeError(f"ensemble point matrix incomplete: {len(rows)}")
    if len(seed_rows) != 45:
        raise RuntimeError(f"single-seed point matrix incomplete: {len(seed_rows)}")
    keys = [
        (row["role"], row["method"], row["mode"], float(row["budget"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate ensemble point rows")
    seed_keys = [
        (row["method"], row["seed"], float(row["budget"])) for row in seed_rows
    ]
    if len(seed_keys) != len(set(seed_keys)):
        raise RuntimeError("duplicate single-seed point rows")
    atomic_csv(output / "results.csv", rows)
    atomic_csv(output / "seed_results.csv", seed_rows)
    atomic_json(output / "selections.json", selections)
    report = {
        "complete": True,
        "ensemble_rows": len(rows),
        "single_seed_rows": len(seed_rows),
        "selections": len(selections),
        "workers": workers,
        "results_sha256": file_sha256(output / "results.csv"),
        "seed_results_sha256": file_sha256(output / "seed_results.csv"),
        "selections_sha256": file_sha256(output / "selections.json"),
        "runtime_domain": PROTOCOL["runtime_domain"],
        "rknn_artifacts_read": 0,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "ensemble_rows": len(rows)})
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
    token = hashlib.sha256(
        f"{PROTOCOL['bootstrap']['seed']}:{job['id']}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
    image_ids = _BOOT_STATE["image_ids"]
    for _ in values:
        rng.integers(0, len(image_ids), size=len(image_ids))
    for replicate in range(len(values), replicates):
        sampled = [
            image_ids[int(index)]
            for index in rng.integers(0, len(image_ids), size=len(image_ids))
        ]
        scores = {}
        for method in (str(job["left"]), str(job["right"])):
            gt, prediction, _ = BOOTSTRAP_CORE._bootstrap_remap(
                _BOOT_STATE["ground_truth"],
                _BOOT_STATE["predictions"][float(job["budget"])][method],
                _BOOT_STATE["predictions"][float(job["budget"])][method],
                sampled,
            )
            scores[method] = float(BOOTSTRAP_CORE._in_memory_ap(gt, prediction))
        values.append(scores[str(job["left"])] - scores[str(job["right"])])
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
        "runtime_domain": PROTOCOL["runtime_domain"],
        "rknn_artifacts_read": 0,
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
    run_evaluate()
    stored = read_json(RUNS / "evaluation/selections.json")
    data = paper_data("final")
    methods = {
        CAVS_REPORT_NAME,
        *CORE.METHODS,
        *PROTOCOL["scalar_controls"],
    }
    predictions: dict[float, dict[str, list[dict[str, Any]]]] = {}
    for budget in map(float, PROTOCOL["budgets"]):
        predictions[budget] = {}
        for method in methods:
            key = f"final|{method}|fixed_byte|{budget:g}"
            selection = {
                int(image_id): str(action) for image_id, action in stored[key].items()
            }
            _, values = evaluate_selection(
                "final", method, "fixed_byte", budget, selection
            )
            predictions[budget][method] = values
    jobs = CORE.bootstrap_jobs(PROTOCOL)
    if len(jobs) != 15:
        raise RuntimeError(f"bootstrap job matrix incomplete: {len(jobs)}")
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        ground_truth=read_json(
            FACTORIAL.SOURCE_ROOT / "runs/common/coco/final/gt.json"
        ),
        image_ids=[int(sample["image_id"]) for sample in data["samples"]],
        predictions=predictions,
    )
    workers = min(int(PROTOCOL["cpu"]["bootstrap_workers"]), len(jobs))
    with mp.get_context("fork").Pool(workers) as pool:
        results = list(pool.imap_unordered(bootstrap_job, jobs, chunksize=1))
    _BOOT_STATE.clear()
    results.sort(key=lambda row: (float(row["budget"]), str(row["id"])))
    report = {
        "complete": True,
        "jobs": len(results),
        "replicates": int(PROTOCOL["bootstrap"]["replicates"]),
        "comparisons": results,
        "workers": workers,
        "runtime_domain": PROTOCOL["runtime_domain"],
        "rknn_artifacts_read": 0,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(results)})
    return report


def run_report() -> dict[str, Any]:
    output = RUNS / "report"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    bootstrap = run_bootstrap()
    rows = read_csv(RUNS / "evaluation/results.csv")
    seed_rows = read_csv(RUNS / "evaluation/seed_results.csv")
    final = {
        (row["method"], float(row["budget"])): float(row["official_ap50_95"])
        for row in rows
        if row["role"] == "final" and row["mode"] == "fixed_byte"
    }
    methods = [
        CAVS_REPORT_NAME,
        *CORE.METHODS,
        *PROTOCOL["scalar_controls"],
    ]
    lines = [
        "# COCO ORIC/MORIC Action-Conditioned OQR 论文实验",
        "",
        "> 仅使用 PyTorch 论文缓存域；本实验未读取 RKNN/RK3588 训练或部署产物。",
        "",
        "## Final Fixed-byte 主结果",
        "",
        "| 方法 | 10% | 25% | 40% |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            f"| {method} | {final[(method, 0.1)]:.6f} | "
            f"{final[(method, 0.25)]:.6f} | {final[(method, 0.4)]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 三种子单模型",
            "",
            "| 方法 | 预算 | 均值 | SD | Ensemble |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        for budget in map(float, PROTOCOL["budgets"]):
            values = np.asarray(
                [
                    float(row["official_ap50_95"])
                    for row in seed_rows
                    if row["method"] == method and abs(float(row["budget"]) - budget) < 1e-12
                ],
                dtype=np.float64,
            )
            if len(values) != 3:
                raise RuntimeError(f"single-seed rows incomplete: {method}/{budget}")
            lines.append(
                f"| {method} | {budget:g} | {values.mean():.6f} | "
                f"{values.std(ddof=1):.6f} | {final[(method, budget)]:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            "正值表示左侧方法更好。",
            "",
            "| 对比 | 预算 | 均值差 | 95% CI | P(>0) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap["comparisons"]:
        lines.append(
            f"| {row['left']} - {row['right']} | {float(row['budget']):g} | "
            f"{float(row['mean_delta']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
            f"{float(row['probability_gt_zero']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 实验边界",
            "",
            "- ORIC-AC/MORIC-AC 是本文动作条件适配，不是原论文官方方法。",
            "- CAVS 与 SharedAction controls 为哈希冻结的同协议 anchor。",
            "- final 未参与训练、CDF、早停、门控或方法选择。",
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "complete": True,
        "results_sha256": file_sha256(output / "RESULTS_ZH.md"),
        "runtime_domain": PROTOCOL["runtime_domain"],
        "rknn_artifacts_read": 0,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"]})
    atomic_json(
        RUNS / "COMPLETED.json",
        {
            "complete": True,
            "protocol_id": PROTOCOL["protocol_id"],
            "runtime_domain": PROTOCOL["runtime_domain"],
            "rknn_artifacts_read": 0,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "evaluate", "bootstrap", "report", "all"),
        default="all",
    )
    args = parser.parse_args()
    stages = {
        "preflight": run_preflight,
        "train": run_train,
        "evaluate": run_evaluate,
        "bootstrap": run_bootstrap,
        "report": run_report,
        "all": run_report,
    }
    result = stages[args.stage]()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
