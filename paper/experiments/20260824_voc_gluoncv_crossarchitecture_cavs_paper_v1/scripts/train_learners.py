#!/usr/bin/env python3
"""Retrain all CAVS-O/OQ and ORIC/MORIC learners under the frozen VOC split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("voc_gluoncv_train_core", ROOT / "scripts/core.py")
DATA = load_module("voc_gluoncv_train_data", ROOT / "scripts/data_core.py")


def _adapter() -> Any:
    if not hasattr(_adapter, "value"):
        setattr(
            _adapter,
            "value",
            load_module(
                "voc_gluoncv_train_adapter", ROOT / "scripts/training_adapter.py"
            ),
        )
    return getattr(_adapter, "value")


def fold_train_holdout_indices(
    identities: Sequence[str], folds: Sequence[Sequence[str]], fold: int | str
) -> tuple[list[int], list[int]]:
    if fold == "full":
        return list(range(len(identities))), []
    index = int(fold)
    if not 0 <= index < len(folds):
        raise ValueError(fold)
    held = set(map(str, folds[index]))
    holdout = [position for position, identity in enumerate(identities) if str(identity) in held]
    train = [position for position, identity in enumerate(identities) if str(identity) not in held]
    if len(train) + len(holdout) != len(identities) or not holdout:
        raise RuntimeError("invalid OOF train/holdout split")
    return train, holdout


def _role(role: str, action_space: str) -> dict[str, Any]:
    common = RUNS / f"common/{role}"
    gt = DATA.read_json(RUNS / f"data/roles/{role}/gt.json")
    return {
        "samples": DATA.read_json(common / "samples.json"),
        "records": DATA.load_pickle(common / f"records_{action_space}.pkl"),
        "category_ids": [int(row["id"]) for row in gt["categories"]],
    }


def _reward_matrix(
    records: Sequence[Mapping[str, Any]], reward_map: Mapping[int, Mapping[str, float]]
) -> np.ndarray:
    values = np.zeros(
        (len(records), int(PROTOCOL["model"]["max_actions"])), dtype=np.float32
    )
    for row, record in enumerate(records):
        image_id = int(record["image_id"])
        for column, action in enumerate(record["actions"]):
            action_id = str(action["action_id"])
            if action_id != "edge":
                values[row, column] = float(reward_map[image_id][action_id])
    return values


def _fit_folds(samples: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    return CORE.deterministic_folds(
        [str(row["identity"]) for row in samples],
        fold_count=int(PROTOCOL["fold_count"]),
        seed=int(PROTOCOL["fold_seed"]),
    )


def _job_root(job: Mapping[str, Any]) -> Path:
    return (
        RUNS
        / "learners/jobs"
        / str(job["method"])
        / f"seed_{int(job['seed'])}"
        / f"fold_{job['fold']}"
    )


def _identity_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


_STATE: dict[str, Any] = {}


def _external_target(method: str, image_ids: Sequence[int]) -> np.ndarray:
    metric_map = _STATE["metric_map"]
    metric = "ap50" if str(method).endswith("AP50") else "ap50_95"
    return np.asarray(
        [float(metric_map[int(image_id)]["original"][metric]) for image_id in image_ids],
        dtype=np.float64,
    )


def train_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(int(PROTOCOL["resource"]["torch_threads_per_worker"]))
    output = _job_root(job)
    completion = output / "completion.json"
    artifact_path = output / "artifact.pkl"
    scores_path = output / "scores.pkl"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("artifact_sha256") != DATA.file_sha256(artifact_path):
            raise RuntimeError(f"completed learner artifact changed: {job['id']}")
        if report.get("scores_sha256") != DATA.file_sha256(scores_path):
            raise RuntimeError(f"completed learner scores changed: {job['id']}")
        return report
    started = time.time()
    method = str(job["method"])
    seed = int(job["seed"])
    fold = job["fold"]
    adapter = _adapter()
    if method in CORE.CAVS_METHODS:
        action_space = "O" if method == "CAVS-O-Core" else "OQ"
        fit = _STATE["roles"][action_space]["fit"]
        identities = [str(row["identity"]) for row in fit["samples"]]
        train_indices, holdout_indices = fold_train_holdout_indices(
            identities, _STATE["fit_folds"], fold
        )
        records = [fit["records"][index] for index in train_indices]
        rewards = _STATE["rewards"][action_space][train_indices]
        artifact = adapter.fit_model(
            "current_action",
            records,
            rewards,
            PROTOCOL,
            category_count=len(fit["category_ids"]),
            seed=seed,
        )
        scores: dict[str, Any] = {
            "method": method,
            "seed": seed,
            "fold": fold,
            "action_space": action_space,
        }
        if fold == "full":
            for role in ("calibration", "final"):
                data = _STATE["roles"][action_space][role]
                scores[role] = {
                    "image_ids": [int(row["image_id"]) for row in data["records"]],
                    "values": np.asarray(
                        adapter.predict_model(
                            artifact,
                            data["records"],
                            PROTOCOL,
                            category_count=len(data["category_ids"]),
                        ),
                        dtype=np.float64,
                    ),
                }
        else:
            holdout_records = [fit["records"][index] for index in holdout_indices]
            scores["fit_oof"] = {
                "image_ids": [int(row["image_id"]) for row in holdout_records],
                "values": np.asarray(
                    adapter.predict_model(
                        artifact,
                        holdout_records,
                        PROTOCOL,
                        category_count=len(fit["category_ids"]),
                    ),
                    dtype=np.float64,
                ),
            }
        trained_identities = [identities[index] for index in train_indices]
    else:
        fit = _STATE["roles"]["O"]["fit"]
        image_ids = [int(row["image_id"]) for row in fit["records"]]
        artifact = adapter.fit_external(
            method,
            _STATE["external_features"]["fit"],
            _external_target(method, image_ids),
            PROTOCOL,
            seed=seed,
        )
        scores = {"method": method, "seed": seed, "fold": "full", "action_space": "O"}
        for role in ("calibration", "final"):
            data = _STATE["roles"]["O"][role]
            scores[role] = {
                "image_ids": [int(row["image_id"]) for row in data["records"]],
                "values": np.asarray(
                    adapter.predict_external(artifact, _STATE["external_features"][role]),
                    dtype=np.float64,
                ),
            }
        trained_identities = [str(row["identity"]) for row in fit["samples"]]
    DATA.atomic_pickle(
        artifact_path,
        {
            "method": method,
            "seed": seed,
            "fold": fold,
            "train_images": len(trained_identities),
            "train_identity_sha256": _identity_hash(trained_identities),
            "old_learner_checkpoint_reused": False,
            "artifact": artifact,
        },
    )
    DATA.atomic_pickle(scores_path, scores)
    report = {
        "complete": True,
        "id": str(job["id"]),
        "method": method,
        "seed": seed,
        "fold": fold,
        "train_images": len(trained_identities),
        "prediction_roles": sorted(
            scores.keys() - {"method", "seed", "fold", "action_space"}
        ),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": DATA.file_sha256(artifact_path),
        "scores_sha256": DATA.file_sha256(scores_path),
        "old_learner_checkpoint_reused": False,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    print(f"[learner] {job['id']} complete in {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def _initialize_state() -> None:
    _STATE.clear()
    _STATE["roles"] = {
        action_space: {
            role: _role(role, action_space) for role in PROTOCOL["roles"]
        }
        for action_space in ("O", "OQ")
    }
    fit_o = _STATE["roles"]["O"]["fit"]
    _STATE["fit_folds"] = _fit_folds(fit_o["samples"])
    _STATE["rewards"] = {
        action_space: _reward_matrix(
            _STATE["roles"][action_space]["fit"]["records"],
            DATA.load_pickle(RUNS / f"rewards/fit/reward_map_{action_space}.pkl"),
        )
        for action_space in ("O", "OQ")
    }
    _STATE["metric_map"] = DATA.load_pickle(RUNS / "rewards/fit/metric_map.pkl")
    adapter = _adapter()
    _STATE["external_features"] = {
        role: adapter.external_features(
            _STATE["roles"]["O"][role]["records"],
            _STATE["roles"]["O"][role]["category_ids"],
            PROTOCOL,
        )
        for role in PROTOCOL["roles"]
    }


def _assemble_oof() -> dict[str, Any]:
    output = RUNS / "learners/oof"
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    fit_samples = _STATE["roles"]["O"]["fit"]["samples"]
    expected = [int(row["image_id"]) for row in fit_samples]
    for method in CORE.CAVS_METHODS:
        for seed in map(int, PROTOCOL["model_seeds"]):
            rows = []
            for fold in range(int(PROTOCOL["fold_count"])):
                payload = DATA.load_pickle(
                    _job_root({"method": method, "seed": seed, "fold": fold}) / "scores.pkl"
                )["fit_oof"]
                for image_id, values in zip(payload["image_ids"], payload["values"].transpose(1, 0, 2)):
                    rows.append((int(image_id), np.asarray(values, dtype=np.float64)))
            if len(rows) != len(expected) or {row[0] for row in rows} != set(expected):
                raise RuntimeError(f"OOF score population changed: {method}/{seed}")
            by_id = {image_id: values for image_id, values in rows}
            path = output / f"{method}_seed_{seed}.pkl"
            DATA.atomic_pickle(
                path,
                {
                    "method": method,
                    "seed": seed,
                    "image_ids": expected,
                    "values": np.stack([by_id[image_id] for image_id in expected], axis=1),
                    "fold_count": int(PROTOCOL["fold_count"]),
                },
            )
            reports[f"{method}/{seed}"] = DATA.file_sha256(path)
    DATA.atomic_json(output / "report.json", {"complete": True, "files": reports})
    DATA.atomic_json(output / "completion.json", {"complete": True, "files": reports})
    return reports


def run_all() -> dict[str, Any]:
    output = RUNS / "learners"
    completion = output / "completion.json"
    if completion.is_file():
        return DATA.read_json(output / "report.json")
    for required in (RUNS / "common/completion.json", RUNS / "rewards/completion.json"):
        if not required.is_file():
            raise RuntimeError(f"required stage is incomplete: {required}")
    _initialize_state()
    jobs = CORE.training_jobs(PROTOCOL)
    workers = min(
        int(PROTOCOL["resource"]["training_workers"]),
        len(jobs),
        max(1, os.cpu_count() or 1),
    )
    with mp.get_context("fork").Pool(workers) as pool:
        reports = list(pool.imap_unordered(train_job, jobs, chunksize=1))
    if len(reports) != len(jobs):
        raise RuntimeError("learner job matrix incomplete")
    oof = _assemble_oof()
    report = {
        "complete": True,
        "jobs": len(reports),
        "workers": workers,
        "methods": sorted({str(row["method"]) for row in reports}),
        "oof_files": len(oof),
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.job:
        _initialize_state()
        job = next(row for row in CORE.training_jobs(PROTOCOL) if row["id"] == args.job)
        result = train_job(job)
    else:
        result = run_all()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
