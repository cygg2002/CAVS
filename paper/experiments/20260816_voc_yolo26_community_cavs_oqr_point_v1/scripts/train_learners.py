#!/usr/bin/env python3
"""Retrain all VOC CAVS and ORIC/MORIC learners from scratch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_point_train_data", ROOT / "scripts/data_core.py")
CORE = load_module("voc_point_train_protocol", ROOT / "scripts/protocol_core.py")
ADAPTER = load_module("voc_point_train_adapter", ROOT / "scripts/training_adapter.py")


def _role(role: str) -> dict[str, Any]:
    root = RUNS / f"common/{role}"
    gt = DATA.read_json(RUNS / f"data/roles/{role}/gt.json")
    return {
        "samples": DATA.read_json(root / "samples.json"),
        "records": DATA.load_pickle(root / "records.pkl"),
        "category_ids": [int(row["id"]) for row in gt["categories"]],
    }


def _job_root(job: Mapping[str, Any]) -> Path:
    return RUNS / f"learners/jobs/{job['method']}/seed_{int(job['seed'])}"


def _reward_matrix(
    records: list[dict[str, Any]], reward_map: Mapping[int, Mapping[str, float]]
) -> np.ndarray:
    values = np.zeros((len(records), int(PROTOCOL["model"]["max_actions"])), dtype=np.float32)
    for row, record in enumerate(records):
        image_id = int(record["image_id"])
        for column, action in enumerate(record["actions"]):
            action_id = str(action["action_id"])
            if action_id != "edge":
                values[row, column] = float(reward_map[image_id][action_id])
    return values


def _external_target(method: str, image_ids: list[int]) -> np.ndarray:
    metric_map = DATA.load_pickle(RUNS / "rewards/fit/metric_map.pkl")
    metric = "ap50" if method.endswith("AP50") else "ap50_95"
    return np.asarray(
        [float(metric_map[image_id]["original"][metric]) for image_id in image_ids],
        dtype=np.float64,
    )


_STATE: dict[str, Any] = {}


def train_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(int(PROTOCOL["resource"]["torch_threads_per_worker"]))
    root = _job_root(job)
    completion = root / "completion.json"
    artifact_path = root / "artifact.pkl"
    scores_path = root / "scores.pkl"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report["artifact_sha256"] != DATA.file_sha256(artifact_path):
            raise RuntimeError(f"learner artifact changed: {job['id']}")
        if report["scores_sha256"] != DATA.file_sha256(scores_path):
            raise RuntimeError(f"learner scores changed: {job['id']}")
        return report
    started = time.time()
    method = str(job["method"])
    seed = int(job["seed"])
    fit = _STATE["fit"]
    image_ids = [int(record["image_id"]) for record in fit["records"]]
    if method == "CAVS-OQR-MAUR":
        artifact = ADAPTER.fit_model(
            "current_action",
            fit["records"],
            _STATE["fit_rewards"],
            PROTOCOL,
            category_count=len(fit["category_ids"]),
            seed=seed,
        )
    else:
        artifact = ADAPTER.fit_external(
            method,
            _STATE["external_features"]["fit"],
            _external_target(method, image_ids),
            PROTOCOL,
            seed=seed,
        )
    scores: dict[str, Any] = {"method": method, "seed": seed}
    for role in ("calibration", "final"):
        data = _STATE[role]
        role_ids = [int(record["image_id"]) for record in data["records"]]
        if method == "CAVS-OQR-MAUR":
            values = ADAPTER.predict_model(
                artifact,
                data["records"],
                PROTOCOL,
                category_count=len(data["category_ids"]),
            )
        else:
            values = ADAPTER.predict_external(
                artifact, _STATE["external_features"][role]
            )
        scores[role] = {
            "image_ids": role_ids,
            "values": np.asarray(values, dtype=np.float64),
        }
    DATA.atomic_pickle(
        artifact_path,
        {
            "method": method,
            "seed": seed,
            "fit_images": len(fit["records"]),
            "cloud_model_sha256": PROTOCOL["checkpoints"]["cloud"]["sha256"],
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
        "fit_images": len(fit["records"]),
        "prediction_roles": ["calibration", "final"],
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": DATA.file_sha256(artifact_path),
        "scores_sha256": DATA.file_sha256(scores_path),
        "old_learner_checkpoint_reused": False,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(root / "report.json", report)
    DATA.atomic_json(completion, report)
    print(f"[learner] {job['id']} complete in {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def run_all() -> dict[str, Any]:
    output = RUNS / "learners"
    completion = output / "completion.json"
    if completion.is_file():
        return DATA.read_json(output / "report.json")
    for required in (
        RUNS / "common/completion.json",
        RUNS / "rewards/completion.json",
    ):
        if not required.is_file():
            raise RuntimeError(f"required stage is incomplete: {required}")
    _STATE.clear()
    _STATE.update({role: _role(role) for role in PROTOCOL["roles"]})
    _STATE["fit_rewards"] = _reward_matrix(
        _STATE["fit"]["records"], DATA.load_pickle(RUNS / "rewards/fit/reward_map.pkl")
    )
    _STATE["external_features"] = {
        role: ADAPTER.external_features(
            _STATE[role]["records"], _STATE[role]["category_ids"], PROTOCOL
        )
        for role in PROTOCOL["roles"]
    }
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
    report = {
        "complete": True,
        "jobs": len(reports),
        "workers": workers,
        "methods": sorted({str(row["method"]) for row in reports}),
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
        job = next(job for job in CORE.training_jobs(PROTOCOL) if job["id"] == args.job)
        _STATE.update({role: _role(role) for role in PROTOCOL["roles"]})
        _STATE["fit_rewards"] = _reward_matrix(
            _STATE["fit"]["records"], DATA.load_pickle(RUNS / "rewards/fit/reward_map.pkl")
        )
        _STATE["external_features"] = {
            role: ADAPTER.external_features(
                _STATE[role]["records"], _STATE[role]["category_ids"], PROTOCOL
            )
            for role in PROTOCOL["roles"]
        }
        result = train_job(job)
    else:
        result = run_all()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
