#!/usr/bin/env python3
"""Retrain Oregon CurrentActionValue-OQR from scratch on Cloud-V4 fit data."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN_CORE = load_module("oregon_cloudv3_train_contract", ROOT / "scripts/train_core.py")
ADAPTER = load_module("oregon_cloudv3_training_adapter", ROOT / "scripts/training_adapter.py")
LEARNER_CORE = load_module("oregon_cloudv3_learner_contract", ROOT / "scripts/learner_core.py")
CANONICAL = load_module(
    "oregon_cloudv3_historical_rankers",
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/run_multiranker_benchmark.py",
)


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


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def group_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in read_json(path):
        output[int(raw["image_id"])].append(
            {
                "image_id": int(raw["image_id"]),
                "category_id": int(raw["category_id"]),
                "bbox": list(map(float, raw["bbox"])),
                "score": float(raw.get("score", 1.0)),
            }
        )
    return dict(output)


_ROLE_CACHE: dict[str, dict[str, Any]] = {}


def load_role(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    if role not in _ROLE_CACHE:
        root = RUNS / f"common/oregon/{role}"
        manifest = read_json(root / "source_manifest.json")
        if manifest["cloud_model_sha256"] != PROTOCOL["cloud_model"]["sha256"]:
            raise RuntimeError(f"wrong cloud cache for {role}")
        samples = read_json(root / "samples.json")
        gt = read_json(root / "gt.json")
        category_ids = [int(row["id"]) for row in gt["categories"]]
        edge = group_predictions(Path(manifest["action_sources"]["edge"]))
        records = TRAIN_CORE.build_action_records(
            samples, edge, category_ids=category_ids
        )
        _ROLE_CACHE[role] = {
            "root": root,
            "manifest": manifest,
            "samples": samples,
            "records": records,
            "category_ids": category_ids,
            "edge": edge,
        }
    return _ROLE_CACHE[role]


def load_rewards(role: str) -> tuple[dict[int, dict[str, float]], dict[int, Any]]:
    if role not in {"fit", "calibration"}:
        raise ValueError("final rewards are forbidden")
    root = RUNS / f"rewards/{role}"
    return load_pickle(root / "reward_map.pkl"), load_pickle(root / "metric_map.pkl")


def fit_folds() -> list[list[int]]:
    values = read_json(RUNS / "rewards/fit/folds.json")["folds"]
    folds = [list(map(int, fold)) for fold in values]
    expected = {int(sample["image_id"]) for sample in load_role("fit")["samples"]}
    observed = [image_id for fold in folds for image_id in fold]
    if len(folds) != int(PROTOCOL["fold_count"]) or set(observed) != expected:
        raise RuntimeError("fit event folds changed")
    if len(observed) != len(set(observed)):
        raise RuntimeError("fit event folds overlap")
    return folds


def _job_root(job: Mapping[str, Any]) -> Path:
    fold = "full" if job["fold"] == "full" else f"fold_{int(job['fold'])}"
    safe = str(job["method"]).replace("/", "_")
    return RUNS / f"learners/jobs/{safe}/seed_{int(job['seed'])}/{fold}"


_STATE: dict[str, Any] = {}


def _subset_rows(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> list[dict[str, Any]]:
    return [dict(records[index]) for index in indices]


def _reward_targets(
    method: str,
    indices: Sequence[int],
    image_ids: Sequence[int],
    metric_map: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> np.ndarray:
    metric = "ap50" if str(method).endswith("AP50") else "ap50_95"
    return np.asarray(
        [
            float(metric_map[int(image_ids[index])]["original"][metric])
            for index in indices
        ],
        dtype=np.float64,
    )


def _train_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    root = _job_root(job)
    completion = root / "completion.json"
    report_path = root / "report.json"
    if completion.is_file():
        report = read_json(report_path)
        for path_key, hash_key in (
            ("artifact.pkl", "artifact_sha256"),
            ("scores.npz", "score_sha256"),
        ):
            if report[hash_key] != file_sha256(root / path_key):
                raise RuntimeError(f"learner job artifact drift: {job['id']}")
        return report

    method = str(job["method"])
    action_ids = TRAIN_CORE.action_space(
        str(job["action_space"]), _STATE["all_actions"]
    )
    fit_records = TRAIN_CORE.filter_records(_STATE["fit_records"], action_ids)
    fit_rewards = TRAIN_CORE.reward_matrix(fit_records, _STATE["fit_reward_map"])
    fit_ids = [int(record["image_id"]) for record in fit_records]
    if job["fold"] == "full":
        train_indices = list(range(len(fit_records)))
        predictions = (("calibration", list(range(len(_STATE["cal_records"])))), ("final", list(range(len(_STATE["final_records"])))))
    else:
        heldout_ids = set(_STATE["folds"][int(job["fold"])])
        heldout_indices = [index for index, image_id in enumerate(fit_ids) if image_id in heldout_ids]
        train_indices = [index for index, image_id in enumerate(fit_ids) if image_id not in heldout_ids]
        if len(heldout_indices) != len(heldout_ids):
            raise RuntimeError(f"OOF population drift: {job['id']}")
        predictions = (("fit_oof", heldout_indices),)
    started = time.time()
    if str(job["trainer"]) == "external":
        train_features = _STATE["fit_external_features"][train_indices]
        targets = _reward_targets(
            method,
            train_indices,
            fit_ids,
            _STATE["fit_metric_map"],
        )
        artifact = ADAPTER.fit_external(
            method,
            train_features,
            targets,
            PROTOCOL,
            seed=int(job["seed"]),
        )
    else:
        artifact = ADAPTER.fit_model(
            str(job["trainer"]),
            _subset_rows(fit_records, train_indices),
            fit_rewards[train_indices],
            PROTOCOL,
            category_count=len(_STATE["category_ids"]),
            seed=int(job["seed"]) + (0 if job["fold"] == "full" else int(job["fold"]) * 10_007),
        )
    arrays: dict[str, np.ndarray] = {}
    ids: dict[str, np.ndarray] = {}
    for role, indices in predictions:
        if role == "fit_oof":
            role_records = _subset_rows(fit_records, indices)
            role_features = _STATE["fit_external_features"][indices]
            role_ids = [fit_ids[index] for index in indices]
        else:
            source_records = _STATE[f"{role}_records"]
            role_records = TRAIN_CORE.filter_records(source_records, action_ids)
            role_features = _STATE[f"{role}_external_features"]
            role_ids = [int(record["image_id"]) for record in role_records]
        if str(job["trainer"]) == "external":
            values = ADAPTER.predict_external(artifact, role_features)
        else:
            values = ADAPTER.predict_model(
                artifact,
                role_records,
                PROTOCOL,
                category_count=len(_STATE["category_ids"]),
            )
        arrays[f"scores_{role}"] = np.asarray(values, dtype=np.float64)
        ids[f"image_ids_{role}"] = np.asarray(role_ids, dtype=np.int64)
    artifact_path = root / "artifact.pkl"
    score_path = root / "scores.npz"
    atomic_pickle(
        artifact_path,
        {
            "method": method,
            "action_space": str(job["action_space"]),
            "trainer": str(job["trainer"]),
            "seed": int(job["seed"]),
            "fold": job["fold"],
            "category_ids": list(map(int, _STATE["category_ids"])),
            "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
            "fit_identity_digest": _STATE["fit_identity_digest"],
            "artifact": artifact,
        },
    )
    atomic_npz(score_path, **ids, **arrays)
    metadata = artifact.get("metadata", artifact.get("checkpoint", {}))
    report = {
        "complete": True,
        "id": str(job["id"]),
        "method": method,
        "action_space": str(job["action_space"]),
        "trainer": str(job["trainer"]),
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "fit_images": len(train_indices),
        "prediction_roles": [role for role, _ in predictions],
        "selected_epoch": int(metadata.get("selected_epoch", -1)),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": file_sha256(artifact_path),
        "score_sha256": file_sha256(score_path),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "old_learner_checkpoint_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "artifact_sha256": report["artifact_sha256"],
            "score_sha256": report["score_sha256"],
        },
    )
    print(f"[learner] {job['id']} complete in {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def run_preflight() -> dict[str, Any]:
    output = RUNS / "learners/preflight"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(output / "report.json")
    required = [
        RUNS / "common/completion.json",
        RUNS / "rewards/completion.json",
        RUNS / "rewards/fit/reward_map.pkl",
        RUNS / "rewards/fit/metric_map.pkl",
        RUNS / "rewards/calibration/reward_map.pkl",
        RUNS / "rewards/calibration/metric_map.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    roles = {role: load_role(role) for role in PROTOCOL["roles"]}
    if any(len(data["samples"]) != int(PROTOCOL["roles"][role]) for role, data in roles.items()):
        raise RuntimeError("canonical Oregon role count drift")
    identities = {
        role: {str(sample["identity"]) for sample in data["samples"]}
        for role, data in roles.items()
    }
    overlaps = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(PROTOCOL["roles"])
        for right in list(PROTOCOL["roles"])[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"identity leakage: {overlaps}")
    folds = fit_folds()
    source_hashes = {
        str(path.resolve()): file_sha256(path)
        for path in (
            ROOT / "configs/protocol.json",
            ROOT / "scripts/train_core.py",
            ROOT / "scripts/learner_core.py",
            ROOT / "scripts/training_adapter.py",
            SCRIPT,
            *required,
        )
    }
    report = {
        "complete": True,
        "role_counts": {role: len(data["samples"]) for role, data in roles.items()},
        "identity_overlap": overlaps,
        "fold_sizes": list(map(len, folds)),
        "training_jobs": len(TRAIN_CORE.training_jobs(PROTOCOL)),
        "source_hashes": source_hashes,
        "source_digest": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


def _initialize_state() -> None:
    fit = load_role("fit")
    calibration = load_role("calibration")
    final = load_role("final")
    fit_rewards, fit_metrics = load_rewards("fit")
    category_ids = fit["category_ids"]
    if calibration["category_ids"] != category_ids or final["category_ids"] != category_ids:
        raise RuntimeError("category schema drift across roles")
    _STATE.clear()
    _STATE.update(
        fit_records=fit["records"],
        calibration_records=calibration["records"],
        cal_records=calibration["records"],
        final_records=final["records"],
        fit_reward_map=fit_rewards,
        fit_metric_map=fit_metrics,
        category_ids=category_ids,
        all_actions=["edge", "original", "q90", *PROTOCOL["roi_actions"]],
        folds=fit_folds(),
        fit_identity_digest=hashlib.sha256(
            "\n".join(str(sample["identity"]) for sample in fit["samples"]).encode()
        ).hexdigest(),
    )
    for role, data in (("fit", fit), ("calibration", calibration), ("final", final)):
        _STATE[f"{role}_external_features"] = ADAPTER.external_features(
            data["records"], category_ids, PROTOCOL
        )


def run_train() -> dict[str, Any]:
    completion = RUNS / "learners/completion.json"
    if completion.is_file():
        return read_json(RUNS / "learners/report.json")
    run_preflight()
    _initialize_state()
    jobs = TRAIN_CORE.training_jobs(PROTOCOL)
    workers = min(int(PROTOCOL["cpu"]["maximum_workers"]), len(jobs), max(1, (os.cpu_count() or 1) - 4))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(_train_job, jobs, chunksize=1))
    if len(reports) != len(jobs):
        raise RuntimeError("learner job matrix incomplete")
    report = {
        "complete": True,
        "jobs": len(jobs),
        "completed_jobs": len(reports),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "learners/report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(jobs)})
    _STATE.clear()
    return report


def run_historical() -> dict[str, Any]:
    output = RUNS / "historical"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(output / "report.json")
    run_preflight()
    CANONICAL.COMMON = RUNS / "common"
    fit = CANONICAL.load_role("oregon", "fit")
    calibration = CANONICAL.load_role("oregon", "calibration")
    final = CANONICAL.load_role("oregon", "final")
    config = {"ranker_hyperparameters": dict(PROTOCOL["ranker_hyperparameters"])}
    models = CANONICAL.fit_rankers(fit, config)
    targets = np.asarray([sample["target"] for sample in fit["samples"]], dtype=np.float64)
    pca = np.asarray([sample["pca32"] for sample in fit["samples"]], dtype=np.float64)
    hp = config["ranker_hyperparameters"]
    controlled = CANONICAL.fit_pairwise_ranker(
        pca,
        targets,
        epochs=int(hp["semantic_epochs"]),
        learning_rate=float(hp["semantic_learning_rate"]),
        ridge=float(hp["semantic_l2"]),
        max_pairs=int(hp["semantic_max_pairs"]),
        seed=int(hp["train_seed"]),
    )
    scores = {}
    for role, data in (("calibration", calibration), ("final", final)):
        values = CANONICAL.ranker_scores(models, data)
        values["Current-Controlled-32D"] = CANONICAL.predict_pairwise_ranker(
            controlled,
            np.asarray([sample["pca32"] for sample in data["samples"]], dtype=np.float64),
        )
        scores[role] = {
            method: np.asarray(score, dtype=np.float64)
            for method, score in values.items()
        }
    artifact = {
        "models": models,
        "controlled": controlled,
        "scores": scores,
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "fit_images": len(fit["samples"]),
        "old_learner_checkpoints_reused": False,
    }
    artifact_path = output / "artifact.pkl"
    atomic_pickle(artifact_path, artifact)
    report = {
        "complete": True,
        "fit_images": len(fit["samples"]),
        "methods": sorted(scores["final"]),
        "artifact_sha256": file_sha256(artifact_path),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "artifact_sha256": report["artifact_sha256"]})
    return report


def run_all() -> dict[str, Any]:
    return {"learners": run_train()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "historical", "train", "all"), default="all")
    args = parser.parse_args()
    result = {
        "preflight": run_preflight,
        "historical": run_historical,
        "train": run_train,
        "all": run_all,
    }[args.stage]()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
