#!/usr/bin/env python3
"""Retrain and evaluate deployment-aligned CAVS-OQ for one RKNN backend."""

from __future__ import annotations

import argparse
from collections import Counter
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

import backend_runtime
import contracts


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_sources(backend: str) -> dict[str, Any]:
    source_root = Path(PROTOCOL["source_training_experiment"])
    token = backend.replace("-", "_")
    train = load_module(
        f"rknn_train_source_{token}", source_root / "scripts/train_learners.py"
    )
    evaluate = load_module(
        f"rknn_eval_source_{token}", source_root / "scripts/evaluate.py"
    )
    return {"train": train, "evaluate": evaluate, "canonical": evaluate.CANONICAL}


_ROLE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def load_role(backend: str, role: str, sources: Mapping[str, Any]) -> dict[str, Any]:
    key = (backend, role)
    if key not in _ROLE_CACHE:
        adapted = backend_runtime.adapt_protocol(PROTOCOL, backend)
        runs = backend_runtime.backend_runs(ROOT, backend)
        canonical = sources["canonical"]
        canonical.COMMON = runs / "common"
        data = canonical.load_role("coco", role)
        manifest = read_json(runs / f"common/coco/{role}/source_manifest.json")
        if manifest.get("backend") != backend:
            raise RuntimeError(f"cross-backend common bundle: {backend}/{role}")
        if manifest.get("rknn_model_sha256") != adapted["rknn_models"][backend]["sha256"]:
            raise RuntimeError(f"wrong RKNN model in common bundle: {backend}/{role}")
        if manifest.get("cloud_model_sha256") != adapted["cloud_model"]["sha256"]:
            raise RuntimeError(f"wrong cloud model in common bundle: {backend}/{role}")
        if len(data["samples"]) != int(adapted["roles"][role]):
            raise RuntimeError(f"COCO role population changed: {backend}/{role}")
        gt = read_json(runs / f"common/coco/{role}/gt.json")
        category_ids = [int(row["id"]) for row in gt["categories"]]
        if len(category_ids) != 80 or len(category_ids) != len(set(category_ids)):
            raise RuntimeError("COCO category schema changed")
        records = sources["train"].TRAIN_CORE.build_action_records(
            data["samples"], data["actions"]["edge"], category_ids=category_ids
        )
        observed = {
            str(action["action_id"])
            for record in records
            for action in record["actions"]
        }
        missing = set(adapted["action_space"]) - observed
        if missing:
            raise RuntimeError(f"required deployment action is absent: {sorted(missing)}")
        _ROLE_CACHE[key] = {
            **data,
            "records": records,
            "category_ids": category_ids,
            "manifest": manifest,
        }
    return _ROLE_CACHE[key]


def fit_folds(backend: str, sources: Mapping[str, Any]) -> list[list[int]]:
    runs = backend_runtime.backend_runs(ROOT, backend)
    folds = [
        list(map(int, fold))
        for fold in read_json(runs / "rewards/fit/folds.json")["folds"]
    ]
    expected = {
        int(sample["image_id"])
        for sample in load_role(backend, "fit", sources)["samples"]
    }
    observed = [image_id for fold in folds for image_id in fold]
    if len(folds) != int(PROTOCOL["fold_count"]):
        raise RuntimeError("OOF fold count changed")
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise RuntimeError("OOF fold population changed")
    return folds


def job_root(backend: str, job: Mapping[str, Any]) -> Path:
    fold = "full" if job["fold"] == "full" else f"fold_{int(job['fold'])}"
    return backend_runtime.backend_runs(ROOT, backend) / (
        f"learners/jobs/seed_{int(job['seed'])}/{fold}"
    )


_STATE: dict[str, Any] = {}
_SOURCES: dict[str, Any] = {}
_ADAPTED: dict[str, Any] = {}


def train_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(int(_ADAPTED["cpu"]["torch_threads_per_worker"]))
    backend = str(job["backend"])
    root = job_root(backend, job)
    completion = root / "completion.json"
    report_path = root / "report.json"
    if completion.is_file():
        report = read_json(report_path)
        if report["artifact_sha256"] != file_sha256(root / "artifact.pkl"):
            raise RuntimeError(f"artifact drift: {job['id']}")
        if report["scores_sha256"] != file_sha256(root / "scores.npz"):
            raise RuntimeError(f"score drift: {job['id']}")
        return report

    train_core = _SOURCES["train"].TRAIN_CORE
    adapter = _SOURCES["train"].ADAPTER
    fit_records = train_core.filter_records(_STATE["fit_records"], job["action_space"])
    rewards = train_core.reward_matrix(fit_records, _STATE["fit_reward_map"])
    ids = [int(record["image_id"]) for record in fit_records]
    if job["fold"] == "full":
        train_indices = list(range(len(ids)))
        predictions = {
            role: train_core.filter_records(_STATE[f"{role}_records"], job["action_space"])
            for role in ("calibration", "final")
        }
        model_seed = int(job["seed"])
    else:
        heldout_ids = set(_STATE["folds"][int(job["fold"])])
        heldout = [index for index, image_id in enumerate(ids) if image_id in heldout_ids]
        train_indices = [index for index, image_id in enumerate(ids) if image_id not in heldout_ids]
        if len(heldout) != len(heldout_ids):
            raise RuntimeError(f"OOF population drift: {job['id']}")
        predictions = {"fit_oof": [fit_records[index] for index in heldout]}
        model_seed = int(job["seed"]) + int(job["fold"]) * 10_007

    started = time.time()
    artifact = adapter.fit_model(
        "current_action",
        [fit_records[index] for index in train_indices],
        rewards[train_indices],
        _ADAPTED,
        category_count=len(_STATE["category_ids"]),
        seed=model_seed,
    )
    arrays: dict[str, np.ndarray] = {}
    for role, records in predictions.items():
        values = adapter.predict_model(
            artifact,
            records,
            _ADAPTED,
            category_count=len(_STATE["category_ids"]),
        )
        arrays[f"scores_{role}"] = np.asarray(values, dtype=np.float64)
        arrays[f"image_ids_{role}"] = np.asarray(
            [int(record["image_id"]) for record in records], dtype=np.int64
        )
    artifact_path = root / "artifact.pkl"
    scores_path = root / "scores.npz"
    atomic_pickle(
        artifact_path,
        {
            "backend": backend,
            "method": job["method"],
            "action_space": list(job["action_space"]),
            "trainer": "current_action",
            "seed": int(job["seed"]),
            "fold": job["fold"],
            "category_ids": _STATE["category_ids"],
            "rknn_model_sha256": _ADAPTED["rknn_models"][backend]["sha256"],
            "cloud_model_sha256": _ADAPTED["cloud_model"]["sha256"],
            "artifact": artifact,
        },
    )
    atomic_npz(scores_path, **arrays)
    metadata = artifact.get("metadata", {})
    report = {
        "complete": True,
        "id": job["id"],
        "backend": backend,
        "method": job["method"],
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "fit_images": len(train_indices),
        "selected_epoch": int(metadata.get("selected_epoch", -1)),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": file_sha256(artifact_path),
        "scores_sha256": file_sha256(scores_path),
        "old_learner_checkpoint_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
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


def run_preflight(backend: str, sources: Mapping[str, Any]) -> dict[str, Any]:
    runs = backend_runtime.backend_runs(ROOT, backend)
    output = runs / "learners/preflight"
    report_path = output / "report.json"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(report_path)
    required = [
        runs / "common/completion.json",
        runs / "rewards/completion.json",
        runs / "rewards/fit/reward_map.pkl",
        runs / "rewards/fit/folds.json",
        runs / "rewards/calibration/reward_map.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    roles = {role: load_role(backend, role, sources) for role in PROTOCOL["roles"]}
    identities = {
        role: {str(sample["identity"]) for sample in data["samples"]}
        for role, data in roles.items()
    }
    names = list(identities)
    overlaps = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"identity leakage: {overlaps}")
    folds = fit_folds(backend, sources)
    reused_action_hashes = {}
    source_current = Path(PROTOCOL["source_current_experiment"])
    for role, data in roles.items():
        source = read_json(source_current / f"runs/common/coco/{role}/source_manifest.json")
        hashes = backend_runtime.validate_reused_action_sources(
            source, ("original", "q90")
        )
        for action_id in hashes:
            if Path(data["manifest"]["action_sources"][action_id]).resolve() != Path(
                source["action_sources"][action_id]
            ).resolve():
                raise RuntimeError(
                    f"reused action path changed: {backend}/{role}/{action_id}"
                )
        reused_action_hashes[role] = hashes
    source_paths = [
        ROOT / "configs/protocol.json",
        ROOT / "scripts/backend_runtime.py",
        SCRIPT,
        Path(PROTOCOL["source_training_experiment"]) / "scripts/train_learners.py",
        Path(PROTOCOL["source_training_experiment"]) / "scripts/training_adapter.py",
        *required,
    ]
    hashes = {str(path.resolve()): file_sha256(path) for path in source_paths}
    report = {
        "complete": True,
        "backend": backend,
        "role_counts": {role: len(data["samples"]) for role, data in roles.items()},
        "identity_overlap": overlaps,
        "fold_sizes": list(map(len, folds)),
        "training_jobs": len(backend_runtime.training_jobs(PROTOCOL, backend)),
        "reused_action_sha256": reused_action_hashes,
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


def initialize_training(backend: str, sources: Mapping[str, Any]) -> None:
    runs = backend_runtime.backend_runs(ROOT, backend)
    fit = load_role(backend, "fit", sources)
    _STATE.clear()
    _STATE.update(
        fit_records=fit["records"],
        fit_reward_map=load_pickle(runs / "rewards/fit/reward_map.pkl"),
        calibration_records=load_role(backend, "calibration", sources)["records"],
        final_records=load_role(backend, "final", sources)["records"],
        folds=fit_folds(backend, sources),
        category_ids=list(map(int, fit["category_ids"])),
    )


def run_train(backend: str, sources: Mapping[str, Any]) -> dict[str, Any]:
    global _ADAPTED, _SOURCES
    runs = backend_runtime.backend_runs(ROOT, backend)
    completion = runs / "learners/completion.json"
    if completion.is_file():
        return read_json(runs / "learners/report.json")
    run_preflight(backend, sources)
    initialize_training(backend, sources)
    _ADAPTED = backend_runtime.adapt_protocol(PROTOCOL, backend)
    _SOURCES = dict(sources)
    jobs = backend_runtime.training_jobs(PROTOCOL, backend)
    workers = min(int(_ADAPTED["cpu"]["maximum_workers"]), len(jobs))
    started = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(train_job, jobs, chunksize=1))
    if len(reports) != len(jobs):
        raise RuntimeError("training matrix incomplete")
    report = {
        "complete": True,
        "backend": backend,
        "jobs": len(jobs),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(runs / "learners/report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(jobs)})
    _STATE.clear()
    return report


def full_seed_scores(
    backend: str, role: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    runs = backend_runtime.backend_runs(ROOT, backend)
    expected = [int(record["image_id"]) for record in records]
    output: dict[str, np.ndarray] = {}
    for seed in map(int, PROTOCOL["seeds"]):
        path = runs / f"learners/jobs/seed_{seed}/full/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            ids = list(map(int, artifact[f"image_ids_{role}"]))
            raw = np.asarray(artifact[f"scores_{role}"], dtype=np.float64)
        if ids != expected:
            raise RuntimeError(f"score/image alignment changed: {backend}/{role}/{seed}")
        values = np.asarray(
            backend_runtime.collapse_static_scores(raw, len(PROTOCOL["budgets"])),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite scores: {backend}/{role}/{seed}")
        output[f"seed_{seed}"] = values
    output["ensemble"] = np.mean(list(output.values()), axis=0)
    return output


def schedule_fixed_byte(
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    budget: float,
    sources: Mapping[str, Any],
) -> tuple[dict[int, str], int, float]:
    groups = []
    for sample in samples:
        image_id = int(sample["image_id"])
        actions = [
            {"action_id": "edge", "image_id": image_id, "value": 0.0, "cost": 0.0}
        ]
        for action_id, value in scores[image_id].items():
            if not sample["available"].get(action_id, False):
                continue
            actions.append(
                {
                    "action_id": action_id,
                    "image_id": image_id,
                    "value": float(value),
                    "cost": float(sample["payload_bytes"][action_id]),
                }
            )
        groups.append(actions)
    capacity = float(budget) * sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    selected = sources["evaluate"].select_actions_for_budget(groups, capacity)["selected"]
    selection = {
        int(row["image_id"]): str(row["action_id"])
        for row in selected
        if str(row["action_id"]) != "edge"
    }
    by_id = {int(sample["image_id"]): sample for sample in samples}
    used = sum(
        int(by_id[image_id]["payload_bytes"][action])
        for image_id, action in selection.items()
    )
    if used > capacity + 1e-6:
        raise RuntimeError("fixed-byte capacity exceeded")
    return selection, used, capacity


def evaluate_selection(
    backend: str,
    role: str,
    variant: str,
    budget: float,
    selection: Mapping[int, str],
    request_bytes: int,
    capacity: float,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    data = load_role(backend, role, sources)
    predictions, _ = sources["canonical"].mixed_predictions(data, selection, 0.001)
    metric = data["evaluator"].evaluate(predictions)
    counts = Counter(selection.values())
    return {
        "backend": backend,
        "role": role,
        "method": f"CAVS-OQ-DeploymentAligned-{backend}",
        "model_variant": variant,
        "fusion": "none",
        "mode": "fixed_byte",
        "budget": float(budget),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "request_bytes": int(request_bytes),
        "capacity_bytes": float(capacity),
        "capacity_utilization": request_bytes / max(1.0, capacity),
        "kb_per_image": request_bytes / 1024 / len(data["samples"]),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["current@640"]),
        "old_learner_checkpoint_reused": False,
        "final_used_for_training_or_selection": False,
    }


def run_evaluate(backend: str, sources: Mapping[str, Any]) -> dict[str, Any]:
    runs = backend_runtime.backend_runs(ROOT, backend)
    completion = runs / "evaluation/completion.json"
    if completion.is_file():
        return read_json(runs / "evaluation/report.json")
    run_train(backend, sources)
    rows = []
    selections: dict[str, dict[str, str]] = {}
    for role in ("calibration", "final"):
        data = load_role(backend, role, sources)
        train_core = sources["train"].TRAIN_CORE
        records = train_core.filter_records(data["records"], backend_runtime.ACTION_SPACE)
        variants = full_seed_scores(backend, role, records)
        for variant, values in variants.items():
            scores = backend_runtime.score_map(records, values)
            for budget in map(float, PROTOCOL["budgets"]):
                selection, used, capacity = schedule_fixed_byte(
                    data["samples"], scores, budget, sources
                )
                key = f"{role}|{variant}|fixed_byte|{budget:g}"
                selections[key] = {str(image_id): action for image_id, action in selection.items()}
                rows.append(
                    evaluate_selection(
                        backend,
                        role,
                        variant,
                        budget,
                        selection,
                        used,
                        capacity,
                        sources,
                    )
                )
    expected = 2 * (len(PROTOCOL["seeds"]) + 1) * len(PROTOCOL["budgets"])
    if len(rows) != expected:
        raise RuntimeError(f"point-result matrix incomplete: {len(rows)}/{expected}")
    atomic_csv(runs / "evaluation/results.csv", rows)
    atomic_json(runs / "evaluation/selections.json", selections)
    final_single = [
        row
        for row in rows
        if row["role"] == "final" and str(row["model_variant"]).startswith("seed_")
    ]
    summary = {}
    for budget in map(float, PROTOCOL["budgets"]):
        values = np.asarray(
            [
                float(row["official_ap50_95"])
                for row in final_single
                if float(row["budget"]) == budget
            ],
            dtype=np.float64,
        )
        ensemble = next(
            row
            for row in rows
            if row["role"] == "final"
            and row["model_variant"] == "ensemble"
            and float(row["budget"]) == budget
        )
        summary[f"{budget:g}"] = {
            "single_model_mean": float(values.mean()),
            "single_model_sd": float(values.std(ddof=1)),
            "ensemble": float(ensemble["official_ap50_95"]),
        }
    report = {
        "complete": True,
        "backend": backend,
        "rows": len(rows),
        "final_ap50_95": summary,
        "fusion": "none",
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(runs / "evaluation/report.json", report)
    atomic_json(completion, {"complete": True, "rows": len(rows)})
    return report


def run_stage(backend: str, stage: str) -> dict[str, Any]:
    sources = load_sources(backend)
    return {
        "preflight": lambda: run_preflight(backend, sources),
        "train": lambda: run_train(backend, sources),
        "evaluate": lambda: run_evaluate(backend, sources),
        "all": lambda: {
            "preflight": run_preflight(backend, sources),
            "train": run_train(backend, sources),
            "evaluate": run_evaluate(backend, sources),
        },
    }[stage]()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=contracts.BACKENDS, required=True)
    parser.add_argument(
        "--stage", choices=("preflight", "train", "evaluate", "all"), default="all"
    )
    args = parser.parse_args()
    print(json.dumps(run_stage(args.backend, args.stage), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
