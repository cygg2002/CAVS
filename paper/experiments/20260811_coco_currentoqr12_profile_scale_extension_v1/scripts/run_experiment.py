#!/usr/bin/env python3
"""Retrain and evaluate COCO CurrentActionValue single-R and R12 methods."""

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
SOURCE_ROOT = Path(PROTOCOL["source_training_experiment"])
HISTORICAL_ROOT = Path(PROTOCOL["source_current_experiment"])


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("coco_oqr12_experiment_core", ROOT / "scripts/core.py")
SOURCE_TRAIN = load_module(
    "coco_oqr12_training_source", SOURCE_ROOT / "scripts/train_learners.py"
)
SOURCE_EVAL = load_module(
    "coco_oqr12_evaluation_source", SOURCE_ROOT / "scripts/evaluate.py"
)
CANONICAL = SOURCE_EVAL.CANONICAL
BOOTSTRAP_CORE = load_module(
    "coco_oqr12_bootstrap_core",
    REPO
    / "paper/experiments/20260804_coco_dualhead_canonical9678_val2017_paper_v1/scripts/canonical_core.py",
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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


_ROLE_CACHE: dict[str, dict[str, Any]] = {}


def load_role(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    if role not in _ROLE_CACHE:
        CANONICAL.COMMON = RUNS / "common"
        data = CANONICAL.load_role("coco", role)
        manifest = read_json(RUNS / f"common/coco/{role}/source_manifest.json")
        if manifest.get("cloud_model_sha256") != PROTOCOL["cloud_model"]["sha256"]:
            raise RuntimeError(f"wrong cloud cache: {role}")
        if len(data["samples"]) != int(PROTOCOL["roles"][role]):
            raise RuntimeError(f"COCO role population drift: {role}")
        category_ids = [
            int(row["id"])
            for row in read_json(RUNS / f"common/coco/{role}/gt.json")["categories"]
        ]
        if len(category_ids) != 80 or len(category_ids) != len(set(category_ids)):
            raise RuntimeError("COCO category schema changed")
        records = SOURCE_TRAIN.TRAIN_CORE.build_action_records(
            data["samples"], data["actions"]["edge"], category_ids=category_ids
        )
        _ROLE_CACHE[role] = {
            **data,
            "records": records,
            "category_ids": category_ids,
            "manifest": manifest,
        }
    return _ROLE_CACHE[role]


def fit_folds() -> list[list[int]]:
    folds = [
        list(map(int, fold))
        for fold in read_json(RUNS / "rewards/fit/folds.json")["folds"]
    ]
    expected = {int(sample["image_id"]) for sample in load_role("fit")["samples"]}
    observed = [image_id for fold in folds for image_id in fold]
    if len(folds) != int(PROTOCOL["fold_count"]):
        raise RuntimeError("OOF fold count changed")
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise RuntimeError("OOF fold population changed")
    return folds


def _source_required() -> list[Path]:
    return [
        RUNS / "common/completion.json",
        RUNS / "rewards/completion.json",
        RUNS / "rewards/fit/reward_map.pkl",
        RUNS / "rewards/fit/folds.json",
        HISTORICAL_ROOT / "runs/calibration/evaluation/results.csv",
        HISTORICAL_ROOT / "runs/calibration/evaluation/selections.json",
        HISTORICAL_ROOT / "runs/final/evaluation/results.csv",
        HISTORICAL_ROOT / "runs/final/evaluation/selections.json",
    ]


def run_preflight() -> dict[str, Any]:
    report_path = RUNS / "preflight/learner_report.json"
    completion = RUNS / "preflight/learner_completion.json"
    if completion.is_file():
        return read_json(report_path)
    CORE.validate_protocol(PROTOCOL)
    required = _source_required()
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])
    roles = {role: load_role(role) for role in PROTOCOL["roles"]}
    expected_actions = set(PROTOCOL["action_spaces"]["OQR12"])
    for role, data in roles.items():
        first = {str(action["action_id"]) for action in data["records"][0]["actions"]}
        if not {"edge", "original", "q90"} <= first:
            raise RuntimeError(f"base action missing: {role}")
        observed_roi = {
            str(action["action_id"])
            for record in data["records"]
            for action in record["actions"]
            if "@" in str(action["action_id"])
        }
        if observed_roi != set(PROTOCOL["roi_actions"]):
            raise RuntimeError(f"R12 action population changed: {role}")
        if not first <= expected_actions:
            raise RuntimeError(f"forbidden action in first record: {role}")
    identities = {
        role: {str(sample["identity"]) for sample in data["samples"]}
        for role, data in roles.items()
    }
    role_names = list(identities)
    overlaps = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(role_names)
        for right in role_names[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"identity leakage: {overlaps}")
    folds = fit_folds()
    sources = [ROOT / "configs/protocol.json", ROOT / "scripts/core.py", SCRIPT, *required]
    hashes = {str(path.resolve()): file_sha256(path) for path in sources}
    report = {
        "complete": True,
        "role_counts": {role: len(data["samples"]) for role, data in roles.items()},
        "identity_overlap": overlaps,
        "fold_sizes": list(map(len, folds)),
        "training_jobs": len(CORE.training_jobs(PROTOCOL)),
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "historical_sources_read_only": True,
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


_TRAIN_STATE: dict[str, Any] = {}


def _filtered(
    records: Sequence[Mapping[str, Any]], actions: Sequence[str]
) -> list[dict[str, Any]]:
    return SOURCE_TRAIN.TRAIN_CORE.filter_records(records, actions)


def _initialize_training() -> None:
    fit = load_role("fit")
    reward_map = load_pickle(RUNS / "rewards/fit/reward_map.pkl")
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        fit_records=fit["records"],
        fit_reward_map=reward_map,
        fit_ids=[int(record["image_id"]) for record in fit["records"]],
        calibration_records=load_role("calibration")["records"],
        final_records=load_role("final")["records"],
        folds=fit_folds(),
        category_count=len(fit["category_ids"]),
        category_ids=list(map(int, fit["category_ids"])),
    )


def _job_root(job: Mapping[str, Any]) -> Path:
    method = str(job["method"]).replace("/", "_")
    fold = "full" if job["fold"] == "full" else f"fold_{int(job['fold'])}"
    return RUNS / f"training/jobs/{method}/seed_{int(job['seed'])}/{fold}"


def _train_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    root = _job_root(job)
    completion = root / "completion.json"
    report_path = root / "report.json"
    if completion.is_file():
        report = read_json(report_path)
        if report["artifact_sha256"] != file_sha256(root / "artifact.pkl"):
            raise RuntimeError(f"artifact drift: {job['id']}")
        if report["scores_sha256"] != file_sha256(root / "scores.npz"):
            raise RuntimeError(f"score drift: {job['id']}")
        return report
    fit_records = _filtered(_TRAIN_STATE["fit_records"], job["action_space"])
    rewards = SOURCE_TRAIN.TRAIN_CORE.reward_matrix(
        fit_records, _TRAIN_STATE["fit_reward_map"]
    )
    ids = [int(record["image_id"]) for record in fit_records]
    if job["fold"] == "full":
        train_indices = list(range(len(ids)))
        predictions = {
            role: _filtered(_TRAIN_STATE[f"{role}_records"], job["action_space"])
            for role in ("calibration", "final")
        }
        model_seed = int(job["seed"])
    else:
        heldout_ids = set(_TRAIN_STATE["folds"][int(job["fold"])])
        heldout = [index for index, image_id in enumerate(ids) if image_id in heldout_ids]
        train_indices = [index for index, image_id in enumerate(ids) if image_id not in heldout_ids]
        if len(heldout) != len(heldout_ids):
            raise RuntimeError(f"OOF population drift: {job['id']}")
        predictions = {"fit_oof": [fit_records[index] for index in heldout]}
        model_seed = int(job["seed"]) + int(job["fold"]) * 10_007
    started = time.time()
    artifact = SOURCE_TRAIN.ADAPTER.fit_model(
        "current_action",
        [fit_records[index] for index in train_indices],
        rewards[train_indices],
        PROTOCOL,
        category_count=int(_TRAIN_STATE["category_count"]),
        seed=model_seed,
    )
    arrays: dict[str, np.ndarray] = {}
    for role, records in predictions.items():
        values = SOURCE_TRAIN.ADAPTER.predict_model(
            artifact,
            records,
            PROTOCOL,
            category_count=int(_TRAIN_STATE["category_count"]),
        )
        arrays[f"scores_{role}"] = np.asarray(values, dtype=np.float64)
        arrays[f"image_ids_{role}"] = np.asarray(
            [int(record["image_id"]) for record in records], dtype=np.int64
        )
    artifact_path = root / "artifact.pkl"
    score_path = root / "scores.npz"
    atomic_pickle(
        artifact_path,
        {
            "method": job["method"],
            "action_space": list(job["action_space"]),
            "seed": int(job["seed"]),
            "fold": job["fold"],
            "category_ids": _TRAIN_STATE["category_ids"],
            "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
            "artifact": artifact,
        },
    )
    atomic_npz(score_path, **arrays)
    metadata = artifact.get("metadata", {})
    report = {
        "complete": True,
        "id": job["id"],
        "method": job["method"],
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "fit_images": len(train_indices),
        "selected_epoch": int(metadata.get("selected_epoch", -1)),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": file_sha256(artifact_path),
        "scores_sha256": file_sha256(score_path),
        "old_learner_checkpoint_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "artifact_sha256": report["artifact_sha256"], "scores_sha256": report["scores_sha256"]})
    print(f"[train] {job['id']} {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def run_train() -> dict[str, Any]:
    completion = RUNS / "training/completion.json"
    if completion.is_file():
        return read_json(RUNS / "training/report.json")
    run_preflight()
    _initialize_training()
    jobs = CORE.training_jobs(PROTOCOL)
    workers = min(int(PROTOCOL["cpu"]["maximum_workers"]), len(jobs))
    started = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(_train_job, jobs, chunksize=1))
    if len(reports) != len(jobs):
        raise RuntimeError("training matrix incomplete")
    report = {
        "complete": True,
        "jobs": len(jobs),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "training/report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(jobs)})
    _TRAIN_STATE.clear()
    return report


def _ensemble_scores(method: str, role: str) -> np.ndarray:
    expected = [int(sample["image_id"]) for sample in load_role(role)["samples"]]
    values = []
    safe = method.replace("/", "_")
    for seed in PROTOCOL["model_seeds"]:
        path = RUNS / f"training/jobs/{safe}/seed_{int(seed)}/full/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            ids = list(map(int, artifact[f"image_ids_{role}"]))
            scores = np.asarray(artifact[f"scores_{role}"], dtype=np.float64)
        if ids != expected:
            raise RuntimeError(f"score/image alignment changed: {method}/{role}/{seed}")
        values.append(scores)
    output = np.mean(values, axis=0)
    output = CORE.collapse_static_budget_scores(
        output, expected_budgets=len(PROTOCOL["budgets"])
    )
    if not np.isfinite(output).all():
        raise RuntimeError(f"non-finite ensemble: {method}/{role}")
    return output


def _score_map(
    records: Sequence[Mapping[str, Any]], matrix: np.ndarray
) -> dict[int, dict[str, float]]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(records):
        raise ValueError(f"unexpected score shape: {values.shape}")
    return {
        int(record["image_id"]): {
            str(action["action_id"]): float(values[row, column])
            for column, action in enumerate(record["actions"])
            if str(action["action_id"]) != "edge"
        }
        for row, record in enumerate(records)
    }


def _schedule(
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    groups = []
    best = []
    for sample in samples:
        image_id = int(sample["image_id"])
        actions = [{"action_id": "edge", "image_id": image_id, "value": 0.0, "cost": 0.0}]
        for action_id, value in scores[image_id].items():
            if not sample["available"].get(action_id, False):
                continue
            if action_id not in sample["payload_bytes"]:
                continue
            actions.append(
                {
                    "action_id": action_id,
                    "image_id": image_id,
                    "value": float(value),
                    "cost": float(sample["payload_bytes"][action_id]),
                }
            )
        if len(actions) == 1:
            raise RuntimeError(f"no cloud action: {image_id}")
        groups.append(actions)
        best.append(max(actions[1:], key=lambda row: (row["value"], -row["cost"], row["action_id"])))
    if mode == "fixed_byte":
        capacity = float(budget) * sum(float(sample["payload_bytes"]["original"]) for sample in samples)
        result = SOURCE_EVAL.select_actions_for_budget(groups, capacity)
        selection = {
            int(row["image_id"]): str(row["action_id"])
            for row in result["selected"]
            if str(row["action_id"]) != "edge"
        }
        used = sum(int(sample["payload_bytes"][selection[int(sample["image_id"])]]) for sample in samples if int(sample["image_id"]) in selection)
        if used > capacity + 1e-6:
            raise RuntimeError("fixed-byte capacity exceeded")
    elif mode == "fixed_call":
        count = int(math.floor(len(samples) * float(budget) + 0.5))
        selected = sorted(best, key=lambda row: (-row["value"], row["cost"], row["image_id"]))[:count]
        selection = {int(row["image_id"]): str(row["action_id"]) for row in selected}
        if len(selection) != count:
            raise RuntimeError("fixed-call population changed")
    else:
        raise ValueError(mode)
    return selection


def _evaluate_selection(
    role: str,
    method: str,
    mode: str,
    budget: float,
    selection: Mapping[int, str],
) -> dict[str, Any]:
    data = load_role(role)
    predictions, _ = CANONICAL.mixed_predictions(data, selection, 0.001)
    metric = data["evaluator"].evaluate(predictions)
    by_id = {int(sample["image_id"]): sample for sample in data["samples"]}
    request_bytes = sum(
        int(by_id[image_id]["payload_bytes"][action])
        for image_id, action in selection.items()
    )
    original_total = sum(
        int(sample["payload_bytes"]["original"]) for sample in data["samples"]
    )
    capacity = float(budget) * original_total
    counts = Counter(selection.values())
    return {
        "role": role,
        "method": method,
        "policy": "Joint",
        "fusion": "Reversible",
        "mode": mode,
        "budget": budget,
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": request_bytes / 1024 / len(data["samples"]),
        "request_bytes": request_bytes,
        "capacity_bytes": capacity,
        "capacity_utilization": request_bytes / max(1.0, capacity) if mode == "fixed_byte" else 1.0,
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": sum(count for action, count in counts.items() if "@" in action),
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }


def _historical_rows() -> list[dict[str, Any]]:
    output = []
    for role in ("calibration", "final"):
        rows = read_csv(HISTORICAL_ROOT / f"runs/{role}/evaluation/results.csv")
        for row in rows:
            if row.get("method") not in PROTOCOL["historical_comparators"]:
                continue
            copied: dict[str, Any] = dict(row)
            copied["fusion"] = "None" if row["method"] == "Current-OQ" else "MAUR"
            copied["policy"] = "Joint"
            output.append(copied)
    return output


def run_evaluate() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    if completion.is_file():
        return read_json(RUNS / "evaluation/report.json")
    run_train()
    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    spaces = {
        PROTOCOL["methods"]["single"]: PROTOCOL["action_spaces"]["OQR"],
        PROTOCOL["methods"]["multi"]: PROTOCOL["action_spaces"]["OQR12"],
    }
    for role in ("calibration", "final"):
        for method, actions in spaces.items():
            records = _filtered(load_role(role)["records"], actions)
            scores = _ensemble_scores(method, role)
            score_map = _score_map(records, scores)
            for budget in map(float, PROTOCOL["budgets"]):
                for mode in PROTOCOL["modes"]:
                    selection = _schedule(
                        load_role(role)["samples"], score_map, mode=mode, budget=budget
                    )
                    key = f"{role}|{method}|{mode}|{budget:g}"
                    selections[key] = {str(image_id): action for image_id, action in selection.items()}
                    rows.append(_evaluate_selection(role, method, mode, budget, selection))
    if len(rows) != CORE.expected_point_rows(PROTOCOL):
        raise RuntimeError("point result matrix incomplete")
    atomic_csv(RUNS / "evaluation/results.csv", rows)
    atomic_json(RUNS / "evaluation/selections.json", selections)
    comparison = _historical_rows() + rows
    atomic_csv(RUNS / "evaluation/comparison_results.csv", comparison)
    report = {
        "complete": True,
        "point_rows": len(rows),
        "comparison_rows": len(comparison),
        "historical_reference_fusions": {"Current-OQ": "None", "Current-OQR": "MAUR"},
        "new_fusion": "Reversible",
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "evaluation/report.json", report)
    atomic_json(completion, {"complete": True, "rows": len(rows)})
    return report


_BOOT_STATE: dict[str, Any] = {}


def _bootstrap_job(job: Mapping[str, Any]) -> dict[str, Any]:
    safe = str(job["id"]).replace("|", "_")
    path = RUNS / f"bootstrap/jobs/{safe}.json"
    if path.is_file():
        return read_json(path)
    token = hashlib.sha256(f"{PROTOCOL['bootstrap']['seed']}:{job['id']}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
    values = []
    started = time.time()
    for replicate in range(int(PROTOCOL["bootstrap"]["replicates"])):
        ids = _BOOT_STATE["image_ids"]
        sampled = [ids[int(index)] for index in rng.integers(0, len(ids), size=len(ids))]
        gt, candidate, comparator = BOOTSTRAP_CORE._bootstrap_remap(
            _BOOT_STATE["ground_truth"],
            _BOOT_STATE["candidate_predictions"][job["id"]],
            _BOOT_STATE["comparator_predictions"][job["id"]],
            sampled,
        )
        values.append(
            float(
                BOOTSTRAP_CORE._in_memory_ap(gt, candidate)
                - BOOTSTRAP_CORE._in_memory_ap(gt, comparator)
            )
        )
        if (replicate + 1) % 100 == 0:
            print(f"[bootstrap] {job['id']} {replicate + 1}/1000", flush=True)
    array = np.asarray(values, dtype=np.float64)
    report = {
        **dict(job),
        "replicates": len(values),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0)),
        "elapsed_seconds": time.time() - started,
        "values": values,
    }
    atomic_json(path, report)
    return report


def run_bootstrap() -> dict[str, Any]:
    completion = RUNS / "bootstrap/completion.json"
    if completion.is_file():
        return read_json(RUNS / "bootstrap/report.json")
    run_evaluate()
    data = load_role("final")
    own = read_json(RUNS / "evaluation/selections.json")
    historical = read_json(HISTORICAL_ROOT / "runs/final/evaluation/selections.json")
    jobs = []
    candidate_predictions: dict[str, Any] = {}
    comparator_predictions: dict[str, Any] = {}
    multi = PROTOCOL["methods"]["multi"]
    single = PROTOCOL["methods"]["single"]
    for budget in map(float, PROTOCOL["budgets"]):
        candidate_key = f"final|{multi}|fixed_byte|{budget:g}"
        candidate_selection = {int(key): value for key, value in own[candidate_key].items()}
        candidate, _ = CANONICAL.mixed_predictions(data, candidate_selection, 0.001)
        comparators = {
            single: {int(key): value for key, value in own[f"final|{single}|fixed_byte|{budget:g}"].items()},
            "Current-OQ": {int(key): value for key, value in historical[f"Current-OQ|fixed_byte|{budget:g}"].items()},
        }
        for comparator_name, selection in comparators.items():
            comparator, _ = CANONICAL.mixed_predictions(data, selection, 0.001)
            job_id = f"{multi}|vs|{comparator_name}|budget={budget:g}|image"
            jobs.append(
                {
                    "id": job_id,
                    "candidate": multi,
                    "comparator": comparator_name,
                    "budget": budget,
                    "unit": "image",
                }
            )
            candidate_predictions[job_id] = candidate
            comparator_predictions[job_id] = comparator
    if len(jobs) != CORE.bootstrap_jobs(PROTOCOL):
        raise RuntimeError("bootstrap matrix incomplete")
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        ground_truth=read_json(RUNS / "common/coco/final/gt.json"),
        image_ids=[int(sample["image_id"]) for sample in data["samples"]],
        candidate_predictions=candidate_predictions,
        comparator_predictions=comparator_predictions,
    )
    workers = min(int(PROTOCOL["cpu"]["bootstrap_workers"]), len(jobs))
    started = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        results = list(pool.imap_unordered(_bootstrap_job, jobs, chunksize=1))
    public = [{key: value for key, value in row.items() if key != "values"} for row in results]
    report = {
        "complete": True,
        "jobs": len(results),
        "replicates": int(PROTOCOL["bootstrap"]["replicates"]),
        "comparisons": public,
        "workers": workers,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(RUNS / "bootstrap/report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(results)})
    _BOOT_STATE.clear()
    return report


def run_report() -> dict[str, Any]:
    run_evaluate()
    rows = read_csv(RUNS / "evaluation/comparison_results.csv")
    methods = [
        "Current-OQ",
        "Current-OQR",
        PROTOCOL["methods"]["single"],
        PROTOCOL["methods"]["multi"],
    ]
    lines = [
        "# COCO CurrentActionValue OQR12 Profile×Scale 补充结果",
        "",
        "本结果为 val2017 post-hoc reuse，不是新的隐藏 final。历史 OQR 使用 MAUR；新单 ROI 与 OQR12 均使用 Reversible Fusion。",
        "",
        "## Fixed-byte final",
        "",
        "| 方法 | 10% | 25% | 40% |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        cells = []
        for budget in map(float, PROTOCOL["budgets"]):
            found = [
                row
                for row in rows
                if row.get("role") == "final"
                and row.get("method") == method
                and row.get("mode") == "fixed_byte"
                and math.isclose(float(row["budget"]), budget)
            ]
            if len(found) != 1:
                raise RuntimeError(f"missing report row: {method}/{budget}")
            cells.append(
                f"{float(found[0]['official_ap50_95']):.6f} / {float(found[0]['kb_per_image']):.2f}"
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    path = ROOT / "RESULTS_ZH.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"complete": True, "result_path": str(path), "rows": len(rows)}
    atomic_json(RUNS / "report.json", report)
    return report


def run_all() -> dict[str, Any]:
    return {
        "preflight": run_preflight(),
        "train": run_train(),
        "evaluate": run_evaluate(),
        "bootstrap": run_bootstrap(),
        "report": run_report(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "evaluate", "bootstrap", "report", "all"),
        default="all",
    )
    args = parser.parse_args()
    result = {
        "preflight": run_preflight,
        "train": run_train,
        "evaluate": run_evaluate,
        "bootstrap": run_bootstrap,
        "report": run_report,
        "all": run_all,
    }[args.stage]()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
