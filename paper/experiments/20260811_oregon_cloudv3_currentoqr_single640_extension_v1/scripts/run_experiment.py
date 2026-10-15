#!/usr/bin/env python3
"""Retrain and evaluate the missing Oregon CurrentActionValue single-ROI cell."""

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
SOURCE = Path(PROTOCOL["source_experiment"])


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("oregon_single640_core", ROOT / "scripts/core.py")
SOURCE_TRAIN = load_module(
    "oregon_single640_source_train", SOURCE / "scripts/train_learners.py"
)
SOURCE_EVAL = load_module(
    "oregon_single640_source_evaluate", SOURCE / "scripts/evaluate.py"
)
BOOTSTRAP_CORE = load_module(
    "oregon_single640_bootstrap_core",
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


def read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_required() -> list[Path]:
    return [
        SOURCE / "runs/common/completion.json",
        SOURCE / "runs/rewards/completion.json",
        SOURCE / "runs/rewards/fit/reward_map.pkl",
        SOURCE / "runs/rewards/fit/folds.json",
        SOURCE / "runs/evaluation/point/results.csv",
        SOURCE / "runs/evaluation/point/selections.json",
    ]


def run_preflight() -> dict[str, Any]:
    output = RUNS / "preflight/report.json"
    completion = RUNS / "preflight/completion.json"
    if completion.is_file():
        return read_json(output)
    CORE.validate_protocol(PROTOCOL)
    missing = [str(path) for path in _source_required() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    roles = {role: SOURCE_TRAIN.load_role(role) for role in PROTOCOL["roles"]}
    if any(
        len(data["samples"]) != int(PROTOCOL["roles"][role])
        for role, data in roles.items()
    ):
        raise RuntimeError("source role population changed")
    for role, data in roles.items():
        actions = {str(action["action_id"]) for action in data["records"][0]["actions"]}
        if "current@640" not in actions:
            raise RuntimeError(f"current@640 missing from {role}")
    source_hashes = {str(path): file_sha256(path) for path in _source_required()}
    report = {
        "complete": True,
        "protocol_id": PROTOCOL["protocol_id"],
        "role_counts": {role: len(data["samples"]) for role, data in roles.items()},
        "training_jobs": len(CORE.training_jobs(PROTOCOL)),
        "source_hashes": source_hashes,
        "source_digest": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "source_opened_read_only": True,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


_TRAIN_STATE: dict[str, Any] = {}


def _filtered(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return SOURCE_TRAIN.TRAIN_CORE.filter_records(records, PROTOCOL["action_space"])


def _initialize_training() -> None:
    fit = SOURCE_TRAIN.load_role("fit")
    calibration = SOURCE_TRAIN.load_role("calibration")
    final = SOURCE_TRAIN.load_role("final")
    reward_map, _ = SOURCE_TRAIN.load_rewards("fit")
    fit_records = _filtered(fit["records"])
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        fit_records=fit_records,
        fit_rewards=SOURCE_TRAIN.TRAIN_CORE.reward_matrix(fit_records, reward_map),
        fit_ids=[int(record["image_id"]) for record in fit_records],
        calibration_records=_filtered(calibration["records"]),
        final_records=_filtered(final["records"]),
        folds=SOURCE_TRAIN.fit_folds(),
        category_count=len(fit["category_ids"]),
        category_ids=list(map(int, fit["category_ids"])),
    )


def _job_root(job: Mapping[str, Any]) -> Path:
    token = "full" if job["fold"] == "full" else f"fold_{int(job['fold'])}"
    return RUNS / f"training/jobs/seed_{int(job['seed'])}/{token}"


def _subset(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


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
    ids = _TRAIN_STATE["fit_ids"]
    if job["fold"] == "full":
        train_indices = list(range(len(ids)))
        prediction_roles = {
            "calibration": _TRAIN_STATE["calibration_records"],
            "final": _TRAIN_STATE["final_records"],
        }
        model_seed = int(job["seed"])
    else:
        heldout = set(_TRAIN_STATE["folds"][int(job["fold"])])
        train_indices = [index for index, image_id in enumerate(ids) if image_id not in heldout]
        heldout_indices = [index for index, image_id in enumerate(ids) if image_id in heldout]
        if len(heldout_indices) != len(heldout):
            raise RuntimeError(f"OOF fold population drift: {job['id']}")
        prediction_roles = {
            "fit_oof": _subset(_TRAIN_STATE["fit_records"], heldout_indices)
        }
        model_seed = int(job["seed"]) + int(job["fold"]) * 10_007
    started = time.time()
    artifact = SOURCE_TRAIN.ADAPTER.fit_model(
        "current_action",
        _subset(_TRAIN_STATE["fit_records"], train_indices),
        _TRAIN_STATE["fit_rewards"][train_indices],
        SOURCE_TRAIN.PROTOCOL,
        category_count=int(_TRAIN_STATE["category_count"]),
        seed=model_seed,
    )
    arrays: dict[str, np.ndarray] = {}
    for role, records in prediction_roles.items():
        values = SOURCE_TRAIN.ADAPTER.predict_model(
            artifact,
            records,
            SOURCE_TRAIN.PROTOCOL,
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
            "method": PROTOCOL["method"],
            "action_space": list(PROTOCOL["action_space"]),
            "seed": int(job["seed"]),
            "fold": job["fold"],
            "category_ids": _TRAIN_STATE["category_ids"],
            "cloud_model_sha256": SOURCE_TRAIN.PROTOCOL["cloud_model"]["sha256"],
            "artifact": artifact,
        },
    )
    atomic_npz(score_path, **arrays)
    metadata = artifact.get("metadata", {})
    report = {
        "complete": True,
        "id": str(job["id"]),
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "fit_images": len(train_indices),
        "prediction_roles": sorted(prediction_roles),
        "selected_epoch": int(metadata.get("selected_epoch", -1)),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": file_sha256(artifact_path),
        "scores_sha256": file_sha256(score_path),
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


def run_train() -> dict[str, Any]:
    completion = RUNS / "training/completion.json"
    if completion.is_file():
        return read_json(RUNS / "training/report.json")
    run_preflight()
    _initialize_training()
    jobs = CORE.training_jobs(PROTOCOL)
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(jobs))
    started = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(_train_job, jobs, chunksize=1))
    if len(reports) != len(jobs):
        raise RuntimeError("training job matrix incomplete")
    report = {
        "complete": True,
        "jobs": len(jobs),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "training/report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(jobs)})
    _TRAIN_STATE.clear()
    return report


def _ensemble_scores(role: str) -> np.ndarray:
    expected = [
        int(sample["image_id"])
        for sample in SOURCE_TRAIN.load_role(role)["samples"]
    ]
    values = []
    for seed in PROTOCOL["model_seeds"]:
        path = RUNS / f"training/jobs/seed_{int(seed)}/full/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            ids = list(map(int, artifact[f"image_ids_{role}"]))
            score = np.asarray(artifact[f"scores_{role}"], dtype=np.float64)
        if ids != expected:
            raise RuntimeError(f"score/image alignment changed: {role}/{seed}")
        values.append(score)
    return np.mean(values, axis=0)


def _evaluate_selection(
    role: str, mode: str, budget: float, selection: Mapping[int, str]
) -> dict[str, Any]:
    CORE.validate_action_selection(selection, PROTOCOL)
    data = SOURCE_EVAL.load_role(role)
    predictions, _ = SOURCE_EVAL.CANONICAL.mixed_predictions(data, selection, 0.001)
    metric = data["evaluator"].evaluate(predictions)
    validation = SOURCE_EVAL.LEARNER_CORE.validate_selection(
        data["samples"], selection, mode=mode, budget=budget
    )
    counts = Counter(selection.values())
    return {
        "role": role,
        "method": PROTOCOL["method"],
        "policy": "Joint",
        "mode": mode,
        "budget": budget,
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": validation["request_bytes"] / 1024 / len(data["samples"]),
        "request_bytes": int(validation["request_bytes"]),
        "capacity_utilization": float(validation["utilization"]),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["current@640"]),
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }


def run_evaluate() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    if completion.is_file():
        return read_json(RUNS / "evaluation/report.json")
    run_train()
    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    for role in ("calibration", "final"):
        data = SOURCE_EVAL.load_role(role)
        records = _filtered(SOURCE_TRAIN.load_role(role)["records"])
        score = _ensemble_scores(role)
        for budget_index, budget in enumerate(PROTOCOL["budgets"]):
            score_map = SOURCE_EVAL._score_map(records, score[budget_index])
            for mode in PROTOCOL["modes"]:
                selection = SOURCE_EVAL.schedule_joint(
                    data["samples"], score_map, mode=mode, budget=float(budget)
                )
                key = f"{role}|{PROTOCOL['method']}|Joint|{mode}|{float(budget):g}"
                selections[key] = {str(key): value for key, value in selection.items()}
                rows.append(_evaluate_selection(role, mode, float(budget), selection))
    if len(rows) != CORE.expected_point_rows(PROTOCOL):
        raise RuntimeError("point result matrix incomplete")
    atomic_csv(RUNS / "evaluation/results.csv", rows)
    atomic_json(RUNS / "evaluation/selections.json", selections)
    source_rows = read_csv(SOURCE / "runs/evaluation/point/results.csv")
    comparison = [
        row
        for row in source_rows
        if row.get("role") in {"calibration", "final"}
        and row.get("method") in set(PROTOCOL["comparators"])
        and row.get("policy") == "Joint"
    ] + rows
    atomic_csv(RUNS / "evaluation/comparison_results.csv", comparison)
    report = {
        "complete": True,
        "point_rows": len(rows),
        "comparison_rows": len(comparison),
        "source_comparators": list(PROTOCOL["comparators"]),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "evaluation/report.json", report)
    atomic_json(completion, {"complete": True, "rows": len(rows)})
    return report


_BOOT_STATE: dict[str, Any] = {}


def _sample_ids(
    rng: np.random.Generator,
    unit: str,
    image_ids: Sequence[int],
    event_groups: Sequence[Sequence[int]],
) -> list[int]:
    if unit == "image":
        indices = rng.integers(0, len(image_ids), size=len(image_ids))
        return [int(image_ids[int(index)]) for index in indices]
    if unit == "event":
        indices = rng.integers(0, len(event_groups), size=len(event_groups))
        return [
            int(image_id)
            for index in indices
            for image_id in event_groups[int(index)]
        ]
    raise ValueError(unit)


def _bootstrap_job(job: Mapping[str, Any]) -> dict[str, Any]:
    safe = str(job["id"]).replace("|", "_")
    path = RUNS / f"bootstrap/jobs/{safe}.json"
    partial_path = RUNS / f"bootstrap/jobs/{safe}.partial.json"
    if path.is_file():
        result = read_json(path)
        if len(result["values"]) != int(PROTOCOL["bootstrap"]["replicates"]):
            raise RuntimeError(f"incomplete bootstrap output: {path}")
        return result
    values = []
    if partial_path.is_file():
        values = list(map(float, read_json(partial_path)["values"]))
    token = hashlib.sha256(
        f"{PROTOCOL['bootstrap']['seed']}:{job['id']}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
    # Advance deterministically to the first unfinished replicate.
    for _ in range(len(values)):
        _sample_ids(
            rng,
            str(job["unit"]),
            _BOOT_STATE["image_ids"],
            _BOOT_STATE["event_groups"],
        )
    started = time.time()
    for replicate in range(len(values), int(PROTOCOL["bootstrap"]["replicates"])):
        sampled = _sample_ids(
            rng,
            str(job["unit"]),
            _BOOT_STATE["image_ids"],
            _BOOT_STATE["event_groups"],
        )
        gt, candidate, comparator = BOOTSTRAP_CORE._bootstrap_remap(
            _BOOT_STATE["ground_truth"],
            _BOOT_STATE["candidate_predictions"][str(job["id"])],
            _BOOT_STATE["comparator_predictions"][str(job["id"])],
            sampled,
        )
        values.append(
            float(
                BOOTSTRAP_CORE._in_memory_ap(gt, candidate)
                - BOOTSTRAP_CORE._in_memory_ap(gt, comparator)
            )
        )
        if (replicate + 1) % 100 == 0:
            atomic_json(partial_path, {"values": values})
            print(f"[bootstrap] {job['id']} {replicate + 1}/1000", flush=True)
    array = np.asarray(values, dtype=np.float64)
    result = {
        **dict(job),
        "values": values,
        "replicates": len(values),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0)),
        "elapsed_seconds": time.time() - started,
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }
    atomic_json(path, result)
    partial_path.unlink(missing_ok=True)
    return result


def run_bootstrap() -> dict[str, Any]:
    completion = RUNS / "bootstrap/completion.json"
    if completion.is_file():
        return read_json(RUNS / "bootstrap/report.json")
    run_evaluate()
    data = SOURCE_EVAL.load_role("final")
    own_selections = read_json(RUNS / "evaluation/selections.json")
    source_selections = read_json(SOURCE / "runs/evaluation/point/selections.json")
    predictions: dict[str, dict[int, list[dict[str, Any]]]] = {}
    jobs = []
    for budget in PROTOCOL["budgets"]:
        own_key = f"final|{PROTOCOL['method']}|Joint|fixed_byte|{float(budget):g}"
        own_map = {int(key): value for key, value in own_selections[own_key].items()}
        predictions[f"{PROTOCOL['method']}@{float(budget):g}"], _ = (
            SOURCE_EVAL.CANONICAL.mixed_predictions(data, own_map, 0.001)
        )
        for comparator in PROTOCOL["comparators"]:
            source_key = f"final|{comparator}|Joint|fixed_byte|{float(budget):g}"
            source_map = {
                int(key): value for key, value in source_selections[source_key].items()
            }
            predictions[f"{comparator}@{float(budget):g}"], _ = (
                SOURCE_EVAL.CANONICAL.mixed_predictions(data, source_map, 0.001)
            )
            for unit in PROTOCOL["bootstrap"]["units"]:
                jobs.append(
                    {
                        "id": f"{PROTOCOL['method']}|vs|{comparator}|budget={float(budget):g}|{unit}",
                        "candidate": PROTOCOL["method"],
                        "comparator": comparator,
                        "budget": float(budget),
                        "unit": str(unit),
                    }
                )
    if len(jobs) != CORE.bootstrap_jobs(PROTOCOL):
        raise RuntimeError("bootstrap job matrix incomplete")
    event_map: dict[str, list[int]] = {}
    for sample in data["samples"]:
        event_map.setdefault(str(sample["event_id"]), []).append(int(sample["image_id"]))
    candidate_predictions = {}
    comparator_predictions = {}
    for job in jobs:
        budget = float(job["budget"])
        candidate_predictions[str(job["id"])] = predictions[
            f"{PROTOCOL['method']}@{budget:g}"
        ]
        comparator_predictions[str(job["id"])] = predictions[
            f"{job['comparator']}@{budget:g}"
        ]
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        ground_truth=read_json(SOURCE / "runs/common/oregon/final/gt.json"),
        image_ids=[int(sample["image_id"]) for sample in data["samples"]],
        event_groups=[event_map[key] for key in sorted(event_map)],
        candidate_predictions=candidate_predictions,
        comparator_predictions=comparator_predictions,
    )
    started = time.time()
    workers = min(int(PROTOCOL["cpu"]["bootstrap_workers"]), len(jobs))
    with mp.get_context("fork").Pool(processes=workers) as pool:
        results = list(pool.imap_unordered(_bootstrap_job, jobs, chunksize=1))
    public = [{key: value for key, value in row.items() if key != "values"} for row in results]
    report = {
        "complete": True,
        "jobs": len(results),
        "comparisons": public,
        "replicates": int(PROTOCOL["bootstrap"]["replicates"]),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "event_bootstrap_primary": True,
        "image_bootstrap_sensitivity_only": True,
    }
    atomic_json(RUNS / "bootstrap/report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(results)})
    _BOOT_STATE.clear()
    return report


def run_report() -> dict[str, Any]:
    run_evaluate()
    rows = read_csv(RUNS / "evaluation/comparison_results.csv")
    lines = [
        "# Oregon Cloud-V3 CurrentActionValue 单 ROI 补充结果",
        "",
        "本实验为 post-hoc final reuse；新 OQR 仅使用 `current@640`，所有学习器训练只使用 fit。",
        "",
        "## Fixed-byte final",
        "",
        "| 方法 | 10% | 25% | 40% |",
        "|---|---:|---:|---:|",
    ]
    methods = [
        "CurrentActionValue-OQ",
        PROTOCOL["method"],
        "CurrentActionValue-OQR12",
    ]
    for method in methods:
        cells = []
        for budget in PROTOCOL["budgets"]:
            found = [
                row
                for row in rows
                if row.get("role") == "final"
                and row.get("method") == method
                and row.get("mode") == "fixed_byte"
                and abs(float(row["budget"]) - float(budget)) < 1e-12
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
