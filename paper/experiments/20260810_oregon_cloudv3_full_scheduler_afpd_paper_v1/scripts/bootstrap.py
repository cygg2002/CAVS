#!/usr/bin/env python3
"""Run 1,000-replicate event-primary and image-sensitivity paired bootstrap."""

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


PROTOCOL_CORE = load_module("oregon_cloudv3_bootstrap_protocol", ROOT / "scripts/protocol_core.py")
EVALUATE = load_module("oregon_cloudv3_bootstrap_evaluate", ROOT / "scripts/evaluate.py")
BOOTSTRAP_CORE = load_module(
    "oregon_cloudv3_bootstrap_core",
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


def read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            for key in (
                "budget",
                "official_ap50_95",
                "official_ap50",
                "official_ap75",
                "kb_per_image",
            ):
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
            rows.append(row)
        return rows


def _selection_key(role: str, method: str, policy: str, budget: float) -> str:
    return f"{role}|{method}|{policy}|fixed_byte|{float(budget):g}"


def _row(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    method: str,
    policy: str,
    budget: float,
) -> Mapping[str, Any]:
    found = [
        row
        for row in rows
        if row["role"] == role
        and row["method"] == method
        and row["policy"] == policy
        and row["mode"] == "fixed_byte"
        and abs(float(row["budget"]) - float(budget)) < 1e-12
    ]
    if len(found) != 1:
        raise RuntimeError(f"missing result row: {role}/{method}/{policy}/{budget:g}")
    return found[0]


def frozen_policy(
    rows: Sequence[Mapping[str, Any]], method: str, budget: float
) -> str:
    if method in {
        "AFPD-OQR12",
        "Raw-ActionValue-OQ",
        "OccupancyFree-AFPD-OQR12",
        "OccupancyAware-AFPD-OQR12",
        "CurrentActionValue-OQ",
        "CurrentActionValue-OQR12",
    }:
        return "Joint"
    candidates = [
        row
        for row in rows
        if row["role"] == "calibration"
        and row["method"] == method
        and row["mode"] == "fixed_byte"
        and abs(float(row["budget"]) - float(budget)) < 1e-12
    ]
    if not candidates:
        raise RuntimeError(f"no calibration policy candidates: {method}/{budget:g}")
    return str(
        max(
            candidates,
            key=lambda row: (
                float(row["official_ap50_95"]),
                -float(row["kb_per_image"]),
                str(row["policy"]),
            ),
        )["policy"]
    )


_STATE: dict[str, Any] = {}


def prepare_jobs() -> list[dict[str, Any]]:
    rows = read_rows(RUNS / "evaluation/point/results.csv")
    selections = read_json(RUNS / "evaluation/point/selections.json")
    first_random_seed = int(PROTOCOL["fold_seed"])
    jobs = []
    for comparison in PROTOCOL_CORE.bootstrap_comparisons(PROTOCOL):
        budget = float(comparison["budget"])
        candidate = str(comparison["candidate"])
        comparator = str(comparison["comparator"])
        candidate_policy = frozen_policy(rows, candidate, budget)
        comparator_policy = frozen_policy(rows, comparator, budget)
        candidate_key = _selection_key("final", candidate, candidate_policy, budget)
        if comparator == "Random":
            comparator_key = f"final|Random[seed={first_random_seed}]|{comparator_policy}|fixed_byte|{budget:g}"
        else:
            comparator_key = _selection_key("final", comparator, comparator_policy, budget)
        if candidate_key not in selections or comparator_key not in selections:
            raise RuntimeError(f"missing bootstrap selection: {candidate_key}/{comparator_key}")
        for unit in PROTOCOL["bootstrap"]["units"]:
            jobs.append(
                {
                    **comparison,
                    "unit": str(unit),
                    "candidate_policy": candidate_policy,
                    "comparator_policy": comparator_policy,
                    "candidate_selection": selections[candidate_key],
                    "comparator_selection": selections[comparator_key],
                    "candidate_point_ap": float(
                        _row(
                            rows,
                            role="final",
                            method=candidate,
                            policy=candidate_policy,
                            budget=budget,
                        )["official_ap50_95"]
                    ),
                    "comparator_point_ap": float(
                        _row(
                            rows,
                            role="final",
                            method=comparator,
                            policy=comparator_policy,
                            budget=budget,
                        )["official_ap50_95"]
                    ),
                }
            )
    if len(jobs) != 54:
        raise RuntimeError(f"expected 54 bootstrap jobs, got {len(jobs)}")
    return jobs


def _sample_ids(
    rng: np.random.Generator,
    *,
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


def _job_path(job: Mapping[str, Any]) -> Path:
    safe = str(job["id"]).replace("/", "_").replace("|", "_")
    return RUNS / f"bootstrap/jobs/{job['unit']}/{safe}.json"


def _bootstrap_job(job: Mapping[str, Any]) -> dict[str, Any]:
    path = _job_path(job)
    if path.is_file():
        value = read_json(path)
        if len(value["values"]) != int(PROTOCOL["bootstrap"]["replicates"]):
            raise RuntimeError(f"incomplete bootstrap cache: {path}")
        return value
    data = _STATE["data"]
    candidate_selection = {int(key): str(value) for key, value in job["candidate_selection"].items()}
    comparator_selection = {int(key): str(value) for key, value in job["comparator_selection"].items()}
    candidate_predictions, _ = EVALUATE.CANONICAL.mixed_predictions(data, candidate_selection, 0.001)
    comparator_predictions, _ = EVALUATE.CANONICAL.mixed_predictions(data, comparator_selection, 0.001)
    ground_truth = _STATE["ground_truth"]
    image_ids = _STATE["image_ids"]
    groups = _STATE["event_groups"]
    seed_token = hashlib.sha256(
        f"{PROTOCOL['bootstrap']['seed']}:{job['id']}:{job['unit']}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(seed_token[:8], "big"))
    values = []
    started = time.time()
    for replicate in range(int(PROTOCOL["bootstrap"]["replicates"])):
        sampled_ids = _sample_ids(
            rng,
            unit=str(job["unit"]),
            image_ids=image_ids,
            event_groups=groups,
        )
        remapped_gt, candidate, comparator = BOOTSTRAP_CORE._bootstrap_remap(
            ground_truth,
            candidate_predictions,
            comparator_predictions,
            sampled_ids,
        )
        delta = BOOTSTRAP_CORE._in_memory_ap(
            remapped_gt, candidate
        ) - BOOTSTRAP_CORE._in_memory_ap(remapped_gt, comparator)
        values.append(float(delta))
        if (replicate + 1) % 100 == 0:
            print(
                f"[bootstrap:{job['unit']}:{job['id']}] {replicate + 1}/{PROTOCOL['bootstrap']['replicates']}",
                flush=True,
            )
    array = np.asarray(values, dtype=np.float64)
    result = {
        **{key: value for key, value in job.items() if not key.endswith("_selection")},
        "replicates": len(values),
        "values": values,
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0.0)),
        "point_delta": float(job["candidate_point_ap"] - job["comparator_point_ap"]),
        "elapsed_seconds": time.time() - started,
        "resampling_unit": str(job["unit"]),
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }
    atomic_json(path, result)
    return result


def run_bootstrap() -> dict[str, Any]:
    completion = RUNS / "bootstrap/completion.json"
    if completion.is_file():
        return read_json(RUNS / "bootstrap/report.json")
    if not (RUNS / "evaluation/point/completion.json").is_file():
        raise RuntimeError("point evaluation must complete before bootstrap")
    data = EVALUATE.load_role("final")
    ground_truth = read_json(RUNS / "common/oregon/final/gt.json")
    event_map: dict[str, list[int]] = {}
    for sample in data["samples"]:
        event_map.setdefault(str(sample["event_id"]), []).append(int(sample["image_id"]))
    _STATE.clear()
    _STATE.update(
        data=data,
        ground_truth=ground_truth,
        image_ids=[int(sample["image_id"]) for sample in data["samples"]],
        event_groups=[event_map[key] for key in sorted(event_map)],
    )
    jobs = prepare_jobs()
    workers = min(
        int(PROTOCOL["cpu"]["bootstrap_workers"]),
        len(jobs),
        max(1, (os.cpu_count() or 1) - 4),
    )
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        raw = list(pool.imap_unordered(_bootstrap_job, jobs, chunksize=1))
    comparisons = [
        {key: value for key, value in row.items() if key != "values"}
        for row in raw
    ]
    comparisons.sort(key=lambda row: (row["unit"], row["id"]))
    report = {
        "complete": True,
        "comparisons": comparisons,
        "jobs": len(jobs),
        "replicates_per_comparison": int(PROTOCOL["bootstrap"]["replicates"]),
        "units": list(PROTOCOL["bootstrap"]["units"]),
        "events": len(_STATE["event_groups"]),
        "images": len(_STATE["image_ids"]),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "event_bootstrap_primary": True,
        "image_bootstrap_sensitivity_only": True,
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }
    atomic_json(RUNS / "bootstrap/report.json", report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(RUNS / "bootstrap/report.json")})
    _STATE.clear()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("bootstrap", "all"), default="all")
    parser.parse_args()
    print(json.dumps(run_bootstrap(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
