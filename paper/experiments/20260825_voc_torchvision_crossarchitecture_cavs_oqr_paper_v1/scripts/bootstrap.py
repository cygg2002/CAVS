#!/usr/bin/env python3
"""Run the frozen image-level paired and nested Random-50 VOC bootstrap."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
BOOTSTRAP_JOBS_PER_BUDGET = 12
DEFAULT_BOOTSTRAP_WORKERS = 4


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_gluoncv_boot_data", ROOT / "scripts/data_core.py")
METRIC = load_module("voc_gluoncv_boot_metric", ROOT / "scripts/voc_metric.py")


def bootstrap_remap(
    ground_truth: Mapping[str, Any],
    prediction_sets: Sequence[Sequence[Mapping[str, Any]]],
    sampled_image_ids: Sequence[int],
) -> tuple[dict[str, Any], tuple[list[dict[str, Any]], ...]]:
    images = {int(row["id"]): row for row in ground_truth["images"]}
    annotations: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ground_truth["annotations"]:
        annotations[int(row["image_id"])].append(row)
    prediction_groups = []
    for predictions in prediction_sets:
        groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in predictions:
            groups[int(row["image_id"])].append(row)
        prediction_groups.append(groups)
    remapped_gt = {
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(ground_truth["categories"]),
    }
    remapped_predictions: list[list[dict[str, Any]]] = [
        [] for _ in prediction_sets
    ]
    annotation_id = 1
    for new_id, source_id in enumerate(map(int, sampled_image_ids), start=1):
        image = copy.deepcopy(images[source_id])
        image["id"] = new_id
        remapped_gt["images"].append(image)
        for source in annotations[source_id]:
            row = copy.deepcopy(source)
            row["id"] = annotation_id
            row["image_id"] = new_id
            annotation_id += 1
            remapped_gt["annotations"].append(row)
        for target, groups in zip(remapped_predictions, prediction_groups):
            for source in groups[source_id]:
                row = copy.deepcopy(source)
                row["image_id"] = new_id
                target.append(row)
    return remapped_gt, tuple(remapped_predictions)


def _in_memory_ap(
    ground_truth: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> float:
    evaluator = METRIC.VocMetricBackend(ground_truth)
    return float(
        evaluator.evaluate(
            predictions, [int(row["id"]) for row in ground_truth["images"]]
        )["map50_95"]
    )


def _selection_key(
    method: str, policy: str, seed: Any, budget: float
) -> str:
    return f"final|{method}|{policy}|seed={seed}|fixed_byte|{float(budget):g}"


def _rng(job_id: str) -> np.random.Generator:
    token = hashlib.sha256(
        f"{int(PROTOCOL['bootstrap']['seed'])}:{job_id}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "big"))


def _sample(rng: np.random.Generator) -> list[int]:
    values = _BOOT_STATE["image_ids"]
    return rng.choice(values, size=len(values), replace=True).astype(np.int64).tolist()


def _job_paths(job: Mapping[str, Any]) -> tuple[Path, Path]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(job["id"]))
    return RUNS / f"bootstrap/jobs/{safe}.json", RUNS / f"bootstrap/jobs/{safe}.partial.json"


def _summary(job: Mapping[str, Any], values: Sequence[float]) -> dict[str, Any]:
    delta = np.asarray(values, dtype=np.float64)
    lower, upper = np.percentile(delta, [2.5, 97.5])
    probability_nonpositive = float(np.mean(delta <= 0.0))
    probability_nonnegative = float(np.mean(delta >= 0.0))
    return {
        "id": str(job["id"]),
        "kind": str(job["kind"]),
        "candidate": str(job["candidate"]),
        "comparator": str(job["comparator"]),
        "budget": float(job["budget"]),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "probability_delta_le_zero": probability_nonpositive,
        "two_sided_sign_probability": min(
            1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)
        ),
        "replicates": len(delta),
        "resampling_unit": "image",
    }


_BOOT_STATE: dict[str, Any] = {}


def _bootstrap_worker_count(configured: int, job_count: int) -> int:
    raw = os.environ.get("VOC_BOOTSTRAP_WORKERS", str(DEFAULT_BOOTSTRAP_WORKERS))
    try:
        requested = int(raw)
    except ValueError as error:
        raise ValueError(
            f"VOC_BOOTSTRAP_WORKERS must be an integer, got {raw!r}"
        ) from error
    if requested < 1:
        raise ValueError("VOC_BOOTSTRAP_WORKERS must be positive")
    return min(
        int(configured), requested, int(job_count), max(1, os.cpu_count() or 1)
    )


def _standard(job: Mapping[str, Any]) -> dict[str, Any]:
    path, partial = _job_paths(job)
    if path.is_file():
        return DATA.read_json(path)
    values = list(map(float, DATA.read_json(partial)["values"])) if partial.is_file() else []
    rng = _rng(str(job["id"]))
    for _ in values:
        _sample(rng)
    candidate, comparator = _BOOT_STATE["pairs"][int(job["pair_index"])]
    total = int(PROTOCOL["bootstrap"]["replicates"])
    for replicate in range(len(values), total):
        sampled = _sample(rng)
        gt, predictions = bootstrap_remap(
            _BOOT_STATE["ground_truth"], [candidate, comparator], sampled
        )
        values.append(_in_memory_ap(gt, predictions[0]) - _in_memory_ap(gt, predictions[1]))
        if (replicate + 1) % 100 == 0:
            DATA.atomic_json(partial, {"values": values})
    result = _summary(job, values)
    DATA.atomic_json(path, result)
    partial.unlink(missing_ok=True)
    return result


def _nested_random(job: Mapping[str, Any]) -> dict[str, Any]:
    path, partial = _job_paths(job)
    if path.is_file():
        return DATA.read_json(path)
    values = list(map(float, DATA.read_json(partial)["values"])) if partial.is_file() else []
    rng = _rng(str(job["id"]))
    for _ in values:
        _sample(rng)
    candidate = _BOOT_STATE["candidate_predictions"][int(job["candidate_index"])]
    random_predictions = [
        _BOOT_STATE["random_predictions"][int(index)]
        for index in job["random_indices"]
    ]
    if len(random_predictions) != int(PROTOCOL["random"]["seeds"]):
        raise RuntimeError("nested Random-50 prediction population changed")
    total = int(PROTOCOL["bootstrap"]["replicates"])
    for replicate in range(len(values), total):
        sampled = _sample(rng)
        gt, predictions = bootstrap_remap(
            _BOOT_STATE["ground_truth"], [candidate, *random_predictions], sampled
        )
        candidate_ap = _in_memory_ap(gt, predictions[0])
        random_ap = [_in_memory_ap(gt, rows) for rows in predictions[1:]]
        values.append(float(candidate_ap - np.mean(random_ap)))
        if (replicate + 1) % 25 == 0:
            DATA.atomic_json(partial, {"values": values})
    result = _summary(job, values)
    result["random_seed_count"] = len(random_predictions)
    result["random_nested_per_resample"] = True
    DATA.atomic_json(path, result)
    partial.unlink(missing_ok=True)
    return result


def _worker(job: Mapping[str, Any]) -> dict[str, Any]:
    os.nice(19)
    return _nested_random(job) if job["kind"] == "nested_random" else _standard(job)


def run() -> dict[str, Any]:
    output = RUNS / "bootstrap"
    completion = output / "completion.json"
    if completion.is_file():
        return DATA.read_json(output / "report.json")
    if not (RUNS / "point/completion.json").is_file():
        raise RuntimeError("point matrix must complete before bootstrap")
    point = load_module("voc_gluoncv_boot_point", ROOT / "scripts/evaluate_point.py")
    data = point._role("final")
    stored = DATA.load_pickle(RUNS / "point/selections.pkl")
    random_stored = DATA.load_pickle(RUNS / "point/random_selections.pkl")

    def selection(key: str, *, random: bool = False) -> dict[int, str]:
        source = random_stored if random else stored
        if key not in source:
            raise RuntimeError(f"missing frozen point selection: {key}")
        return {int(image_id): str(action) for image_id, action in source[key].items()}

    prediction_cache: dict[tuple[str, bool], list[dict[str, Any]]] = {}

    def predictions(key: str, *, random: bool = False) -> list[dict[str, Any]]:
        cache_key = (key, random)
        if cache_key not in prediction_cache:
            prediction_cache[cache_key] = point.mixed_predictions(
                data, selection(key, random=random)
            )
        return prediction_cache[cache_key]

    jobs = []
    pairs = []
    candidate_predictions = []
    random_predictions = []
    for budget in map(float, PROTOCOL["budgets"]):
        cavs_o_key = _selection_key("CAVS-O-Core", "Joint-O", "ensemble", budget)
        cavs_o = predictions(cavs_o_key)
        cavs_oq_key = _selection_key("CAVS-OQ-Core", "Joint-OQ", "ensemble", budget)
        cavs_oq = predictions(cavs_oq_key)
        cavs_oqr_key = _selection_key("CAVS-OQR-Core", "Joint-OQR", "ensemble", budget)
        cavs_oqr = predictions(cavs_oqr_key)

        def add_comparator_block(
            candidate_name: str,
            candidate_rows: list[dict[str, Any]],
            policy: str,
        ) -> None:
            comparators = {
                f"ORIC-AP5095-3Seed|{policy}": _selection_key(
                    "ORIC-AP5095", policy, "ensemble", budget
                ),
                f"MORIC-AP5095-3Seed|{policy}": _selection_key(
                    "MORIC-AP5095", policy, "ensemble", budget
                ),
                f"DCSB-Score-Adapted|{policy}": _selection_key(
                    "DCSB-Score-Adapted", policy, "fixed", budget
                ),
                f"Confidence|{policy}": _selection_key(
                    "Confidence", policy, "fixed", budget
                ),
            }
            for comparator_name, comparator_key in comparators.items():
                pair_index = len(pairs)
                pairs.append((candidate_rows, predictions(comparator_key)))
                jobs.append(
                    {
                        "id": f"{candidate_name}|vs|{comparator_name}|budget={budget:g}",
                        "kind": "paired",
                        "candidate": candidate_name,
                        "comparator": comparator_name,
                        "budget": budget,
                        "pair_index": pair_index,
                    }
                )
            random_indices = []
            for offset in range(int(PROTOCOL["random"]["seeds"])):
                seed = int(PROTOCOL["random"]["base_seed"]) + offset
                key = _selection_key("Random", policy, seed, budget)
                random_indices.append(len(random_predictions))
                random_predictions.append(predictions(key, random=True))
            candidate_index = len(candidate_predictions)
            candidate_predictions.append(candidate_rows)
            jobs.append(
                {
                    "id": f"{candidate_name}|vs|Random-Mean50|{policy}|budget={budget:g}",
                    "kind": "nested_random",
                    "candidate": candidate_name,
                    "comparator": f"Random-Mean50|{policy}",
                    "budget": budget,
                    "candidate_index": candidate_index,
                    "random_indices": random_indices,
                }
            )

        # Same-action fairness block: O versus scalar Original-only baselines.
        add_comparator_block("CAVS-O-Core", cavs_o, "Original-only")

        pair_index = len(pairs)
        pairs.append((cavs_oq, cavs_o))
        jobs.append(
            {
                "id": f"CAVS-OQ-Core|vs|CAVS-O-Core|budget={budget:g}",
                "kind": "paired",
                "candidate": "CAVS-OQ-Core",
                "comparator": "CAVS-O-Core",
                "budget": budget,
                "pair_index": pair_index,
            }
        )

        pair_index = len(pairs)
        pairs.append((cavs_oqr, cavs_oq))
        jobs.append(
            {
                "id": f"CAVS-OQR-Core|vs|CAVS-OQ-Core|budget={budget:g}",
                "kind": "paired",
                "candidate": "CAVS-OQR-Core",
                "comparator": "CAVS-OQ-Core",
                "budget": budget,
                "pair_index": pair_index,
            }
        )

        # Main-system block: OQR versus each baseline's strongest registered OQ policy.
        add_comparator_block("CAVS-OQR-Core", cavs_oqr, "SharedAction-OQ")
    expected_jobs = len(PROTOCOL["budgets"]) * BOOTSTRAP_JOBS_PER_BUDGET
    if len(jobs) != expected_jobs or len({str(job["id"]) for job in jobs}) != expected_jobs:
        raise RuntimeError(f"bootstrap job matrix changed: {len(jobs)}")
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        ground_truth=DATA.read_json(data["gt"]),
        image_ids=list(map(int, data["image_ids"])),
        pairs=pairs,
        candidate_predictions=candidate_predictions,
        random_predictions=random_predictions,
    )
    workers = _bootstrap_worker_count(
        int(PROTOCOL["bootstrap"]["workers"]), len(jobs)
    )
    with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        results = list(pool.imap_unordered(_worker, jobs, chunksize=1))
    results.sort(key=lambda row: (float(row["budget"]), str(row["candidate"]), str(row["comparator"])))
    if any(int(row["replicates"]) != int(PROTOCOL["bootstrap"]["replicates"]) for row in results):
        raise RuntimeError("bootstrap replicate matrix incomplete")
    DATA.atomic_csv(output / "results.csv", results)
    report = {
        "complete": True,
        "jobs": len(results),
        "paired_jobs": sum(row["kind"] == "paired" for row in results),
        "nested_random_jobs": sum(row["kind"] == "nested_random" for row in results),
        "replicates_per_job": int(PROTOCOL["bootstrap"]["replicates"]),
        "random_seed_count": int(PROTOCOL["random"]["seeds"]),
        "workers": workers,
        "results_sha256": DATA.file_sha256(output / "results.csv"),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
