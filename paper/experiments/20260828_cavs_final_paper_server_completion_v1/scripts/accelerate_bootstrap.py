#!/usr/bin/env python3
"""Resume frozen paper bootstraps with deterministic replicate-level parallelism."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import multiprocessing as multiprocessing_module
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
REAL_FORK_CONTEXT = multiprocessing_module.get_context("fork")

_TARGET: Any = None
_TARGET_NAME = ""
_TARGET_KIND = ""
_VOC_FAST: "FastVocBootstrap | None" = None


class VerificationOnly(RuntimeError):
    """Internal control-flow signal used after a read-only equivalence check."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def validate_source(target_name: str) -> tuple[dict[str, Any], Path, Path]:
    if target_name not in PROTOCOL["targets"]:
        raise ValueError(f"unknown target: {target_name}")
    config = dict(PROTOCOL["targets"][target_name])
    experiment = REPO / str(config["experiment"])
    script = experiment / str(config["script"])
    protocol = experiment / str(config["protocol"])
    for path, expected in (
        (script, str(config["script_sha256"])),
        (protocol, str(config["protocol_sha256"])),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen source hash changed: {path}\nexpected={expected}\nactual={actual}"
            )
    return config, experiment, script


def chunk_ranges(start: int, stop: int, size: int) -> list[tuple[int, int]]:
    if not 0 <= start <= stop:
        raise ValueError(f"invalid range: {start}/{stop}")
    if size < 1:
        raise ValueError(f"invalid chunk size: {size}")
    return [(left, min(stop, left + size)) for left in range(start, stop, size)]


def merge_contiguous(
    values: list[float], pending: dict[int, list[float]], start: int, chunk: Sequence[float]
) -> bool:
    """Merge a chunk and extend only the durable contiguous prefix."""
    if start < len(values) or start in pending:
        raise RuntimeError(f"overlapping bootstrap chunk: {start}/{len(values)}")
    pending[start] = list(map(float, chunk))
    changed = False
    while len(values) in pending:
        values.extend(pending.pop(len(values)))
        changed = True
    return changed


def _state() -> Mapping[str, Any]:
    if _TARGET_KIND in {"coco_oregon", "voc"}:
        return _TARGET._BOOT_STATE
    if _TARGET_KIND == "native":
        return _TARGET._BOOT
    raise RuntimeError(_TARGET_KIND)


def _job_paths(job: Mapping[str, Any]) -> tuple[Path, Path]:
    if _TARGET_KIND == "coco_oregon":
        return _TARGET._bootstrap_output(job)
    if _TARGET_KIND == "voc":
        return _TARGET._job_paths(job)
    if _TARGET_KIND == "native":
        return _TARGET.bootstrap_path(job)
    raise RuntimeError(_TARGET_KIND)


def _source_read_json(path: Path) -> Any:
    if _TARGET_KIND == "voc":
        return _TARGET.DATA.read_json(path)
    return _TARGET.read_json(path)


def _source_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if _TARGET_KIND == "voc":
        _TARGET.DATA.atomic_json(path, payload)
    else:
        _TARGET.atomic_json(path, payload)


def _rng(job: Mapping[str, Any]) -> np.random.Generator:
    if _TARGET_KIND == "coco_oregon":
        return _TARGET._bootstrap_rng(str(job["id"]))
    if _TARGET_KIND == "voc":
        return _TARGET._rng(str(job["id"]))
    if _TARGET_KIND == "native":
        return _TARGET.bootstrap_rng(job)
    raise RuntimeError(_TARGET_KIND)


def _sample(rng: np.random.Generator, job: Mapping[str, Any]) -> list[int]:
    if _TARGET_KIND == "coco_oregon":
        return _TARGET._bootstrap_sample(rng, str(job["unit"]))
    if _TARGET_KIND == "voc":
        return _TARGET._sample(rng)
    if _TARGET_KIND == "native":
        image_ids = _state()["image_ids"]
        indices = rng.integers(0, len(image_ids), size=len(image_ids))
        return [int(image_ids[int(index)]) for index in indices]
    raise RuntimeError(_TARGET_KIND)


def _core_value(job: Mapping[str, Any], sampled: Sequence[int]) -> float:
    state = _state()
    if str(job["kind"]) == "nested_random":
        candidate = state["candidate_predictions"][int(job["candidate_index"])]
        comparators = [state["random_predictions"][int(index)] for index in job["random_indices"]]
        if len(comparators) != int(_TARGET.PROTOCOL["bootstrap"]["random_seeds"]):
            raise RuntimeError("nested Random prediction population changed")
        candidate_ap: float | None = None
        comparator_ap = []
        for comparator in comparators:
            gt, left, right = state["core"]._bootstrap_remap(
                state["ground_truth"], candidate, comparator, sampled
            )
            if candidate_ap is None:
                candidate_ap = float(state["core"]._in_memory_ap(gt, left))
            comparator_ap.append(float(state["core"]._in_memory_ap(gt, right)))
        if candidate_ap is None:
            raise RuntimeError("empty nested Random comparison")
        return float(candidate_ap - np.mean(comparator_ap))
    candidate, comparator = state["pairs"][int(job["pair_index"])]
    gt, left, right = state["core"]._bootstrap_remap(
        state["ground_truth"], candidate, comparator, sampled
    )
    return float(state["core"]._in_memory_ap(gt, left) - state["core"]._in_memory_ap(gt, right))


def _voc_value(job: Mapping[str, Any], sampled: Sequence[int]) -> float:
    state = _state()
    if _VOC_FAST is None:
        raise RuntimeError("fast VOC bootstrap state was not initialized")
    if str(job["kind"]) == "nested_random":
        candidate = state["candidate_predictions"][int(job["candidate_index"])]
        comparators = [state["random_predictions"][int(index)] for index in job["random_indices"]]
        if len(comparators) != int(_TARGET.PROTOCOL["random"]["seeds"]):
            raise RuntimeError("nested Random prediction population changed")
        candidate_ap = _VOC_FAST.evaluate(candidate, sampled)
        comparator_ap = [_VOC_FAST.evaluate(comparator, sampled) for comparator in comparators]
        return float(candidate_ap - np.mean(comparator_ap))
    candidate, comparator = state["pairs"][int(job["pair_index"])]
    return float(
        _VOC_FAST.evaluate(candidate, sampled) - _VOC_FAST.evaluate(comparator, sampled)
    )


def _native_value(job: Mapping[str, Any], sampled: Sequence[int]) -> float:
    state = _state()
    left = state["cavs"][float(job["budget"])]
    comparators = state["comparators"][float(job["budget"])][str(job["right"])]
    left_ap: float | None = None
    right_ap = []
    for comparator in comparators:
        gt, candidate, right = _TARGET.AC.BOOTSTRAP_CORE._bootstrap_remap(
            state["ground_truth"], left, comparator, sampled
        )
        if left_ap is None:
            left_ap = float(_TARGET.AC.BOOTSTRAP_CORE._in_memory_ap(gt, candidate))
        right_ap.append(float(_TARGET.AC.BOOTSTRAP_CORE._in_memory_ap(gt, right)))
    if left_ap is None:
        raise RuntimeError("empty native comparison")
    return float(left_ap - np.mean(right_ap))


def _value(job: Mapping[str, Any], sampled: Sequence[int]) -> float:
    if _TARGET_KIND == "coco_oregon":
        return _core_value(job, sampled)
    if _TARGET_KIND == "voc":
        return _voc_value(job, sampled)
    if _TARGET_KIND == "native":
        return _native_value(job, sampled)
    raise RuntimeError(_TARGET_KIND)


def _compute_chunk(task: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    job, start, stop = task
    rng = _rng(job)
    for _ in range(start):
        _sample(rng, job)
    started = time.time()
    values = []
    for _ in range(start, stop):
        sampled = _sample(rng, job)
        value = float(_value(job, sampled))
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite bootstrap value: {job['id']}/{value}")
        values.append(value)
    return {
        "job_id": str(job["id"]),
        "start": start,
        "stop": stop,
        "values": values,
        "pid": os.getpid(),
        "elapsed_seconds": time.time() - started,
    }


def _summary(job: Mapping[str, Any], values: Sequence[float]) -> dict[str, Any]:
    if _TARGET_KIND == "coco_oregon":
        result = _TARGET._bootstrap_summary(job, values)
        if str(job["kind"]) == "nested_random":
            result["random_seed_count"] = len(job["random_indices"])
            result["random_nested_per_resample"] = True
        return result
    if _TARGET_KIND == "voc":
        result = _TARGET._summary(job, values)
        if str(job["kind"]) == "nested_random":
            result["random_seed_count"] = len(job["random_indices"])
            result["random_nested_per_resample"] = True
        return result
    if _TARGET_KIND == "native":
        array = np.asarray(values, dtype=np.float64)
        comparators = _state()["comparators"][float(job["budget"])][str(job["right"])]
        result = {
            **dict(job),
            "replicates": len(values),
            "mean_delta": float(array.mean()),
            "median_delta": float(np.median(array)),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
            "random_seed_count": len(comparators) if str(job["kind"]) == "nested_random" else "",
            "final_used_for_training_gate_or_selection": False,
        }
        if str(job["kind"]) == "nested_random":
            result["random_nested_per_resample"] = True
        return result
    raise RuntimeError(_TARGET_KIND)


class FastVocBootstrap:
    """Exact pre-indexed execution of the frozen VOC metric for bootstrap samples."""

    def __init__(self, target: Any, ground_truth: Mapping[str, Any]) -> None:
        self.target = target
        self.categories = tuple(int(row["id"]) for row in ground_truth["categories"])
        grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
        for row in ground_truth["annotations"]:
            grouped[(int(row["image_id"]), int(row["category_id"]))].append(row)
        self.truth: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for key, rows in grouped.items():
            boxes = np.asarray(
                [target.METRIC._xywh_to_xyxy(row["bbox"]) for row in rows],
                dtype=np.float64,
            ).reshape((-1, 4))
            difficult = np.asarray(
                [bool(row.get("difficult", row.get("ignore", False))) for row in rows],
                dtype=bool,
            )
            self.truth[key] = {"boxes": boxes, "difficult": difficult}
        self.maximum_ground_truth = {
            category: 1
            + max(
                (
                    len(value["difficult"])
                    for (unused_image_id, cat), value in self.truth.items()
                    if cat == category
                ),
                default=0,
            )
            for category in self.categories
        }
        self.compiled: dict[int, dict[int, dict[int, dict[str, np.ndarray]]]] = {}

    def compile(self, predictions: Sequence[Mapping[str, Any]]) -> None:
        token = id(predictions)
        if token in self.compiled:
            return
        grouped: dict[tuple[int, int], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for order, row in enumerate(predictions):
            category = int(row["category_id"])
            if category in self.categories:
                grouped[(category, int(row["image_id"]))].append((order, row))
        by_category: dict[int, dict[int, dict[str, np.ndarray]]] = {
            category: {} for category in self.categories
        }
        for (category, image_id), rows in grouped.items():
            truth = self.truth.get((image_id, category))
            boxes = np.asarray([row["bbox"] for _, row in rows], dtype=np.float64).reshape((-1, 4))
            best_iou = np.zeros(len(rows), dtype=np.float64)
            best_gt = np.full(len(rows), -1, dtype=np.int32)
            best_difficult = np.zeros(len(rows), dtype=bool)
            if truth is not None and len(truth["boxes"]):
                prediction_xyxy = np.column_stack(
                    (
                        boxes[:, 0],
                        boxes[:, 1],
                        boxes[:, 0] + np.maximum(0.0, boxes[:, 2]),
                        boxes[:, 1] + np.maximum(0.0, boxes[:, 3]),
                    )
                )
                truth_boxes = truth["boxes"]
                left_top = np.maximum(prediction_xyxy[:, None, :2], truth_boxes[None, :, :2])
                right_bottom = np.minimum(
                    prediction_xyxy[:, None, 2:], truth_boxes[None, :, 2:]
                )
                intersection_size = np.maximum(0.0, right_bottom - left_top)
                intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
                prediction_size = np.maximum(0.0, prediction_xyxy[:, 2:] - prediction_xyxy[:, :2])
                prediction_area = prediction_size[:, 0] * prediction_size[:, 1]
                truth_size = np.maximum(0.0, truth_boxes[:, 2:] - truth_boxes[:, :2])
                truth_area = truth_size[:, 0] * truth_size[:, 1]
                union = prediction_area[:, None] + truth_area[None, :] - intersection
                overlaps = np.divide(
                    intersection,
                    union,
                    out=np.zeros_like(intersection),
                    where=union > 0,
                )
                match = np.argmax(overlaps, axis=1).astype(np.int32)
                best_iou = overlaps[np.arange(len(rows)), match]
                best_gt = match
                best_difficult = truth["difficult"][match]
            by_category[category][image_id] = {
                "score": np.asarray([float(row.get("score", 0.0)) for _, row in rows]),
                "bbox": boxes,
                "order": np.asarray([order for order, _ in rows], dtype=np.int64),
                "best_iou": best_iou,
                "best_gt": best_gt,
                "best_difficult": best_difficult,
            }
        self.compiled[token] = by_category

    def compile_state(self, state: Mapping[str, Any]) -> None:
        for left, right in state["pairs"]:
            self.compile(left)
            self.compile(right)
        for predictions in state["candidate_predictions"]:
            self.compile(predictions)
        for predictions in state["random_predictions"]:
            self.compile(predictions)

    def _category_ap(
        self,
        groups: Mapping[int, Mapping[str, np.ndarray]],
        sampled: Sequence[int],
        category: int,
    ) -> list[float] | None:
        positives = 0
        score_parts: list[np.ndarray] = []
        occurrence_parts: list[np.ndarray] = []
        bbox_parts: list[np.ndarray] = []
        order_parts: list[np.ndarray] = []
        iou_parts: list[np.ndarray] = []
        gt_parts: list[np.ndarray] = []
        difficult_parts: list[np.ndarray] = []
        for occurrence, source_id in enumerate(map(int, sampled), start=1):
            truth = self.truth.get((source_id, category))
            if truth is not None:
                positives += int(np.count_nonzero(~truth["difficult"]))
            rows = groups.get(source_id)
            if rows is None or not len(rows["score"]):
                continue
            count = len(rows["score"])
            score_parts.append(rows["score"])
            occurrence_parts.append(np.full(count, occurrence, dtype=np.int64))
            bbox_parts.append(rows["bbox"])
            order_parts.append(rows["order"])
            iou_parts.append(rows["best_iou"])
            gt_parts.append(rows["best_gt"])
            difficult_parts.append(rows["best_difficult"])
        if positives == 0:
            return None
        if not score_parts:
            return [0.0 for _ in self.target.METRIC.DEFAULT_IOUS]
        scores = np.concatenate(score_parts)
        occurrences = np.concatenate(occurrence_parts)
        boxes = np.concatenate(bbox_parts)
        original_order = np.concatenate(order_parts)
        best_iou = np.concatenate(iou_parts)
        best_gt = np.concatenate(gt_parts)
        best_difficult = np.concatenate(difficult_parts)
        order = np.lexsort(
            (
                original_order,
                boxes[:, 3],
                boxes[:, 2],
                boxes[:, 1],
                boxes[:, 0],
                occurrences,
                -scores,
            )
        )
        occurrences = occurrences[order]
        best_iou = best_iou[order]
        best_gt = best_gt[order]
        best_difficult = best_difficult[order]
        maximum_ground_truth = self.maximum_ground_truth[category]
        output = []
        for threshold in self.target.METRIC.DEFAULT_IOUS:
            difficult = (best_iou >= threshold) & best_difficult
            eligible = (best_iou >= threshold) & (~best_difficult) & (best_gt >= 0)
            true_positive = np.zeros(len(order), dtype=bool)
            eligible_indices = np.flatnonzero(eligible)
            if len(eligible_indices):
                keys = occurrences[eligible_indices] * maximum_ground_truth + best_gt[eligible_indices]
                _, first = np.unique(keys, return_index=True)
                true_positive[eligible_indices[first]] = True
            keep = ~difficult
            tp = np.cumsum(true_positive[keep].astype(np.float64))
            fp = np.cumsum((~true_positive[keep]).astype(np.float64))
            recall = tp / float(positives) if len(tp) else np.empty(0, dtype=np.float64)
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            output.append(float(self.target.METRIC._interpolated_ap(recall, precision, 101)))
        return output

    def evaluate(self, predictions: Sequence[Mapping[str, Any]], sampled: Sequence[int]) -> float:
        if id(predictions) not in self.compiled:
            self.compile(predictions)
        compiled = self.compiled[id(predictions)]
        by_threshold = [[] for _ in self.target.METRIC.DEFAULT_IOUS]
        for category in self.categories:
            values = self._category_ap(compiled[category], sampled, category)
            if values is None:
                continue
            for target, value in zip(by_threshold, values):
                target.append(value)
        threshold_map = [float(np.mean(values)) if values else 0.0 for values in by_threshold]
        return float(np.mean(threshold_map))


class AcceleratedPool:
    def __init__(
        self,
        *,
        workers: int,
        standard_chunk: int,
        nested_chunk: int,
        expected_jobs: int,
        expected_nested: int,
        verify_existing: bool,
        verify_only: bool,
    ) -> None:
        self.workers = workers
        self.standard_chunk = standard_chunk
        self.nested_chunk = nested_chunk
        self.expected_jobs = expected_jobs
        self.expected_nested = expected_nested
        self.verify_existing = verify_existing
        self.verify_only = verify_only

    def __enter__(self) -> "AcceleratedPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False

    def _load_values(self, job: Mapping[str, Any]) -> list[float]:
        path, partial = _job_paths(job)
        if path.is_file():
            return []
        if not partial.is_file():
            return []
        payload = _source_read_json(partial)
        values = list(map(float, payload.get("values", ())))
        total = int(PROTOCOL["replicates_per_job"])
        if len(values) > total or any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"invalid partial bootstrap: {partial}")
        return values

    def _verify(self, jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not self.verify_existing:
            return {"performed": False}
        for job in jobs:
            values = self._load_values(job)
            if not values:
                continue
            index = 0
            result = _compute_chunk((dict(job), index, index + 1))
            actual = float(result["values"][0])
            expected = float(values[index])
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError(
                    f"replicate equivalence failed: {job['id']}[{index}] "
                    f"expected={expected:.17g} actual={actual:.17g}"
                )
            report = {
                "performed": True,
                "job_id": str(job["id"]),
                "replicate": index,
                "expected": expected,
                "actual": actual,
                "absolute_error": abs(actual - expected),
            }
            print(f"[verify] {json.dumps(report, ensure_ascii=False)}", flush=True)
            return report
        return {"performed": False, "reason": "no existing partial replicate"}

    def imap_unordered(
        self, unused_worker: Any, jobs: Iterable[Mapping[str, Any]], chunksize: int = 1
    ) -> Iterator[dict[str, Any]]:
        del unused_worker, chunksize
        job_list = [dict(job) for job in jobs]
        nested = sum(str(job["kind"]) == "nested_random" for job in job_list)
        if len(job_list) != self.expected_jobs or nested != self.expected_nested:
            raise RuntimeError(
                f"registered job matrix changed: jobs={len(job_list)}/{self.expected_jobs}, "
                f"nested={nested}/{self.expected_nested}"
            )
        global _VOC_FAST
        if _TARGET_KIND == "voc":
            _VOC_FAST = FastVocBootstrap(_TARGET, _state()["ground_truth"])
        verification = self._verify(job_list)
        atomic_json(ROOT / f"runs/{_TARGET_NAME}/verification.json", verification)
        if self.verify_only:
            raise VerificationOnly(_TARGET_NAME)
        if _TARGET_KIND == "voc":
            started = time.time()
            _VOC_FAST.compile_state(_state())
            print(f"[voc-index] compiled in {time.time() - started:.1f}s", flush=True)

        total = int(PROTOCOL["replicates_per_job"])
        complete_results: dict[str, dict[str, Any]] = {}
        progress: dict[str, dict[str, Any]] = {}
        tasks: list[tuple[dict[str, Any], int, int]] = []
        for job in job_list:
            job_id = str(job["id"])
            path, _ = _job_paths(job)
            if path.is_file():
                result = _source_read_json(path)
                if int(result.get("replicates", -1)) != total:
                    raise RuntimeError(f"completed job has wrong replicate count: {path}")
                complete_results[job_id] = result
                continue
            values = self._load_values(job)
            progress[job_id] = {"job": job, "values": values, "pending": {}}
            size = self.nested_chunk if str(job["kind"]) == "nested_random" else self.standard_chunk
            tasks.extend((job, start, stop) for start, stop in chunk_ranges(len(values), total, size))
        print(
            f"[plan] target={_TARGET_NAME} jobs={len(job_list)} complete={len(complete_results)} "
            f"chunks={len(tasks)} workers={min(self.workers, max(1, len(tasks)))}",
            flush=True,
        )
        started = time.time()
        finished_chunks = 0
        if tasks:
            worker_count = min(self.workers, len(tasks), max(1, os.cpu_count() or 1))
            with REAL_FORK_CONTEXT.Pool(processes=worker_count) as pool:
                for result in pool.imap_unordered(_compute_chunk, tasks, chunksize=1):
                    job_id = str(result["job_id"])
                    item = progress[job_id]
                    changed = merge_contiguous(
                        item["values"], item["pending"], int(result["start"]), result["values"]
                    )
                    if changed:
                        _, partial = _job_paths(item["job"])
                        _source_atomic_json(partial, {"values": item["values"]})
                    finished_chunks += 1
                    if finished_chunks == 1 or finished_chunks % max(1, worker_count) == 0:
                        elapsed = time.time() - started
                        print(
                            f"[progress] target={_TARGET_NAME} chunks={finished_chunks}/{len(tasks)} "
                            f"elapsed={elapsed:.1f}s",
                            flush=True,
                        )
        for job_id, item in progress.items():
            values = item["values"]
            if item["pending"] or len(values) != total:
                raise RuntimeError(
                    f"bootstrap prefix incomplete after workers: {job_id}/{len(values)}/{item['pending'].keys()}"
                )
            job = item["job"]
            path, partial = _job_paths(job)
            result = _summary(job, values)
            if int(result.get("replicates", -1)) != total:
                raise RuntimeError(f"summary replicate count changed: {job_id}")
            _source_atomic_json(path, result)
            partial.unlink(missing_ok=True)
            complete_results[job_id] = result
        if len(complete_results) != len(job_list):
            raise RuntimeError("bootstrap result matrix incomplete")
        return iter(complete_results[str(job["id"])] for job in job_list)


class FakeContext:
    def __init__(self, pool: AcceleratedPool) -> None:
        self.pool = pool

    def Pool(self, *args: Any, **kwargs: Any) -> AcceleratedPool:
        del args, kwargs
        return self.pool


def run_target(args: argparse.Namespace) -> dict[str, Any]:
    global _TARGET, _TARGET_KIND, _TARGET_NAME
    config, experiment, source_script = validate_source(args.target)
    _TARGET_NAME = args.target
    _TARGET_KIND = str(config["kind"])
    _TARGET = load_module(f"server_completion_{args.target}", source_script)
    workers = int(args.workers)
    maximum = int(PROTOCOL["maximum_workers"])
    if not 1 <= workers <= min(maximum, max(1, os.cpu_count() or 1)):
        raise ValueError(f"workers must be in [1, {min(maximum, os.cpu_count() or 1)}]")
    pool = AcceleratedPool(
        workers=workers,
        standard_chunk=int(args.standard_chunk),
        nested_chunk=int(args.nested_chunk),
        expected_jobs=int(config["expected_jobs"]),
        expected_nested=int(config["expected_nested_jobs"]),
        verify_existing=not args.skip_existing_verification,
        verify_only=bool(args.verify_only),
    )
    _TARGET.mp = SimpleNamespace(get_context=lambda unused_method: FakeContext(pool))
    started = time.time()
    try:
        if _TARGET_KIND == "coco_oregon":
            canonical = _TARGET.run_bootstrap(str(config["dataset"]))
        elif _TARGET_KIND == "voc":
            canonical = _TARGET.run()
        elif _TARGET_KIND == "native":
            canonical = _TARGET.run_bootstrap()
        else:
            raise RuntimeError(_TARGET_KIND)
    except VerificationOnly:
        return {
            "complete": False,
            "verification_only": True,
            "target": args.target,
            "elapsed_seconds": time.time() - started,
        }
    report = {
        "complete": True,
        "target": args.target,
        "source_experiment": str(experiment.relative_to(REPO)),
        "source_script_sha256": file_sha256(source_script),
        "workers": workers,
        "standard_chunk_replicates": int(args.standard_chunk),
        "nested_chunk_replicates": int(args.nested_chunk),
        "elapsed_seconds": time.time() - started,
        "canonical_report": canonical,
        "source_experiments_modified": False,
    }
    atomic_json(ROOT / f"runs/{args.target}/completion.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=tuple(PROTOCOL["targets"]), required=True)
    parser.add_argument("--workers", type=int, default=int(PROTOCOL["default_workers"]))
    parser.add_argument(
        "--standard-chunk", type=int, default=int(PROTOCOL["standard_chunk_replicates"])
    )
    parser.add_argument(
        "--nested-chunk", type=int, default=int(PROTOCOL["nested_chunk_replicates"])
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-existing-verification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_target(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
