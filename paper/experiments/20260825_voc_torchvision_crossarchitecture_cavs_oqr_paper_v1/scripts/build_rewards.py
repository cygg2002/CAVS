#!/usr/bin/env python3
"""Build fold-contained contextual VOC AP rewards for independent O/OQ/OQR learners."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


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


CORE = load_module("voc_gluoncv_reward_core", ROOT / "scripts/core.py")
DATA = load_module("voc_gluoncv_reward_data", ROOT / "scripts/data_core.py")
METRIC = load_module("voc_gluoncv_reward_metric", ROOT / "scripts/voc_metric.py")


def contextual_reward(
    evaluator: Any,
    image_ids: Sequence[int],
    baseline_predictions: Sequence[Mapping[str, Any]],
    candidate_predictions: Sequence[Mapping[str, Any]],
    *,
    baseline_metrics: Mapping[str, float] | None = None,
) -> dict[str, float]:
    baseline = (
        dict(baseline_metrics)
        if baseline_metrics is not None
        else evaluator.evaluate(baseline_predictions, image_ids)
    )
    candidate = evaluator.evaluate(candidate_predictions, image_ids)
    scale = float(len(image_ids))
    return {
        "ap50_95": (float(candidate["map50_95"]) - float(baseline["map50_95"])) * scale,
        "ap50": (float(candidate["map50"]) - float(baseline["map50"])) * scale,
        "baseline_ap50_95": float(baseline["map50_95"]),
        "candidate_ap50_95": float(candidate["map50_95"]),
    }


def build_reward_views(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[int, dict[str, dict[str, float]]]]:
    by_id = {int(row["image_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("duplicate contextual reward image")
    actions_by_space = {
        "O": ("original",),
        "OQ": ("original", "q90"),
        "OQR": ("original", "q90", "current@official"),
    }
    views = {
        action_space: {
            image_id: {
                action: float(row["rewards"][action]["ap50_95"])
                for action in actions
            }
            for image_id, row in by_id.items()
        }
        for action_space, actions in actions_by_space.items()
    }
    metric_map = {
        image_id: {
            action: {
                "ap50": float(values["ap50"]),
                "ap50_95": float(values["ap50_95"]),
            }
            for action, values in row["rewards"].items()
        }
        for image_id, row in by_id.items()
    }
    return views, metric_map


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid reward checkpoint line {number}: {path}") from error
    return rows


def _load_role(role: str) -> dict[str, Any]:
    CORE.reward_roles(role)
    common = RUNS / f"common/{role}"
    manifest = DATA.read_json(common / "source_manifest.json")
    samples = DATA.read_json(common / "samples.json")
    image_ids = [int(row["image_id"]) for row in samples]
    actions = {
        action: DATA.group_predictions(DATA.read_json(path), image_ids=image_ids)
        for action, path in manifest["actions"].items()
    }
    return {
        "gt": Path(manifest["gt"]),
        "samples": samples,
        "actions": actions,
    }


def _folds(role: str, samples: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    if role == "calibration":
        return [sorted(int(row["image_id"]) for row in samples)]
    identities = [str(row["identity"]) for row in samples]
    identity_folds = CORE.deterministic_folds(
        identities,
        fold_count=int(PROTOCOL["fold_count"]),
        seed=int(PROTOCOL["fold_seed"]),
    )
    by_identity = {str(row["identity"]): int(row["image_id"]) for row in samples}
    return [[by_identity[identity] for identity in fold] for fold in identity_folds]


_STATE: dict[str, Any] = {}


def _init_worker(gt_path: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _STATE["evaluator"] = METRIC.VocMetricBackend(Path(gt_path))


def _flatten(action: str, image_ids: Sequence[int]) -> list[dict[str, Any]]:
    groups = _STATE["actions"][action]
    return [dict(row) for image_id in image_ids for row in groups.get(int(image_id), ())]


def _worker(image_id: int) -> dict[str, Any]:
    target = int(image_id)
    context_ids = list(map(int, _STATE["contexts"][target]))
    evaluation_ids = [*context_ids, target]
    context_edge = _flatten("edge", context_ids)
    baseline = [*context_edge, *_flatten("edge", [target])]
    baseline_metrics = _STATE["evaluator"].evaluate(baseline, evaluation_ids)
    rewards = {}
    for action in ("original", "q90", "current@official"):
        available = bool(_STATE["availability"][target].get(action, False))
        target_predictions = _flatten(action, [target]) if available else _flatten("edge", [target])
        candidate = [*context_edge, *target_predictions]
        rewards[action] = contextual_reward(
            _STATE["evaluator"],
            evaluation_ids,
            baseline,
            candidate,
            baseline_metrics=baseline_metrics,
        )
    return {
        "image_id": target,
        "fold_index": int(_STATE["membership"][target]),
        "context_image_ids": context_ids,
        "context_size": len(context_ids),
        "rewards": rewards,
    }


def run_role(role: str) -> dict[str, Any]:
    CORE.reward_roles(role)
    output = RUNS / f"rewards/{role}"
    completion = output / "completion.json"
    rows_path = output / "rows.jsonl"
    o_path = output / "reward_map_O.pkl"
    oq_path = output / "reward_map_OQ.pkl"
    oqr_path = output / "reward_map_OQR.pkl"
    metric_path = output / "metric_map.pkl"
    if completion.is_file():
        report = DATA.read_json(completion)
        for key, path in (
            ("rows_sha256", rows_path),
            ("reward_map_O_sha256", o_path),
            ("reward_map_OQ_sha256", oq_path),
            ("reward_map_OQR_sha256", oqr_path),
            ("metric_map_sha256", metric_path),
        ):
            if report.get(key) != DATA.file_sha256(path):
                raise RuntimeError(f"completed reward artifact changed: {role}/{path.name}")
        return report
    data = _load_role(role)
    folds = _folds(role, data["samples"])
    contexts = DATA.fold_contained_contexts(
        folds,
        context_size=int(PROTOCOL["reward"]["context_size"]),
        seed=int(PROTOCOL["fold_seed"]),
    )
    membership = {
        int(image_id): fold_index
        for fold_index, fold in enumerate(folds)
        for image_id in fold
    }
    expected = {int(row["image_id"]) for row in data["samples"]}
    if set(contexts) != expected or set(membership) != expected:
        raise RuntimeError(f"reward fold population changed: {role}")
    output.mkdir(parents=True, exist_ok=True)
    DATA.atomic_json(output / "folds.json", {"folds": folds})
    DATA.atomic_json(
        output / "contexts.json", {str(key): value for key, value in contexts.items()}
    )
    existing_rows = _read_jsonl(rows_path)
    existing = {int(row["image_id"]) for row in existing_rows}
    if len(existing) != len(existing_rows) or not existing <= expected:
        raise RuntimeError(f"invalid reward checkpoint: {role}")
    tasks = sorted(expected - existing)
    _STATE.clear()
    _STATE.update(
        actions=data["actions"],
        contexts=contexts,
        membership=membership,
        availability={
            int(row["image_id"]): {
                "original": True,
                "q90": bool(row.get("available", {}).get("q90", False)),
                "current@official": bool(
                    row.get("available", {}).get("current@official", False)
                ),
            }
            for row in data["samples"]
        },
    )
    workers = min(
        int(PROTOCOL["reward"]["workers"]),
        max(1, os.cpu_count() or 1),
        max(1, len(tasks)),
    )
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        if tasks:
            with mp.get_context("fork").Pool(
                workers, initializer=_init_worker, initargs=(str(data["gt"]),)
            ) as pool:
                for count, row in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    if count % int(PROTOCOL["reward"]["checkpoint_interval"]) == 0:
                        handle.flush()
                    if count % 500 == 0 or count == len(tasks):
                        print(f"[reward {role}] {len(existing) + count}/{len(expected)}", flush=True)
    rows = _read_jsonl(rows_path)
    if len(rows) != len(expected) or {int(row["image_id"]) for row in rows} != expected:
        raise RuntimeError(f"incomplete contextual rewards: {role}")
    reward_views, metric_map = build_reward_views(rows)
    DATA.atomic_pickle(o_path, reward_views["O"])
    DATA.atomic_pickle(oq_path, reward_views["OQ"])
    DATA.atomic_pickle(oqr_path, reward_views["OQR"])
    DATA.atomic_pickle(metric_path, metric_map)
    report = {
        "complete": True,
        "role": role,
        "images": len(rows),
        "actions": ["original", "q90", "current@official"],
        "folds": len(folds),
        "context_size": int(PROTOCOL["reward"]["context_size"]),
        "fold_contained": True,
        "workers": workers,
        "elapsed_seconds_this_invocation": time.time() - started,
        "rows_sha256": DATA.file_sha256(rows_path),
        "reward_map_O_sha256": DATA.file_sha256(o_path),
        "reward_map_OQ_sha256": DATA.file_sha256(oq_path),
        "reward_map_OQR_sha256": DATA.file_sha256(oqr_path),
        "metric_map_sha256": DATA.file_sha256(metric_path),
        "final_accessed": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def run_all() -> dict[str, Any]:
    reports = {role: run_role(role) for role in CORE.reward_roles("all")}
    report = {
        "complete": True,
        "roles": {role: int(value["images"]) for role, value in reports.items()},
        "final_accessed": False,
    }
    DATA.atomic_json(RUNS / "rewards/report.json", report)
    DATA.atomic_json(RUNS / "rewards/completion.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("fit", "calibration", "all"), default="all")
    args = parser.parse_args()
    result = run_all() if args.role == "all" else run_role(args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
