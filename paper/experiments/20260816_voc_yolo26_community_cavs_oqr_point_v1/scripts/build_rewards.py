#!/usr/bin/env python3
"""Generate fold-contained VOC contextual AP rewards for cloud actions."""

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


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(REPO))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_point_reward_data", ROOT / "scripts/data_core.py")
CORE = load_module("voc_point_reward_protocol", ROOT / "scripts/protocol_core.py")
PAPER = load_module(
    "voc_point_reward_paper",
    REPO / "paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/paper_core.py",
)
METRIC = load_module(
    "voc_point_reward_metric", REPO / "paper/experiments/CA_AORIC/ca_aoric.py"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid reward line {number}: {path}") from error
    return rows


def load_role(role: str) -> dict[str, Any]:
    DATA.reward_roles(role)
    root = RUNS / f"common/{role}"
    manifest = DATA.read_json(root / "source_manifest.json")
    samples = DATA.read_json(root / "samples.json")
    image_ids = [int(sample["image_id"]) for sample in samples]
    actions = {
        action: DATA.group_predictions(DATA.read_json(path), image_ids=image_ids)
        for action, path in manifest["action_sources"].items()
    }
    return {
        "gt": Path(manifest["gt"]),
        "samples": samples,
        "actions": actions,
    }


def reward_folds(role: str, samples: list[dict[str, Any]]) -> list[list[int]]:
    if role == "calibration":
        return [sorted(int(sample["image_id"]) for sample in samples)]
    identities = [str(sample["identity"]) for sample in samples]
    identity_folds = CORE.deterministic_folds(
        identities,
        fold_count=int(PROTOCOL["fold_count"]),
        seed=int(PROTOCOL["fold_seed"]),
    )
    id_by_identity = {str(sample["identity"]): int(sample["image_id"]) for sample in samples}
    return [[id_by_identity[identity] for identity in fold] for fold in identity_folds]


_STATE: dict[str, Any] = {}


def _init_worker(gt_path: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _STATE["metric"] = METRIC.CocoMetricBackend(Path(gt_path))


def _worker(task: tuple[int, str]) -> dict[str, Any]:
    image_id, action = int(task[0]), str(task[1])
    context = [
        {"image_id": value, "edge_predictions": _STATE["actions"]["edge"].get(value, [])}
        for value in _STATE["contexts"][image_id]
    ]
    target = {
        "image_id": image_id,
        "edge_predictions": _STATE["actions"]["edge"].get(image_id, []),
        "cloud_predictions": _STATE["actions"][action].get(image_id, []),
    }
    row = PAPER.dual_metric_context_reward(_STATE["metric"], context, target)
    row.update(
        action_id=action,
        reward=float(row["reward_ap50_95"]),
        fold_index=int(_STATE["membership"][image_id]),
    )
    return row


def run_action(
    role: str,
    action: str,
    data: Mapping[str, Any],
    contexts: Mapping[int, list[int]],
    membership: Mapping[int, int],
) -> dict[str, Any]:
    output = RUNS / f"rewards/{role}/{action.replace('@', '__')}"
    completion = output / "completion.json"
    rows_path = output / "rows.jsonl"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("rows_sha256") != DATA.file_sha256(rows_path):
            raise RuntimeError(f"reward shard changed: {role}/{action}")
        return report
    existing_rows = _read_jsonl(rows_path)
    existing = {int(row["target_image_id"]) for row in existing_rows}
    expected = {int(sample["image_id"]) for sample in data["samples"]}
    if len(existing) != len(existing_rows) or not existing <= expected:
        raise RuntimeError(f"invalid reward checkpoint: {role}/{action}")
    tasks = [(image_id, action) for image_id in sorted(expected - existing)]
    _STATE.clear()
    _STATE.update(actions=data["actions"], contexts=dict(contexts), membership=dict(membership))
    output.mkdir(parents=True, exist_ok=True)
    workers = min(int(PROTOCOL["reward"]["workers"]), max(1, os.cpu_count() or 1))
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        if tasks:
            with mp.get_context("fork").Pool(
                workers, initializer=_init_worker, initargs=(str(data["gt"]),)
            ) as pool:
                for count, row in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2), 1):
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    if count % int(PROTOCOL["reward"]["checkpoint_interval"]) == 0:
                        handle.flush()
                    if count % 500 == 0 or count == len(tasks):
                        print(f"[reward {role}/{action}] {len(existing) + count}/{len(expected)}", flush=True)
    rows = _read_jsonl(rows_path)
    if len(rows) != len(expected) or {int(row["target_image_id"]) for row in rows} != expected:
        raise RuntimeError(f"incomplete rewards: {role}/{action}")
    report = {
        "complete": True,
        "role": role,
        "action_id": action,
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "rows_sha256": DATA.file_sha256(rows_path),
        "final_accessed": False,
    }
    DATA.atomic_json(completion, report)
    return report


def run_role(role: str) -> dict[str, Any]:
    DATA.reward_roles(role)
    output = RUNS / f"rewards/{role}"
    completion = output / "completion.json"
    reward_path = output / "reward_map.pkl"
    metric_path = output / "metric_map.pkl"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("reward_map_sha256") != DATA.file_sha256(reward_path):
            raise RuntimeError(f"reward map changed: {role}")
        return report
    data = load_role(role)
    folds = reward_folds(role, data["samples"])
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
    DATA.atomic_json(output / "folds.json", {"folds": folds})
    DATA.atomic_json(output / "contexts.json", {str(key): value for key, value in contexts.items()})
    reward_map = {int(sample["image_id"]): {} for sample in data["samples"]}
    metric_map = {int(sample["image_id"]): {} for sample in data["samples"]}
    for action in ("original", "q90", "current@640"):
        run_action(role, action, data, contexts, membership)
        for row in _read_jsonl(output / action.replace("@", "__") / "rows.jsonl"):
            image_id = int(row["target_image_id"])
            reward_map[image_id][action] = float(row["reward_ap50_95"])
            metric_map[image_id][action] = {
                "ap50": float(row["reward_ap50"]),
                "ap50_95": float(row["reward_ap50_95"]),
            }
    DATA.atomic_pickle(reward_path, reward_map)
    DATA.atomic_pickle(metric_path, metric_map)
    report = {
        "complete": True,
        "role": role,
        "images": len(reward_map),
        "actions": 3,
        "tasks": len(reward_map) * 3,
        "folds": len(folds),
        "fold_contained": True,
        "reward_map_sha256": DATA.file_sha256(reward_path),
        "metric_map_sha256": DATA.file_sha256(metric_path),
        "final_accessed": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def run_all() -> dict[str, Any]:
    reports = {role: run_role(role) for role in DATA.reward_roles("all")}
    report = {"complete": True, "roles": {role: value["tasks"] for role, value in reports.items()}, "final_accessed": False}
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
