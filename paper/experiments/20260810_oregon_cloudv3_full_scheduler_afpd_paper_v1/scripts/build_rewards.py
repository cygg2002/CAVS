#!/usr/bin/env python3
"""Generate event-fold-contained contextual AP50-95 rewards for O/Q/R12."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
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


PAPER = load_module(
    "oregon_cloudv3_reward_paper_core",
    REPO
    / "paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/paper_core.py",
)
DATA_CORE = load_module("oregon_cloudv3_reward_data_core", ROOT / "scripts/data_core.py")
PROTOCOL_CORE = load_module(
    "oregon_cloudv3_reward_protocol_core", ROOT / "scripts/protocol_core.py"
)
CA_AORIC = load_module(
    "oregon_cloudv3_reward_metric",
    REPO / "paper/experiments/CA_AORIC/ca_aoric.py",
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


def group_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for raw in read_json(path):
        image_id = int(raw["image_id"])
        output.setdefault(image_id, []).append(
            {
                "image_id": image_id,
                "category_id": int(raw["category_id"]),
                "bbox": list(map(float, raw["bbox"])),
                "score": float(raw.get("score", 1.0)),
            }
        )
    return output


def load_role(role: str) -> dict[str, Any]:
    root = RUNS / f"common/oregon/{role}"
    manifest = read_json(root / "source_manifest.json")
    if manifest.get("cloud_model_sha256") != PROTOCOL["cloud_model"]["sha256"]:
        raise RuntimeError(f"wrong cloud model in common role: {role}")
    samples = read_json(root / "samples.json")
    actions = {
        action_id: group_predictions(Path(path))
        for action_id, path in manifest["action_sources"].items()
    }
    return {
        "root": root,
        "gt": root / "gt.json",
        "samples": samples,
        "actions": actions,
    }


def reward_actions() -> tuple[str, ...]:
    return ("original", "q90", *map(str, PROTOCOL["roi_actions"]))


def reward_folds(role: str, samples: list[dict[str, Any]]) -> list[list[int]]:
    if role == "calibration":
        return [sorted(int(sample["image_id"]) for sample in samples)]
    identity_folds = PROTOCOL_CORE.event_group_folds(
        samples,
        fold_count=int(PROTOCOL["fold_count"]),
        seed=int(PROTOCOL["fold_seed"]),
    )
    id_by_identity = {str(sample["identity"]): int(sample["image_id"]) for sample in samples}
    folds = [[id_by_identity[identity] for identity in fold] for fold in identity_folds]
    membership = {
        int(image_id): fold_index
        for fold_index, fold in enumerate(folds)
        for image_id in fold
    }
    event_membership: dict[str, set[int]] = {}
    for sample in samples:
        event_membership.setdefault(str(sample["event_id"]), set()).add(
            membership[int(sample["image_id"])]
        )
    violations = {event: values for event, values in event_membership.items() if len(values) != 1}
    if violations:
        raise RuntimeError(f"events cross OOF folds: {list(violations)[:3]}")
    return folds


_STATE: dict[str, Any] = {}


def _init_worker(gt_path: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _STATE["metric"] = CA_AORIC.CocoMetricBackend(Path(gt_path))


def _reward_worker(task: tuple[int, str]) -> dict[str, Any]:
    image_id, action_id = int(task[0]), str(task[1])
    context_ids = _STATE["contexts"][image_id]
    edge = _STATE["actions"]["edge"]
    context = [
        {"image_id": value, "edge_predictions": edge.get(value, [])}
        for value in context_ids
    ]
    target = {
        "image_id": image_id,
        "edge_predictions": edge.get(image_id, []),
        "cloud_predictions": _STATE["actions"][action_id].get(image_id, []),
    }
    row = PAPER.dual_metric_context_reward(_STATE["metric"], context, target)
    row.update(
        action_id=action_id,
        reward=float(row["reward_ap50_95"]),
        fold_index=int(_STATE["membership"][image_id]),
    )
    return row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid reward shard line {number}: {path}") from error
    return rows


def run_action(role: str, action_id: str, data: Mapping[str, Any], contexts: Mapping[int, list[int]], membership: Mapping[int, int]) -> dict[str, Any]:
    output = RUNS / f"rewards/{role}/{action_id.replace('@', '__')}"
    completion = output / "completion.json"
    rows_path = output / "rows.jsonl"
    if completion.is_file():
        state = read_json(completion)
        if state.get("rows_sha256") != file_sha256(rows_path):
            raise RuntimeError(f"reward shard changed: {rows_path}")
        return state
    existing_rows = read_jsonl(rows_path)
    existing = {int(row["target_image_id"]) for row in existing_rows}
    expected = {int(sample["image_id"]) for sample in data["samples"]}
    if not existing <= expected or len(existing) != len(existing_rows):
        raise RuntimeError(f"invalid existing reward population: {role}/{action_id}")
    tasks = [(image_id, action_id) for image_id in sorted(expected - existing)]
    _STATE.clear()
    _STATE.update(
        actions=data["actions"],
        contexts=dict(contexts),
        membership=dict(membership),
    )
    output.mkdir(parents=True, exist_ok=True)
    workers = min(int(PROTOCOL["reward"]["workers"]), max(1, os.cpu_count() or 1))
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        if tasks:
            context = mp.get_context("fork")
            with context.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(str(data["gt"]),),
            ) as pool:
                for count, row in enumerate(
                    pool.imap_unordered(_reward_worker, tasks, chunksize=2), start=1
                ):
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    if count % 100 == 0:
                        handle.flush()
                    if count % 500 == 0 or count == len(tasks):
                        print(
                            f"[reward {role}/{action_id}] {len(existing) + count}/{len(expected)}",
                            flush=True,
                        )
    rows = read_jsonl(rows_path)
    ids = {int(row["target_image_id"]) for row in rows}
    if ids != expected or len(rows) != len(expected):
        raise RuntimeError(f"incomplete rewards: {role}/{action_id}")
    report = {
        "complete": True,
        "role": role,
        "action_id": action_id,
        "rows": len(rows),
        "fold_contained": True,
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "rows_sha256": file_sha256(rows_path),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "final_used_for_training_or_selection": False,
    }
    atomic_json(completion, report)
    return report


def run_role(role: str) -> dict[str, Any]:
    if role not in ("fit", "calibration"):
        raise ValueError("contextual rewards are forbidden for final")
    output = RUNS / f"rewards/{role}"
    completion = output / "completion.json"
    reward_map_path = output / "reward_map.pkl"
    metric_map_path = output / "metric_map.pkl"
    if completion.is_file():
        report = read_json(completion)
        if report.get("reward_map_sha256") != file_sha256(reward_map_path):
            raise RuntimeError(f"reward map changed: {role}")
        if report.get("metric_map_sha256") != file_sha256(metric_map_path):
            raise RuntimeError(f"reward metric map changed: {role}")
        return report
    data = load_role(role)
    folds = reward_folds(role, data["samples"])
    contexts = DATA_CORE.fold_contained_contexts(
        folds,
        context_size=int(PROTOCOL["reward"]["context_size"]),
        seed=int(PROTOCOL["fold_seed"]),
    )
    membership = {
        int(image_id): fold_index
        for fold_index, fold in enumerate(folds)
        for image_id in fold
    }
    if set(contexts) != set(membership):
        raise RuntimeError(f"reward context population mismatch: {role}")
    atomic_json(output / "folds.json", {"folds": folds})
    atomic_json(output / "contexts.json", {str(key): value for key, value in contexts.items()})
    reports = []
    reward_map = {int(sample["image_id"]): {} for sample in data["samples"]}
    metric_map = {int(sample["image_id"]): {} for sample in data["samples"]}
    for action_id in reward_actions():
        reports.append(run_action(role, action_id, data, contexts, membership))
        rows_path = output / action_id.replace("@", "__") / "rows.jsonl"
        for row in read_jsonl(rows_path):
            image_id = int(row["target_image_id"])
            reward_map[image_id][action_id] = float(row["reward"])
            metric_map[image_id][action_id] = {
                "ap50": float(row["reward_ap50"]),
                "ap50_95": float(row["reward_ap50_95"]),
            }
    expected_actions = set(reward_actions())
    if any(set(values) != expected_actions for values in reward_map.values()):
        raise RuntimeError(f"reward action matrix incomplete: {role}")
    atomic_pickle(reward_map_path, reward_map)
    atomic_pickle(metric_map_path, metric_map)
    report = {
        "complete": True,
        "role": role,
        "images": len(reward_map),
        "actions": len(expected_actions),
        "tasks": len(reward_map) * len(expected_actions),
        "folds": len(folds),
        "event_grouped": role == "fit",
        "fold_contained": True,
        "reward_map_sha256": file_sha256(reward_map_path),
        "metric_map_sha256": file_sha256(metric_map_path),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "final_accessed": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, report)
    return report


def run_all() -> dict[str, Any]:
    if not (RUNS / "common/completion.json").is_file():
        raise RuntimeError("Cloud-V3 common cache must complete before rewards")
    reports = {role: run_role(role) for role in ("fit", "calibration")}
    report = {
        "complete": True,
        "roles": {role: value["tasks"] for role, value in reports.items()},
        "final_accessed": False,
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
    }
    atomic_json(RUNS / "rewards/report.json", report)
    atomic_json(
        RUNS / "rewards/completion.json",
        {"complete": True, "report_sha256": file_sha256(RUNS / "rewards/report.json")},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("fit", "calibration", "all"), default="all")
    args = parser.parse_args()
    result = run_all() if args.role == "all" else run_role(args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
