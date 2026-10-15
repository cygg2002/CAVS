#!/usr/bin/env python3
"""Run the cache-only COCO Selection-Context JointAction teacher diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"

sys.path[:0] = [
    str(ROOT / "scripts"),
    str(
        REPO
        / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts"
    ),
    str(
        REPO
        / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines/scripts"
    ),
]

from teacher_core import (  # noqa: E402
    build_context_ids,
    evaluate_gate,
    schedule_joint_actions,
    selection_overlap,
    validate_protocol,
)
import run_multiranker_benchmark as canonical  # noqa: E402
from baseline_core import DetectionStats, detection_stats, oric_reward  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


PROTOCOL = read_json(PROTOCOL_PATH)
validate_protocol(PROTOCOL)
SOURCE_ROOT = REPO / PROTOCOL["source_experiment"]
COMMON = SOURCE_ROOT / PROTOCOL["canonical_common"]
BASELINE_ROOT = REPO / PROTOCOL["external_baseline_experiment"]
CANDIDATES = tuple(PROTOCOL["candidate_actions"])
ALL_ACTIONS = tuple(PROTOCOL["actions"])


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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _completion(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    return value if value.get("complete") is True else None


def _configure_canonical() -> None:
    canonical.COMMON = COMMON.parent
    if Path(canonical.COMMON) / "coco" != COMMON:
        raise RuntimeError("Canonical COCO cache binding failed")


def load_role(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(f"Forbidden role: {role}")
    _configure_canonical()
    data = canonical.load_role("coco", role)
    expected = int(PROTOCOL["roles"][role])
    if len(data["samples"]) != expected:
        raise RuntimeError(
            f"Canonical {role} population changed: {len(data['samples'])} != {expected}"
        )
    return data


def _role_source_files(role: str) -> list[Path]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(f"Forbidden role: {role}")
    root = COMMON / role
    manifest_path = root / "source_manifest.json"
    manifest = read_json(manifest_path)
    paths = [manifest_path, root / "samples.json", root / "gt.json"]
    for action in ALL_ACTIONS:
        source = Path(manifest["action_sources"][action])
        if not source.is_file():
            raise FileNotFoundError(f"Missing canonical {role}/{action}: {source}")
        expected = manifest.get("source_paths", {}).get(str(source))
        if expected is None and action != "roi":
            raise RuntimeError(f"Unregistered canonical source for {role}/{action}")
        if action == "roi" and source.name != "roi_maur_predictions.json":
            raise RuntimeError(f"Unexpected canonical ROI cache for {role}")
        paths.append(source)
    return paths


def required_source_files() -> list[Path]:
    paths = {
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/teacher_core.py",
        SOURCE_ROOT / "scripts/run_multiranker_benchmark.py",
        BASELINE_ROOT / "scripts/baseline_core.py",
        REPO / "paper/experiments/GAIR_v5/gair/selector.py",
    }
    for role in PROTOCOL["roles"]:
        paths.update(_role_source_files(role))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen sources:\n" + "\n".join(sorted(missing)))
    return sorted(path.resolve() for path in paths)


def capture_source_hashes() -> dict[str, str]:
    return {str(path): file_sha256(path) for path in required_source_files()}


def validate_registered_action_hashes(hashes: Mapping[str, str]) -> None:
    for role in PROTOCOL["roles"]:
        manifest = read_json(COMMON / role / "source_manifest.json")
        registered = manifest.get("source_paths", {})
        for action in ALL_ACTIONS:
            source = Path(manifest["action_sources"][action]).resolve()
            expected = registered.get(str(source)) or registered.get(str(Path(source)))
            if expected is None:
                if action != "roi":
                    raise RuntimeError(f"Unregistered canonical source for {role}/{action}")
                continue
            if hashes[str(source)] != str(expected):
                raise RuntimeError(f"Registered hash changed for {role}/{action}")


def source_digest(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_source_hashes() -> dict[str, str]:
    report = read_json(RUNS / "preflight/report.json")
    current = capture_source_hashes()
    if current != report["source_sha256"]:
        raise RuntimeError("Frozen source hashes changed after preflight")
    return current


def run_preflight() -> dict[str, Any]:
    completion_path = RUNS / "preflight/completion.json"
    if _completion(completion_path):
        return read_json(RUNS / "preflight/report.json")
    role_counts = {}
    role_availability = {}
    for role, expected_count in PROTOCOL["roles"].items():
        samples = read_json(COMMON / role / "samples.json")
        if len(samples) != int(expected_count):
            raise RuntimeError(f"Canonical {role} population changed")
        role_counts[role] = len(samples)
        role_availability[role] = {
            action: sum(bool(sample["available"].get(action, False)) for sample in samples)
            for action in CANDIDATES
        }
    hashes = capture_source_hashes()
    validate_registered_action_hashes(hashes)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "role_counts": role_counts,
        "role_availability": role_availability,
        "source_sha256": hashes,
        "source_digest": source_digest(hashes),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gpu_inference": False,
        "detectors_retrained": False,
        "rankers_trained": False,
        "roles_accessed": sorted(PROTOCOL["roles"]),
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(RUNS / "preflight/report.json", report)
    atomic_json(
        completion_path,
        {"complete": True, "source_digest": report["source_digest"]},
    )
    return report


_STATS_STATE: dict[str, Any] = {}


def _stats_worker(image_id: int) -> tuple[int, dict[str, DetectionStats]]:
    data = _STATS_STATE["data"]
    threshold = float(_STATS_STATE["confidence"])
    iou_thresholds = _STATS_STATE["iou_thresholds"]
    sample = data["sample_by_id"][int(image_id)]
    target = data["gt_map"].get(int(image_id), [])
    output = {}
    for action in ALL_ACTIONS:
        if action != "edge" and not bool(sample["available"].get(action, False)):
            continue
        predictions = canonical.filter_conf(
            data["actions"][action].get(int(image_id), []), threshold
        )
        output[action] = detection_stats(predictions, target, iou_thresholds)
    return int(image_id), output


def _fork_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError as error:
        raise RuntimeError("This cache pipeline requires multiprocessing fork") from error


def build_role_stats(role: str, workers: int) -> dict[str, Any]:
    output_path = RUNS / "stats" / f"{role}.pkl"
    completion_path = RUNS / "stats" / f"{role}_completion.json"
    if _completion(completion_path):
        return load_pickle(output_path)
    data = load_role(role)
    _STATS_STATE.clear()
    _STATS_STATE.update(
        {
            "data": data,
            "confidence": PROTOCOL["reward"]["confidence"],
            "iou_thresholds": tuple(PROTOCOL["reward"]["iou_thresholds"]),
        }
    )
    started = time.time()
    stats_by_action = {action: {} for action in ALL_ACTIONS}
    with ProcessPoolExecutor(
        max_workers=int(workers), mp_context=_fork_context()
    ) as executor:
        for completed, (image_id, action_stats) in enumerate(
            executor.map(_stats_worker, data["image_ids"], chunksize=8), start=1
        ):
            for action, stats in action_stats.items():
                stats_by_action[action][int(image_id)] = stats
            if completed % 512 == 0 or completed == len(data["image_ids"]):
                print(
                    f"[stats/{role}] {completed}/{len(data['image_ids'])}",
                    flush=True,
                )
    payload = {
        "role": role,
        "image_ids": list(map(int, data["image_ids"])),
        "stats": stats_by_action,
        "elapsed_seconds": time.time() - started,
    }
    atomic_pickle(output_path, payload)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "images": len(data["image_ids"]),
            "sha256": file_sha256(output_path),
            "elapsed_seconds": payload["elapsed_seconds"],
        },
    )
    _STATS_STATE.clear()
    return payload


def run_stats(workers: int) -> None:
    if not _completion(RUNS / "preflight/completion.json"):
        raise RuntimeError("Preflight must complete before stats")
    for role in ("fit", "calibration"):
        build_role_stats(role, workers)
    atomic_json(RUNS / "stats/completion.json", {"complete": True})


_REWARD_STATE: dict[str, Any] = {}


def _reward_worker(image_id: int) -> tuple[int, dict[str, float]]:
    fit_ids = _REWARD_STATE["fit_ids"]
    context_ids = build_context_ids(
        fit_ids,
        int(image_id),
        context_size=int(_REWARD_STATE["context_size"]),
        seed=int(_REWARD_STATE["seed"]),
    )
    fit_stats = _REWARD_STATE["fit_stats"]
    context_actions = _REWARD_STATE["context_actions"]
    context = []
    for context_id in context_ids:
        action = context_actions.get(int(context_id), "edge")
        if int(context_id) not in fit_stats[action]:
            raise RuntimeError(f"Missing fit context stats for {context_id}/{action}")
        context.append(fit_stats[action][int(context_id)])
    target_stats = _REWARD_STATE["target_stats"]
    target_edge = target_stats["edge"][int(image_id)]
    rewards = {}
    for action in CANDIDATES:
        cloud = target_stats[action].get(int(image_id))
        if cloud is None:
            continue
        rewards[action] = float(oric_reward(context, target_edge, cloud)["reward"])
    return int(image_id), rewards


def compute_rewards(
    *,
    label: str,
    target_role: str,
    fit_stats: dict[str, Any],
    target_stats: dict[str, Any],
    context_actions: Mapping[int, str],
    workers: int,
) -> dict[int, dict[str, float]]:
    output_path = RUNS / "rewards" / f"{label}.pkl"
    completion_path = RUNS / "rewards" / f"{label}_completion.json"
    if _completion(completion_path):
        return load_pickle(output_path)
    checkpoint_path = RUNS / "rewards" / f"{label}_partial.pkl"
    rewards: dict[int, dict[str, float]] = (
        load_pickle(checkpoint_path) if checkpoint_path.is_file() else {}
    )
    target_ids = list(map(int, target_stats["image_ids"]))
    remaining = [image_id for image_id in target_ids if image_id not in rewards]
    _REWARD_STATE.clear()
    _REWARD_STATE.update(
        {
            "fit_ids": list(map(int, fit_stats["image_ids"])),
            "fit_stats": fit_stats["stats"],
            "target_stats": target_stats["stats"],
            "context_actions": dict(context_actions),
            "context_size": int(PROTOCOL["reward"]["context_size"]),
            "seed": int(PROTOCOL["reward"]["context_seed"]),
        }
    )
    started = time.time()
    checkpoint_interval = int(PROTOCOL["reward"]["checkpoint_interval"])
    if remaining:
        with ProcessPoolExecutor(
            max_workers=int(workers), mp_context=_fork_context()
        ) as executor:
            for completed, (image_id, values) in enumerate(
                executor.map(_reward_worker, remaining, chunksize=4), start=1
            ):
                rewards[int(image_id)] = values
                total_completed = len(rewards)
                if completed % checkpoint_interval == 0 or completed == len(remaining):
                    atomic_pickle(checkpoint_path, rewards)
                    print(
                        f"[rewards/{label}] {total_completed}/{len(target_ids)}",
                        flush=True,
                    )
    if set(rewards) != set(target_ids):
        raise RuntimeError(f"Incomplete reward cache for {label}")
    atomic_pickle(output_path, rewards)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "role": target_role,
            "images": len(rewards),
            "context_actions": len(context_actions),
            "sha256": file_sha256(output_path),
            "elapsed_seconds": time.time() - started,
        },
    )
    checkpoint_path.unlink(missing_ok=True)
    _REWARD_STATE.clear()
    return rewards


def role_samples(role: str) -> list[dict[str, Any]]:
    samples = read_json(COMMON / role / "samples.json")
    if len(samples) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"Canonical {role} sample population changed")
    return samples


def role_capacity(samples: list[dict[str, Any]], budget: float) -> float:
    return float(budget) * sum(float(sample["payload_bytes"]["original"]) for sample in samples)


def run_rewards(workers: int) -> None:
    if not _completion(RUNS / "stats/completion.json"):
        raise RuntimeError("Stats must complete before rewards")
    fit_stats = load_pickle(RUNS / "stats/fit.pkl")
    calibration_stats = load_pickle(RUNS / "stats/calibration.pkl")
    fit_samples = role_samples("fit")
    fit_rewards = compute_rewards(
        label="fit_static",
        target_role="fit",
        fit_stats=fit_stats,
        target_stats=fit_stats,
        context_actions={},
        workers=workers,
    )
    fit_schedules = {}
    for budget in PROTOCOL["budgets"]:
        schedule = schedule_joint_actions(
            fit_samples,
            fit_rewards,
            candidate_actions=CANDIDATES,
            capacity_bytes=role_capacity(fit_samples, float(budget)),
        )
        fit_schedules[str(budget)] = schedule
    atomic_json(RUNS / "rewards/fit_schedules.json", fit_schedules)
    compute_rewards(
        label="calibration_static",
        target_role="calibration",
        fit_stats=fit_stats,
        target_stats=calibration_stats,
        context_actions={},
        workers=workers,
    )
    for budget in PROTOCOL["budgets"]:
        compute_rewards(
            label=f"calibration_selection_context_{budget}",
            target_role="calibration",
            fit_stats=fit_stats,
            target_stats=calibration_stats,
            context_actions={
                int(image_id): action
                for image_id, action in fit_schedules[str(budget)]["selection"].items()
            },
            workers=workers,
        )
    atomic_json(RUNS / "rewards/completion.json", {"complete": True})


def evaluate_selection(
    data: dict[str, Any],
    *,
    method: str,
    budget: float,
    schedule: Mapping[str, Any],
    capacity_bytes: float,
) -> dict[str, Any]:
    selection = {int(image_id): str(action) for image_id, action in schedule["selection"].items()}
    predictions, _ = canonical.mixed_predictions(
        data, selection, float(PROTOCOL["reward"]["confidence"])
    )
    metrics = data["evaluator"].evaluate(predictions)
    request_bytes = sum(
        int(data["sample_by_id"][image_id]["payload_bytes"][action])
        for image_id, action in selection.items()
    )
    tolerance = float(PROTOCOL["gate"]["capacity_tolerance_bytes"])
    counts = Counter(selection.values())
    images = len(data["image_ids"])
    return {
        "method": method,
        "role": "calibration",
        "mode": "fixed_byte",
        "budget": float(budget),
        "official_ap50_95": float(metrics["map50_95"]),
        "official_ap50": float(metrics["map50"]),
        "official_ap75": float(metrics["map75"]),
        "kb_per_image": request_bytes / 1024.0 / images,
        "upload_ratio": request_bytes
        / max(1.0, sum(float(sample["payload_bytes"]["original"]) for sample in data["samples"])),
        "request_bytes": int(request_bytes),
        "capacity_bytes": float(capacity_bytes),
        "capacity_ok": bool(request_bytes <= capacity_bytes + tolerance),
        "selected_images": len(selection),
        "selected_fraction": len(selection) / images,
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "teacher_objective": float(schedule["total_value"]),
        "scheduler_lambda": float(schedule["lambda"]),
    }


def _results_markdown(rows: list[dict[str, Any]], gate: dict[str, Any]) -> str:
    by_key = {(row["method"], float(row["budget"])): row for row in rows}
    lines = [
        "# COCO Selection-Context JointAction Teacher 诊断结果",
        "",
        "本实验只使用 canonical train2017 fit=9,678 与 calibration=571 缓存；",
        "未访问 audit、val2017/final 或 registered final。结果是 GT-aware Teacher ceiling，",
        "不是可部署学习器结果。",
        "",
        "## Fixed-byte 主结果",
        "",
        "| 预算 | 方法 | Official AP | AP50 | AP75 | KB/图 | 上传率 | Original/Q90/ROI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in PROTOCOL["budgets"]:
        for method in ("Static-JointAction", "SelectionContext-JointAction"):
            row = by_key[(method, float(budget))]
            lines.append(
                "| {budget:.0%} | {method} | {ap:.6f} | {ap50:.6f} | {ap75:.6f} | "
                "{kb:.2f} | {ratio:.2%} | {original}/{q90}/{roi} |".format(
                    budget=float(budget),
                    method=method,
                    ap=row["official_ap50_95"],
                    ap50=row["official_ap50"],
                    ap75=row["official_ap75"],
                    kb=row["kb_per_image"],
                    ratio=row["upload_ratio"],
                    original=row["original_count"],
                    q90=row["q90_count"],
                    roi=row["roi_count"],
                )
            )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- passed: `{str(gate['passed']).lower()}`",
            f"- 达到 AP +0.002 的预算点：{gate['gain_points']}/3",
            f"- 其余点满足不劣于 -0.001：`{str(gate['all_noninferior']).lower()}`",
            f"- 所有字节容量约束通过：`{str(gate['capacity_ok']).lower()}`",
            "",
            "| 预算 | SelectionContext − Static AP | 图片选择 Jaccard | 精确动作 Jaccard |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in gate["budget_rows"]:
        lines.append(
            f"| {float(item['budget']):.0%} | {float(item['delta_ap']):+.6f} | "
            f"{float(item['selected_image_jaccard']):.3f} | "
            f"{float(item['exact_action_jaccard']):.3f} |"
        )
    lines.extend(
        [
            "",
            "通过时才进入 Selection-Contextual SharedAction 学习器实现；失败则停止该分支，",
            "继续保留 SharedAction、CostEligible-AllROI 和 GBDT-Hierarchical 作为现有动作框架。",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation() -> dict[str, Any]:
    completion_path = RUNS / "evaluation/completion.json"
    if _completion(completion_path):
        return read_json(RUNS / "evaluation/report.json")
    if not _completion(RUNS / "rewards/completion.json"):
        raise RuntimeError("Rewards must complete before evaluation")
    data = load_role("calibration")
    samples = data["samples"]
    static_rewards = load_pickle(RUNS / "rewards/calibration_static.pkl")
    rows = []
    selections = {}
    budget_rows = []
    for budget_value in PROTOCOL["budgets"]:
        budget = float(budget_value)
        capacity = role_capacity(samples, budget)
        contextual_rewards = load_pickle(
            RUNS / "rewards" / f"calibration_selection_context_{budget_value}.pkl"
        )
        static_schedule = schedule_joint_actions(
            samples,
            static_rewards,
            candidate_actions=CANDIDATES,
            capacity_bytes=capacity,
        )
        contextual_schedule = schedule_joint_actions(
            samples,
            contextual_rewards,
            candidate_actions=CANDIDATES,
            capacity_bytes=capacity,
        )
        static_row = evaluate_selection(
            data,
            method="Static-JointAction",
            budget=budget,
            schedule=static_schedule,
            capacity_bytes=capacity,
        )
        contextual_row = evaluate_selection(
            data,
            method="SelectionContext-JointAction",
            budget=budget,
            schedule=contextual_schedule,
            capacity_bytes=capacity,
        )
        overlap = selection_overlap(
            static_schedule["selection"], contextual_schedule["selection"]
        )
        rows.extend([static_row, contextual_row])
        selections[str(budget_value)] = {
            "static": static_schedule["selection"],
            "selection_context": contextual_schedule["selection"],
        }
        budget_rows.append(
            {
                "budget": budget,
                "delta_ap": contextual_row["official_ap50_95"]
                - static_row["official_ap50_95"],
                "capacity_ok": static_row["capacity_ok"]
                and contextual_row["capacity_ok"],
                **overlap,
            }
        )
    gate = evaluate_gate(budget_rows, PROTOCOL)
    gate["budget_rows"] = budget_rows
    hashes = verify_source_hashes()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_type": "gt_aware_teacher_ceiling_diagnostic",
        "rows": rows,
        "gate": gate,
        "source_digest": source_digest(hashes),
        "roles_accessed": ["fit", "calibration"],
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(RUNS / "evaluation/report.json", report)
    atomic_json(RUNS / "evaluation/gate.json", gate)
    atomic_json(RUNS / "evaluation/selections.json", selections)
    atomic_csv(RUNS / "evaluation/results.csv", rows)
    note = _results_markdown(rows, gate)
    note_path = ROOT / "notes/results_zh.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = note_path.with_name(f".{note_path.name}.{os.getpid()}.tmp")
    temporary.write_text(note, encoding="utf-8")
    os.replace(temporary, note_path)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "passed": gate["passed"],
            "source_digest": report["source_digest"],
        },
    )
    atomic_json(
        ROOT / "COMPLETED.json",
        {
            "complete": True,
            "passed": gate["passed"],
            "result_type": report["result_type"],
            "audit_accessed": False,
            "final_accessed": False,
            "registered_final_accessed": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "stats", "rewards", "evaluation", "all"),
        default="all",
    )
    parser.add_argument(
        "--workers", type=int, default=int(PROTOCOL["reward"]["workers"])
    )
    arguments = parser.parse_args()
    if arguments.workers <= 0:
        raise ValueError("workers must be positive")
    if arguments.stage in ("preflight", "all"):
        run_preflight()
    if arguments.stage in ("stats", "all"):
        run_stats(arguments.workers)
    if arguments.stage in ("rewards", "all"):
        run_rewards(arguments.workers)
    if arguments.stage in ("evaluation", "all"):
        report = run_evaluation()
        print(json.dumps(report["gate"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
