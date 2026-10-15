#!/usr/bin/env python3
"""Run the cache-only COCO DualHead-v2 No-ROI OOF diagnostic."""

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
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
sys.path.insert(0, str(ROOT / "scripts"))

from dualheadv2_core import (  # noqa: E402
    assert_no_roi_selection,
    blend_mean_lower,
    choose_alphas,
    evaluate_dualheadv2_gate,
    evaluate_selection_context_gate,
    no_roi_score_map,
    validate_protocol,
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


PROTOCOL = read_json(PROTOCOL_PATH)
validate_protocol(PROTOCOL)
SOURCES = {
    name: REPO / value for name, value in PROTOCOL["source_experiments"].items()
}
CANONICAL_ROOT = SOURCES["canonical"]
TEACHER_ROOT = SOURCES["teacher"]
ACTION_ROOT = SOURCES["action_value"]
COMMON = CANONICAL_ROOT / PROTOCOL["canonical_common"]


def _load_module(name: str, path: Path):
    source_scripts = str(path.parent)
    if source_scripts not in sys.path:
        sys.path.insert(0, source_scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_module(
    "dualheadv2_frozen_actionvalue_runner", ACTION_ROOT / "scripts/run_oof.py"
)
TEACHER_CORE = _load_module(
    "dualheadv2_frozen_teacher_core", TEACHER_ROOT / "scripts/teacher_core.py"
)


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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _complete(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    return value if value.get("complete") is True else None


def _teacher_reward_path(key: str, *, budget: float | None = None) -> Path:
    value = str(PROTOCOL["teacher_rewards"][key])
    if budget is not None:
        value = value.format(budget=f"{float(budget):g}")
    return TEACHER_ROOT / value


def _score_path(role: str) -> Path:
    return ACTION_ROOT / str(PROTOCOL["source_scores"][role])


def required_source_files() -> list[Path]:
    paths = {
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/dualheadv2_core.py",
        ACTION_ROOT / "configs/protocol.json",
        ACTION_ROOT / "scripts/run_oof.py",
        ACTION_ROOT / "scripts/action_core.py",
        ACTION_ROOT / "runs/oof/completion.json",
        ACTION_ROOT / "runs/calibration_scores/completion.json",
        _score_path("fit"),
        _score_path("calibration"),
        TEACHER_ROOT / "configs/protocol.json",
        TEACHER_ROOT / "scripts/teacher_core.py",
        TEACHER_ROOT / "COMPLETED.json",
        _teacher_reward_path("static_fit"),
        _teacher_reward_path("static_calibration"),
    }
    for budget in map(float, PROTOCOL["budgets"]):
        paths.add(
            _teacher_reward_path("selection_context_calibration_pattern", budget=budget)
        )
    for role in PROTOCOL["roles"]:
        root = COMMON / role
        manifest_path = root / "source_manifest.json"
        manifest = read_json(manifest_path)
        paths.update({manifest_path, root / "samples.json", root / "gt.json"})
        for action in PROTOCOL["actions"]:
            source = Path(manifest["action_sources"][action])
            paths.add(source)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen sources:\n" + "\n".join(sorted(missing)))
    return sorted(path.resolve() for path in paths)


def capture_hashes() -> dict[str, str]:
    return {str(path): file_sha256(path) for path in required_source_files()}


def source_digest(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_hashes() -> dict[str, str]:
    expected = read_json(RUNS / "preflight/report.json")["source_sha256"]
    observed = capture_hashes()
    if observed != expected:
        raise RuntimeError("Frozen source hashes changed after preflight")
    return observed


def _load_score_artifact(role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = load_pickle(_score_path(role))
    records = SOURCE.records_for_role(role)
    record_ids = [int(record["image_id"]) for record in records]
    if list(map(int, artifact["image_ids"])) != record_ids:
        raise RuntimeError(f"Frozen {role} score/image ordering changed")
    return artifact, records


def run_preflight() -> dict[str, Any]:
    completion = RUNS / "preflight/completion.json"
    if _complete(completion):
        return read_json(RUNS / "preflight/report.json")
    action_oof = read_json(ACTION_ROOT / "runs/oof/completion.json")
    action_calibration = read_json(
        ACTION_ROOT / "runs/calibration_scores/completion.json"
    )
    teacher = read_json(TEACHER_ROOT / "COMPLETED.json")
    if not all(
        value.get("complete") is True
        for value in (action_oof, action_calibration, teacher)
    ):
        raise RuntimeError("A frozen source experiment is incomplete")
    role_counts = {}
    score_shapes = {}
    for role, expected in PROTOCOL["roles"].items():
        data = SOURCE.load_role(role)
        if len(data["samples"]) != int(expected):
            raise RuntimeError(f"Canonical {role} population changed")
        artifact, records = _load_score_artifact(role)
        scores = SOURCE.numeric_score_arrays(artifact)
        expected_shape = (len(records), 4)
        for key in ("value", "risk_mean", "risk_lower"):
            if scores[key].shape != expected_shape:
                raise RuntimeError(f"Unexpected {role}/{key} shape: {scores[key].shape}")
        role_counts[role] = len(records)
        score_shapes[role] = {key: list(value.shape) for key, value in scores.items()}
    hashes = capture_hashes()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "role_counts": role_counts,
        "score_shapes": score_shapes,
        "source_sha256": hashes,
        "source_digest": source_digest(hashes),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "detectors_retrained": False,
        "scores_retrained": False,
        "roles_accessed": ["fit", "calibration"],
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(RUNS / "preflight/report.json", report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


def _original_capacity(samples: Sequence[Mapping[str, Any]], budget: float) -> float:
    return float(budget) * sum(
        float(sample["payload_bytes"]["original"]) for sample in samples
    )


def _with_capacity_utilization(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    capacity = float(output.get("capacity_bytes", 0.0))
    output["capacity_utilization"] = (
        float(output.get("request_bytes", 0.0)) / capacity if capacity > 0.0 else 1.0
    )
    if int(output.get("roi_count", 0)) != 0:
        raise RuntimeError("ROI survived the No-ROI evaluation")
    return output


def run_ceiling() -> dict[str, Any]:
    """Formally re-evaluate the frozen selection-context teacher without ROI."""
    completion = RUNS / "ceiling/completion.json"
    if _complete(completion):
        return read_json(RUNS / "ceiling/report.json")
    verify_hashes()
    data = SOURCE.load_role("calibration")
    samples = data["samples"]
    static_rewards = load_pickle(_teacher_reward_path("static_calibration"))
    rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    budget_rows = []
    for budget in map(float, PROTOCOL["budgets"]):
        contextual_rewards = load_pickle(
            _teacher_reward_path(
                "selection_context_calibration_pattern", budget=budget
            )
        )
        capacity = _original_capacity(samples, budget)
        static = TEACHER_CORE.schedule_joint_actions(
            samples,
            static_rewards,
            candidate_actions=PROTOCOL["candidate_actions"],
            capacity_bytes=capacity,
        )
        contextual = TEACHER_CORE.schedule_joint_actions(
            samples,
            contextual_rewards,
            candidate_actions=PROTOCOL["candidate_actions"],
            capacity_bytes=capacity,
        )
        assert_no_roi_selection(static["selection"])
        assert_no_roi_selection(contextual["selection"])
        static_row = _with_capacity_utilization(
            SOURCE._evaluate_selection(
                role="calibration",
                method="StaticTeacher-OQ",
                mode="fixed_byte",
                budget=budget,
                data=data,
                selection=static["selection"],
                schedule=static,
                rewards=static_rewards,
                teacher_value=float(static["total_value"]),
                capacity_bytes=capacity,
            )
        )
        contextual_row = _with_capacity_utilization(
            SOURCE._evaluate_selection(
                role="calibration",
                method="SelectionContextTeacher-OQ",
                mode="fixed_byte",
                budget=budget,
                data=data,
                selection=contextual["selection"],
                schedule=contextual,
                rewards=contextual_rewards,
                teacher_value=float(contextual["total_value"]),
                capacity_bytes=capacity,
            )
        )
        overlap = TEACHER_CORE.selection_overlap(
            static["selection"], contextual["selection"]
        )
        rows.extend((static_row, contextual_row))
        selections[f"{budget:g}"] = {
            "static": static["selection"],
            "selection_context": contextual["selection"],
        }
        budget_rows.append(
            {
                "budget": budget,
                "delta_ap": float(contextual_row["official_ap50_95"])
                - float(static_row["official_ap50_95"]),
                "capacity_ok": bool(static_row["capacity_ok"])
                and bool(contextual_row["capacity_ok"]),
                **overlap,
            }
        )
    gate = evaluate_selection_context_gate(budget_rows, PROTOCOL)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_type": "gt_aware_noroi_teacher_ceiling_diagnostic",
        "rows": rows,
        "gate": gate,
        "sc_branch_open": bool(gate["passed"]),
        "sc_models_trained": False,
        "roles_accessed": ["fit", "calibration"],
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_csv(RUNS / "ceiling/results.csv", rows)
    atomic_json(RUNS / "ceiling/selections.json", selections)
    atomic_json(RUNS / "ceiling/report.json", report)
    atomic_json(
        completion,
        {"complete": True, "passed": bool(gate["passed"]), "sc_models_trained": False},
    )
    return report


_POINT_STATE: dict[str, Any] = {}


def _point_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    if not _POINT_STATE:
        raise RuntimeError("Point-evaluation state is not configured")
    row = SOURCE._evaluate_selection(
        role=str(task["role"]),
        method=str(task["method"]),
        mode=str(task["mode"]),
        budget=float(task["budget"]),
        data=_POINT_STATE["data"],
        selection=task["selection"],
        schedule=task["schedule"],
        rewards=_POINT_STATE["rewards"],
        teacher_value=float(task["teacher_value"]),
        capacity_bytes=float(task["capacity_bytes"]),
    )
    output = _with_capacity_utilization(row)
    output["role"] = str(task["output_role"])
    if task.get("alpha") is not None:
        output["alpha"] = float(task["alpha"])
    return output


def _evaluate_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    data: dict[str, Any],
    rewards: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    _POINT_STATE.clear()
    _POINT_STATE.update({"data": data, "rewards": rewards})
    workers = min(
        int(PROTOCOL["cpu_workers"]), max(1, (os.cpu_count() or 1) - 2), len(tasks)
    )
    context = mp.get_context("fork")
    rows = []
    try:
        with context.Pool(processes=workers) as pool:
            for count, row in enumerate(
                pool.imap_unordered(_point_worker, tasks, chunksize=1), start=1
            ):
                rows.append(row)
                if count % 8 == 0 or count == len(tasks):
                    print(f"[point] {count}/{len(tasks)}", flush=True)
    finally:
        _POINT_STATE.clear()
    return rows


def _filtered_rewards(
    rewards: Mapping[int, Mapping[str, float]]
) -> dict[int, dict[str, float]]:
    return {
        int(image_id): {
            action: float(value)
            for action, value in values.items()
            if str(action) in PROTOCOL["candidate_actions"]
        }
        for image_id, values in rewards.items()
    }


def _score_maps(
    records: Sequence[Mapping[str, Any]], scores: Mapping[str, np.ndarray]
) -> dict[str, dict[int, dict[str, float]]]:
    output = {
        "ActionValue-OQ": no_roi_score_map(records, scores["value"]),
        "RiskMean-OQ": no_roi_score_map(records, scores["risk_mean"]),
    }
    for alpha in map(float, PROTOCOL["alpha_grid"]):
        method = f"DualHeadV2-alpha={alpha:g}"
        blended = blend_mean_lower(scores["risk_mean"], scores["risk_lower"], alpha)
        output[method] = no_roi_score_map(records, blended)
    return output


def _schedule(
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
    capacity: float,
) -> dict[str, Any]:
    if mode == "fixed_call":
        result = SOURCE.schedule_fixed_call(samples, scores, budget=budget)
    elif mode == "fixed_byte":
        result = SOURCE.schedule_fixed_byte(
            samples, scores, capacity_bytes=capacity
        )
    else:
        raise ValueError(mode)
    assert_no_roi_selection(result["selection"])
    return result


def _build_tasks(
    role: str,
    *,
    output_role: str,
    method_maps: Mapping[str, Mapping[int, Mapping[str, float]]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[int, str]],
    dict[tuple[str, float], dict[str, Any]],
]:
    data = SOURCE.load_role(role)
    samples = data["samples"]
    rewards = _filtered_rewards(SOURCE.reward_map(role))
    tasks = []
    selections = {}
    teachers = {}
    for budget in map(float, PROTOCOL["budgets"]):
        capacity = _original_capacity(samples, budget)
        for mode in PROTOCOL["modes"]:
            teacher = _schedule(
                samples,
                rewards,
                mode=str(mode),
                budget=budget,
                capacity=capacity,
            )
            teachers[(str(mode), budget)] = teacher
            for method, values in method_maps.items():
                schedule = _schedule(
                    samples,
                    values,
                    mode=str(mode),
                    budget=budget,
                    capacity=capacity,
                )
                key = f"{method}|{mode}|{budget:g}"
                selections[key] = dict(schedule["selection"])
                alpha = (
                    float(method.split("=", 1)[1])
                    if method.startswith("DualHeadV2-alpha=")
                    else None
                )
                tasks.append(
                    {
                        "role": role,
                        "output_role": output_role,
                        "method": method,
                        "mode": str(mode),
                        "budget": budget,
                        "selection": schedule["selection"],
                        "schedule": schedule,
                        "teacher_value": float(teacher["total_value"]),
                        "capacity_bytes": capacity,
                        "alpha": alpha,
                    }
                )
    return tasks, selections, teachers


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, float]:
    return str(row["method"]), str(row["mode"]), float(row["budget"])


def _chosen_oof_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    frozen: Mapping[float, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {_row_key(row): row for row in raw_rows}
    output = []
    for budget in map(float, PROTOCOL["budgets"]):
        alpha = float(frozen[budget]["alpha"])
        source_method = f"DualHeadV2-alpha={alpha:g}"
        for mode in PROTOCOL["modes"]:
            for method in ("ActionValue-OQ", "RiskMean-OQ"):
                output.append(dict(by_key[(method, str(mode), budget)]))
            candidate = dict(by_key[(source_method, str(mode), budget)])
            candidate["source_method"] = source_method
            candidate["method"] = "DualHeadV2-OQ"
            candidate["alpha"] = alpha
            output.append(candidate)
    if len(output) != 18 or len({_row_key(row) for row in output}) != 18:
        raise RuntimeError("Chosen OOF result matrix is incomplete")
    return output


def run_sweep() -> dict[str, Any]:
    completion = RUNS / "sweep/completion.json"
    if _complete(completion):
        return read_json(RUNS / "sweep/report.json")
    verify_hashes()
    if not _complete(RUNS / "ceiling/completion.json"):
        raise RuntimeError("Teacher ceiling must complete before the risk sweep")
    artifact, records = _load_score_artifact("fit")
    scores = SOURCE.numeric_score_arrays(artifact)
    maps = _score_maps(records, scores)
    tasks, selections, _ = _build_tasks(
        "fit", output_role="fit_oof", method_maps=maps
    )
    expected_rows = len(maps) * len(PROTOCOL["modes"]) * len(PROTOCOL["budgets"])
    if len(tasks) != expected_rows:
        raise RuntimeError("OOF sweep task matrix is incomplete")
    raw_rows = _evaluate_tasks(
        tasks, data=SOURCE.load_role("fit"), rewards=_filtered_rewards(SOURCE.reward_map("fit"))
    )
    if len(raw_rows) != expected_rows or len({_row_key(row) for row in raw_rows}) != expected_rows:
        raise RuntimeError("OOF sweep result matrix is incomplete or duplicated")
    frozen = choose_alphas(raw_rows, PROTOCOL)
    chosen_rows = _chosen_oof_rows(raw_rows, frozen)
    gate = evaluate_dualheadv2_gate(chosen_rows, PROTOCOL)
    chosen_selections = {}
    for budget in map(float, PROTOCOL["budgets"]):
        alpha = float(frozen[budget]["alpha"])
        for mode in PROTOCOL["modes"]:
            for method in ("ActionValue-OQ", "RiskMean-OQ"):
                key = f"{method}|{mode}|{budget:g}"
                chosen_selections[key] = selections[key]
            source_key = f"DualHeadV2-alpha={alpha:g}|{mode}|{budget:g}"
            chosen_selections[f"DualHeadV2-OQ|{mode}|{budget:g}"] = selections[
                source_key
            ]
    for selection in chosen_selections.values():
        assert_no_roi_selection(selection)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "raw_rows": raw_rows,
        "chosen_rows": chosen_rows,
        "frozen_alphas": {str(key): value for key, value in frozen.items()},
        "gate": gate,
        "alpha_selected_on": "fit_oof_only",
        "calibration_used": False,
        "reward_context_may_cross_oof_folds": True,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_csv(RUNS / "sweep/raw_results.csv", raw_rows)
    atomic_csv(RUNS / "sweep/chosen_results.csv", chosen_rows)
    atomic_pickle(RUNS / "sweep/chosen_selections.pkl", chosen_selections)
    atomic_json(RUNS / "sweep/frozen_alphas.json", report["frozen_alphas"])
    atomic_json(RUNS / "sweep/report.json", report)
    atomic_json(
        completion,
        {
            "complete": True,
            "rows": len(raw_rows),
            "frozen_alphas_sha256": file_sha256(RUNS / "sweep/frozen_alphas.json"),
            "calibration_used": False,
        },
    )
    return report


def run_calibration() -> dict[str, Any]:
    completion = RUNS / "calibration/completion.json"
    if _complete(completion):
        return read_json(RUNS / "calibration/report.json")
    verify_hashes()
    sweep_completion = _complete(RUNS / "sweep/completion.json")
    if not sweep_completion:
        raise RuntimeError("Fit-OOF alpha sweep must complete first")
    alpha_path = RUNS / "sweep/frozen_alphas.json"
    if sweep_completion.get("frozen_alphas_sha256") != file_sha256(alpha_path):
        raise RuntimeError("Frozen alpha artifact changed")
    frozen_raw = read_json(alpha_path)
    frozen = {float(key): dict(value) for key, value in frozen_raw.items()}
    if any(value.get("calibration_used") is not False for value in frozen.values()):
        raise RuntimeError("Calibration leaked into alpha selection")

    artifact, records = _load_score_artifact("calibration")
    scores = SOURCE.numeric_score_arrays(artifact)
    data = SOURCE.load_role("calibration")
    samples = data["samples"]
    rewards = _filtered_rewards(SOURCE.reward_map("calibration"))
    value_map = no_roi_score_map(records, scores["value"])
    risk_mean_map = no_roi_score_map(records, scores["risk_mean"])
    tasks = []
    selections = {}
    for budget in map(float, PROTOCOL["budgets"]):
        alpha = float(frozen[budget]["alpha"])
        candidate_map = no_roi_score_map(
            records,
            blend_mean_lower(scores["risk_mean"], scores["risk_lower"], alpha),
        )
        method_maps = {
            "ActionValue-OQ": value_map,
            "RiskMean-OQ": risk_mean_map,
            "DualHeadV2-OQ": candidate_map,
        }
        capacity = _original_capacity(samples, budget)
        for mode in PROTOCOL["modes"]:
            teacher = _schedule(
                samples,
                rewards,
                mode=str(mode),
                budget=budget,
                capacity=capacity,
            )
            for method, values in method_maps.items():
                schedule = _schedule(
                    samples,
                    values,
                    mode=str(mode),
                    budget=budget,
                    capacity=capacity,
                )
                key = f"{method}|{mode}|{budget:g}"
                selections[key] = dict(schedule["selection"])
                tasks.append(
                    {
                        "role": "calibration",
                        "output_role": "calibration",
                        "method": method,
                        "mode": str(mode),
                        "budget": budget,
                        "selection": schedule["selection"],
                        "schedule": schedule,
                        "teacher_value": float(teacher["total_value"]),
                        "capacity_bytes": capacity,
                        "alpha": alpha if method == "DualHeadV2-OQ" else None,
                    }
                )
    rows = _evaluate_tasks(tasks, data=data, rewards=rewards)
    if len(rows) != 18 or len({_row_key(row) for row in rows}) != 18:
        raise RuntimeError("Frozen calibration matrix is incomplete or duplicated")
    for selection in selections.values():
        assert_no_roi_selection(selection)
    gate = evaluate_dualheadv2_gate(rows, PROTOCOL)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "calibration",
        "rows": rows,
        "frozen_alphas": frozen_raw,
        "gate": gate,
        "alpha_selected_on": "fit_oof_only",
        "calibration_used_for_training_or_selection": False,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_csv(RUNS / "calibration/results.csv", rows)
    atomic_json(RUNS / "calibration/selections.json", selections)
    atomic_json(RUNS / "calibration/report.json", report)
    atomic_json(
        completion,
        {
            "complete": True,
            "passed": bool(gate["passed"]),
            "frozen_alphas_sha256": file_sha256(alpha_path),
            "calibration_used_for_training_or_selection": False,
        },
    )
    return report


def _result_table(rows: Sequence[Mapping[str, Any]], role: str) -> list[str]:
    selected = [row for row in rows if str(row["role"]) == role]
    by_key = {_row_key(row): row for row in selected}
    lines = [
        f"## {role}",
        "",
        "| 模式 | 预算 | 方法 | Official AP | AP75 | KB/图 | O/Q90 | 容量利用率 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for mode in PROTOCOL["modes"]:
        for budget in map(float, PROTOCOL["budgets"]):
            for method in PROTOCOL["methods"]:
                row = by_key[(str(method), str(mode), budget)]
                lines.append(
                    f"| {mode} | {budget:.0%} | {method} | "
                    f"{float(row['official_ap50_95']):.6f} | "
                    f"{float(row['official_ap75']):.6f} | "
                    f"{float(row['kb_per_image']):.2f} | "
                    f"{int(row['original_count'])}/{int(row['q90_count'])} | "
                    f"{float(row['capacity_utilization']):.2%} |"
                )
    return lines


def run_report() -> dict[str, Any]:
    completion = RUNS / "COMPLETED.json"
    if _complete(completion):
        return read_json(completion)
    for stage in ("ceiling", "sweep", "calibration"):
        if not _complete(RUNS / f"{stage}/completion.json"):
            raise RuntimeError(f"Stage {stage} is incomplete")
    hashes = verify_hashes()
    ceiling = read_json(RUNS / "ceiling/report.json")
    sweep = read_json(RUNS / "sweep/report.json")
    calibration = read_json(RUNS / "calibration/report.json")
    open_sc = bool(ceiling["gate"]["passed"])
    open_dualheadv2 = bool(
        sweep["gate"]["passed"] and calibration["gate"]["passed"]
    )
    report = {
        "complete": True,
        "protocol_id": PROTOCOL["protocol_id"],
        "result_type": PROTOCOL["result_type"],
        "selection_context_gate": ceiling["gate"],
        "fit_oof_gate": sweep["gate"],
        "calibration_gate": calibration["gate"],
        "frozen_alphas": calibration["frozen_alphas"],
        "open_selection_context_model_branch": open_sc,
        "selection_context_models_trained": False,
        "open_dualheadv2_three_seed_protocol": open_dualheadv2,
        "source_digest": source_digest(hashes),
        "reward_context_may_cross_oof_folds": True,
        "calibration_used_for_training_or_selection": False,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    lines = [
        "# COCO DualHead-v2 No-ROI OOF 诊断结果",
        "",
        "> 仅使用 canonical train2017 fit=9,678 的冻结5-fold OOF分数与 calibration=571；未访问 audit、val2017/final 或 registered final。",
        "",
        "## No-ROI Selection-Context Teacher gate",
        "",
        f"- passed: `{str(open_sc).lower()}`",
        f"- gain points: {int(ceiling['gate']['gain_points'])}/3",
        "",
        "| 预算 | SelectionContext − Static AP |",
        "|---:|---:|",
    ]
    for row in ceiling["gate"]["budget_rows"]:
        lines.append(
            f"| {float(row['budget']):.0%} | {float(row['delta_ap']):+.6f} |"
        )
    lines.extend([""] + _result_table(sweep["chosen_rows"], "fit_oof"))
    lines.extend([""] + _result_table(calibration["rows"], "calibration"))
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Selection-Context branch: `{str(open_sc).lower()}`",
            f"- DualHead-v2 fit OOF: `{str(bool(sweep['gate']['passed'])).lower()}`",
            f"- DualHead-v2 frozen calibration: `{str(bool(calibration['gate']['passed'])).lower()}`",
            f"- Open three-seed DualHead-v2 protocol: `{str(open_dualheadv2).lower()}`",
            "",
            "Selection-Context gate 失败时没有训练 SC-Value/SC-VC；该停止规则在查看 calibration 学习器结果前已冻结。",
            "",
        ]
    )
    note = ROOT / "notes/results_zh.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    temporary = note.with_name(f".{note.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, note)
    atomic_json(RUNS / "report.json", report)
    atomic_json(completion, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "ceiling", "sweep", "calibration", "report", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    reports = {}
    for stage in ("preflight", "ceiling", "sweep", "calibration", "report"):
        if arguments.stage not in (stage, "all"):
            continue
        print(f"[stage] {stage}", flush=True)
        reports[stage] = globals()[f"run_{stage}"]()
    if "report" in reports:
        print(json.dumps(reports["report"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
