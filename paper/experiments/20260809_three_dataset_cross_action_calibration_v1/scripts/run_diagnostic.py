#!/usr/bin/env python3
"""Run one cache-only cross-action calibration diagnostic dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from calibration_core import (
    ACTIONS,
    BUDGETS,
    DATASETS,
    DEPLOYABLE_METHODS,
    METHODS,
    apply_isotonic,
    assert_fixed_membership,
    build_fixed_membership_groups,
    crossfit_isotonic,
    fit_isotonic,
    restrict_to_oq_values,
    validate_action_availability,
    validate_protocol,
    validate_result_rows,
)


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
validate_protocol(PROTOCOL)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE_ROOT = REPO / PROTOCOL["source_experiments"]["actionvalue_roi_gate"]
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
SOURCE = _load_module(
    "cross_action_calibration_roi_gate_source", SOURCE_ROOT / "scripts/run_roi_gate.py"
)

sys.path.insert(0, str(REPO / "paper/experiments/GAIR_v5"))
from gair.selector import select_actions_for_budget  # noqa: E402


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
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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


def complete(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text(encoding="utf-8")).get("complete"))


def dataset_run(dataset: str) -> Path:
    return RUNS / dataset


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def reward_digest(values: Mapping[int, Mapping[str, float]]) -> str:
    normalized = {
        str(int(image_id)): {
            action: float(actions[action]) for action in ACTIONS[1:] if action in actions
        }
        for image_id, actions in sorted(values.items())
    }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_preflight(dataset: str) -> dict[str, Any]:
    output = dataset_run(dataset) / "preflight"
    marker = output / "completion.json"
    if complete(marker):
        return json.loads((output / "report.json").read_text(encoding="utf-8"))
    source_completion = SOURCE_ROOT / f"runs/{dataset}/COMPLETED.json"
    if not complete(source_completion):
        raise RuntimeError(f"Frozen source experiment is incomplete for {dataset}")
    fit = SOURCE.load_role(dataset, "fit_oof")
    calibration = SOURCE.load_role(dataset, "calibration")
    expected = DATASETS[dataset]
    if len(fit["samples"]) != expected["fit"] or len(calibration["samples"]) != expected["calibration"]:
        raise RuntimeError(f"Role population changed for {dataset}")
    fit_identities = {str(sample["identity"]) for sample in fit["samples"]}
    calibration_identities = {str(sample["identity"]) for sample in calibration["samples"]}
    if fit_identities & calibration_identities:
        raise RuntimeError(f"Fit/calibration identity overlap for {dataset}")
    raw_fit = SOURCE.load_oq_scores(dataset, "fit_oof")
    raw_calibration = SOURCE.load_oq_scores(dataset, "calibration")
    if raw_fit.shape != (len(BUDGETS), expected["fit"], len(ACTIONS)):
        raise RuntimeError(f"Fit score shape changed for {dataset}: {raw_fit.shape}")
    if raw_calibration.shape != (len(BUDGETS), expected["calibration"], len(ACTIONS)):
        raise RuntimeError(f"Calibration score shape changed for {dataset}: {raw_calibration.shape}")
    fit_rewards = SOURCE.get_rewards(dataset, "fit_oof")
    calibration_rewards = SOURCE.get_rewards(dataset, "calibration")
    for role, data, rewards in (
        ("fit_oof", fit, fit_rewards),
        ("calibration", calibration, calibration_rewards),
    ):
        image_ids = {int(sample["image_id"]) for sample in data["samples"]}
        if set(map(int, rewards)) != image_ids:
            raise RuntimeError(f"Reward population changed for {dataset}/{role}")
        validate_action_availability(data["samples"])
        for sample in data["samples"]:
            image_id = int(sample["image_id"])
            for action in ACTIONS[1:]:
                if bool(sample["available"].get(action, False)) and action not in rewards[image_id]:
                    raise RuntimeError(
                        f"Available action lacks reward for {dataset}/{role}/{image_id}/{action}"
                    )
    source_paths = [
        SOURCE_ROOT / "configs/protocol.json",
        source_completion,
        *SOURCE.source_score_paths(dataset, "fit_oof"),
        *SOURCE.source_score_paths(dataset, "calibration"),
    ]
    source_text = "\n".join(str(path).lower() for path in source_paths)
    for forbidden in PROTOCOL["forbidden_source_substrings"]:
        if str(forbidden).lower() in source_text:
            raise RuntimeError(f"Forbidden source path configured: {forbidden}")
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "dataset": dataset,
        "role_counts": {"fit_oof": len(fit["samples"]), "calibration": len(calibration["samples"])},
        "identity_overlap": 0,
        "source_files": {str(path.resolve()): file_sha256(path) for path in sorted(set(source_paths))},
        "score_digests": {"fit_oof": array_digest(raw_fit), "calibration": array_digest(raw_calibration)},
        "reward_digests": {"fit_oof": reward_digest(fit_rewards), "calibration": reward_digest(calibration_rewards)},
        "final_accessed": False,
        "detector_inference": False,
        "gpu_inference": False,
        "passed": True,
    }
    atomic_json(output / "report.json", report)
    atomic_json(marker, {"complete": True, "report_sha256": file_sha256(output / "report.json")})
    return report


def reward_targets(
    data: Mapping[str, Any], rewards: Mapping[int, Mapping[str, float]], action: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(data["samples"]), np.nan, dtype=np.float64)
    valid = np.zeros(len(data["samples"]), dtype=bool)
    for index, sample in enumerate(data["samples"]):
        image_id = int(sample["image_id"])
        if bool(sample["available"].get(action, False)) and action in rewards[image_id]:
            values[index] = float(rewards[image_id][action])
            valid[index] = True
    if not valid.any():
        raise RuntimeError(f"No valid reward targets for action {action}")
    return values, valid


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def score_diagnostics(
    raw: np.ndarray,
    calibrated: np.ndarray,
    data: Mapping[str, Any],
    rewards: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    target_by_action = {
        action: reward_targets(data, rewards, action) for action in ACTIONS[1:]
    }
    both_valid = target_by_action["original"][1] & target_by_action["q90"][1]
    oracle_action = np.asarray(
        [
            int(target_by_action["q90"][0][index] > target_by_action["original"][0][index])
            for index in range(len(data["samples"]))
        ]
    )
    for budget_index, budget in enumerate(BUDGETS):
        for label, values in (("raw", raw), ("calibrated", calibrated)):
            predicted_action = (values[budget_index, :, 2] > values[budget_index, :, 1]).astype(int)
            for action_index, action in enumerate(ACTIONS[1:], start=1):
                prediction = values[budget_index, :, action_index]
                target, valid = target_by_action[action]
                rows.append(
                    {
                        "score_type": label,
                        "budget": float(budget),
                        "action": action,
                        "available_images": int(valid.sum()),
                        "mae": float(np.mean(np.abs(prediction[valid] - target[valid]))),
                        "pearson": correlation(prediction[valid], target[valid]),
                        "spearman": correlation(_rankdata(prediction[valid]), _rankdata(target[valid])),
                        "action_accuracy": float(np.mean(predicted_action[both_valid] == oracle_action[both_valid])) if both_valid.any() else 0.0,
                    }
                )
    return rows


def run_calibration(dataset: str) -> None:
    output = dataset_run(dataset) / "calibration"
    marker = output / "completion.json"
    if complete(marker):
        return
    if not complete(dataset_run(dataset) / "preflight/completion.json"):
        raise RuntimeError("Preflight must complete before calibration")
    fit = SOURCE.load_role(dataset, "fit_oof")
    calibration = SOURCE.load_role(dataset, "calibration")
    raw_fit = np.asarray(SOURCE.load_oq_scores(dataset, "fit_oof"), dtype=np.float64)
    raw_calibration = np.asarray(SOURCE.load_oq_scores(dataset, "calibration"), dtype=np.float64)
    fit_rewards = SOURCE.get_rewards(dataset, "fit_oof")
    calibration_rewards = SOURCE.get_rewards(dataset, "calibration")
    folds = SOURCE.deterministic_folds(fit)
    if len(folds) != int(PROTOCOL["calibration"]["crossfit_folds"]):
        raise RuntimeError("Source fold count changed")
    calibrated_fit = raw_fit.copy()
    calibrated_calibration = raw_calibration.copy()
    calibrated_fit[:, :, 0] = 0.0
    calibrated_calibration[:, :, 0] = 0.0
    models: dict[str, Any] = {}
    for budget_index, budget in enumerate(BUDGETS):
        for action_index, action in enumerate(ACTIONS[1:], start=1):
            targets, valid = reward_targets(fit, fit_rewards, action)
            crossfit, fold_reports = crossfit_isotonic(
                raw_fit[budget_index, :, action_index],
                targets,
                folds,
                bins=int(PROTOCOL["calibration"]["bins"]),
                valid_mask=valid,
            )
            full_model = fit_isotonic(
                raw_fit[budget_index, valid, action_index],
                targets[valid],
                bins=int(PROTOCOL["calibration"]["bins"]),
            )
            calibrated_fit[budget_index, :, action_index] = crossfit
            calibrated_calibration[budget_index, :, action_index] = apply_isotonic(
                full_model, raw_calibration[budget_index, :, action_index]
            )
            key = f"{float(budget):g}|{action}"
            models[key] = {
                "budget": float(budget),
                "action": action,
                "full_fit": full_model,
                "crossfit": [
                    {
                        "fold": int(report["fold"]),
                        "train_count": len(report["train_indices"]),
                        "heldout_count": len(report["heldout_indices"]),
                        "heldout_sha256": hashlib.sha256(
                            np.asarray(report["heldout_indices"], dtype=np.int64).tobytes()
                        ).hexdigest(),
                        "model": report["model"],
                    }
                    for report in fold_reports
                ],
            }
    if not np.isfinite(calibrated_fit).all() or not np.isfinite(calibrated_calibration).all():
        raise RuntimeError(f"Non-finite calibrated scores for {dataset}")
    artifact = {
        "raw_fit_oof_digest": array_digest(raw_fit),
        "raw_calibration_digest": array_digest(raw_calibration),
        "calibrated_fit_oof": calibrated_fit,
        "calibrated_calibration": calibrated_calibration,
        "models": models,
        "fit_only": True,
        "final_accessed": False,
    }
    atomic_pickle(output / "scores.pkl", artifact)
    diagnostics = [
        {"dataset": dataset, "role": "fit_oof", **row}
        for row in score_diagnostics(raw_fit, calibrated_fit, fit, fit_rewards)
    ]
    diagnostics.extend(
        {"dataset": dataset, "role": "calibration", **row}
        for row in score_diagnostics(
            raw_calibration, calibrated_calibration, calibration, calibration_rewards
        )
    )
    atomic_csv(output / "diagnostics.csv", diagnostics)
    atomic_json(
        output / "report.json",
        {
            "dataset": dataset,
            "models": len(models),
            "bins": int(PROTOCOL["calibration"]["bins"]),
            "crossfit_folds": len(folds),
            "fit_only": True,
            "calibration_used_for_fit_or_selection": False,
            "final_accessed": False,
        },
    )
    atomic_json(marker, {"complete": True, "scores_sha256": file_sha256(output / "scores.pkl")})


def fixed_membership_selection(
    data: Mapping[str, Any],
    image_ids: set[int],
    values: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
) -> dict[str, Any]:
    groups = build_fixed_membership_groups(data["samples"], image_ids, values)
    if mode == "fixed_byte":
        capacity = float(budget) * sum(
            float(sample["payload_bytes"]["original"]) for sample in data["samples"]
        )
    elif mode == "fixed_call":
        capacity = float("inf")
    else:
        raise ValueError(mode)
    scheduled = select_actions_for_budget(groups, capacity)
    selection = {
        int(item["image_id"]): str(item["action_id"]) for item in scheduled["selected"]
    }
    assert_fixed_membership(selection, image_ids)
    if mode == "fixed_byte" and float(scheduled["total_cost"]) > capacity + 1e-6:
        raise RuntimeError("Fixed-membership capacity exceeded")
    return {
        "selection": selection,
        "total_value": float(scheduled["total_value"]),
        "total_cost_bytes": float(scheduled["total_cost"]),
    }


def membership_overlap(left: Mapping[int, str], right: Mapping[int, str]) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    return float(len(a & b) / max(1, len(a | b)))


def action_agreement(left: Mapping[int, str], right: Mapping[int, str]) -> float:
    common = set(map(int, left)) & set(map(int, right))
    return float(
        sum(str(left[image_id]) == str(right[image_id]) for image_id in common)
        / max(1, len(common))
    )


def evaluate_role(
    dataset: str,
    role: str,
    raw: np.ndarray,
    calibrated: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = SOURCE.load_role(dataset, role)
    rewards = SOURCE.get_rewards(dataset, role)
    oq_rewards = restrict_to_oq_values(rewards)
    rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for budget_index, budget in enumerate(BUDGETS):
        raw_map = SOURCE.score_map(data, raw[budget_index], include_roi=False)
        calibrated_map = SOURCE.score_map(data, calibrated[budget_index], include_roi=False)
        for mode in PROTOCOL["modes"]:
            raw_result = SOURCE.schedule(data, raw_map, mode, float(budget))
            calibrated_result = SOURCE.schedule(data, calibrated_map, mode, float(budget))
            teacher_result = SOURCE.schedule(data, oq_rewards, mode, float(budget))
            calibrated_oracle_action = fixed_membership_selection(
                data,
                set(map(int, calibrated_result["selection"])),
                oq_rewards,
                mode=mode,
                budget=float(budget),
            )
            oracle_calibrated_action = fixed_membership_selection(
                data,
                set(map(int, teacher_result["selection"])),
                calibrated_map,
                mode=mode,
                budget=float(budget),
            )
            methods = {
                "Raw-ActionValue-OQ": (raw_result, True, False, "learned", "learned"),
                "PerAction-Isotonic-OQ": (calibrated_result, True, False, "learned_calibrated", "learned_calibrated"),
                "CalibratedMembership-OracleAction": (calibrated_oracle_action, False, True, "learned_calibrated", "oracle"),
                "OracleMembership-CalibratedAction": (oracle_calibrated_action, False, True, "oracle", "learned_calibrated"),
                "FullOracle-OQ": (teacher_result, False, True, "oracle", "oracle"),
            }
            teacher_selection = teacher_result["selection"]
            for method, (result, deployable, gt_aware, membership_source, action_source) in methods.items():
                selection = result["selection"]
                row = SOURCE.evaluate_selection(
                    dataset,
                    role,
                    method,
                    mode,
                    float(budget),
                    data,
                    rewards,
                    selection,
                    float(result["total_value"]),
                    None,
                )
                row.update(
                    {
                        "deployable": bool(deployable),
                        "gt_aware": bool(gt_aware),
                        "membership_source": membership_source,
                        "action_source": action_source,
                        "membership_jaccard_with_teacher": membership_overlap(selection, teacher_selection),
                        "action_agreement_with_teacher_on_common": action_agreement(selection, teacher_selection),
                        "calibration_source": "fit_oof_only" if method != "Raw-ActionValue-OQ" else None,
                    }
                )
                rows.append(row)
                key = f"{role}|{method}|{mode}|{float(budget):g}"
                selections[key] = {str(image_id): str(action) for image_id, action in selection.items()}
                print(
                    f"[{dataset}] {role} {mode} {budget:g} {method}: "
                    f"AP={float(row['official_ap50_95']):.6f} KB={float(row['kb_per_image']):.2f}",
                    flush=True,
                )
    return rows, selections


def write_notes(dataset: str, rows: Sequence[Mapping[str, Any]]) -> None:
    output = dataset_run(dataset) / "results_zh.md"
    lines = [
        f"# {dataset.upper()} Cross-Action Calibration 诊断",
        "",
        "主表仅列 deployable Raw 与 PerAction-Isotonic；Oracle 行仅用于误差分解。",
        "",
    ]
    for role in ("fit_oof", "calibration"):
        for mode in ("fixed_byte", "fixed_call"):
            lines.extend([f"## {role} / {mode}", "", "| 方法 | 10% AP | 25% AP | 40% AP |", "|---|---:|---:|---:|"])
            for method in DEPLOYABLE_METHODS:
                values = {
                    float(row["budget"]): float(row["official_ap50_95"])
                    for row in rows
                    if row["role"] == role and row["mode"] == mode and row["method"] == method
                }
                lines.append(
                    f"| {method} | {values[0.1]:.6f} | {values[0.25]:.6f} | {values[0.4]:.6f} |"
                )
            lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(dataset: str) -> None:
    output = dataset_run(dataset) / "evaluation"
    marker = output / "completion.json"
    if complete(marker):
        return
    if not complete(dataset_run(dataset) / "calibration/completion.json"):
        raise RuntimeError("Calibration must complete before evaluation")
    artifact = load_pickle(dataset_run(dataset) / "calibration/scores.pkl")
    raw_fit = np.asarray(SOURCE.load_oq_scores(dataset, "fit_oof"), dtype=np.float64)
    raw_calibration = np.asarray(SOURCE.load_oq_scores(dataset, "calibration"), dtype=np.float64)
    calibrated_fit = np.asarray(artifact["calibrated_fit_oof"], dtype=np.float64)
    calibrated_calibration = np.asarray(artifact["calibrated_calibration"], dtype=np.float64)
    rows, selections = evaluate_role(dataset, "fit_oof", raw_fit, calibrated_fit)
    calibration_rows, calibration_selections = evaluate_role(
        dataset, "calibration", raw_calibration, calibrated_calibration
    )
    rows.extend(calibration_rows)
    selections.update(calibration_selections)
    validate_result_rows(rows, PROTOCOL, dataset)
    atomic_csv(output / "results.csv", rows)
    atomic_json(output / "selections.json", selections)
    atomic_json(
        output / "report.json",
        {
            "dataset": dataset,
            "rows": len(rows),
            "methods": list(METHODS),
            "deployable_methods": list(DEPLOYABLE_METHODS),
            "oracle_methods_gt_aware": True,
            "final_accessed": False,
            "result_status": PROTOCOL["result_status"],
        },
    )
    write_notes(dataset, rows)
    atomic_json(marker, {"complete": True, "results_sha256": file_sha256(output / "results.csv")})
    atomic_json(
        dataset_run(dataset) / "COMPLETED.json",
        {
            "complete": True,
            "dataset": dataset,
            "preflight": True,
            "calibration": True,
            "evaluation": True,
            "final_accessed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument(
        "--stage", choices=("preflight", "calibrate", "evaluate", "all"), default="all"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ("preflight", "all"):
        run_preflight(args.dataset)
    if args.stage in ("calibrate", "all"):
        run_calibration(args.dataset)
    if args.stage in ("evaluate", "all"):
        run_evaluation(args.dataset)


if __name__ == "__main__":
    main()
