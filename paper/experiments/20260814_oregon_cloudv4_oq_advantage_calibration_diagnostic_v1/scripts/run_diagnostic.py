#!/usr/bin/env python3
"""Run the read-only Cloud-V4 Original/Q90 calibration diagnostic."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
SOURCE = REPO / str(PROTOCOL["source_experiment"])
sys.dont_write_bytecode = True


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("cloudv4_oq_advantage_core", ROOT / "scripts/calibration_core.py")
EVAL = load_module("cloudv4_oq_advantage_source_eval", SOURCE / "scripts/evaluate.py")
CAL = EVAL.CALIBRATION_CORE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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


def preflight() -> dict[str, Any]:
    CORE.validate_protocol(PROTOCOL)
    source_protocol = json.loads((SOURCE / "configs/protocol.json").read_text(encoding="utf-8"))
    if source_protocol["cloud_model"]["sha256"] != PROTOCOL["source_cloud_sha256"]:
        raise RuntimeError("Cloud-V4 source hash changed")
    if source_protocol["model_seeds"] != PROTOCOL["seeds"]:
        raise RuntimeError("source seed matrix changed")
    required = [
        SOURCE / "runs/learners/completion.json",
        SOURCE / "runs/rewards/fit/reward_map.pkl",
        SOURCE / "runs/rewards/calibration/reward_map.pkl",
        SOURCE / "runs/point/results.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"source experiment incomplete: {missing}")
    return {
        "complete": True,
        "source_protocol_sha256": sha256(SOURCE / "configs/protocol.json"),
        "source_learner_completion_sha256": sha256(SOURCE / "runs/learners/completion.json"),
        "source_point_results_sha256": sha256(SOURCE / "runs/point/results.csv"),
        "source_modification_allowed": False,
    }


def fit_oof_scores() -> tuple[list[int], np.ndarray]:
    expected_ids = [int(sample["image_id"]) for sample in EVAL.load_role("fit")["samples"]]
    per_seed = []
    for seed in map(int, PROTOCOL["seeds"]):
        score_by_id: dict[int, np.ndarray] = {}
        for fold in range(int(PROTOCOL["folds"])):
            path = (
                SOURCE
                / f"runs/learners/jobs/{PROTOCOL['source_method']}/seed_{seed}/fold_{fold}/scores.npz"
            )
            with np.load(path, allow_pickle=False) as artifact:
                image_ids = np.asarray(artifact["image_ids_fit_oof"], dtype=np.int64)
                scores = np.asarray(artifact["scores_fit_oof"], dtype=np.float64)
            if scores.ndim != 3 or scores.shape[0] != len(PROTOCOL["budgets"]):
                raise RuntimeError(f"OOF score shape changed: {path}/{scores.shape}")
            for index, image_id in enumerate(map(int, image_ids)):
                if image_id in score_by_id:
                    raise RuntimeError(f"duplicate OOF image for seed {seed}: {image_id}")
                score_by_id[image_id] = scores[:, index, :]
        if set(score_by_id) != set(expected_ids):
            raise RuntimeError(f"OOF coverage changed for seed {seed}")
        per_seed.append(np.stack([score_by_id[image_id] for image_id in expected_ids], axis=1))
    output = np.mean(np.stack(per_seed, axis=0), axis=0)
    if not np.isfinite(output[:, :, :3]).all():
        raise RuntimeError("non-finite OOF Edge/Original/Q90 scores")
    return expected_ids, output


def fit_calibrators() -> tuple[dict[str, Any], dict[str, Any]]:
    image_ids, scores = fit_oof_scores()
    reward_map = load_pickle(SOURCE / "runs/rewards/fit/reward_map.pkl")
    target_original = np.asarray(
        [reward_map[image_id]["original"] for image_id in image_ids], dtype=np.float64
    )
    target_q90 = np.asarray(
        [reward_map[image_id]["q90"] for image_id in image_ids], dtype=np.float64
    )
    target_advantage = target_original - target_q90
    models: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        original = scores[budget_index, :, 1]
        q90 = scores[budget_index, :, 2]
        predicted_advantage = original - q90
        original_model = CAL.fit_isotonic(
            original, target_original, bins=int(PROTOCOL["isotonic_bins"])
        )
        q90_model = CAL.fit_isotonic(
            q90, target_q90, bins=int(PROTOCOL["isotonic_bins"])
        )
        advantage_model = CAL.fit_isotonic(
            predicted_advantage,
            target_advantage,
            bins=int(PROTOCOL["isotonic_bins"]),
        )
        calibrated_advantage = CAL.apply_isotonic(advantage_model, predicted_advantage)
        models[f"{budget:g}"] = {
            "original": original_model,
            "q90": q90_model,
            "advantage": advantage_model,
        }
        diagnostics[f"{budget:g}"] = {
            "raw": CORE.advantage_metrics(predicted_advantage, target_advantage),
            "isotonic_in_sample": CORE.advantage_metrics(
                calibrated_advantage, target_advantage
            ),
        }
    return models, diagnostics


def margin_candidates() -> list[float]:
    _, scores = fit_oof_scores()
    advantage = scores[0, :, 1] - scores[0, :, 2]
    positive = advantage[advantage > 0.0]
    if not len(positive):
        return [0.0]
    candidates = {
        0.0,
        *map(
            float,
            np.quantile(
                positive,
                np.asarray(PROTOCOL["margin_quantiles"], dtype=np.float64),
            ),
        ),
        float(positive.max() + 1e-6),
    }
    return sorted(candidates)


def method_scores(
    raw: np.ndarray,
    models: Mapping[str, Any],
    budget_index: int,
    budget: float,
    *,
    original_margin: float,
) -> dict[str, np.ndarray]:
    matrix = np.asarray(raw[budget_index], dtype=np.float64)
    model = models[f"{budget:g}"]
    calibrated_original = CAL.apply_isotonic(model["original"], matrix[:, 1])
    calibrated_q90 = CAL.apply_isotonic(model["q90"], matrix[:, 2])
    calibrated_advantage = CAL.apply_isotonic(
        model["advantage"], matrix[:, 1] - matrix[:, 2]
    )
    return {
        "Raw-CAVS-OQR": matrix.copy(),
        "PerAction-Isotonic-OQR": CORE.replace_per_action(
            matrix, calibrated_original, calibrated_q90
        ),
        "Advantage-Isotonic-Midpoint-OQR": CORE.replace_oq_advantage(
            matrix, calibrated_advantage, anchor="midpoint"
        ),
        "Advantage-Isotonic-QAnchor-OQR": CORE.replace_oq_advantage(
            matrix, calibrated_advantage, anchor="q90"
        ),
        "Calibration-Frozen-OQMargin-OQR": CORE.apply_original_margin(
            matrix, margin=original_margin
        ),
        "Q90-Preferred-Diagnostic": CORE.prefer_q90(matrix),
    }


def calibrate_original_margin() -> dict[str, Any]:
    role = str(PROTOCOL["margin_selection_role"])
    if role != "calibration":
        raise RuntimeError("Original margin must be selected on calibration")
    data = EVAL.load_role(role, fusion="MAUR")
    records = EVAL.role_records(role, "OQR12")
    raw = EVAL._budget_matrix(EVAL.full_job_score(PROTOCOL["source_method"], role))
    EVAL._EVAL_STATE.clear()
    EVAL._EVAL_STATE[role] = data
    candidates = []
    for margin in margin_candidates():
        points = []
        for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
            values = CORE.apply_original_margin(raw[budget_index], margin=margin)
            selection = EVAL.schedule_joint(
                data["samples"],
                EVAL._score_map(records, values),
                mode="fixed_byte",
                budget=budget,
            )
            row = EVAL._evaluate(
                {
                    "role": role,
                    "method": f"margin={margin:.12g}",
                    "policy": "Joint-OQR",
                    "mode": "fixed_byte",
                    "budget": budget,
                    "selection": selection,
                }
            )
            points.append(
                {
                    "budget": budget,
                    "official_ap50_95": float(row["official_ap50_95"]),
                    "original_count": int(row["original_count"]),
                    "q90_count": int(row["q90_count"]),
                    "roi_count": int(row["roi_count"]),
                    "capacity_utilization": float(row["capacity_utilization"]),
                }
            )
        candidates.append({"margin": float(margin), "points": points})
    EVAL._EVAL_STATE.clear()
    raw = next(row for row in candidates if row["margin"] == 0.0)
    raw_ap = {float(point["budget"]): float(point["official_ap50_95"]) for point in raw["points"]}
    eligible = []
    for candidate in candidates:
        points = {float(point["budget"]): point for point in candidate["points"]}
        safe = (
            float(points[0.1]["official_ap50_95"]) >= raw_ap[0.1] - float(PROTOCOL["maximum_low_budget_loss"])
            and float(points[0.25]["official_ap50_95"]) >= raw_ap[0.25] - float(PROTOCOL["maximum_low_budget_loss"])
            and float(points[0.4]["official_ap50_95"]) >= float(points[0.25]["official_ap50_95"])
            and min(float(point["capacity_utilization"]) for point in points.values())
            >= float(PROTOCOL["minimum_capacity_utilization"])
        )
        candidate["eligible"] = bool(safe)
        candidate["mean_ap"] = float(
            np.mean([float(point["official_ap50_95"]) for point in points.values()])
        )
        candidate["original_total"] = int(
            sum(int(point["original_count"]) for point in points.values())
        )
        if safe:
            eligible.append(candidate)
    pool = eligible or [raw]
    chosen = max(
        pool,
        key=lambda row: (
            float(row["mean_ap"]),
            float(next(point for point in row["points"] if float(point["budget"]) == 0.4)["official_ap50_95"]),
            -int(row["original_total"]),
            -float(row["margin"]),
        ),
    )
    return {
        "fit_candidate_source": "fit_oof_advantage_quantiles",
        "selection_role": role,
        "final_used_for_selection": False,
        "chosen_margin": float(chosen["margin"]),
        "eligible_candidates": len(eligible),
        "candidates": candidates,
    }


def evaluate_role(
    role: str, models: Mapping[str, Any], *, original_margin: float
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    data = EVAL.load_role(role, fusion="MAUR")
    records = EVAL.role_records(role, "OQR12")
    raw = EVAL._budget_matrix(EVAL.full_job_score(PROTOCOL["source_method"], role))
    EVAL._EVAL_STATE.clear()
    EVAL._EVAL_STATE[role] = data
    rows = []
    selections: dict[str, dict[str, str]] = {}
    score_diagnostics: dict[str, Any] = {}
    reward_map = (
        load_pickle(SOURCE / "runs/rewards/calibration/reward_map.pkl")
        if role == "calibration"
        else None
    )
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        variants = method_scores(
            raw,
            models,
            budget_index,
            budget,
            original_margin=original_margin,
        )
        if role == "calibration" and reward_map is not None:
            image_ids = [int(record["image_id"]) for record in records]
            target = np.asarray(
                [reward_map[image_id]["original"] - reward_map[image_id]["q90"] for image_id in image_ids],
                dtype=np.float64,
            )
            score_diagnostics[f"{budget:g}"] = {
                method: CORE.advantage_metrics(values[:, 1] - values[:, 2], target)
                for method, values in variants.items()
                if method != "Q90-Preferred-Diagnostic"
            }
        for method in PROTOCOL["methods"]:
            score_map = EVAL._score_map(records, variants[method])
            selection = EVAL.schedule_joint(
                data["samples"], score_map, mode="fixed_byte", budget=budget
            )
            row = EVAL._evaluate(
                {
                    "role": role,
                    "method": method,
                    "policy": "Joint-OQR",
                    "mode": "fixed_byte",
                    "budget": budget,
                    "selection": selection,
                }
            )
            row.update(
                {
                    "cloud_model": "Cloud-V4-epoch29-best",
                    "primary_roi_fusion": "MAUR",
                    "calibrator_fit_role": "fit_oof",
                    "final_already_consumed": True,
                    "confirmatory": False,
                }
            )
            row["capacity_gate_passed"] = float(row["capacity_utilization"]) >= float(
                PROTOCOL["minimum_capacity_utilization"]
            )
            rows.append(row)
            selections[f"{role}|{method}|fixed_byte|{budget:g}"] = {
                str(image_id): str(action) for image_id, action in selection.items()
            }
    EVAL._EVAL_STATE.clear()
    return rows, selections, score_diagnostics


def source_point_rows() -> dict[float, dict[str, Any]]:
    with (SOURCE / "runs/point/results.csv").open(newline="", encoding="utf-8") as handle:
        return {float(row["budget"]): row for row in csv.DictReader(handle)}


def validate_raw_reproduction(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = source_point_rows()
    observed = {
        float(row["budget"]): row
        for row in rows
        if row["role"] == "final" and row["method"] == "Raw-CAVS-OQR"
    }
    if set(observed) != set(expected):
        raise RuntimeError("raw final reproduction budget set changed")
    for budget in expected:
        if abs(float(observed[budget]["official_ap50_95"]) - float(expected[budget]["official_ap50_95"])) > 1e-12:
            raise RuntimeError(f"raw final AP did not reproduce at {budget}")


def gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {
        (str(row["role"]), str(row["method"]), float(row["budget"])): row
        for row in rows
    }
    output = {}
    for method in PROTOCOL["deployable_methods"]:
        final_delta = {
            str(budget): float(lookup[("final", method, budget)]["official_ap50_95"])
            - float(lookup[("final", "Raw-CAVS-OQR", budget)]["official_ap50_95"])
            for budget in map(float, PROTOCOL["budgets"])
        }
        calibration_delta = {
            str(budget): float(
                lookup[("calibration", method, budget)]["official_ap50_95"]
            )
            - float(
                lookup[("calibration", "Raw-CAVS-OQR", budget)]["official_ap50_95"]
            )
            for budget in map(float, PROTOCOL["budgets"])
        }
        calibration_safe = all(
            value >= -float(PROTOCOL["maximum_low_budget_loss"])
            for value in calibration_delta.values()
        )
        calibration_monotonic = float(
            lookup[("calibration", method, 0.4)]["official_ap50_95"]
        ) >= float(lookup[("calibration", method, 0.25)]["official_ap50_95"])
        capacity_passed = all(
            float(lookup[(role, method, budget)]["capacity_utilization"])
            >= float(PROTOCOL["minimum_capacity_utilization"])
            for role in ("calibration", "final")
            for budget in map(float, PROTOCOL["budgets"])
        )
        passed = (
            final_delta["0.1"] >= -float(PROTOCOL["maximum_low_budget_loss"])
            and final_delta["0.25"] >= -float(PROTOCOL["maximum_low_budget_loss"])
            and final_delta["0.4"] >= float(PROTOCOL["minimum_40_gain"])
            and float(lookup[("final", method, 0.4)]["official_ap50_95"])
            >= float(lookup[("final", method, 0.25)]["official_ap50_95"])
            and calibration_safe
            and calibration_monotonic
            and capacity_passed
        )
        output[method] = {
            "posthoc_passed": bool(passed),
            "final_delta_vs_raw": final_delta,
            "calibration_delta_vs_raw": calibration_delta,
            "calibration_safe": bool(calibration_safe),
            "calibration_monotonic": bool(calibration_monotonic),
            "capacity_passed": bool(capacity_passed),
        }
    return {
        "exploratory_only": True,
        "final_used_for_fit_or_selection": False,
        "methods": output,
    }


def write_report(rows: Sequence[Mapping[str, Any]], gate_report: Mapping[str, Any]) -> None:
    lookup = {
        (str(row["role"]), str(row["method"]), float(row["budget"])): row
        for row in rows
    }
    lines = [
        "# Oregon Cloud-V4 O/Q Advantage Calibration 诊断",
        "",
        "本实验仅使用 fit OOF 拟合校准器；final 已被历史点实验访问，因此结果仅作 post-hoc 机制诊断。",
    ]
    for role in ("calibration", "final"):
        lines.extend(
            [
                "",
                f"## {role}",
                "",
                "| 方法 | 10% AP / O-Q-R | 25% AP / O-Q-R | 40% AP / O-Q-R |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in PROTOCOL["methods"]:
            cells = []
            for budget in map(float, PROTOCOL["budgets"]):
                row = lookup[(role, method, budget)]
                cells.append(
                    f"{float(row['official_ap50_95']):.6f} / "
                    f"{int(row['original_count'])}-{int(row['q90_count'])}-{int(row['roi_count'])}"
                )
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Post-hoc gate", ""])
    for method, result in gate_report["methods"].items():
        delta = result["final_delta_vs_raw"]
        lines.append(
            f"- {method}: passed={str(result['posthoc_passed']).lower()}, "
            f"final delta={delta['0.1']:+.6f}/{delta['0.25']:+.6f}/{delta['0.4']:+.6f}."
        )
    lines.extend(
        [
            "",
            "任何通过结果都必须在新注册终评上重新验证，不能直接作为论文确认性结论。",
        ]
    )
    (RUNS / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    completion = RUNS / "completion.json"
    if completion.is_file():
        return json.loads((RUNS / "report.json").read_text(encoding="utf-8"))
    preflight_report = preflight()
    models, fit_diagnostics = fit_calibrators()
    margin_control = calibrate_original_margin()
    rows = []
    selections = {}
    role_diagnostics = {}
    for role in ("calibration", "final"):
        role_rows, role_selections, diagnostics = evaluate_role(
            role,
            models,
            original_margin=float(margin_control["chosen_margin"]),
        )
        rows.extend(role_rows)
        selections.update(role_selections)
        role_diagnostics[role] = diagnostics
    if len(rows) != 36:
        raise RuntimeError(f"result row count changed: {len(rows)}")
    validate_raw_reproduction(rows)
    gate_report = gate(rows)
    atomic_csv(RUNS / "results.csv", rows)
    atomic_json(RUNS / "selections.json", selections)
    atomic_json(
        RUNS / "calibrators.json",
        {
            "fit_role": "fit_oof",
            "final_used_for_fit_or_selection": False,
            "models": models,
            "original_margin": margin_control,
        },
    )
    atomic_json(RUNS / "diagnostics.json", {"fit": fit_diagnostics, **role_diagnostics})
    atomic_json(RUNS / "gate.json", gate_report)
    write_report(rows, gate_report)
    report = {
        "complete": True,
        "rows": len(rows),
        "preflight": preflight_report,
        "original_margin": margin_control,
        "gate": gate_report,
        "results_sha256": sha256(RUNS / "results.csv"),
        "source_modified": False,
        "confirmatory": False,
    }
    atomic_json(RUNS / "report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"]})
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
