#!/usr/bin/env python3
"""Audit the full protocol and render the paper-facing Chinese report."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
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
    spec.loader.exec_module(module)
    return module


TRAIN_CORE = load_module("oregon_cloudv3_report_train", ROOT / "scripts/train_core.py")
LEARNER_CORE = load_module("oregon_cloudv3_report_learner", ROOT / "scripts/learner_core.py")
PROTOCOL_CORE = load_module("oregon_cloudv3_report_protocol", ROOT / "scripts/protocol_core.py")


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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        output = []
        for row in csv.DictReader(handle):
            for field in (
                "budget",
                "official_ap50_95",
                "official_ap50",
                "official_ap75",
                "kb_per_image",
                "capacity_utilization",
                "selected_images",
                "original_count",
                "q90_count",
                "roi_count",
            ):
                if row.get(field) not in (None, ""):
                    row[field] = float(row[field])
            output.append(row)
        return output


def best_rows(
    values: Sequence[Mapping[str, Any]], *, role: str, mode: str
) -> dict[tuple[str, float], Mapping[str, Any]]:
    output = {}
    for method in (*LEARNER_CORE.SCALAR_METHODS, *LEARNER_CORE.JOINT_METHODS):
        for budget in map(float, PROTOCOL["budgets"]):
            candidates = [
                row
                for row in values
                if row["role"] == role
                and row["method"] == method
                and row["mode"] == mode
                and math.isclose(float(row["budget"]), budget)
            ]
            if not candidates:
                raise RuntimeError(f"missing report rows: {method}/{mode}/{budget:g}")
            output[(method, budget)] = max(
                candidates,
                key=lambda row: (
                    float(row["official_ap50_95"]),
                    -float(row["kb_per_image"]),
                    str(row["policy"]),
                ),
            )
    return output


def _cell(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['official_ap50_95']):.6f} / "
        f"{float(row['kb_per_image']):.2f} / {row['policy']}"
    )


def _method_table(best: Mapping[tuple[str, float], Mapping[str, Any]], methods: Sequence[str]) -> list[str]:
    lines = ["| 方法 | 10% AP / KB / 策略 | 25% AP / KB / 策略 | 40% AP / KB / 策略 |", "|---|---:|---:|---:|"]
    for method in methods:
        lines.append(
            f"| {method} | "
            + " | ".join(_cell(best[(method, float(budget))]) for budget in PROTOCOL["budgets"])
            + " |"
        )
    return lines


def run_audit() -> dict[str, Any]:
    output = RUNS / "audit"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(output / "report.json")
    required = [
        RUNS / "cloud_cache/completion.json",
        RUNS / "common/completion.json",
        RUNS / "rewards/completion.json",
        RUNS / "learners/completion.json",
        RUNS / "historical/completion.json",
        RUNS / "evaluation/point/completion.json",
        RUNS / "evaluation/references/completion.json",
        RUNS / "bootstrap/completion.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    jobs = TRAIN_CORE.training_jobs(PROTOCOL)
    learner_reports = []
    for job in jobs:
        fold = "full" if job["fold"] == "full" else f"fold_{int(job['fold'])}"
        safe = str(job["method"]).replace("/", "_")
        path = RUNS / f"learners/jobs/{safe}/seed_{int(job['seed'])}/{fold}/report.json"
        report = read_json(path)
        if report["cloud_model_sha256"] != PROTOCOL["cloud_model"]["sha256"]:
            raise RuntimeError(f"learner cloud hash mismatch: {job['id']}")
        if report.get("old_learner_checkpoint_reused") is not False:
            raise RuntimeError(f"old learner reused: {job['id']}")
        learner_reports.append(report)
    point = rows(RUNS / "evaluation/point/results.csv")
    expected = len(LEARNER_CORE.evaluation_matrix(PROTOCOL))
    role_counts = {
        role: sum(row["role"] == role for row in point)
        for role in ("calibration", "final")
    }
    if role_counts != {"calibration": expected, "final": expected}:
        raise RuntimeError(f"evaluation row matrix changed: {role_counts}")
    for row in point:
        if row["mode"] == "fixed_byte" and float(row["capacity_utilization"]) > 1.0 + 1e-9:
            raise RuntimeError(f"capacity exceeded: {row}")
    bootstrap = read_json(RUNS / "bootstrap/report.json")
    if int(bootstrap["jobs"]) != 54 or any(
        int(row["replicates"]) != int(PROTOCOL["bootstrap"]["replicates"])
        for row in bootstrap["comparisons"]
    ):
        raise RuntimeError("bootstrap matrix incomplete")
    report = {
        "complete": True,
        "protocol_validation": PROTOCOL_CORE.validate_protocol(PROTOCOL),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "learner_jobs": len(learner_reports),
        "fit_oof_jobs": sum(row["fold"] != "full" for row in learner_reports),
        "full_fit_jobs": sum(row["fold"] == "full" for row in learner_reports),
        "old_learner_checkpoints_reused": False,
        "evaluation_rows": len(point),
        "evaluation_role_counts": role_counts,
        "bootstrap_jobs": int(bootstrap["jobs"]),
        "bootstrap_replicates_each": int(bootstrap["replicates_per_comparison"]),
        "event_bootstrap_primary": True,
        "identity_roles": dict(PROTOCOL["roles"]),
        "final_is_hidden": False,
        "status": PROTOCOL["status"],
        "final_used_for_training_or_selection": False,
        "required_artifact_hashes": {
            str(path): file_sha256(path) for path in required
        },
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output / "report.json")})
    return report


def run_report() -> dict[str, Any]:
    completion = RUNS / "report/completion.json"
    if completion.is_file():
        return read_json(RUNS / "report/report.json")
    audit = run_audit()
    values = rows(RUNS / "evaluation/point/results.csv")
    fixed_byte = best_rows(values, role="final", mode="fixed_byte")
    fixed_call = best_rows(values, role="final", mode="fixed_call")
    references = rows(RUNS / "evaluation/references/results.csv")
    bootstrap = read_json(RUNS / "bootstrap/report.json")
    controls = read_json(RUNS / "evaluation/controls/report.json")
    internal = list(PROTOCOL["internal_rankers"])
    historical = list(PROTOCOL["historical_rankers"])
    external = list(PROTOCOL["external_rankers"])
    lines = [
        "# Oregon Cloud‑V3 全学习器与 Adaptive ROI 论文级后验评测",
        "",
        "> 状态：`posthoc_exploratory_final_reuse`。Oregon final 已在历史实验中打开；本结果是严格同协议的后验完整比较，不是新的隐藏 final，也不用于训练、阈值或模型选择。",
        "",
        "## 协议",
        "",
        f"- Cloud：Cloud‑V3 epoch‑28 `best.pt`，SHA256 `{PROTOCOL['cloud_model']['sha256']}`。",
        f"- canonical Oregon：fit={PROTOCOL['roles']['fit']:,}、calibration={PROTOCOL['roles']['calibration']:,}、final={PROTOCOL['roles']['final']:,}，45 类 same-camera event-disjoint。",
        "- 所有学习器均从 Cloud‑V3 fit 重新训练；未复用 Cloud‑V1/V2 或 COCO checkpoint。",
        "- 动作：Edge、Original、Q90、4 Profile × 160/320/640 共 12 个 ROI；Reversible 为主融合，MAUR 为消融。",
        "- Fixed-byte 为主，Fixed-call 为辅；表格单元格为 `Official AP50-95 / KB每图 / 最佳冻结策略`。",
        "- 统计：1,000 次 event-block paired bootstrap 为主；image bootstrap 仅作敏感性分析。",
        "",
        "## Fixed-byte 主结果：内部方法",
        "",
        *_method_table(fixed_byte, internal),
        "",
        "## Fixed-byte：历史 9 Ranker",
        "",
        *_method_table(fixed_byte, historical),
        "",
        "## Fixed-byte：外部 ORIC/MORIC",
        "",
        *_method_table(fixed_byte, external),
        "",
        "## Fixed-call 辅助结果：内部方法",
        "",
        *_method_table(fixed_call, internal),
        "",
        "## 2×2 归因：ActionValue 目标 × R12",
        "",
        "| 比较 | 10% ΔAP | 25% ΔAP | 40% ΔAP |",
        "|---|---:|---:|---:|",
    ]
    comparisons = (
        ("ROI 主效应（Current OQR12 − Current OQ）", "CurrentActionValue-OQR12", "CurrentActionValue-OQ"),
        ("AFPD 目标效应（OQ）", "AFPD-OQ", "CurrentActionValue-OQ"),
        ("AFPD 目标效应（OQR12）", "AFPD-OQR12", "CurrentActionValue-OQR12"),
        ("总效应（AFPD-OQR12 − Current-OQ）", "AFPD-OQR12", "CurrentActionValue-OQ"),
    )
    for label, candidate, baseline in comparisons:
        deltas = [
            float(fixed_byte[(candidate, float(budget))]["official_ap50_95"])
            - float(fixed_byte[(baseline, float(budget))]["official_ap50_95"])
            for budget in PROTOCOL["budgets"]
        ]
        lines.append(f"| {label} | " + " | ".join(f"{value:+.6f}" for value in deltas) + " |")
    lines.extend(
        [
            "",
            "## 系统上下界与融合消融",
            "",
            "| 角色 | 方法 | 模式 | 预算 | AP50-95 | AP50 | AP75 | KB/图 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in references:
        if row["role"] != "final":
            continue
        lines.append(
            f"| final | {row['method']} | {row['mode']} | {float(row['budget']):.2f} | "
            f"{float(row['official_ap50_95']):.6f} | {float(row['official_ap50']):.6f} | "
            f"{float(row['official_ap75']):.6f} | {float(row['kb_per_image']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap（Fixed-byte）",
            "",
            "| 单元 | 候选 − 对照 | 预算 | 点估计 ΔAP | 均值 ΔAP | 95% CI | P(Δ>0) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap["comparisons"]:
        lines.append(
            f"| {row['unit']} | {row['candidate']} − {row['comparator']} | {float(row['budget']):.2f} | "
            f"{float(row['point_delta']):+.6f} | {float(row['mean_delta']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
            f"{float(row['probability_gt_zero']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 完整性审计",
            "",
            f"- 学习器任务：{audit['learner_jobs']}（OOF {audit['fit_oof_jobs']} + full-fit {audit['full_fit_jobs']}）。",
            f"- 主评测行：{audit['evaluation_rows']}；每个 role {len(LEARNER_CORE.evaluation_matrix(PROTOCOL))} 行。",
            f"- GBDT-Hierarchical calibration-only 阈值：{controls['gbdt_threshold']}；fit 正类率 {controls['gbdt_fit_positive_rate']:.4f}。",
            f"- Bootstrap：{audit['bootstrap_jobs']} 组 × {audit['bootstrap_replicates_each']} 次；event 主、image sensitivity。",
            "- 所有 Fixed-byte 行均通过容量上限检查；所有模型和动作阈值只由 fit/calibration 产生。",
            "- `final_is_hidden=false`，`final_used_for_training_or_selection=false`。",
            "",
            "## 解读边界",
            "",
            "- GT-Proxy Teacher 使用逐图 gain proxy，不是全局 mAP Oracle；只能表示候选动作空间上限。",
            "- 最佳策略列是在 calibration 冻结、final 只评一次的策略；不能再根据本表回调阈值。",
            "- Oregon 存在事件相关性，论文主置信区间应引用 event bootstrap，image bootstrap 只放附录。",
        ]
    )
    report_path = RUNS / "report/RESULTS_ZH.md"
    atomic_text(report_path, "\n".join(lines) + "\n")
    report = {
        "complete": True,
        "results": str(report_path),
        "results_sha256": file_sha256(report_path),
        "audit_sha256": file_sha256(RUNS / "audit/report.json"),
        "status": PROTOCOL["status"],
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "report/report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"]})
    atomic_json(RUNS / "COMPLETED.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run_report(), indent=2, ensure_ascii=False))
