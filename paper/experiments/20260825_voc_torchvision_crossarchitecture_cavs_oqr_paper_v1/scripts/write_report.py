#!/usr/bin/env python3
"""Write the auditable Chinese result summary after all frozen stages complete."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any


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


DATA = load_module("voc_gluoncv_report_data", ROOT / "scripts/data_core.py")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run() -> dict[str, Any]:
    required = (
        RUNS / "endpoint_gate/completion.json",
        RUNS / "point/completion.json",
        RUNS / "bootstrap/completion.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot report incomplete experiment: {missing}")
    points = _csv(RUNS / "point/point_results.csv")
    bootstrap = _csv(RUNS / "bootstrap/results.csv")
    endpoints = DATA.read_json(RUNS / "endpoint_gate/report.json")["actions"]
    final = [
        row
        for row in points
        if row["role"] == "final" and row["mode"] == "fixed_byte"
    ]
    main_methods = [
        ("CAVS-O-Core", "Joint-O", "ensemble"),
        ("CAVS-OQ-Core", "Joint-OQ", "ensemble"),
        ("CAVS-OQR-Core", "Joint-OQR", "ensemble"),
        ("ORIC-AP5095", "Original-only", "ensemble"),
        ("MORIC-AP5095", "Original-only", "ensemble"),
        ("ORIC-AP5095", "SharedAction-OQ", "ensemble"),
        ("MORIC-AP5095", "SharedAction-OQ", "ensemble"),
        ("DCSB-Score-Adapted", "SharedAction-OQ", "fixed"),
        ("Confidence", "SharedAction-OQ", "fixed"),
        ("Random", "SharedAction-OQ", "mean50"),
    ]
    lines = [
        "# VOC 异构检测器 CAVS 论文实验结果",
        "",
        "## 协议",
        "",
        "- Edge：Torchvision SSDLite320-MobileNetV3-Large COCO_V1；Cloud：Torchvision Faster R-CNN-ResNet50-FPN-V2 COCO_V1。",
        "- 检测器使用官方固定 COCO 权重且未训练；CAVS-O、CAVS-OQ、CAVS-OQR 与 ORIC/MORIC 学习器全部从头训练。",
        "- fit/calibration/final = 15,629 / 922 / 4,952；final 未参与 reward、训练或校准。",
        "- 主指标为 difficult-aware VOC-AP50–95；主预算为 Fixed-byte。",
        "",
        "## 检测端点门禁",
        "",
        "| 动作 | AP50–95 | AP50 | AP75 | VOC07 11-point AP50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for action in ("edge", "original", "q90", "current@official"):
        row = endpoints[action]
        lines.append(
            f"| {action} | {float(row['map50_95']):.6f} | {float(row['map50']):.6f} | "
            f"{float(row['map75']):.6f} | {float(row['voc07_map50']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-byte final 主结果",
            "",
            "| 方法 | 策略 | 10% | 25% | 40% |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for method, policy, seed in main_methods:
        values = {
            float(row["budget"]): float(row["official_ap50_95"])
            for row in final
            if row["method"] == method
            and row["policy"] == policy
            and row["seed"] == seed
        }
        if set(values) != {0.1, 0.25, 0.4}:
            raise RuntimeError(f"main table row incomplete: {method}/{policy}/{seed}")
        lines.append(
            f"| {method} | {policy} | {values[0.1]:.6f} | {values[0.25]:.6f} | {values[0.4]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 三种子单模型稳定性",
            "",
            "| 方法 | 策略 | 预算 | mean ± sample SD | ensemble |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for method, policy in (
        ("CAVS-O-Core", "Joint-O"),
        ("CAVS-OQ-Core", "Joint-OQ"),
        ("CAVS-OQR-Core", "Joint-OQR"),
        ("ORIC-AP5095", "Original-only"),
        ("MORIC-AP5095", "Original-only"),
    ):
        for budget in map(float, PROTOCOL["budgets"]):
            singles = [
                float(row["official_ap50_95"])
                for row in final
                if row["method"] == method
                and row["policy"] == policy
                and float(row["budget"]) == budget
                and row["seed"] in set(map(str, PROTOCOL["model_seeds"]))
            ]
            ensemble = next(
                float(row["official_ap50_95"])
                for row in final
                if row["method"] == method
                and row["policy"] == policy
                and float(row["budget"]) == budget
                and row["seed"] == "ensemble"
            )
            if len(singles) != 3:
                raise RuntimeError(f"three-seed row incomplete: {method}/{budget}")
            lines.append(
                f"| {method} | {policy} | {budget:.0%} | "
                f"{statistics.mean(singles):.6f} ± {statistics.stdev(singles):.6f} | {ensemble:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Paired bootstrap（Fixed-byte final）",
            "",
            "| 候选 | 对照 | 预算 | ΔAP mean | 95% CI | sign p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap:
        lines.append(
            f"| {row['candidate']} | {row['comparator']} | {float(row['budget']):.0%} | "
            f"{float(row['delta_mean']):+.6f} | [{float(row['ci95_low']):+.6f}, "
            f"{float(row['ci95_high']):+.6f}] | {float(row['two_sided_sign_probability']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 声明边界",
            "",
            "该实验验证跨检测器架构可迁移性，但不是新的隐藏终评：VOC2007 test 已在本项目此前探索中被查看。两份检测器权重只在 COCO 上训练；本实验的路由器、reward、阈值与动作规则没有访问 final。SharedAction-OQ 是统一系统适配，不应称为 ORIC/MORIC 官方原始动作空间。Current@official 端点行仅在 ROI 可用图执行 ROI，其余图回退 Edge。",
            "",
        ]
    )
    result_path = ROOT / "RESULTS_ZH.md"
    result_path.write_text("\n".join(lines), encoding="utf-8")
    report = {
        "complete": True,
        "point_rows": len(points),
        "bootstrap_rows": len(bootstrap),
        "results_sha256": DATA.file_sha256(result_path),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(RUNS / "report/report.json", report)
    DATA.atomic_json(RUNS / "report/completion.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
