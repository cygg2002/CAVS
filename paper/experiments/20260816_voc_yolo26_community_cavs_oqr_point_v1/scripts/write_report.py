#!/usr/bin/env python3
"""Write the concise Chinese report for the VOC point benchmark."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def read_rows() -> list[dict[str, str]]:
    with (RUNS / "point/point_results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run() -> dict[str, object]:
    output = RUNS / "report"
    completion = output / "completion.json"
    report_path = output / "RESULTS_ZH.md"
    if completion.is_file():
        return json.loads(completion.read_text(encoding="utf-8"))
    rows = read_rows()
    main = [
        row
        for row in rows
        if row["role"] == "final"
        and row["mode"] == "fixed_byte"
        and row["seed"] in {"ensemble", "fixed"}
        and row["method"]
        in {
            "CAVS-OQR-MAUR",
            "CAVS-OQ-from-OQR",
            "ORIC-AP5095",
            "MORIC-AP5095",
            "ORIC-AP50",
            "MORIC-AP50",
            "DCSB-Score-Adapted",
            "Confidence",
            "Random",
        }
    ]
    methods = [
        "CAVS-OQR-MAUR",
        "CAVS-OQ-from-OQR",
        "ORIC-AP5095",
        "MORIC-AP5095",
        "ORIC-AP50",
        "MORIC-AP50",
        "DCSB-Score-Adapted",
        "Confidence",
        "Random",
    ]
    lookup = {(row["method"], float(row["budget"])): row for row in main}
    lines = [
        "# VOC community-YOLO26 CAVS-OQR 点结果",
        "",
        "本实验在 VOC2007+2012 trainval 上重新训练全部上传学习器，并在 VOC2007 test 上仅报告点估计。",
        "该测试集曾被社区检测器发布者用于模型选择，因此属于 post-hoc benchmark，不作隐藏终评或显著性声明。",
        "",
        "## Fixed-byte 主结果",
        "",
        "| 方法 | 10% AP50-95 / KB每图 | 25% AP50-95 / KB每图 | 40% AP50-95 / KB每图 |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        cells = []
        for budget in map(float, PROTOCOL["budgets"]):
            row = lookup[(method, budget)]
            cells.append(f"{float(row['official_ap50_95']):.6f} / {float(row['kb_per_image']):.2f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 协议说明",
            "",
            "- CAVS、ORIC、MORIC 均为 3 个独立 VOC 重训练种子的分数集成；完整单种子行保存在 CSV。",
            "- Random 为 50 套冻结随机选择分别计算 AP 后的均值。",
            "- 主指标为 Fixed-byte 10%/25%/40%；Fixed-call、AP50、AP75、动作计数与容量利用率均在完整 CSV 中。",
            "- ROI 为 Current@640，融合规则为 MAUR；CAVS-OQ-from-OQR 是同一学习器的 No-ROI 硬掩码消融。",
            "- bootstrap_performed=false；final_used_for_training_or_selection=false。",
            "",
            "完整结果：`runs/point/point_results.csv`。",
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    result = {
        "complete": True,
        "rows": len(rows),
        "report": str(report_path.resolve()),
        "bootstrap_performed": False,
        "final_used_for_training_or_selection": False,
    }
    temporary_completion = completion.with_name(f".{completion.name}.{os.getpid()}.tmp")
    temporary_completion.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary_completion, completion)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
