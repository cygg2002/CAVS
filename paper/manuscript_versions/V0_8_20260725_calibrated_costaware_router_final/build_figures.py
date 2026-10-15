#!/usr/bin/env python3
"""Build source-backed figures for the JRTIP V0.8 Chinese manuscript."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
EXPERIMENT = (
    REPO
    / "paper/experiments/20260725_three_dataset_costaware_selector_riskgate"
)
SOURCE_MATRIX = (
    REPO
    / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
)
DATASETS = ("coco", "ena24", "oregon")
DATASET_LABELS = {"coco": "COCO", "ena24": "ENA24", "oregon": "Oregon"}
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
FONT_SERIF = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
FONT_BOLD = Path("/System/Library/Fonts/STHeiti Medium.ttc")

INK = "#17212B"
MUTED = "#66727F"
GRID = "#D9DEE3"
PANEL = "#F5F7F8"
BLUE = "#246BCE"
GREEN = "#20845A"
RED = "#C7473A"
AMBER = "#D58B22"
TEAL = "#238C8C"
PURPLE = "#7256A8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def number(row: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return default


def identical_gate(row: Mapping[str, Any]) -> bool:
    return any(
        key.endswith("identical_action_shortcut") and truthy(value)
        for key, value in row.items()
    )


def load_evidence() -> dict[str, Any]:
    final: dict[str, list[dict[str, str]]] = {}
    gates: dict[str, list[dict[str, str]]] = {}
    baselines: dict[str, list[dict[str, str]]] = {}
    regret: dict[str, dict[str, float]] = {}
    gate_summary: dict[str, dict[str, int]] = {}

    for dataset in DATASETS:
        run = EXPERIMENT / "runs" / dataset
        final[dataset] = read_csv(run / "final_results.csv")
        gate_file = "strict_risk_gate.csv" if dataset == "ena24" else "risk_gate.csv"
        gates[dataset] = read_csv(run / gate_file)
        baselines[dataset] = read_csv(
            SOURCE_MATRIX / "runs" / dataset / "canonical_system_reference.csv"
        )
        oracle = read_csv(run / "final_oracle_diagnostic.csv")
        values = [float(row["final_ap_regret"]) for row in oracle]
        regret[dataset] = {"mean": sum(values) / len(values), "max": max(values)}

        passed = [row for row in gates[dataset] if truthy(row.get("gate_passed"))]
        identical = sum(identical_gate(row) for row in passed)
        gate_summary[dataset] = {
            "non_identical": len(passed) - identical,
            "identical": identical,
            "fallback": len(gates[dataset]) - len(passed),
        }

    return {
        "final": final,
        "gates": gates,
        "baselines": baselines,
        "regret": regret,
        "gate_summary": gate_summary,
    }


def font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_SERIF if serif else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str = INK,
) -> None:
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), value, font=text_font, spacing=6, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        value,
        font=text_font,
        fill=fill,
        spacing=6,
        align="center",
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str = MUTED,
    width: int = 5,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 15
    spread = 0.55
    points = [
        end,
        (
            end[0] - length * math.cos(angle - spread),
            end[1] - length * math.sin(angle - spread),
        ),
        (
            end[0] - length * math.cos(angle + spread),
            end[1] - length * math.sin(angle + spread),
        ),
    ]
    draw.polygon(points, fill=fill)


def save(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))
    return path


def build_architecture(path: Path) -> Path:
    image = Image.new("RGB", (2100, 1180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "校准式成本感知端云协同架构", font=font(48, bold=True), fill=INK)
    draw.text(
        (70, 104),
        "在线执行只运行一个冻结配置；离线阶段负责候选选择、风险门禁与一次性 final 评测",
        font=font(25),
        fill=MUTED,
    )

    draw.rounded_rectangle((55, 175, 2045, 610), radius=22, fill="#F0F6FC", outline="#9BBCE0", width=3)
    draw.text((82, 196), "在线端云执行", font=font(30, bold=True), fill=BLUE)
    online = [
        (90, 310, 330, 475, "输入图像\nI_i"),
        (385, 310, 670, 475, "端侧检测器\nD_edge 与特征 z_i"),
        (725, 310, 970, 475, "样本 Ranker\n哪些图上传"),
        (1025, 310, 1265, 475, "预算调度\nCall / Online / Byte"),
        (1320, 310, 1580, 475, "载荷策略\nOriginal / Q90 / ROI"),
        (1635, 310, 1870, 475, "云端检测与\n坐标融合"),
    ]
    for index, (left, top, right, bottom, label) in enumerate(online):
        fill = "white" if index not in {2, 3, 4} else "#E5F0FB"
        draw.rounded_rectangle((left, top, right, bottom), radius=16, fill=fill, outline="#477EAE", width=3)
        text_center(draw, (left, top, right, bottom), label, font(24, bold=index in {2, 3, 4}))
        if index:
            previous = online[index - 1]
            arrow(draw, (previous[2] + 8, 392), (left - 8, 392), fill="#477EAE")
    draw.text((742, 505), "冻结配置 theta_frozen", font=font(22, bold=True), fill=BLUE)

    draw.rounded_rectangle((55, 665, 2045, 1100), radius=22, fill="#F7F8F5", outline="#AAB5A5", width=3)
    draw.text((82, 685), "离线校准与审计", font=font(30, bold=True), fill=GREEN)
    offline = [
        (105, 805, 425, 970, "8 Rankers × 3 Policies\n每工作点 24 个候选"),
        (510, 805, 800, 970, "Selector\n配置级 AP / KB 选择"),
        (885, 805, 1175, 970, "Gate / Audit\n200 次 paired bootstrap"),
        (1260, 805, 1535, 970, "冻结或回退\n写入配置与哈希"),
        (1620, 805, 1970, 970, "Final 一次性评测\nOracle 仅诊断 regret"),
    ]
    for index, (left, top, right, bottom, label) in enumerate(offline):
        draw.rounded_rectangle((left, top, right, bottom), radius=16, fill="white", outline="#59836A", width=3)
        text_center(draw, (left, top, right, bottom), label, font(23, bold=index in {1, 2, 3}))
        if index:
            previous = offline[index - 1]
            arrow(draw, (previous[2] + 8, 887), (left - 8, 887), fill="#59836A")
    arrow(draw, (1397, 790), (1397, 620), fill=GREEN, width=6)
    draw.text((1430, 705), "部署", font=font(22, bold=True), fill=GREEN)
    return save(image, path)


def method_name(row: Mapping[str, Any]) -> str:
    return str(row.get("method") or row.get("ranker") or "")


def policy_name(row: Mapping[str, Any]) -> str:
    return str(row.get("action_policy") or row.get("policy") or "")


def mode_name(row: Mapping[str, Any]) -> str:
    return str(row.get("mode") or "")


def budget_value(row: Mapping[str, Any]) -> float:
    return number(row, "nominal_budget", "budget")


def build_ap_kb(path: Path, evidence: Mapping[str, Any]) -> Path:
    image = Image.new("RGB", (2280, 920), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "冻结配置的精度-请求成本工作点", font=font(44, bold=True), fill=INK)
    draw.text(
        (70, 92),
        "每个面板独立缩放；系统参考使用灰色，冻结配置按预算模式着色",
        font=font(23),
        fill=MUTED,
    )
    mode_colors = {"fixed_call": BLUE, "online": GREEN, "fixed_byte": AMBER}
    mode_labels = {"fixed_call": "Fixed-call", "online": "Online", "fixed_byte": "Fixed-byte"}

    for panel_index, dataset in enumerate(DATASETS):
        left = 65 + panel_index * 745
        top, width, height = 180, 675, 620
        plot = (left + 80, top + 70, left + width - 35, top + height - 75)
        rows = list(evidence["final"][dataset])
        base_rows = list(evidence["baselines"][dataset])
        points = [
            (number(row, "request_kb_per_image"), number(row, "official_map50_95"))
            for row in rows + base_rows
        ]
        xs = [math.log10(value + 1.0) for value, _ in points]
        ys = [value for _, value in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        y_pad = max(0.003, (y_max - y_min) * 0.08)
        y_min -= y_pad
        y_max += y_pad

        draw.rounded_rectangle((left, top, left + width, top + height), radius=16, fill=PANEL, outline=GRID, width=2)
        draw.text((left + 22, top + 18), DATASET_LABELS[dataset], font=font(29, bold=True), fill=INK)
        for tick in range(5):
            y = plot[1] + tick * (plot[3] - plot[1]) / 4
            value = y_max - tick * (y_max - y_min) / 4
            draw.line((plot[0], y, plot[2], y), fill=GRID, width=2)
            draw.text((left + 5, y - 13), f"{value:.3f}", font=font(17), fill=MUTED)
        draw.line((plot[0], plot[1], plot[0], plot[3]), fill=INK, width=2)
        draw.line((plot[0], plot[3], plot[2], plot[3]), fill=INK, width=2)

        def project(kb: float, ap: float) -> tuple[float, float]:
            x_value = math.log10(kb + 1.0)
            x = plot[0] + (x_value - x_min) / max(1e-9, x_max - x_min) * (plot[2] - plot[0])
            y = plot[3] - (ap - y_min) / max(1e-9, y_max - y_min) * (plot[3] - plot[1])
            return x, y

        baseline_labels = {
            "Edge-only": "Edge",
            "Cloud-only Original": "Cloud Original",
            "Always 640-Q90": "Q90",
            "Always ROI fallback": "ROI fallback",
            "Shared ContextM2E all": "Shared ContextM2E",
        }
        label_offsets = {
            "Edge-only": (8, -18),
            "Cloud-only Original": (-118, -18),
            "Always 640-Q90": (8, -22),
            "Always ROI fallback": (8, 5),
            "Shared ContextM2E all": (-118, 6),
        }
        for row in base_rows:
            kb = number(row, "request_kb_per_image")
            ap = number(row, "official_map50_95")
            x, y = project(kb, ap)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#8A939C", outline="white", width=2)
            method = method_name(row)
            offset_x, offset_y = label_offsets.get(method, (8, -18))
            draw.text(
                (x + offset_x, y + offset_y),
                baseline_labels.get(method, method),
                font=font(13),
                fill=MUTED,
            )

        for row in rows:
            mode = mode_name(row)
            kb = number(row, "request_kb_per_image")
            ap = number(row, "official_map50_95")
            x, y = project(kb, ap)
            color = mode_colors[mode]
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline="white", width=3)
            budget = round(budget_value(row) * 100)
            offset_x, offset_y = {
                "fixed_call": (8, -27),
                "online": (8, 8),
                "fixed_byte": (8, 8),
            }[mode]
            offset_x, offset_y = {
                ("coco", "fixed_call", 40): (10, 12),
                ("coco", "online", 40): (12, -25),
                ("coco", "fixed_byte", 40): (-43, -25),
            }.get((dataset, mode, budget), (offset_x, offset_y))
            draw.text((x + offset_x, y + offset_y), f"{budget}%", font=font(14, bold=True), fill=color)

        tick_values = [0, 10, 100, 1000]
        for value in tick_values:
            transformed = math.log10(value + 1.0)
            if x_min <= transformed <= x_max:
                x = plot[0] + (transformed - x_min) / max(1e-9, x_max - x_min) * (plot[2] - plot[0])
                draw.line((x, plot[3], x, plot[3] + 7), fill=INK, width=2)
                draw.text((x - 15, plot[3] + 10), str(value), font=font(16), fill=MUTED)
        draw.text((left + 255, top + height - 40), "请求 KB/image（log₁₀(KB+1)）", font=font(18), fill=MUTED)

    legend_x = 720
    for index, mode in enumerate(("fixed_call", "online", "fixed_byte")):
        x = legend_x + index * 230
        draw.ellipse((x, 850, x + 18, 868), fill=mode_colors[mode])
        draw.text((x + 28, 844), mode_labels[mode], font=font(19), fill=INK)
    return save(image, path)


def build_gate_summary(path: Path, evidence: Mapping[str, Any]) -> Path:
    image = Image.new("RGB", (1700, 950), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "风险门禁通过与回退构成", font=font(46, bold=True), fill=INK)
    draw.text(
        (70, 102),
        "13/30 个通过点中，6 个在 gate 上动作完全相同；真正比较不同动作并通过的只有 7 个",
        font=font(25),
        fill=MUTED,
    )
    left, top, right, bottom = 180, 220, 1580, 780
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    for tick in range(0, 11, 2):
        y = bottom - tick / 10 * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((125, y - 14), str(tick), font=font(20), fill=MUTED)

    colors = {"non_identical": GREEN, "identical": AMBER, "fallback": RED}
    labels = {"non_identical": "非同动作通过", "identical": "相同动作通过", "fallback": "回退 SharedAction"}
    bar_width = 230
    centers = [430, 850, 1270]
    for center, dataset in zip(centers, DATASETS):
        y_cursor = bottom
        summary = evidence["gate_summary"][dataset]
        for key in ("non_identical", "identical", "fallback"):
            value = summary[key]
            segment = value / 10 * (bottom - top)
            if value:
                draw.rectangle((center - bar_width // 2, y_cursor - segment, center + bar_width // 2, y_cursor), fill=colors[key])
                text_center(
                    draw,
                    (center - bar_width // 2, int(y_cursor - segment), center + bar_width // 2, int(y_cursor)),
                    str(value),
                    font(25, bold=True),
                    fill="white",
                )
            y_cursor -= segment
        draw.text((center - 62, bottom + 28), DATASET_LABELS[dataset], font=font(27, bold=True), fill=INK)

    legend_x = 350
    for index, key in enumerate(("non_identical", "identical", "fallback")):
        x = legend_x + index * 360
        draw.rectangle((x, 860, x + 26, 886), fill=colors[key])
        draw.text((x + 40, 852), labels[key], font=font(22), fill=INK)
    return save(image, path)


def ordered_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    mode_order = {"fixed_call": 0, "online": 1, "fixed_byte": 2}
    return sorted(rows, key=lambda row: (mode_order[mode_name(row)], budget_value(row)))


def build_action_composition(path: Path, evidence: Mapping[str, Any]) -> Path:
    image = Image.new("RGB", (2200, 1520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "冻结配置的动作组成", font=font(46, bold=True), fill=INK)
    draw.text(
        (70, 96),
        "比例按全部输入图像计算；Edge 表示未上传，Original/Q90/ROI 表示云端请求载荷",
        font=font(25),
        fill=MUTED,
    )
    colors = {"edge": "#B8C0C7", "original": BLUE, "q90": GREEN, "roi": AMBER}
    labels = {"edge": "Edge", "original": "Original", "q90": "Q90", "roi": "ROI"}
    mode_short = {"fixed_call": "FC", "online": "ON", "fixed_byte": "FB"}

    for panel_index, dataset in enumerate(DATASETS):
        left = 65 + panel_index * 710
        top, width, height = 175, 655, 1220
        draw.rounded_rectangle((left, top, left + width, top + height), radius=15, fill=PANEL, outline=GRID, width=2)
        draw.text((left + 20, top + 18), DATASET_LABELS[dataset], font=font(29, bold=True), fill=INK)
        rows = ordered_rows(evidence["final"][dataset])
        bar_left, bar_right = left + 135, left + width - 28
        for row_index, row in enumerate(rows):
            y = top + 92 + row_index * 105
            budget = budget_value(row)
            label = f"{mode_short[mode_name(row)]} {budget:.0%}"
            draw.text((left + 18, y + 13), label, font=font(19, bold=True), fill=INK)
            upload = number(row, "upload_ratio")
            parts = {
                "edge": max(0.0, 1.0 - upload),
                "original": number(row, "full_ratio"),
                "q90": number(row, "q90_ratio"),
                "roi": number(row, "roi_ratio"),
            }
            x = bar_left
            for key in ("edge", "original", "q90", "roi"):
                segment = max(0.0, min(1.0, parts[key])) * (bar_right - bar_left)
                if segment > 0:
                    draw.rectangle((x, y, x + segment, y + 54), fill=colors[key])
                x += segment
            draw.rectangle((bar_left, y, bar_right, y + 54), outline="#7A848D", width=2)
            draw.text((bar_left, y + 60), f"上传 {upload:.1%}  |  ROI {parts['roi']:.2%}", font=font(15), fill=MUTED)
        draw.text((bar_left - 4, top + height - 42), "0%", font=font(17), fill=MUTED)
        draw.text((bar_right - 42, top + height - 42), "100%", font=font(17), fill=MUTED)

    legend_x = 620
    for index, key in enumerate(("edge", "original", "q90", "roi")):
        x = legend_x + index * 230
        draw.rectangle((x, 1440, x + 28, 1468), fill=colors[key])
        draw.text((x + 40, 1434), labels[key], font=font(21), fill=INK)
    return save(image, path)


def has_visual_content(path: Path) -> bool:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        extrema = grayscale.getextrema()
        return extrema[0] < 245 and extrema[1] > 250


def build_all(output_dir: Path | None = None) -> list[Path]:
    output = output_dir or PACKAGE / "figures"
    evidence = load_evidence()
    architecture = output / "Fig1_calibrated_costaware_architecture.png"
    architecture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        PACKAGE / "figures" / "drawio" / "Fig1_calibrated_costaware_router.png",
        architecture,
    )
    return [
        architecture,
        build_ap_kb(output / "Fig2_frozen_ap_request_cost.png", evidence),
        build_gate_summary(output / "Fig3_risk_gate_outcomes.png", evidence),
        build_action_composition(output / "Fig4_frozen_action_composition.png", evidence),
    ]


if __name__ == "__main__":
    for output_path in build_all():
        print(output_path)
