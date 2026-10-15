#!/usr/bin/env python3
"""Build source-backed data figures for the V0.9 Chinese manuscript.

Fixes over the V0.8 figure builder (review findings 4.3):
- no in-figure titles/subtitles (captions carry them);
- palette re-stepped so adjacent categories differ in grayscale luminance
  (asserted below, not eyeballed);
- minimum text >= 8pt print-equivalent at 170 mm column width;
- Fig2: per-mode polylines + distinct marker shapes (circle/square/triangle)
  instead of per-point percentage labels (kills label collisions);
- stacked segments separated by 2px white gaps.

Figure 1 is produced by figures/fig1_architecture.py (draw.io-as-code) and
copied here under the canonical file name.
"""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
EXPERIMENT = REPO / "paper/experiments/20260725_three_dataset_costaware_selector_riskgate"
SOURCE_MATRIX = REPO / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
DATASETS = ("coco", "ena24", "oregon")
DATASET_LABELS = {"coco": "COCO", "ena24": "ENA24", "oregon": "Oregon"}
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
FONT_BOLD = Path("/System/Library/Fonts/STHeiti Medium.ttc")

INK = "#17212B"
MUTED = "#5A6672"
GRID = "#D9DEE3"
PANEL = "#F7F8F9"
REF_GRAY = "#8A939C"

# Grayscale-stepped palette (luminance asserted below).
BLUE = "#246BCE"        # Y ~ 99
DARKGREEN = "#145A32"   # Y ~ 71
AMBER = "#E0A32E"       # Y ~ 160
RED = "#B03A2E"         # Y ~ 84
LIGHTGRAY = "#C4CBD1"   # Y ~ 201

MODE_COLORS = {"fixed_call": BLUE, "online": DARKGREEN, "fixed_byte": AMBER}
MODE_LABELS = {"fixed_call": "Fixed-call", "online": "Online", "fixed_byte": "Fixed-byte"}
MODE_ORDER = ("fixed_call", "online", "fixed_byte")


def luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def assert_grayscale_distinct(sequence: list[str], min_delta: float, context: str) -> None:
    for a, b in zip(sequence, sequence[1:]):
        delta = abs(luminance(a) - luminance(b))
        assert delta >= min_delta, f"{context}: adjacent {a}/{b} grayscale delta {delta:.0f} < {min_delta}"


# Fig2 modes / Fig3 stacks / Fig4 stacks: adjacent categories must survive grayscale print.
assert_grayscale_distinct([DARKGREEN, BLUE, AMBER], 25, "fig2 modes")
assert_grayscale_distinct([DARKGREEN, AMBER, RED], 25, "fig3 stack")
assert_grayscale_distinct([LIGHTGRAY, BLUE, DARKGREEN, AMBER], 25, "fig4 stack")


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
    gate_summary: dict[str, dict[str, int]] = {}

    for dataset in DATASETS:
        run = EXPERIMENT / "runs" / dataset
        final[dataset] = read_csv(run / "final_results.csv")
        gate_file = "strict_risk_gate.csv" if dataset == "ena24" else "risk_gate.csv"
        gates[dataset] = read_csv(run / gate_file)
        baselines[dataset] = read_csv(
            SOURCE_MATRIX / "runs" / dataset / "canonical_system_reference.csv"
        )
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
        "gate_summary": gate_summary,
    }


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def text_center(draw, box, value, text_font, fill=INK) -> None:
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), value, font=text_font, spacing=6, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        value, font=text_font, fill=fill, spacing=6, align="center",
    )


def save(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))
    return path


def method_name(row: Mapping[str, Any]) -> str:
    return str(row.get("method") or row.get("ranker") or "")


def mode_name(row: Mapping[str, Any]) -> str:
    return str(row.get("mode") or "")


def budget_value(row: Mapping[str, Any]) -> float:
    return number(row, "nominal_budget", "budget")


def marker(draw, shape: str, x: float, y: float, r: float, color: str) -> None:
    if shape == "circle":
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="white", width=3)
    elif shape == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), fill=color, outline="white", width=3)
    else:  # triangle
        draw.polygon(
            [(x, y - r - 2), (x - r - 1, y + r - 1), (x + r + 1, y + r - 1)],
            fill=color, outline="white",
        )


MODE_SHAPES = {"fixed_call": "circle", "online": "square", "fixed_byte": "triangle"}

# Short reference labels; caption maps them to full method names.
BASE_LABELS = {
    "Edge-only": "Edge",
    "Cloud-only Original": "Cloud-Orig",
    "Always 640-Q90": "Q90",
    "Always ROI fallback": "ROI-fb",
    "Shared ContextM2E all": "Shared-all",
}
# (dx, dy) per dataset/method, tuned after visual pass; dy in px from point.
BASE_OFFSETS = {
    ("coco", "Edge-only"): (16, -44),
    ("coco", "Cloud-only Original"): (-235, -10),
    ("coco", "Always 640-Q90"): (-118, 18),
    ("coco", "Always ROI fallback"): (16, 16),
    ("coco", "Shared ContextM2E all"): (-250, 24),
    ("ena24", "Edge-only"): (16, -44),
    ("ena24", "Cloud-only Original"): (-235, 22),
    ("ena24", "Always 640-Q90"): (-160, -14),
    ("ena24", "Always ROI fallback"): (22, 26),
    ("ena24", "Shared ContextM2E all"): (-250, -42),
    ("oregon", "Edge-only"): (16, -44),
    ("oregon", "Cloud-only Original"): (-235, -10),
    ("oregon", "Always 640-Q90"): (-158, -10),
    ("oregon", "Always ROI fallback"): (-200, 16),
    ("oregon", "Shared ContextM2E all"): (24, -10),
}


def build_ap_kb(path: Path, evidence: Mapping[str, Any]) -> Path:
    image = Image.new("RGB", (2280, 1120), "white")
    draw = ImageDraw.Draw(image)

    for panel_index, dataset in enumerate(DATASETS):
        left = 60 + panel_index * 745
        top, width, height = 40, 690, 900
        plot = (left + 160, top + 90, left + width - 30, top + height - 120)
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
        y_pad = max(0.003, (y_max - y_min) * 0.09)
        y_min -= y_pad
        y_max += y_pad

        draw.rounded_rectangle((left, top, left + width, top + height), radius=16, fill=PANEL, outline=GRID, width=2)
        draw.text((left + 26, top + 22), DATASET_LABELS[dataset], font=font(44, bold=True), fill=INK)
        for tick in range(5):
            y = plot[1] + tick * (plot[3] - plot[1]) / 4
            value = y_max - tick * (y_max - y_min) / 4
            draw.line((plot[0], y, plot[2], y), fill=GRID, width=2)
            draw.text((left + 18, y - 18), f"{value:.3f}", font=font(32), fill=MUTED)
        draw.line((plot[0], plot[1], plot[0], plot[3]), fill=INK, width=3)
        draw.line((plot[0], plot[3], plot[2], plot[3]), fill=INK, width=3)

        def project(kb: float, ap: float) -> tuple[float, float]:
            x_value = math.log10(kb + 1.0)
            x = plot[0] + (x_value - x_min) / max(1e-9, x_max - x_min) * (plot[2] - plot[0])
            y = plot[3] - (ap - y_min) / max(1e-9, y_max - y_min) * (plot[3] - plot[1])
            return x, y

        # occupied boxes: frozen-config markers + already placed labels
        occupied: list[tuple[float, float, float, float]] = []
        for row in rows:
            mx, my = project(number(row, "request_kb_per_image"), number(row, "official_map50_95"))
            occupied.append((mx - 18, my - 18, mx + 18, my + 18))
        ref_pts = []
        for row in base_rows:
            kb = number(row, "request_kb_per_image")
            ap = number(row, "official_map50_95")
            x, y = project(kb, ap)
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=REF_GRAY, outline="white", width=3)
            ref_pts.append((x, y, BASE_LABELS.get(method_name(row), method_name(row)), len(occupied)))
            occupied.append((x - 13, y - 13, x + 13, y + 13))
        label_font = font(30)
        for x, y, text, own_index in ref_pts:
            tw = draw.textlength(text, font=label_font)
            th = 34
            candidates = [
                (18, -th - 6), (18, 8), (-tw - 18, -th - 6), (-tw - 18, 8),
                (22, -th / 2), (-tw - 22, -th / 2),
                (18, -th - 44), (-tw - 18, -th - 44), (18, 44), (-tw - 18, 44),
                (-tw / 2, -th - 24), (-tw / 2, 32),
            ]
            def in_bounds(box):
                return not (box[0] < left + 14 or box[2] > left + width - 8
                            or box[1] < top + 12 or box[3] > top + height - 60)
            placed = None
            fallback = None
            for dx, dy in candidates:
                box = (x + dx - 4, y + dy - 4, x + dx + tw + 4, y + dy + th + 4)
                if not in_bounds(box):
                    continue
                if fallback is None:
                    fallback = (dx, dy, box)
                hit = any(
                    index != own_index
                    and not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3])
                    for index, o in enumerate(occupied)
                )
                if not hit:
                    placed = (dx, dy, box)
                    break
            dx, dy, box = placed or fallback or (18, -th - 6, (x, y, x + tw, y + th))
            occupied.append(box)
            draw.text((x + dx, y + dy), text, font=label_font, fill=MUTED,
                      stroke_width=6, stroke_fill="white")

        # per-mode polylines, budget ascending; markers by shape
        for mode in MODE_ORDER:
            mode_rows = sorted(
                (row for row in rows if mode_name(row) == mode), key=budget_value
            )
            coords = [
                project(number(r, "request_kb_per_image"), number(r, "official_map50_95"))
                for r in mode_rows
            ]
            color = MODE_COLORS[mode]
            for a, b in zip(coords, coords[1:]):
                draw.line((a, b), fill=color, width=4)
            for x, y in coords:
                marker(draw, MODE_SHAPES[mode], x, y, 14, color)

        for value in (0, 10, 100, 1000):
            transformed = math.log10(value + 1.0)
            if x_min <= transformed <= x_max:
                x = plot[0] + (transformed - x_min) / max(1e-9, x_max - x_min) * (plot[2] - plot[0])
                draw.line((x, plot[3], x, plot[3] + 10), fill=INK, width=3)
                draw.text((x - 22, plot[3] + 16), str(value), font=font(30), fill=MUTED)
        draw.text((left + 200, top + height - 56), "请求 KB/图（log₁₀(KB+1)）", font=font(32), fill=MUTED)

    legend_x = 660
    for index, mode in enumerate(MODE_ORDER):
        x = legend_x + index * 330
        marker(draw, MODE_SHAPES[mode], x + 16, 1022, 15, MODE_COLORS[mode])
        draw.text((x + 44, 1002), MODE_LABELS[mode], font=font(34), fill=INK)
    return save(image, path)


def build_gate_summary(path: Path, evidence: Mapping[str, Any]) -> Path:
    image = Image.new("RGB", (1700, 860), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 190, 60, 1580, 640
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    for tick in range(0, 11, 2):
        y = bottom - tick / 10 * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((120, y - 20), str(tick), font=font(32), fill=MUTED)

    colors = {"non_identical": DARKGREEN, "identical": AMBER, "fallback": RED}
    labels = {"non_identical": "非同动作通过", "identical": "同动作通过", "fallback": "回退参考配置"}
    text_fill = {"non_identical": "white", "identical": INK, "fallback": "white"}
    bar_width = 240
    centers = [440, 880, 1320]
    for center, dataset in zip(centers, DATASETS):
        y_cursor = float(bottom)
        summary = evidence["gate_summary"][dataset]
        for key in ("non_identical", "identical", "fallback"):
            value = summary[key]
            segment = value / 10 * (bottom - top)
            if value:
                seg_top = y_cursor - segment
                draw.rectangle(
                    (center - bar_width // 2, seg_top, center + bar_width // 2, y_cursor - 2),
                    fill=colors[key],
                )
                text_center(
                    draw,
                    (center - bar_width // 2, int(seg_top), center + bar_width // 2, int(y_cursor)),
                    str(value), font(40, bold=True), fill=text_fill[key],
                )
            y_cursor -= segment
        draw.text((center - 70, bottom + 24), DATASET_LABELS[dataset], font=font(40, bold=True), fill=INK)

    legend_x = 260
    for index, key in enumerate(("non_identical", "identical", "fallback")):
        x = legend_x + index * 420
        draw.rectangle((x, 764, x + 34, 798), fill=colors[key])
        draw.text((x + 48, 758), labels[key], font=font(34), fill=INK)
    return save(image, path)


def ordered_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    order = {m: i for i, m in enumerate(MODE_ORDER)}
    return sorted(rows, key=lambda row: (order[mode_name(row)], budget_value(row)))


def build_action_composition(path: Path, evidence: Mapping[str, Any]) -> Path:
    image = Image.new("RGB", (2200, 1760), "white")
    draw = ImageDraw.Draw(image)
    colors = {"edge": LIGHTGRAY, "original": BLUE, "q90": DARKGREEN, "roi": AMBER}
    labels = {"edge": "Edge（未上传）", "original": "Original", "q90": "Q90", "roi": "ROI"}
    mode_short = {"fixed_call": "FC", "online": "ON", "fixed_byte": "FB"}

    for panel_index, dataset in enumerate(DATASETS):
        left = 55 + panel_index * 715
        top, width, height = 40, 665, 1560
        draw.rounded_rectangle((left, top, left + width, top + height), radius=15, fill=PANEL, outline=GRID, width=2)
        draw.text((left + 22, top + 20), DATASET_LABELS[dataset], font=font(42, bold=True), fill=INK)
        rows = ordered_rows(evidence["final"][dataset])
        bar_left, bar_right = left + 175, left + width - 30
        for row_index, row in enumerate(rows):
            y = top + 110 + row_index * 138
            budget = budget_value(row)
            label = f"{mode_short[mode_name(row)]} {budget:.0%}"
            draw.text((left + 20, y + 12), label, font=font(32, bold=True), fill=INK)
            upload = number(row, "upload_ratio")
            parts = {
                "edge": max(0.0, 1.0 - upload),
                "original": number(row, "full_ratio"),
                "q90": number(row, "q90_ratio"),
                "roi": number(row, "roi_ratio"),
            }
            x = float(bar_left)
            for key in ("edge", "original", "q90", "roi"):
                segment = max(0.0, min(1.0, parts[key])) * (bar_right - bar_left)
                if segment > 1.0:
                    gap = 2 if segment > 5.0 else 0
                    draw.rectangle((x, y, x + segment - gap, y + 62), fill=colors[key])
                x += segment
            draw.rectangle((bar_left, y, bar_right, y + 62), outline="#7A848D", width=2)
            draw.text(
                (bar_left, y + 70),
                f"上传 {upload:.1%} · ROI {parts['roi']:.2%}",
                font=font(28), fill=MUTED,
            )
        draw.text((bar_left - 6, top + height - 52), "0%", font=font(30), fill=MUTED)
        draw.text((bar_right - 76, top + height - 52), "100%", font=font(30), fill=MUTED)

    legend_x = 480
    for index, key in enumerate(("edge", "original", "q90", "roi")):
        x = legend_x + index * 330
        draw.rectangle((x, 1660, x + 36, 1696), fill=colors[key])
        draw.text((x + 50, 1654), labels[key], font=font(34), fill=INK)
    return save(image, path)


def build_all(output_dir: Path | None = None) -> list[Path]:
    output = output_dir or PACKAGE / "figures"
    evidence = load_evidence()
    architecture = output / "Fig1_calibrated_costaware_architecture.png"
    architecture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "fig1_v09_architecture.png", architecture)
    return [
        architecture,
        build_ap_kb(output / "Fig2_frozen_ap_request_cost.png", evidence),
        build_gate_summary(output / "Fig3_risk_gate_outcomes.png", evidence),
        build_action_composition(output / "Fig4_frozen_action_composition.png", evidence),
    ]


if __name__ == "__main__":
    for output_path in build_all():
        print(output_path)
