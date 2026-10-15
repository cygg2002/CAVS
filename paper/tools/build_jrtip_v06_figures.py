#!/usr/bin/env python3
"""Build publication figures for the JRTIP V0.6 manuscript."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/manuscript_versions/V0_6_JRTIP_submission/figures"
MAIN_CSV = ROOT / "paper/experiments/E0_coco_main_results/tables/coco_main_results.csv"
NETWORK_CSV = ROOT / "paper/experiments/E16_paper_result_synthesis/tables/e16_table3a_network_realtime.csv"
FAIRNESS_JSON = (
    ROOT
    / "paper/experiments/E15_metric_fairness_check/runs/all_deploy_q80/edge_cloud_eval_summary.json"
)

WIDTH, HEIGHT = 4000, 2400
MARGIN_LEFT, MARGIN_RIGHT = 430, 180
MARGIN_TOP, MARGIN_BOTTOM = 180, 390


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a portable Arial-compatible font."""
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_AXIS = font(75)
FONT_TICK = font(58)
FONT_LABEL = font(58)
FONT_LEGEND = font(58)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, fnt: ImageFont.ImageFont, fill: str = "black") -> None:
    """Draw centered text."""
    box = draw.textbbox((0, 0), value, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, font=fnt, fill=fill)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def draw_axes(
    image: Image.Image,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: list[float],
    y_ticks: list[float],
    x_label: str,
    y_label: str,
) -> tuple[ImageDraw.ImageDraw, callable, callable]:
    """Draw shared chart axes and return coordinate transforms."""
    draw = ImageDraw.Draw(image)
    x0, x1 = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    y0, y1 = MARGIN_TOP, HEIGHT - MARGIN_BOTTOM

    def sx(value: float) -> float:
        return x0 + (value - x_min) / (x_max - x_min) * (x1 - x0)

    def sy(value: float) -> float:
        return y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)

    for tick in x_ticks:
        x = sx(tick)
        draw.line((x, y0, x, y1), fill="#D9D9D9", width=3)
        text_center(draw, (x, y1 + 85), f"{tick:g}", FONT_TICK)
    for tick in y_ticks:
        y = sy(tick)
        draw.line((x0, y, x1, y), fill="#D9D9D9", width=3)
        draw.text((x0 - 190, y - 35), f"{tick:.2f}", font=FONT_TICK, fill="black")
    draw.line((x0, y1, x1, y1), fill="black", width=7)
    draw.line((x0, y0, x0, y1), fill="black", width=7)
    text_center(draw, ((x0 + x1) / 2, HEIGHT - 130), x_label, FONT_AXIS)

    y_layer = Image.new("RGBA", (HEIGHT, 200), "white")
    y_draw = ImageDraw.Draw(y_layer)
    text_center(y_draw, (HEIGHT / 2, 100), y_label, FONT_AXIS)
    y_layer = y_layer.rotate(90, expand=True)
    image.paste(y_layer, (30, 0))
    return draw, sx, sy


def build_tradeoff() -> None:
    """Build mAP versus amortized upload, with compute represented by bubble area."""
    rows = read_csv(MAIN_CSV)
    fairness = json.loads(FAIRNESS_JSON.read_text(encoding="utf-8"))
    edge_deploy_map = fairness["edge_only_metrics"]["metrics/mAP50-95(B)"]
    names = {
        "Edge-only YOLO26s": "Edge-only",
        "Rule aggressive": "Rule-selective",
        "Learned v2 f0.50/r0.05": "CAFRU-Accuracy",
        "Learned v2 f0.60/r0.08": "CAFRU-Balanced",
        "Learned v2 f0.70/r0.10": "CAFRU-Efficient",
        "Learned v2 f0.80/r0.10": "CAFRU-Bandwidth",
    }
    selected = [row for row in rows if row["method"] in names]
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw, sx, sy = draw_axes(
        image,
        x_min=0,
        x_max=38,
        y_min=0.52,
        y_max=0.60,
        x_ticks=[0, 5, 10, 15, 20, 25, 30, 35],
        y_ticks=[0.52, 0.54, 0.56, 0.58, 0.60],
        x_label="Amortized uploaded data (KB/image)",
        y_label="mAP50-95",
    )
    cafru = []
    for row in selected:
        x = float(row["kb_per_image"])
        y = edge_deploy_map if row["method"] == "Edge-only YOLO26s" else float(row["map50_95"])
        gflops = float(row["gflops_per_image"])
        if row["method"].startswith("Learned v2"):
            cafru.append((x, y))
    cafru.sort()
    draw.line([(sx(x), sy(y)) for x, y in cafru], fill="#4472C4", width=10)

    offsets = {
        "Edge-only": (35, -75),
        "Rule-selective": (35, -60),
        "CAFRU-Accuracy": (-350, -115),
        "CAFRU-Balanced": (35, -75),
        "CAFRU-Efficient": (35, -75),
        "CAFRU-Bandwidth": (35, -75),
    }
    colors = {"Edge-only": "#7F7F7F", "Rule-selective": "#ED7D31"}
    for row in selected:
        label = names[row["method"]]
        x = float(row["kb_per_image"])
        y = edge_deploy_map if row["method"] == "Edge-only YOLO26s" else float(row["map50_95"])
        gflops = float(row["gflops_per_image"])
        radius = int(26 + 3.8 * gflops**0.5)
        px, py = sx(x), sy(y)
        fill = colors.get(label, "#5B9BD5")
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill, outline="#1F4E79", width=7)
        dx, dy = offsets[label]
        draw.text((px + dx, py + dy), label, font=FONT_LABEL, fill="black")
    draw.text((WIDTH - 1050, HEIGHT - 270), "Bubble area: amortized GFLOPs/image", font=FONT_LEGEND, fill="#404040")
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / "Fig3_accuracy_upload_compute.png", dpi=(600, 600), optimize=True)


def build_latency() -> None:
    """Build a grouped mean/P95 latency chart without internal method labels."""
    rows = read_csv(NETWORK_CSV)
    rename = {
        "Edge-only YOLO26s": "Edge",
        "Cloud-only YOLO26x@512 Q80": "Cloud Q80",
        "ROI-rule strict": "ROI rule",
        "Learned v2 Q70": "CAFRU Q70",
        "Learned v2 Q80": "CAFRU Q80",
        "Learned v2 Q90": "CAFRU Q90",
    }
    selected = [row for row in rows if row["Method"] in rename]
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    x0, x1 = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    y0, y1 = MARGIN_TOP, HEIGHT - 470
    y_max = 225.0

    def sy(value: float) -> float:
        return y1 - value / y_max * (y1 - y0)

    for tick in [0, 50, 100, 150, 200]:
        y = sy(tick)
        draw.line((x0, y, x1, y), fill="#D9D9D9", width=3)
        draw.text((x0 - 170, y - 35), str(tick), font=FONT_TICK, fill="black")
    draw.line((x0, y1, x1, y1), fill="black", width=7)
    draw.line((x0, y0, x0, y1), fill="black", width=7)
    slot = (x1 - x0) / len(selected)
    bar_w = slot * 0.26
    for index, row in enumerate(selected):
        center = x0 + slot * (index + 0.5)
        mean = float(row["Mean E2E(ms)"])
        p95 = float(row["P95 E2E(ms)"])
        draw.rectangle((center - bar_w - 8, sy(mean), center - 8, y1), fill="#5B9BD5", outline="#1F4E79", width=4)
        draw.rectangle((center + 8, sy(p95), center + bar_w + 8, y1), fill="#ED7D31", outline="#9E480E", width=4)
        label = rename[row["Method"]]
        text_center(draw, (center, y1 + 95), label, FONT_TICK)
    text_center(draw, ((x0 + x1) / 2, HEIGHT - 120), "Method", FONT_AXIS)
    y_layer = Image.new("RGBA", (HEIGHT, 200), "white")
    y_draw = ImageDraw.Draw(y_layer)
    text_center(y_draw, (HEIGHT / 2, 100), "End-to-end latency (ms)", FONT_AXIS)
    y_layer = y_layer.rotate(90, expand=True)
    image.paste(y_layer, (30, 0))
    lx, ly = x0 + 70, y0 + 45
    draw.rectangle((lx, ly, lx + 90, ly + 55), fill="#5B9BD5")
    draw.text((lx + 120, ly - 5), "Mean", font=FONT_LEGEND, fill="black")
    draw.rectangle((lx + 430, ly, lx + 520, ly + 55), fill="#ED7D31")
    draw.text((lx + 550, ly - 5), "P95", font=FONT_LEGEND, fill="black")
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / "Fig4_weak_network_latency.png", dpi=(600, 600), optimize=True)


if __name__ == "__main__":
    build_tradeoff()
    build_latency()
    print(f"Figures written to {OUT}")
