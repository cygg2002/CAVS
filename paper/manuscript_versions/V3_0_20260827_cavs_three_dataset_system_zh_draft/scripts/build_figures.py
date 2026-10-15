#!/usr/bin/env python3
"""Generate manuscript figures from frozen point results only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BUDGETS = np.array([10, 25, 40])

DATASETS = {
    "COCO": {
        "CAVS": [0.480277, 0.532900, 0.560178],
        "MORIC": [0.452230, 0.496914, 0.528170],
        "Confidence": [0.447029, 0.489546, 0.520311],
        "Random": [0.433586, 0.468908, 0.500443],
    },
    "Oregon": {
        "CAVS": [0.506028, 0.517405, 0.516663],
        "MORIC": [0.479826, 0.493286, 0.506171],
        "Confidence": [0.489643, 0.502499, 0.510449],
        "Random": [0.471087, 0.484147, 0.495973],
    },
    "VOC": {
        "CAVS": [0.531486, 0.579678, 0.612721],
        "MORIC": [0.504416, 0.554956, 0.592212],
        "Confidence": [0.495439, 0.532334, 0.569091],
        "Random": [0.482501, 0.503851, 0.532850],
    },
}

COLORS = {
    "CAVS": "#0B6E99",
    "MORIC": "#D95F02",
    "Confidence": "#7570B3",
    "Random": "#777777",
}
MARKERS = {"CAVS": "o", "MORIC": "s", "Confidence": "^", "Random": "x"}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fixed_byte_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), sharex=True)
    for ax, (dataset, series) in zip(axes, DATASETS.items()):
        for method, values in series.items():
            ax.plot(
                BUDGETS,
                values,
                label=method,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.0 if method == "CAVS" else 1.45,
                markersize=5.5,
            )
        ax.set_title(dataset)
        ax.set_xticks(BUDGETS)
        ax.set_xlabel("Fixed-byte budget (%)")
        ax.grid(True, linestyle="--", alpha=0.32)
    axes[0].set_ylabel("AP50–95")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.78, wspace=0.24)
    save(fig, "fig_fixed_byte_three_datasets")


def voc_action_ablation() -> None:
    values = {
        "O": [0.520249, 0.557280, 0.589427],
        "OQ": [0.530787, 0.579142, 0.614054],
        "OQR": [0.531486, 0.579678, 0.612721],
    }
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for name, vals in values.items():
        ax.plot(BUDGETS, vals, marker="o", linewidth=1.8, label=f"CAVS-{name}")
    ax.set_xlabel("Fixed-byte budget (%)")
    ax.set_ylabel("VOC AP50–95")
    ax.set_xticks(BUDGETS)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(frameon=False)
    ax.annotate(
        "ROI is not uniformly beneficial",
        xy=(40, values["OQR"][2]),
        xytext=(20, 0.602),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
    )
    save(fig, "fig_voc_action_space_ablation")


def deployment_gap() -> None:
    series = {
        "Paper cache": [0.480277, 0.532900, 0.560178],
        "Deployment-aligned offline": [0.479095, 0.531125, 0.560189],
        "RK3588 batch HIL": [0.461068, 0.503569, 0.530318],
        "RK3588 causal online pilot": [0.454842, 0.505302, 0.529707],
    }
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    styles = ["-", "--", "-.", ":"]
    for (name, vals), style in zip(series.items(), styles):
        ax.plot(BUDGETS, vals, linestyle=style, marker="o", linewidth=1.9, label=name)
    ax.set_xlabel("Fixed-byte budget (%)")
    ax.set_ylabel("COCO AP50–95")
    ax.set_xticks(BUDGETS)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(frameon=False, fontsize=8.5)
    ax.text(
        0.01,
        0.02,
        "Online values are pilot HIL results with protocol caveats.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    save(fig, "fig_coco_offline_to_online_gap")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fixed_byte_curves()
    voc_action_ablation()
    deployment_gap()


if __name__ == "__main__":
    main()
