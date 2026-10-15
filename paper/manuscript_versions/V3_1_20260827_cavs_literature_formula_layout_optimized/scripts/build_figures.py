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
    fig.text(
        0.5,
        0.015,
        "Point estimates only; paired 95% confidence intervals are pending.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(top=0.78, bottom=0.22, wspace=0.24)
    save(fig, "fig_fixed_byte_three_datasets")


def voc_action_ablation() -> None:
    values = {
        "O": [0.520249, 0.557280, 0.589427],
        "OQ": [0.530787, 0.579142, 0.614054],
        "OQR": [0.531486, 0.579678, 0.612721],
    }
    fig, (ax, delta_ax) = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.7),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    for name, vals in values.items():
        ax.plot(BUDGETS, vals, marker="o", linewidth=1.8, label=f"CAVS-{name}")
    ax.set_xlabel("Fixed-byte budget (%)")
    ax.set_ylabel("VOC AP50–95")
    ax.set_title("(a) Absolute AP")
    ax.set_xticks(BUDGETS)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(frameon=False)

    width = 5.2
    oq_delta = (np.asarray(values["OQ"]) - np.asarray(values["O"])) * 1000
    roi_delta = (np.asarray(values["OQR"]) - np.asarray(values["OQ"])) * 1000
    delta_ax.bar(
        BUDGETS - width / 2,
        oq_delta,
        width=width,
        color="#4C78A8",
        label="OQ − O",
    )
    delta_ax.bar(
        BUDGETS + width / 2,
        roi_delta,
        width=width,
        color="#F58518",
        label="OQR − OQ",
    )
    delta_ax.axhline(0, color="#333333", linewidth=0.8)
    delta_ax.set_title("(b) Incremental action gain")
    delta_ax.set_xlabel("Fixed-byte budget (%)")
    delta_ax.set_ylabel("Δ AP50–95 (×10⁻³)")
    delta_ax.set_xticks(BUDGETS)
    delta_ax.set_ylim(
        min(-3.0, float(roi_delta.min()) - 1.5),
        float(max(oq_delta.max(), roi_delta.max())) * 1.15,
    )
    delta_ax.grid(True, axis="y", linestyle="--", alpha=0.32)
    delta_ax.legend(frameon=False, fontsize=8.5)
    for x, delta in zip(BUDGETS + width / 2, roi_delta):
        delta_ax.text(
            x,
            delta + (0.65 if delta >= 0 else -0.95),
            f"{delta:+.1f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=8,
        )
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_voc_action_space_ablation")


def deployment_gap() -> None:
    series = {
        "Paper cache": [0.480277, 0.532900, 0.560178],
        "Deployment-aligned offline": [0.479095, 0.531125, 0.560189],
        "RK3588 batch HIL": [0.461068, 0.503569, 0.530318],
        "RK3588 causal online pilot": [0.454842, 0.505302, 0.529707],
    }
    fig, (ax, gap_ax) = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.9),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    styles = ["-", "--", "-.", ":"]
    for (name, vals), style in zip(series.items(), styles):
        ax.plot(BUDGETS, vals, linestyle=style, marker="o", linewidth=1.9, label=name)
    ax.set_xlabel("Fixed-byte budget (%)")
    ax.set_ylabel("COCO AP50–95")
    ax.set_title("(a) Protocol ladder")
    ax.set_xticks(BUDGETS)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(frameon=False, fontsize=8.5)
    aligned = np.asarray(series["Deployment-aligned offline"])
    online = np.asarray(series["RK3588 causal online pilot"])
    gaps = aligned - online
    gap_ax.bar(BUDGETS, gaps, width=8.0, color="#B34D4D")
    gap_ax.set_title("(b) Offline → online gap")
    gap_ax.set_xlabel("Fixed-byte budget (%)")
    gap_ax.set_ylabel("AP50–95 loss")
    gap_ax.set_xticks(BUDGETS)
    gap_ax.set_ylim(0, max(gaps) * 1.28)
    gap_ax.grid(True, axis="y", linestyle="--", alpha=0.32)
    for x, gap in zip(BUDGETS, gaps):
        gap_ax.text(x, gap + 0.0008, f"{gap:.4f}", ha="center", fontsize=8.5)
    fig.text(
        0.5,
        0.01,
        "Online values are pilot HIL results with protocol caveats.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.35)
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
