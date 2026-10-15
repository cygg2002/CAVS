#!/usr/bin/env python3
"""Generate V3.4 manuscript figures from frozen point and formal HIL results."""

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

ONLINE_AP = {
    "CAVS-OQR": [0.454831, 0.505348, 0.530900],
    "CAVS-OQ": [0.451892, 0.500310, 0.527717],
    "ORIC": [0.438553, 0.468420, 0.492202],
    "MORIC": [0.430520, 0.463411, 0.485750],
    "Confidence": [0.427864, 0.461410, 0.482862],
    "DCSB": [0.432997, 0.464606, 0.490678],
    "Adaptive Feeding": [0.433734, 0.465847, 0.493665],
    "Random-5": [0.421070, 0.444743, 0.465573],
}

ONLINE_COLORS = {
    "CAVS-OQR": "#0B6E99",
    "CAVS-OQ": "#4C9CC7",
    "ORIC": "#2CA02C",
    "MORIC": "#D95F02",
    "Confidence": "#7570B3",
    "DCSB": "#A6761D",
    "Adaptive Feeding": "#E7298A",
    "Random-5": "#777777",
}


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
        "RK3588 causal online": [0.454831, 0.505348, 0.530900],
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
    online = np.asarray(series["RK3588 causal online"])
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
        "Formal online runs use binary HTTP, zero lookahead, and zero cloud failures.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.35)
    save(fig, "fig_coco_offline_to_online_gap")


def online_system_results() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.85))
    ax, time_ax, byte_ax = axes

    markers = ["o", "s", "^", "D", "v", "P", "X", "x"]
    for (method, values), marker in zip(ONLINE_AP.items(), markers):
        ax.plot(
            BUDGETS,
            values,
            marker=marker,
            linewidth=2.2 if method == "CAVS-OQR" else 1.35,
            markersize=5.0,
            color=ONLINE_COLORS[method],
            label=method,
        )
    ax.set_title("(a) Formal causal-online AP")
    ax.set_xlabel("JPEG payload budget (%)")
    ax.set_ylabel("COCO AP50–95")
    ax.set_xticks(BUDGETS)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    stages = {
        "JPEG decode": [6.85, 7.41, 7.78],
        "Edge pre+RKNN+NMS": [86.80, 90.80, 94.18],
        "Candidate+controller": [16.63, 17.21, 17.43],
        "Cloud RTT": [42.56, 72.15, 108.82],
        "ROI fusion": [4.01, 4.60, 2.80],
        "Commit/log": [5.87, 6.84, 7.47],
    }
    stage_colors = ["#9ECAE1", "#3182BD", "#74C476", "#FD8D3C", "#9E9AC8", "#BDBDBD"]
    bottom = np.zeros(3)
    for (label, values), color in zip(stages.items(), stage_colors):
        vals = np.asarray(values)
        time_ax.bar(BUDGETS, vals, width=8.5, bottom=bottom, label=label, color=color)
        bottom += vals
    total_p95 = np.asarray([305.61, 306.07, 327.94])
    time_ax.plot(BUDGETS, total_p95, color="#111111", marker="x", linewidth=1.2, label="Total P95")
    time_ax.set_title("(b) CAVS-OQR E2E decomposition")
    time_ax.set_xlabel("Budget (%)")
    time_ax.set_ylabel("Latency (ms/input frame)")
    time_ax.set_xticks(BUDGETS)
    time_ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    time_ax.legend(frameon=False, fontsize=6.9, ncol=2, loc="upper left")

    payload = np.asarray([16.256, 40.591, 65.049])
    uplink_overhead = np.asarray([0.069, 0.138, 0.189])
    downlink = np.asarray([4.407, 8.563, 10.518])
    byte_ax.bar(BUDGETS, payload, width=8.5, color="#4C78A8", label="JPEG payload")
    byte_ax.bar(
        BUDGETS,
        uplink_overhead,
        width=8.5,
        bottom=payload,
        color="#F2CF5B",
        label="HTTP uplink overhead",
    )
    byte_ax.bar(
        BUDGETS,
        downlink,
        width=8.5,
        bottom=payload + uplink_overhead,
        color="#E45756",
        label="Detection downlink",
    )
    byte_ax.set_title("(c) Bidirectional bytes per input")
    byte_ax.set_xlabel("Budget (%)")
    byte_ax.set_ylabel("kB/input frame")
    byte_ax.set_xticks(BUDGETS)
    byte_ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    byte_ax.legend(frameon=False, fontsize=7.3)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_online_system_results")


def network_sensitivity() -> None:
    labels = ["LAN", "20 ms/\n20 Mbps", "50 ms/\n5 Mbps"]
    x = np.arange(len(labels))
    mean = {
        "CAVS-OQR": [199.01, 260.78, 350.02],
        "CAVS-OQ": [188.61, 241.76, 329.12],
        "ORIC": [149.70, 186.73, 260.50],
        "MORIC": [153.57, 194.14, 268.55],
    }
    p95 = {
        "CAVS-OQR": [306.07, 427.05, 620.03],
        "CAVS-OQ": [297.32, 413.80, 623.59],
        "ORIC": [281.91, 416.75, 709.14],
        "MORIC": [271.67, 396.48, 647.24],
    }
    fps = {
        "CAVS-OQR": [4.73, 3.67, 2.77],
        "CAVS-OQ": [4.99, 3.94, 2.94],
        "ORIC": [6.22, 5.05, 3.68],
        "MORIC": [6.07, 4.87, 3.58],
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.65), sharex=True)
    for method in mean:
        color = ONLINE_COLORS[method]
        axes[0].plot(x, mean[method], marker="o", color=color, label=method)
        axes[1].plot(x, p95[method], marker="o", color=color, label=method)
        axes[2].plot(x, fps[method], marker="o", color=color, label=method)
    titles = ["(a) E2E mean", "(b) E2E P95", "(c) Stream throughput"]
    ylabels = ["ms", "ms", "FPS"]
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.grid(True, linestyle="--", alpha=0.3)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.text(
        0.5,
        0.015,
        "Budget = 25%; actions and AP remain unchanged because all cloud requests succeed.",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(top=0.78, bottom=0.25, wspace=0.32)
    save(fig, "fig_network_sensitivity")


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
    online_system_results()
    network_sensitivity()


if __name__ == "__main__":
    main()
