# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Summarize multiple edge-cloud strategy evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directories or edge_cloud_eval_summary.json files.")
    parser.add_argument("--save-dir", default="runs/edge_cloud/strategy_comparison", help="Output directory.")
    return parser.parse_args()


def summary_path(value: str) -> Path:
    """Resolve a run directory or a summary JSON path."""
    path = Path(value)
    if path.is_dir():
        path = path / "edge_cloud_eval_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def strategy_label(path: Path) -> str:
    """Create a compact label from a run directory."""
    label = path.parent.name
    for prefix in ("strategy_", "coco128_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label.replace("_", " ")


def flatten(summary: dict[str, Any], path: Path, baseline_map: float | None = None) -> dict[str, Any]:
    """Flatten one summary JSON into one CSV row."""
    fused = summary["edge_cloud_fused_metrics"]
    edge = summary["edge_only_metrics"]
    m_ap = float(fused["metrics/mAP50-95(B)"])
    events_path = path.parent / "fusion_events.jsonl"
    uploaded_images: set[str] = set()
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            image = event.get("image")
            if image:
                uploaded_images.add(str(image))
    row = {
        "strategy": strategy_label(path),
        "run_dir": str(path.parent),
        "images": summary["images"],
        "imgsz": summary.get("imgsz", 640),
        "cloud_imgsz": summary.get("cloud_imgsz", summary.get("imgsz", 640)),
        "adaptive_cloud_imgsz": summary.get("adaptive_cloud_imgsz", False),
        "cloud_compute_scale_per_upload": summary.get("cloud_compute_scale_per_upload", ""),
        "jpeg_quality": summary.get("jpeg_quality", 80),
        "simulate_upload_codec": summary.get("simulate_upload_codec", False),
        "upload_threshold": summary["upload_threshold"],
        "max_roi_per_image": summary["max_roi_per_image"],
        "keyframe_interval": summary["keyframe_interval"],
        "uploads_submitted": summary["uploads_submitted"],
        "upload_ratio": summary["upload_ratio"],
        "uploaded_images": len(uploaded_images),
        "uploaded_image_ratio": round(len(uploaded_images) / max(1, int(summary["images"])), 4),
        "roi_uploads": summary["roi_uploads"],
        "keyframe_uploads": summary["keyframe_uploads"],
        "proposal_uploads": summary.get("proposal_uploads", 0),
        "tile_uploads": summary.get("tile_uploads", 0),
        "total_payload_mb": summary["total_payload_mb"],
        "avg_payload_kb": summary["avg_payload_kb"],
        "edge_latency_ms_avg": summary["edge_latency_ms_avg"],
        "cloud_latency_ms_avg": summary["cloud_latency_ms_avg"],
        "cloud_latency_ms_p95": summary["cloud_latency_ms_p95"],
        "precision": float(fused["metrics/precision(B)"]),
        "recall": float(fused["metrics/recall(B)"]),
        "map50": float(fused["metrics/mAP50(B)"]),
        "map50_95": m_ap,
        "edge_map50_95": float(edge["metrics/mAP50-95(B)"]),
        "delta_map50_95_vs_edge": m_ap - float(edge["metrics/mAP50-95(B)"]),
    }
    if baseline_map is not None:
        row["delta_map50_95_vs_baseline"] = m_ap - baseline_map
    return row


def plot_rows(rows: list[dict[str, Any]], save_dir: Path) -> None:
    """Plot mAP/cost working points."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = [max(30.0, float(row["total_payload_mb"]) * 180.0) for row in rows]
    scatter = ax.scatter(
        [float(row["upload_ratio"]) for row in rows],
        [float(row["map50_95"]) for row in rows],
        s=sizes,
        c=[float(row["recall"]) for row in rows],
        cmap="viridis",
        alpha=0.85,
    )
    for row in rows:
        ax.annotate(
            row["strategy"],
            (float(row["upload_ratio"]), float(row["map50_95"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Edge-Cloud Upload Strategy Trade-off")
    ax.set_xlabel("Upload tasks per image")
    ax.set_ylabel("mAP50-95")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Recall")
    fig.tight_layout()
    fig.savefig(save_dir / "strategy_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Create CSV/JSON/plot summaries."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = [summary_path(run) for run in args.runs]
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    baseline_map = float(summaries[0]["edge_cloud_fused_metrics"]["metrics/mAP50-95(B)"])
    rows = [flatten(summary, path, baseline_map) for summary, path in zip(summaries, paths)]

    (save_dir / "strategy_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = save_dir / "strategy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        plot_rows(rows, save_dir)
    except Exception as exc:  # noqa: BLE001 - summary files are still valid without plotting
        print(f"Plotting failed: {exc}")

    for row in sorted(rows, key=lambda item: float(item["map50_95"]), reverse=True):
        print(
            "{strategy}: mAP50-95={map50_95:.4f}, recall={recall:.4f}, "
            "uploads={uploads_submitted}, payload={total_payload_mb:.3f}MB".format(**row)
        )
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
