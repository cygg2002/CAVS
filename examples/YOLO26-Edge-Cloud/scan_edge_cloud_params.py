# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Scan edge-cloud offloading parameters and plot accuracy/cost trade-offs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

from evaluate_edge_cloud import run_evaluation


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated float list."""
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=r"F:\IDE\trae\datasets\coco128", help="Dataset image directory.")
    parser.add_argument("--edge-model", default="yolo26s.pt", help="Edge model.")
    parser.add_argument("--cloud-model", default="yolo26x.pt", help="Cloud model.")
    parser.add_argument("--save-dir", default="runs/edge_cloud/coco128_scan", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Edge confidence threshold.")
    parser.add_argument("--cloud-conf", type=float, default=0.25, help="Cloud confidence threshold.")
    parser.add_argument("--device", default="0", help="Ultralytics device string.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images; 0 means all.")
    parser.add_argument("--upload-thresholds", default="0.25,0.35,0.45,0.55", help="Comma-separated thresholds.")
    parser.add_argument("--max-roi-values", default="1,2,3", help="Comma-separated max ROI values.")
    parser.add_argument("--keyframe-intervals", default="0,30,60", help="Comma-separated keyframe intervals. 0 disables.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality used for upload byte accounting.")
    parser.add_argument("--roi-expand", type=float, default=1.35, help="ROI expansion factor.")
    parser.add_argument("--iou-match", type=float, default=0.5, help="Fusion IoU threshold.")
    parser.add_argument("--fused-nms-iou", type=float, default=0.7, help="Final class-aware NMS IoU threshold.")
    return parser.parse_args()


def flatten_row(summary: dict) -> dict[str, float | int | str]:
    """Flatten one evaluation summary for CSV/table output."""
    fused = summary["edge_cloud_fused_metrics"]
    edge = summary["edge_only_metrics"]
    cloud = summary.get("cloud_only_metrics") or {}
    return {
        "upload_threshold": summary["upload_threshold"],
        "max_roi_per_image": summary["max_roi_per_image"],
        "keyframe_interval": summary["keyframe_interval"],
        "uploads_submitted": summary["uploads_submitted"],
        "upload_ratio": summary["upload_ratio"],
        "roi_uploads": summary["roi_uploads"],
        "keyframe_uploads": summary["keyframe_uploads"],
        "proposal_uploads": summary.get("proposal_uploads", 0),
        "tile_uploads": summary.get("tile_uploads", 0),
        "total_payload_mb": summary["total_payload_mb"],
        "avg_payload_kb": summary["avg_payload_kb"],
        "edge_latency_ms_avg": summary["edge_latency_ms_avg"],
        "cloud_latency_ms_avg": summary["cloud_latency_ms_avg"],
        "cloud_latency_ms_p95": summary["cloud_latency_ms_p95"],
        "fused_precision": fused["metrics/precision(B)"],
        "fused_recall": fused["metrics/recall(B)"],
        "fused_map50": fused["metrics/mAP50(B)"],
        "fused_map50_95": fused["metrics/mAP50-95(B)"],
        "edge_precision": edge["metrics/precision(B)"],
        "edge_recall": edge["metrics/recall(B)"],
        "edge_map50": edge["metrics/mAP50(B)"],
        "edge_map50_95": edge["metrics/mAP50-95(B)"],
        "cloud_precision": cloud.get("metrics/precision(B)", ""),
        "cloud_recall": cloud.get("metrics/recall(B)", ""),
        "cloud_map50": cloud.get("metrics/mAP50(B)", ""),
        "cloud_map50_95": cloud.get("metrics/mAP50-95(B)", ""),
    }


def plot_results(rows: list[dict], save_dir: Path) -> None:
    """Create trade-off plots."""
    import matplotlib.pyplot as plt

    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["keyframe_interval"]), []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for keyframe_interval, items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: (float(r["upload_ratio"]), int(r["max_roi_per_image"])))
        label = f"keyframe={keyframe_interval}"
        x = [float(r["upload_ratio"]) for r in items]
        axes[0, 0].plot(x, [float(r["fused_map50_95"]) for r in items], marker="o", label=label)
        axes[0, 1].plot(x, [float(r["fused_map50"]) for r in items], marker="o", label=label)
        axes[1, 0].plot(x, [float(r["total_payload_mb"]) for r in items], marker="o", label=label)
        axes[1, 1].plot(x, [float(r["cloud_latency_ms_avg"]) for r in items], marker="o", label=label)

    axes[0, 0].set_title("Upload Ratio vs mAP50-95")
    axes[0, 1].set_title("Upload Ratio vs mAP50")
    axes[1, 0].set_title("Upload Ratio vs Total Payload")
    axes[1, 1].set_title("Upload Ratio vs Cloud Latency")
    axes[0, 0].set_ylabel("mAP50-95")
    axes[0, 1].set_ylabel("mAP50")
    axes[1, 0].set_ylabel("Payload (MB)")
    axes[1, 1].set_ylabel("Cloud latency avg (ms)")
    for ax in axes.ravel():
        ax.set_xlabel("Upload ratio")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "tradeoff_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        [float(r["upload_ratio"]) for r in rows],
        [float(r["fused_map50_95"]) for r in rows],
        c=[int(r["max_roi_per_image"]) for r in rows],
        s=[max(20.0, float(r["total_payload_mb"]) * 600.0) for r in rows],
        cmap="viridis",
        alpha=0.85,
    )
    ax.set_title("Edge-Cloud Working Points")
    ax.set_xlabel("Upload ratio")
    ax.set_ylabel("Fused mAP50-95")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="max_roi_per_image")
    fig.tight_layout()
    fig.savefig(save_dir / "working_points.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the parameter scan."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    upload_thresholds = parse_float_list(args.upload_thresholds)
    max_roi_values = parse_int_list(args.max_roi_values)
    keyframe_intervals = parse_int_list(args.keyframe_intervals)

    print(f"Loading edge model once: {args.edge_model}")
    edge_model = YOLO(args.edge_model)
    print(f"Loading cloud model once: {args.cloud_model}")
    cloud_model = YOLO(args.cloud_model)

    summaries = []
    rows = []
    total = len(upload_thresholds) * len(max_roi_values) * len(keyframe_intervals)
    index = 0
    for upload_threshold in upload_thresholds:
        for max_roi in max_roi_values:
            for keyframe_interval in keyframe_intervals:
                index += 1
                run_name = f"thr{upload_threshold:.2f}_roi{max_roi}_kf{keyframe_interval}".replace(".", "p")
                run_args = argparse.Namespace(
                    source=args.source,
                    edge_model=args.edge_model,
                    cloud_model=args.cloud_model,
                    save_dir=str(save_dir / run_name),
                    imgsz=args.imgsz,
                    cloud_imgsz=args.imgsz,
                    conf=args.conf,
                    cloud_conf=args.cloud_conf,
                    device=args.device,
                    max_images=args.max_images,
                    upload_threshold=upload_threshold,
                    max_roi_per_image=max_roi,
                    keyframe_interval=keyframe_interval,
                    enable_complexity_probe=False,
                    complexity_threshold=0.18,
                    empty_complexity_threshold=0.08,
                    complexity_max_detections=1,
                    complexity_coverage_threshold=0.02,
                    enable_learned_policy=False,
                    learned_policy_model="",
                    learned_roi_threshold=0.5,
                    learned_frame_threshold=0.5,
                    learned_cost_weight=0.02,
                    learned_max_roi_per_image=1,
                    enable_learned_adaptive_cloud_imgsz=False,
                    learned_cloud_imgsz_low=416,
                    learned_cloud_imgsz_high=512,
                    learned_cloud_imgsz_switch=0.70,
                    enable_stability_probe=False,
                    stability_imgsz=384,
                    stability_threshold=0.45,
                    stability_iou=0.5,
                    enable_lowconf_proposal=False,
                    proposal_conf=0.05,
                    proposal_score_threshold=0.55,
                    max_proposals_per_image=2,
                    proposal_cloud_conf=0.35,
                    proposal_iou_match=0.25,
                    enable_tile_probe=False,
                    tile_grid="2x2",
                    max_tiles_per_image=1,
                    tile_score_threshold=0.12,
                    tile_cloud_conf=0.45,
                    tile_inner_margin=0.03,
                    tile_min_area_ratio=0.0,
                    jpeg_quality=args.jpeg_quality,
                    simulate_upload_codec=False,
                    roi_expand=args.roi_expand,
                    enable_adaptive_roi=False,
                    roi_small_gain=2.0,
                    roi_edge_gain=1.65,
                    roi_large_gain=1.15,
                    roi_extra_cloud_conf=-1.0,
                    roi_veto_edge_conf=1.0,
                    iou_match=args.iou_match,
                    fused_nms_iou=args.fused_nms_iou,
                    eval_cloud_only=False,
                )
                print(f"[{index}/{total}] threshold={upload_threshold} max_roi={max_roi} keyframe={keyframe_interval}")
                summary = run_evaluation(run_args, edge_model=edge_model, cloud_model=cloud_model, print_summary=False)
                summaries.append(summary)
                row = flatten_row(summary)
                rows.append(row)
                print(
                    "  upload={upload_ratio:.4f} mAP50={fused_map50:.4f} "
                    "mAP50-95={fused_map50_95:.4f} payload={total_payload_mb:.3f}MB".format(**row)
                )

    (save_dir / "scan_results.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = save_dir / "scan_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        plot_results(rows, save_dir)
    except Exception as exc:  # noqa: BLE001 - plotting should not invalidate scan data
        print(f"Plotting failed: {exc}")

    best_map = max(rows, key=lambda r: float(r["fused_map50_95"]))
    efficient = max(
        rows,
        key=lambda r: float(r["fused_map50_95"]) - 0.02 * float(r["upload_ratio"]) - 0.01 * float(r["total_payload_mb"]),
    )
    summary = {"best_map50_95": best_map, "balanced_working_point": efficient}
    (save_dir / "scan_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"CSV: {csv_path}")
    print(f"Plots: {save_dir / 'tradeoff_curves.png'}, {save_dir / 'working_points.png'}")


if __name__ == "__main__":
    main()
