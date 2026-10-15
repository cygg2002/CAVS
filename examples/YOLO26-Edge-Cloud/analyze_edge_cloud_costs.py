# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Analyze accuracy, latency, bandwidth, and compute trade-offs for edge-cloud runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


DEFAULT_NETWORKS = {
    "field_2Mbps": {"uplink_mbps": 2.0, "rtt_ms": 100.0},
    "rural_5Mbps": {"uplink_mbps": 5.0, "rtt_ms": 80.0},
    "mobile_10Mbps": {"uplink_mbps": 10.0, "rtt_ms": 50.0},
    "good_50Mbps": {"uplink_mbps": 50.0, "rtt_ms": 20.0},
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default="runs/edge_cloud/strategy_comparison_learned/strategy_summary.csv",
        help="Strategy summary CSV from summarize_strategy_results.py.",
    )
    parser.add_argument("--save-dir", default="runs/edge_cloud/cost_analysis", help="Output directory.")
    parser.add_argument("--edge-model", default="yolo26s.pt", help="Edge model path.")
    parser.add_argument("--cloud-model", default="yolo26x.pt", help="Cloud model path.")
    parser.add_argument("--learned-policy", default="", help="Optional learned policy JSON path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Model profile image size.")
    parser.add_argument(
        "--networks",
        default="field_2Mbps:2:100,rural_5Mbps:5:80,mobile_10Mbps:10:50,good_50Mbps:50:20",
        help="Comma-separated NAME:UPLINK_MBPS:RTT_MS network scenarios.",
    )
    return parser.parse_args()


def profile_model(path: str, imgsz: int) -> dict[str, Any]:
    """Profile an Ultralytics model for layers, parameters, gradients, and GFLOPs."""
    model = YOLO(path)
    try:
        layers, params, gradients, gflops = model.info(detailed=False, verbose=True, imgsz=imgsz)
    except TypeError:
        layers, params, gradients, gflops = model.info(detailed=False, verbose=True)
    weight_mb = round(Path(path).stat().st_size / (1024.0 * 1024.0), 3) if Path(path).exists() else None
    return {
        "path": path,
        "layers": int(layers),
        "params": int(params),
        "gradients": int(gradients),
        "gflops": float(gflops),
        "profile_imgsz": int(imgsz),
        "weight_mb": weight_mb,
    }


def parse_networks(value: str) -> dict[str, dict[str, float]]:
    """Parse network scenario text."""
    networks: dict[str, dict[str, float]] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        name, mbps, rtt = item.split(":")
        networks[name] = {"uplink_mbps": float(mbps), "rtt_ms": float(rtt)}
    return networks or DEFAULT_NETWORKS


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read strategy CSV rows with numeric conversion."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = value
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def learned_policy_size(path: str) -> dict[str, Any] | None:
    """Return learned policy parameter counts and size."""
    if not path:
        return None
    policy_path = Path(path)
    if not policy_path.exists():
        return None
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    roi_params = len(policy["roi_model"]["weights"]) + 1
    frame_params = len(policy["frame_model"]["weights"]) + 1
    return {
        "path": str(policy_path),
        "roi_linear_params": roi_params,
        "frame_linear_params": frame_params,
        "total_linear_params": roi_params + frame_params,
        "json_kb": round(policy_path.stat().st_size / 1024.0, 3),
    }


def compute_cost_rows(
    rows: list[dict[str, Any]],
    edge_profile: dict[str, Any],
    cloud_profile: dict[str, Any],
    networks: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Compute cost metrics for each strategy and network scenario."""
    edge_map = float(rows[0].get("edge_map50_95", 0.0))
    edge_gflops = float(edge_profile["gflops"])
    cloud_gflops = float(cloud_profile["gflops"])
    out: list[dict[str, Any]] = []
    for row in rows:
        images = max(1.0, float(row["images"]))
        upload_tasks = float(row["uploads_submitted"])
        upload_ratio = float(row["upload_ratio"])
        payload_mb = float(row["total_payload_mb"])
        payload_kb_per_image = payload_mb * 1024.0 / images
        row_cloud_imgsz = float(row.get("cloud_imgsz", 640.0) or 640.0)
        profiled_imgsz = float(edge_profile.get("profile_imgsz", 640.0))
        scale_value = row.get("cloud_compute_scale_per_upload", "")
        if scale_value != "":
            row_cloud_gflops = cloud_gflops * float(scale_value)
        else:
            row_cloud_gflops = cloud_gflops * (row_cloud_imgsz / max(1.0, profiled_imgsz)) ** 2
        cloud_gflops_per_image = upload_ratio * row_cloud_gflops
        total_gflops_per_image = edge_gflops + cloud_gflops_per_image
        map_gain = float(row["map50_95"]) - edge_map
        base = {
            "strategy": row["strategy"],
            "images": int(images),
            "map50_95": float(row["map50_95"]),
            "map50": float(row["map50"]),
            "recall": float(row["recall"]),
            "precision": float(row["precision"]),
            "map50_95_gain_vs_edge": map_gain,
            "uploads_submitted": int(upload_tasks),
            "upload_tasks_per_image": upload_ratio,
            "uploaded_image_ratio": float(row.get("uploaded_image_ratio", upload_ratio)),
            "payload_mb_total": payload_mb,
            "payload_kb_per_image": payload_kb_per_image,
            "avg_payload_kb_per_upload": float(row["avg_payload_kb"]),
            "edge_latency_ms_avg": float(row["edge_latency_ms_avg"]),
            "cloud_latency_ms_avg_per_upload": float(row["cloud_latency_ms_avg"]),
            "cloud_latency_ms_p95_per_upload": float(row["cloud_latency_ms_p95"]),
            "edge_gflops_per_image": edge_gflops,
            "cloud_imgsz": row_cloud_imgsz,
            "cloud_gflops_per_upload": row_cloud_gflops,
            "cloud_gflops_per_image": cloud_gflops_per_image,
            "total_system_gflops_per_image": total_gflops_per_image,
            "system_gflops_ratio_vs_cloud_only": total_gflops_per_image / cloud_gflops,
            "map_gain_per_payload_mb": map_gain / payload_mb if payload_mb > 0 else 0.0,
            "map_gain_per_100_cloud_gflops_per_image": map_gain / (cloud_gflops_per_image / 100.0)
            if cloud_gflops_per_image > 0
            else 0.0,
        }
        for network_name, network in networks.items():
            uplink_mbps = network["uplink_mbps"]
            rtt_ms = network["rtt_ms"]
            upload_transfer_ms_per_image = payload_kb_per_image * 8.0 / max(1e-6, uplink_mbps)
            rtt_ms_per_image = upload_ratio * rtt_ms
            cloud_work_ms_per_image = upload_ratio * float(row["cloud_latency_ms_avg"])
            blocking_ms = float(row["edge_latency_ms_avg"]) + upload_transfer_ms_per_image + rtt_ms_per_image + cloud_work_ms_per_image
            async_cloud_network_work_ms = upload_transfer_ms_per_image + rtt_ms_per_image + cloud_work_ms_per_image
            item = dict(base)
            item.update(
                {
                    "network": network_name,
                    "uplink_mbps": uplink_mbps,
                    "rtt_ms": rtt_ms,
                    "upload_transfer_ms_per_image": upload_transfer_ms_per_image,
                    "rtt_ms_per_image": rtt_ms_per_image,
                    "cloud_work_ms_per_image": cloud_work_ms_per_image,
                    "blocking_e2e_latency_ms_est": blocking_ms,
                    "blocking_fps_est": 1000.0 / blocking_ms if blocking_ms > 0 else 0.0,
                    "async_extra_cloud_network_ms_per_image": async_cloud_network_work_ms,
                    "map_gain_per_blocking_second": map_gain / (blocking_ms / 1000.0) if blocking_ms > 0 else 0.0,
                }
            )
            out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_costs(rows: list[dict[str, Any]], save_dir: Path) -> None:
    """Plot core cost/accuracy trade-offs."""
    import matplotlib.pyplot as plt

    mobile_rows = [row for row in rows if row["network"] == "mobile_10Mbps"]
    if not mobile_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x_payload = [row["payload_kb_per_image"] for row in mobile_rows]
    y_map = [row["map50_95"] for row in mobile_rows]
    x_latency = [row["blocking_e2e_latency_ms_est"] for row in mobile_rows]
    x_compute = [row["total_system_gflops_per_image"] for row in mobile_rows]
    labels = [str(row["strategy"]) for row in mobile_rows]

    axes[0].scatter(x_payload, y_map, s=60)
    axes[1].scatter(x_latency, y_map, s=60)
    axes[2].scatter(x_compute, y_map, s=60)
    for ax, xs in zip(axes, [x_payload, x_latency, x_compute]):
        for x, y, label in zip(xs, y_map, labels):
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_xlabel("Upload KB / image")
    axes[1].set_xlabel("Estimated blocking latency @10Mbps (ms)")
    axes[2].set_xlabel("Total system GFLOPs / image")
    axes[0].set_ylabel("mAP50-95")
    axes[0].set_title("Bandwidth vs Accuracy")
    axes[1].set_title("Latency vs Accuracy")
    axes[2].set_title("Compute vs Accuracy")
    fig.tight_layout()
    fig.savefig(save_dir / "cost_accuracy_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run cost analysis."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    strategy_rows = read_rows(Path(args.summary_csv))
    networks = parse_networks(args.networks)
    edge_profile = profile_model(args.edge_model, args.imgsz)
    cloud_profile = profile_model(args.cloud_model, args.imgsz)
    policy_profile = learned_policy_size(args.learned_policy)
    cost_rows = compute_cost_rows(strategy_rows, edge_profile, cloud_profile, networks)

    write_csv(save_dir / "cost_analysis.csv", cost_rows)
    report = {
        "model_profiles": {"edge": edge_profile, "cloud": cloud_profile, "learned_policy": policy_profile},
        "networks": networks,
        "rows": cost_rows,
    }
    (save_dir / "cost_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        plot_costs(cost_rows, save_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Plotting failed: {exc}")

    mobile_rows = [row for row in cost_rows if row["network"] == "mobile_10Mbps"]
    for row in sorted(mobile_rows, key=lambda item: item["map50_95"], reverse=True):
        print(
            "{strategy}: mAP={map50_95:.4f}, KB/img={payload_kb_per_image:.2f}, "
            "GFLOPs/img={total_system_gflops_per_image:.1f}, E2E@10Mbps={blocking_e2e_latency_ms_est:.1f}ms".format(**row)
        )
    print(f"CSV: {save_dir / 'cost_analysis.csv'}")


if __name__ == "__main__":
    main()
