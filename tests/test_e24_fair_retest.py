# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the E24 frozen fair-retest summarizer."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "paper" / "experiments" / "E24_frozen_fair_retest" / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_e24_results import comparison_indicators, derive_deploy_row, run_summary  # noqa: E402


def synthetic_summary() -> dict:
    """Return a compact evaluator summary with easy-to-check arithmetic."""
    return {
        "images": 100,
        "imgsz": 640,
        "cloud_imgsz": 512,
        "edge_only_metrics": {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.4},
        "edge_only_deploy_metrics": {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.4},
        "cloud_only_metrics": {"metrics/mAP50(B)": 0.7, "metrics/mAP50-95(B)": 0.6},
        "cloud_only_deploy_metrics": {"metrics/mAP50(B)": 0.7, "metrics/mAP50-95(B)": 0.6},
        "edge_cloud_fused_metrics": {"metrics/mAP50(B)": 0.65, "metrics/mAP50-95(B)": 0.55},
        "edge_cloud_fused_deploy_metrics": {"metrics/mAP50(B)": 0.65, "metrics/mAP50-95(B)": 0.55},
        "edge_only_operating_point": {
            "tp": 40,
            "fp": 10,
            "fn": 20,
            "precision": 0.8,
            "recall": 2 / 3,
            "f1": 8 / 11,
        },
        "cloud_only_operating_point": {
            "tp": 50,
            "fp": 10,
            "fn": 10,
            "precision": 5 / 6,
            "recall": 5 / 6,
            "f1": 5 / 6,
        },
        "edge_cloud_fused_operating_point": {
            "tp": 48,
            "fp": 12,
            "fn": 12,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
        },
        "uploads_submitted": 20,
        "upload_ratio": 0.2,
        "uploaded_images": 15,
        "uploaded_image_ratio": 0.15,
        "avg_payload_kb": 51.2,
        "total_payload_mb": 1.0,
        "cloud_compute_scale_sum": 12.8,
        "edge_latency_ms_avg": 10.0,
        "pipeline_latency_ms_avg": 15.0,
        "pipeline_latency_ms_p95": 25.0,
        "cloud_only_total_payload_mb": 2.0,
        "cloud_only_avg_payload_kb": 20.48,
        "cloud_only_latency_ms_avg": 30.0,
        "cloud_only_encode_ms_avg": 2.0,
        "cloud_only_decode_ms_avg": 1.0,
    }


def test_deploy_row_uses_cloud_invocations_for_compute_and_rtt():
    """System cost should count every cloud request, not only unique uploaded images."""
    row = derive_deploy_row(
        "learned",
        "system",
        synthetic_summary(),
        edge_gflops=20.0,
        cloud_gflops_640=200.0,
        uplink_mbps=10.0,
        rtt_ms=50.0,
    )
    assert math.isclose(row["payload_kb_per_source_image"], 10.24)
    assert math.isclose(row["cloud_gflops_per_source_image"], 25.6)
    assert math.isclose(row["total_gflops_per_source_image"], 45.6)
    assert math.isclose(row["network_transfer_ms_per_source_image"], 8.192)
    assert math.isclose(row["rtt_ms_per_source_image"], 10.0)
    assert math.isclose(row["blocking_e2e_ms_est"], 33.192)


def test_cloud_only_row_includes_codec_payload_compute_and_latency():
    """Cloud-only Q80 must include full-frame bytes, codec time, cloud compute, and one RTT."""
    row = derive_deploy_row(
        "cloud",
        "cloud",
        synthetic_summary(),
        edge_gflops=20.0,
        cloud_gflops_640=200.0,
        uplink_mbps=10.0,
        rtt_ms=50.0,
    )
    assert math.isclose(row["payload_kb_per_source_image"], 20.48)
    assert math.isclose(row["total_gflops_per_source_image"], 128.0)
    assert math.isclose(row["local_pipeline_ms_avg"], 33.0)
    assert math.isclose(row["blocking_e2e_ms_est"], 99.384)


def test_comparison_indicators_quantify_accuracy_cost_tradeoff():
    """Headline indicators should report retained cloud accuracy and recovered cloud gain."""
    edge = {
        "map50_95": 0.5,
        "uploaded_image_ratio": 0.0,
        "total_gflops_per_source_image": 20.0,
        "blocking_e2e_ms_est": 20.0,
    }
    cloud = {
        "map50_95": 0.6,
        "uploaded_image_ratio": 1.0,
        "total_gflops_per_source_image": 100.0,
        "blocking_e2e_ms_est": 100.0,
    }
    system = {
        "map50_95": 0.58,
        "uploaded_image_ratio": 0.4,
        "total_gflops_per_source_image": 60.0,
        "blocking_e2e_ms_est": 70.0,
    }
    indicators = comparison_indicators(edge, cloud, system)
    assert math.isclose(indicators["map_gain_vs_edge"], 0.08)
    assert math.isclose(indicators["map_gap_vs_cloud"], -0.02)
    assert math.isclose(indicators["cloud_accuracy_retained"], 0.58 / 0.6)
    assert math.isclose(indicators["cloud_gain_recovered"], 0.8)
    assert math.isclose(indicators["upload_saved_vs_cloud"], 0.6)
    assert math.isclose(indicators["gflops_saved_vs_cloud"], 0.4)
    assert math.isclose(indicators["e2e_saved_vs_cloud"], 0.3)


def test_run_summary_ignores_unselected_stale_artifact(tmp_path):
    """A partial/core report must not consume old results from runs it did not select."""
    stale_summary = tmp_path / "runs" / "baseline_q80_640" / "edge_cloud_eval_summary.json"
    stale_summary.parent.mkdir(parents=True)
    stale_summary.write_text('{"images": 5000, "end2end": true, "head_mode": "end2end"}', encoding="utf-8")

    result = run_summary(
        tmp_path,
        "baseline_q80_640",
        required=False,
        selected_runs={"baseline_q80_512"},
    )

    assert result is None
