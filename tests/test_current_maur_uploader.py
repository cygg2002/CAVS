# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Behavior contracts for the deployable CAVS Current+MAUR policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO_DIR))

import gradio_demo
from current_maur_uploader import (
    _validate_npz_archive,
    current_roi_xyxy,
    load_current_maur_policy,
    maur_fuse_boxes,
    predict_current_maur_scores,
    select_current_maur_action,
    validate_manifest,
)


def test_current_roi_matches_the_frozen_profile_geometry():
    """The live crop uses the exact current@640 expansion and 65% area gate."""
    box = {"xyxy": [40.0, 40.0, 60.0, 60.0], "cls": 0, "conf": 0.9}

    assert current_roi_xyxy(100, 100, [box]) == [34, 34, 66, 66]
    assert current_roi_xyxy(100, 100, [{**box, "conf": 0.1}]) is None
    assert current_roi_xyxy(
        100,
        100,
        [{"xyxy": [0.0, 0.0, 90.0, 90.0], "cls": 0, "conf": 0.9}],
    ) is None


def test_maur_fusion_keeps_only_outside_nonduplicate_edge_boxes():
    """MAUR removes Edge boxes covered by the ROI or duplicated by Cloud, then runs class NMS."""
    edge = [
        {"xyxy": [10.0, 10.0, 20.0, 20.0], "cls": 0, "conf": 0.8},
        {"xyxy": [61.0, 61.0, 89.0, 89.0], "cls": 0, "conf": 0.7},
        {"xyxy": [61.0, 61.0, 89.0, 89.0], "cls": 1, "conf": 0.6},
    ]
    cloud = [{"xyxy": [60.0, 60.0, 90.0, 90.0], "cls": 0, "conf": 0.95}]

    fused = maur_fuse_boxes(edge, cloud, [0, 0, 50, 50], duplicate_iou=0.5, nms_iou=0.7)

    assert [(box["cls"], box["conf"]) for box in fused] == [(0, 0.95), (1, 0.6)]


def test_current_maur_action_uses_value_minus_normalized_upload_cost():
    """A live frame uses the CAVS value head with Edge as the zero-cost fallback."""
    decision = select_current_maur_action(
        {"edge": 0.0, "original": 0.05, "q90": 0.04, "current@640": 0.06},
        {"edge": 0, "original": 1000, "q90": 500, "current@640": 200},
        full_payload_bytes=1000,
        cost_weight=0.1,
    )

    assert decision["action"] == "current@640"
    assert decision["utilities"]["current@640"] == pytest.approx(0.04)

    edge = select_current_maur_action(
        {"edge": 0.0, "original": 0.01, "q90": 0.01},
        {"edge": 0, "original": 1000, "q90": 500},
        full_payload_bytes=1000,
        cost_weight=0.2,
    )
    assert edge["action"] == "edge"


def test_current_maur_candidate_exposes_exact_action_bytes_without_selecting(monkeypatch):
    """The first video pass can score all available actions without issuing a cloud request."""
    monkeypatch.setattr(
        gradio_demo,
        "predict_current_maur_scores",
        lambda *_args, **_kwargs: {
            "edge": 0.0,
            "original": 0.3,
            "q90": 0.25,
            "current@640": 0.2,
        },
    )
    monkeypatch.setattr(
        gradio_demo,
        "encode_jpeg",
        lambda frame, quality: b"x" * ({95: 1000, 90: 500}[int(quality)] if frame.shape[0] == 100 else 120),
    )

    candidate = gradio_demo.prepare_current_maur_candidate(
        np.zeros((100, 100, 3), dtype=np.uint8),
        {
            "roi": {"confidence": 0.25, "max_area_ratio": 0.65, "imgsz": 640},
        },
        [{"xyxy": [40.0, 40.0, 60.0, 60.0], "cls": 0, "conf": 0.9}],
    )

    assert candidate["roi_xyxy"] == [34, 34, 66, 66]
    assert candidate["payload_bytes"] == {"edge": 0, "original": 1000, "q90": 500, "current@640": 120}
    assert candidate["scores"]["current@640"] == pytest.approx(0.2)


def test_forced_paper_action_bypasses_the_online_lambda_choice(monkeypatch):
    """The second video pass executes the globally frozen action, not a new local decision."""
    monkeypatch.setattr(
        gradio_demo,
        "predict_current_maur_scores",
        lambda *_args, **_kwargs: {"edge": 0.0, "original": 0.01, "q90": 0.01},
    )
    monkeypatch.setattr(
        gradio_demo,
        "encode_jpeg",
        lambda _frame, quality: b"x" * (1000 if int(quality) == 95 else 500),
    )
    observed = {}

    def fake_cloud_request(*args, **_kwargs):
        observed["quality"] = args[3]
        return (
            {"status": "ok", "cloud_boxes": [], "cloud_latency_ms": 1.0, "roundtrip_ms": 2.0},
            500,
            {"roundtrip_ms": 2.0},
        )

    monkeypatch.setattr(gradio_demo, "cloud_request_timed", fake_cloud_request)
    result = gradio_demo.run_current_maur_action(
        np.zeros((16, 16, 3), dtype=np.uint8),
        1,
        {
            "roi": {"confidence": 0.25, "max_area_ratio": 0.65, "imgsz": 640},
            "maur": {"duplicate_iou": 0.5, "nms_iou": 0.7},
        },
        [],
        [],
        "http://127.0.0.1:8765/infer",
        1.0,
        0.25,
        100.0,
        None,
        use_temporal_memory=False,
        frame_temporal_stable=False,
        temporal_tracks=[],
        runtime_args=argparse.Namespace(),
        forced_action="q90",
    )

    assert result["selected_action"] == "q90"
    assert result["decision_reason"] == "paper_fixed_byte_q90"
    assert observed["quality"] == 90


def test_online_fixed_byte_budget_blocks_actions_until_credit_is_available(monkeypatch):
    """Live camera routing keeps every prefix within the selected raw-JPEG byte budget."""
    monkeypatch.setattr(
        gradio_demo,
        "predict_current_maur_scores",
        lambda *_args, **_kwargs: {"edge": 0.0, "original": 0.8, "q90": 0.7},
    )
    monkeypatch.setattr(
        gradio_demo,
        "encode_jpeg",
        lambda _frame, quality: b"x" * (1000 if int(quality) == 95 else 500),
    )
    observed = []

    def fake_cloud_request(*_args, **_kwargs):
        observed.append("upload")
        return (
            {"status": "ok", "cloud_boxes": [], "cloud_latency_ms": 1.0, "roundtrip_ms": 2.0},
            500,
            {"roundtrip_ms": 2.0},
        )

    monkeypatch.setattr(gradio_demo, "cloud_request_timed", fake_cloud_request)
    common = dict(
        frame=np.zeros((16, 16, 3), dtype=np.uint8),
        frame_id=1,
        policy={
            "roi": {"confidence": 0.25, "max_area_ratio": 0.65, "imgsz": 640},
            "maur": {"duplicate_iou": 0.5, "nms_iou": 0.7},
        },
        policy_edge_boxes=[],
        display_edge_boxes=[],
        cloud_url="http://127.0.0.1:8765/infer",
        cloud_timeout=1.0,
        cloud_conf=0.25,
        cost_weight=0.0,
        class_ids=None,
        use_temporal_memory=False,
        frame_temporal_stable=False,
        temporal_tracks=[],
        runtime_args=argparse.Namespace(),
        online_budget_fraction=0.1,
    )

    blocked = gradio_demo.run_current_maur_action(**common, online_budget_credit_bytes=0.0)
    allowed = gradio_demo.run_current_maur_action(**common, online_budget_credit_bytes=400.0)

    assert blocked["selected_action"] == "edge"
    assert blocked["online_budget_credit_bytes"] == pytest.approx(100.0)
    assert allowed["selected_action"] == "q90"
    assert allowed["online_budget_credit_bytes"] == pytest.approx(0.0)
    assert observed == ["upload"]


def test_manifest_rejects_pickle_and_path_escape():
    """An editable UI path may reference only three hash-pinned NPZ seed files."""
    manifest = {
        "schema_version": 1,
        "policy_type": "cavs_current_maur",
        "policy_id": "cavs_current_maur_controlled_retrain_v2",
        "method": "CAVS-Current-MAUR",
        "profile": "current",
        "fusion": "MAUR",
        "action_ids": ["edge", "original", "q90", "current@640"],
        "category_count": 80,
        "edge_model": {
            "name": "yolo26n.pt",
            "sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        },
        "cloud_model_sha256": "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92",
        "architecture": {
            "class_embedding_dim": 8,
            "box_hidden_dim": 32,
            "action_hidden_dim": 32,
            "joint_hidden_dim": 64,
            "dropout": 0.05,
            "max_boxes": 64,
        },
        "seeds": [
            {"seed": 20260809, "file": "seed_20260809.npz", "sha256": "a" * 64},
            {"seed": 20260810, "file": "seed_20260810.npz", "sha256": "b" * 64},
            {"seed": 20260811, "file": "seed_20260811.npz", "sha256": "c" * 64},
        ],
    }

    validate_manifest(manifest)

    manifest["seeds"][0]["file"] = "model.pkl"
    with pytest.raises(ValueError, match="NPZ"):
        validate_manifest(manifest)

    manifest["seeds"][0]["file"] = "../seed_20260809.npz"
    with pytest.raises(ValueError, match="relative"):
        validate_manifest(manifest)

    manifest["seeds"][0]["file"] = "seed_20260809.npz"
    manifest["cloud_model_sha256"] = "not-a-checksum"
    with pytest.raises(ValueError, match="cloud model"):
        validate_manifest(manifest)


def test_seed_archive_rejects_unexpected_tensor_members(tmp_path):
    """A hash-matched NPZ still cannot expand arbitrary members in the UI process."""
    archive = tmp_path / "seed.npz"
    np.savez_compressed(archive, expected=np.zeros(1), unexpected=np.zeros(1))

    with pytest.raises(ValueError, match="members"):
        _validate_npz_archive(archive, {"expected"})


def test_default_current_maur_package_loads_three_seed_ensemble():
    """The shipped package is executable data, not a label over the old learner."""
    manifest_path = DEMO_DIR / "policies" / "current_maur_v2" / "manifest.json"

    policy = load_current_maur_policy(manifest_path)
    roi = current_roi_xyxy(
        640,
        480,
        [{"xyxy": [100.0, 80.0, 180.0, 200.0], "cls": 0, "conf": 0.9}],
    )
    scores = predict_current_maur_scores(
        policy,
        [{"xyxy": [100.0, 80.0, 180.0, 200.0], "cls": 0, "conf": 0.9}],
        width=640,
        height=480,
        roi_xyxy=roi,
    )

    assert policy["method"] == "CAVS-Current-MAUR"
    assert policy["edge_model"] == {
        "name": "yolo26n.pt",
        "sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    }
    assert policy["ensemble_seeds"] == [20260809, 20260810, 20260811]
    assert set(scores) == {"edge", "original", "q90", "current@640"}
    assert scores["edge"] == 0.0
    assert all(-1.0 < value < 1.0 for value in scores.values())


def test_current_action_sends_one_merged_roi_and_applies_maur(monkeypatch):
    """The online Current action is not the legacy one-box ROI path."""
    observed = {}

    monkeypatch.setattr(
        gradio_demo,
        "predict_current_maur_scores",
        lambda *_args, **_kwargs: {
            "edge": 0.0,
            "original": 0.01,
            "q90": 0.02,
            "current@640": 0.5,
        },
    )

    def fake_cloud_request(*args, **kwargs):
        observed["payload_type"] = args[5]
        observed["crop_xyxy"] = kwargs["crop_xyxy"]
        observed["cloud_imgsz"] = kwargs["cloud_imgsz"]
        return (
            {
                "status": "ok",
                "cloud_boxes": [
                    {"xyxy": [42.0, 42.0, 58.0, 58.0], "cls": 0, "conf": 0.95}
                ],
                "cloud_latency_ms": 10.0,
                "roundtrip_ms": 15.0,
                "request_metadata": {},
            },
            1234,
            {"roundtrip_ms": 15.0},
        )

    monkeypatch.setattr(gradio_demo, "cloud_request_timed", fake_cloud_request)
    edge = [{"xyxy": [40.0, 40.0, 60.0, 60.0], "cls": 0, "conf": 0.9}]
    result = gradio_demo.run_current_maur_action(
        np.zeros((100, 100, 3), dtype=np.uint8),
        1,
        {
            "roi": {"confidence": 0.25, "max_area_ratio": 0.65, "imgsz": 640},
            "maur": {"duplicate_iou": 0.5, "nms_iou": 0.7},
        },
        edge,
        edge,
        "http://127.0.0.1:8765/infer",
        1.0,
        0.25,
        0.02,
        None,
        use_temporal_memory=False,
        frame_temporal_stable=False,
        temporal_tracks=[],
        runtime_args=argparse.Namespace(),
    )

    assert result["selected_action"] == "current@640"
    assert result["stat_updates"]["collab_roi_ok"] == 1
    assert result["display_boxes"] == [
        {"xyxy": [42.0, 42.0, 58.0, 58.0], "cls": 0, "conf": 0.95}
    ]
    assert observed == {"payload_type": "roi", "crop_xyxy": [34, 34, 66, 66], "cloud_imgsz": 640}


def test_q90_is_unavailable_when_it_is_not_smaller_than_original(monkeypatch):
    """Live action eligibility matches the frozen training record contract."""
    monkeypatch.setattr(
        gradio_demo,
        "predict_current_maur_scores",
        lambda *_args, **_kwargs: {"edge": 0.0, "original": 0.1, "q90": 1.0},
    )
    monkeypatch.setattr(
        gradio_demo,
        "encode_jpeg",
        lambda _frame, quality: b"x" * (100 if int(quality) == 95 else 120),
    )
    observed = {}

    def fake_cloud_request(*args, **_kwargs):
        observed["quality"] = args[3]
        return (
            {"status": "ok", "cloud_boxes": [], "cloud_latency_ms": 1.0, "roundtrip_ms": 2.0},
            100,
            {"roundtrip_ms": 2.0},
        )

    monkeypatch.setattr(gradio_demo, "cloud_request_timed", fake_cloud_request)
    result = gradio_demo.run_current_maur_action(
        np.zeros((16, 16, 3), dtype=np.uint8),
        1,
        {
            "roi": {"confidence": 0.25, "max_area_ratio": 0.65, "imgsz": 640},
            "maur": {"duplicate_iou": 0.5, "nms_iou": 0.7},
        },
        [],
        [],
        "http://127.0.0.1:8765/infer",
        1.0,
        0.25,
        0.0,
        None,
        use_temporal_memory=False,
        frame_temporal_stable=False,
        temporal_tracks=[],
        runtime_args=argparse.Namespace(),
    )

    assert result["selected_action"] == "original"
    assert "q90" not in result["action_scores"]
    assert observed["quality"] == 95


def test_current_manifest_detection_does_not_depend_on_filename(tmp_path, monkeypatch):
    """Renaming a valid deployment manifest must not silently invoke the legacy learner."""
    manifest = tmp_path / "renamed-policy.json"
    manifest.write_text('{"policy_type": "cavs_current_maur"}', encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        gradio_demo,
        "load_current_maur_policy",
        lambda path: observed.append(("current", Path(path).name)) or {"kind": "current_maur"},
    )
    monkeypatch.setattr(
        gradio_demo,
        "load_policy",
        lambda path: observed.append(("legacy", Path(path).name)) or {"kind": "legacy"},
    )

    policy = gradio_demo.get_policy(str(manifest))

    assert policy["kind"] == "current_maur"
    assert observed == [("current", "renamed-policy.json")]
