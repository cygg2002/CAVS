# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the YOLO26 edge-cloud demo helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO_DIR))

from fusion import fuse_detections  # noqa: E402
from edge_client import (  # noqa: E402
    detection_upload_score,
    edge_boxes_for_fusion,
    make_payload,
    normalize_capture_source,
    select_cloud_imgsz,
    temporal_skip_roi_upload,
)
from evaluate_edge_cloud import (  # noqa: E402
    apply_tile_cloud,
    fixed_threshold_counts,
    fixed_threshold_metrics,
    full_frame_fusion_boxes,
    make_evaluation_payload,
    oracle_full_frame_fusion_boxes,
    prepare_cloud_only_input,
    resolve_end2end_modes,
    uploaded_image_for_cloud,
)
from learned_uploader import (  # noqa: E402
    decision_score,
    fit_logistic,
    fit_value_regressor,
    image_context,
    policy_decision,
    predict as learned_predict,
)
from protocol import b64_to_bytes, crop_image, decode_jpeg, encode_jpeg, expand_xyxy, shift_boxes  # noqa: E402
from upload_strategies import adaptive_roi_gain, low_conf_proposals, parse_grid, stability_probe_score, tile_candidates  # noqa: E402


def test_roi_expand_encode_decode_and_shift():
    """ROI payload helpers should preserve dimensions and map boxes back to full-frame coordinates."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    crop_xyxy = expand_xyxy([10, 10, 30, 30], image.shape[:2], gain=1.5)
    assert crop_xyxy == [5, 5, 35, 35]

    encoded = encode_jpeg(crop_image(image, crop_xyxy), quality=80)
    decoded = decode_jpeg(encoded)
    assert decoded.shape[:2] == (30, 30)

    shifted = shift_boxes([{"xyxy": [1, 2, 3, 4], "cls": 0, "conf": 0.9}], (5, 7))
    assert shifted[0]["xyxy"] == [6.0, 9.0, 8.0, 11.0]


def test_fixed_threshold_metrics_count_duplicate_predictions_as_false_positives():
    """Deployment metrics should expose duplicate boxes instead of hiding them in AP aggregation."""
    labels = [{"xyxy": [0, 0, 10, 10], "cls": 0}]
    predictions = [
        {"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9},
        {"xyxy": [1, 1, 9, 9], "cls": 0, "conf": 0.8},
    ]
    counts = fixed_threshold_counts(predictions, labels, iou_threshold=0.5)
    assert counts == {"tp": 1, "fp": 1, "fn": 0}
    assert fixed_threshold_metrics(counts) == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2.0 / 3.0,
    }


def test_cloud_only_codec_simulation_returns_decoded_pixels_and_payload_size():
    """Cloud-only fairness mode should evaluate the transmitted JPEG rather than the source pixels."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    raw_args = argparse.Namespace(simulate_cloud_only_codec=False, jpeg_quality=40)
    raw_image, raw_bytes, raw_encode_ms, raw_decode_ms = prepare_cloud_only_input(image, raw_args)
    assert raw_image is image
    assert raw_bytes == 0
    assert raw_encode_ms == 0.0
    assert raw_decode_ms == 0.0

    codec_args = argparse.Namespace(simulate_cloud_only_codec=True, jpeg_quality=40)
    decoded, payload_bytes, encode_ms, decode_ms = prepare_cloud_only_input(image, codec_args)
    assert decoded.shape == image.shape
    assert decoded is not image
    assert payload_bytes > 0
    assert encode_ms >= 0.0
    assert decode_ms >= 0.0


def test_evaluator_resolves_edge_and_cloud_heads_independently():
    """Mixed M2E evaluation must override the shared legacy head setting."""
    mixed = argparse.Namespace(end2end=False, edge_end2end=False, cloud_end2end=True)
    assert resolve_end2end_modes(mixed) == (False, True)

    shared = argparse.Namespace(end2end=False, edge_end2end=None, cloud_end2end=None)
    assert resolve_end2end_modes(shared) == (False, False)


def test_upload_codec_decodes_the_serialized_payload_without_reencoding_source():
    """Cloud simulation should consume the payload bytes produced by the edge serializer."""
    source = np.zeros((24, 24, 3), dtype=np.uint8)
    source[:, 12:] = 255
    unrelated = np.full_like(source, 127)
    args = argparse.Namespace(
        simulate_upload_codec=True,
        jpeg_quality=50,
        _upload_encode_latencies=[],
        _upload_decode_latencies=[],
    )
    payload, payload_bytes = make_evaluation_payload(
        args,
        source,
        0,
        "keyframe",
        ["test"],
        [],
        0.5,
        args.jpeg_quality,
    )
    decoded = uploaded_image_for_cloud(unrelated, args, payload)
    expected = decode_jpeg(b64_to_bytes(payload["image_b64"]))
    assert np.array_equal(decoded, expected)
    assert not np.array_equal(decoded, unrelated)
    assert payload_bytes > 0
    assert len(args._upload_encode_latencies) == 1
    assert len(args._upload_decode_latencies) == 1


def test_fusion_marks_class_corrections_and_misses():
    """Fusion should separate class corrections from missed cloud detections."""
    edge_boxes = [{"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.7}]
    cloud_boxes = [
        {"xyxy": [0, 0, 10, 10], "cls": 1, "conf": 0.9},
        {"xyxy": [20, 20, 30, 30], "cls": 2, "conf": 0.8},
    ]
    fusion = fuse_detections(edge_boxes, cloud_boxes, payload_type="roi", iou_threshold=0.5)
    assert fusion["counts"]["class_corrected"] == 1
    assert fusion["counts"]["missed_by_edge"] == 1


def test_roi_fusion_uses_only_target_edge_box():
    """ROI fusion should not treat unrelated full-frame edge boxes as ROI false positives."""
    metadata = {
        "payload_type": "roi",
        "edge_boxes": [
            {"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.7},
            {"xyxy": [50, 50, 60, 60], "cls": 1, "conf": 0.8},
        ],
        "target_edge_box": {"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.7},
    }
    assert edge_boxes_for_fusion(metadata) == [metadata["target_edge_box"]]


def test_request_cloud_imgsz_metadata_for_online_client():
    """Online uploads should be able to request static or learned-adaptive cloud compute."""
    args = argparse.Namespace(
        cloud_imgsz=512,
        enable_learned_adaptive_cloud_imgsz=False,
        learned_cloud_imgsz_switch=0.75,
        learned_cloud_imgsz_low=416,
        learned_cloud_imgsz_high=512,
    )
    assert select_cloud_imgsz(0.2, args) == 512

    args.enable_learned_adaptive_cloud_imgsz = True
    assert select_cloud_imgsz(0.70, args, adaptive=True) == 416
    assert select_cloud_imgsz(0.80, args, adaptive=True) == 512

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    payload, payload_bytes = make_payload(image, 0, "keyframe", ["test"], [], 0.8, 80, cloud_imgsz=512)
    assert payload_bytes > 0
    assert payload["metadata"]["cloud_imgsz"] == 512


def test_detection_score_can_disable_temporal_signal_for_image_sets():
    """Independent image datasets should not be penalized as temporally unstable video."""
    box = {"xyxy": [20, 20, 80, 80], "cls": 0, "conf": 0.9}
    score_with_temporal, reasons_with_temporal = detection_upload_score(box, (100, 100), [], use_temporal=True)
    score_without_temporal, reasons_without_temporal = detection_upload_score(box, (100, 100), [], use_temporal=False)
    assert score_with_temporal > score_without_temporal
    assert "temporal_instability" in reasons_with_temporal
    assert "temporal_instability" not in reasons_without_temporal


def test_tile_candidate_and_cloud_acceptance_are_budgeted():
    """Tile probes should be bounded and reject weak or crop-border cloud boxes."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = 255
    assert parse_grid("2x2") == (2, 2)
    candidates = tile_candidates(image, [], grid=(2, 2), max_tiles=1, score_threshold=0.01)
    assert len(candidates) <= 1

    cloud_boxes = [
        {"xyxy": [20, 20, 40, 40], "cls": 0, "conf": 0.44},
        {"xyxy": [20, 20, 40, 40], "cls": 0, "conf": 0.90},
        {"xyxy": [0, 20, 8, 40], "cls": 0, "conf": 0.95},
    ]
    final_boxes, accepted = apply_tile_cloud([], cloud_boxes, [0, 0, 50, 50], image.shape[:2], 0.45, 0.10)
    assert accepted == 1
    assert final_boxes == [cloud_boxes[1]]


def test_adaptive_roi_and_low_conf_proposals():
    """Adaptive ROI and suppressed proposal selection should be deterministic."""
    shape = (100, 100)
    small_edge = {"xyxy": [0, 10, 5, 15], "cls": 0, "conf": 0.2}
    assert adaptive_roi_gain(small_edge, shape, base_gain=1.35, small_gain=1.6, edge_gain=1.45) == 1.6

    deployed = [{"xyxy": [60, 60, 90, 90], "cls": 0, "conf": 0.8}]
    proposals = low_conf_proposals([small_edge, deployed[0]], deployed, shape, min_conf=0.1, max_conf=0.25)
    assert len(proposals) == 1
    assert proposals[0]["proposal_reason"][0] == "suppressed_low_conf"


def test_stability_probe_scores_unmatched_boxes():
    """Scale disagreement should produce an instability score and reason."""
    primary = [{"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.8}]
    secondary = [{"xyxy": [50, 50, 60, 60], "cls": 0, "conf": 0.8}]
    score, reasons = stability_probe_score(primary, secondary, iou_threshold=0.5)
    assert score > 0
    assert "scale_unmatched_boxes" in reasons


def test_learned_policy_helpers_are_lightweight():
    """The learned policy should fit and score without external ML dependencies."""
    model = fit_logistic([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1], epochs=50, lr=0.1)
    assert learned_predict(model, [3.0]) > learned_predict(model, [0.0])
    assert decision_score(0.8, estimated_kb=100.0, cost_weight=0.1) == 0.7000000000000001

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    context = image_context(image, [{"xyxy": [5, 5, 20, 20], "cls": 0, "conf": 0.8}])
    assert context["det_count"] > 0


def test_dual_head_policy_uses_expected_gain_and_cost():
    """The upgraded learner should combine probability, value, and upload cost."""
    classifier = fit_logistic([[0.0], [0.5], [1.0], [1.5]], [0, 0, 1, 1], epochs=80, lr=0.1)
    value_model = fit_value_regressor([[0.0], [0.5], [1.0], [1.5]], [0.0, 0.1, 0.5, 0.8], epochs=120, lr=0.08)
    policy = {
        "frame_model": classifier,
        "frame_value_model": value_model,
        "roi_model": classifier,
        "roi_value_model": value_model,
    }
    high = policy_decision(policy, "frame", [1.5], estimated_kb=10.0, cost_weight=0.02)
    low = policy_decision(policy, "frame", [0.0], estimated_kb=10.0, cost_weight=0.02)
    assert high["probability"] > low["probability"]
    assert high["expected_gain"] > low["expected_gain"]
    assert high["decision"] > low["decision"]


def test_full_frame_fusion_upper_bound_keeps_unmatched_edge_and_cloud_boxes():
    """Upper-bound fusion should keep unmatched detections from both models after matched replacement."""
    edge_boxes = [
        {"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.6},
        {"xyxy": [40, 40, 50, 50], "cls": 1, "conf": 0.7},
    ]
    cloud_boxes = [
        {"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9},
        {"xyxy": [70, 70, 80, 80], "cls": 2, "conf": 0.8},
    ]
    fused = full_frame_fusion_boxes(edge_boxes, cloud_boxes, 0.5, 0.7, 0.25, 0.25)
    classes = sorted(int(box["cls"]) for box in fused)
    assert classes == [0, 1, 2]


def test_oracle_full_frame_fusion_upper_bound_keeps_only_correct_union_signal():
    """The oracle upper bound should keep only detections that can explain a ground-truth label."""
    labels = [{"xyxy": [0, 0, 10, 10], "cls": 0}, {"xyxy": [50, 50, 60, 60], "cls": 1}]
    edge_boxes = [
        {"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.6},
        {"xyxy": [80, 80, 90, 90], "cls": 3, "conf": 0.9},
    ]
    cloud_boxes = [{"xyxy": [50, 50, 60, 60], "cls": 1, "conf": 0.8}]
    oracle = oracle_full_frame_fusion_boxes(edge_boxes, cloud_boxes, labels, 0.7)
    classes = sorted(int(box["cls"]) for box in oracle)
    assert classes == [0, 1]


def test_temporal_memory_suppresses_repeat_upload_until_confidence_drops():
    """Recently cloud-verified stable tracks should suppress redundant ROI uploads, but not after a confidence drop."""
    track = {
        "cloud_verified": True,
        "stable_hits": 4,
        "last_cloud_frame": 10,
        "verified_cls": 0,
        "verified_xyxy": [10, 10, 30, 30],
        "last_cloud_conf": 0.9,
        "last_conf": 0.9,
    }
    box = {"xyxy": [11, 11, 31, 31], "cls": 0, "conf": 0.84}
    assert temporal_skip_roi_upload(box, track, 18, 0.5, 3, 15, 0.12)

    dropped = dict(box)
    dropped["conf"] = 0.6
    assert not temporal_skip_roi_upload(dropped, track, 18, 0.5, 3, 15, 0.12)


def test_normalize_capture_source_supports_webcam_indices():
    """The webcam demo should treat bare integers as camera ids."""
    assert normalize_capture_source("0") == 0
