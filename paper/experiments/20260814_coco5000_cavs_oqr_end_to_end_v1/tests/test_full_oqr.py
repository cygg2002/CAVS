from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from full_oqr import (  # noqa: E402
    build_action_payload,
    candidate_oqr,
    load_latest_jsonl,
    resolve_oqr_output,
    selected_request_keys,
)


def test_candidate_oqr_records_exact_roi_bytes(monkeypatch):
    monkeypatch.setattr("full_oqr.current_roi_xyxy", lambda *_args, **_kwargs: [1, 1, 5, 5])
    monkeypatch.setattr(
        "full_oqr.predict_current_maur_scores",
        lambda *_args, **_kwargs: {"edge": 0.0, "original": 0.3, "q90": 0.2, "current@640": 0.4},
    )
    monkeypatch.setattr(
        "full_oqr.encode_jpeg",
        lambda image, quality: b"x" * ({95: 100, 90: 60}[int(quality)] if image.shape[:2] == (8, 8) else 20),
    )

    result = candidate_oqr(
        np.zeros((8, 8, 3), dtype=np.uint8),
        image_id=7,
        policy={"roi": {"confidence": 0.25, "max_area_ratio": 0.65}},
        edge_boxes=[],
    )

    assert result["roi_xyxy"] == [1, 1, 5, 5]
    assert result["payload_bytes"] == {"edge": 0, "original": 100, "q90": 60, "current@640": 20}
    assert set(result["scores"]) == {"edge", "original", "q90", "current@640"}


def test_roi_payload_crops_image_and_preserves_geometry(monkeypatch):
    observed = {}

    def fake_encode(image, quality):
        observed["shape"] = image.shape
        observed["quality"] = quality
        return b"jpeg"

    monkeypatch.setattr("full_oqr.encode_jpeg", fake_encode)
    payload, payload_bytes, _ = build_action_payload(
        np.zeros((10, 12, 3), dtype=np.uint8),
        image_id=9,
        action="current@640",
        edge_boxes=[],
        roi_xyxy=[2, 3, 8, 9],
        score=0.4,
        cloud_imgsz=640,
    )

    assert observed == {"shape": (6, 6, 3), "quality": 90}
    assert payload_bytes == 4
    assert payload["metadata"]["payload_type"] == "roi"
    assert payload["metadata"]["crop_xyxy"] == [2, 3, 8, 9]


def test_maur_is_applied_only_to_successful_roi(monkeypatch):
    edge = [{"xyxy": [0, 0, 2, 2], "conf": 0.5, "cls": 0}]
    cloud = [{"xyxy": [1, 1, 3, 3], "conf": 0.8, "cls": 0}]
    expected = [{"xyxy": [1, 1, 3, 3], "conf": 0.9, "cls": 0}]
    monkeypatch.setattr("full_oqr.maur_fuse_boxes", lambda *_args, **_kwargs: expected)

    fused, status, _ = resolve_oqr_output(
        "current@640",
        {"status": "ok", "cloud_boxes": cloud},
        edge,
        [0, 0, 4, 4],
        {"maur": {"duplicate_iou": 0.5, "nms_iou": 0.7}},
    )
    fallback, fallback_status, _ = resolve_oqr_output(
        "current@640",
        {"status": "error"},
        edge,
        [0, 0, 4, 4],
        {"maur": {"duplicate_iou": 0.5, "nms_iou": 0.7}},
    )

    assert fused == expected
    assert status == "ok"
    assert fallback == edge
    assert fallback_status == "fallback"


def test_selected_request_keys_deduplicates_budget_plans():
    plans = {
        "10": {1: "q90", 2: "edge"},
        "25": {1: "q90", 2: "original"},
        "40": {1: "original", 2: "original"},
    }

    assert selected_request_keys(plans) == {(1, "q90"), (1, "original"), (2, "original")}


def test_latest_jsonl_record_wins_and_partial_tail_is_ignored(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"image_id": 1, "status": "error"})
        + "\n"
        + json.dumps({"image_id": 1, "status": "ok"})
        + "\n{",
        encoding="utf-8",
    )

    rows = load_latest_jsonl(path, lambda row: int(row["image_id"]))

    assert rows == {1: {"image_id": 1, "status": "ok"}}


def test_instrumented_server_maps_roi_boxes_back_to_full_image():
    path = (
        ROOT.parents[0]
        / "20260814_mac_remote_end_to_end_system_benchmark_v1/scripts/instrumented_cloud_server.py"
    )
    spec = importlib.util.spec_from_file_location("instrumented_cloud_server_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    mapped = module.map_boxes_to_frame(
        [{"xyxy": [1.0, 2.0, 3.0, 4.0], "conf": 0.8, "cls": 0}],
        {"payload_type": "roi", "crop_xyxy": [10, 20, 30, 40]},
    )

    assert mapped[0]["xyxy"] == [11.0, 22.0, 13.0, 24.0]
