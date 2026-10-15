from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import (  # noqa: E402
    action_plan_for_budget,
    coco_predictions,
    deterministic_sample,
    percentile,
    request_body,
    resolve_output_boxes,
    summarize_values,
)


def test_deterministic_sample_is_sorted_unique_and_repeatable(tmp_path):
    paths = []
    for image_id in range(1, 21):
        path = tmp_path / f"{image_id:012d}.jpg"
        path.touch()
        paths.append(path)

    first = deterministic_sample(paths, size=7, seed=123)
    second = deterministic_sample(reversed(paths), size=7, seed=123)

    assert first == second
    assert len(first) == len(set(first)) == 7
    assert first == sorted(first)


def test_request_body_reports_raw_payload_and_actual_json_bytes():
    payload = {
        "metadata": {"frame_id": 1, "payload_bytes": 3},
        "image_b64": "YWJj",
    }

    body = request_body(payload)

    assert body == json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(body) > 3


def test_percentiles_and_summary_are_deterministic():
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])
    assert summary == {
        "count": 4,
        "mean": pytest.approx(2.5),
        "median": pytest.approx(2.5),
        "p95": pytest.approx(3.85),
        "p99": pytest.approx(3.97),
    }


def test_oq_plan_never_selects_roi_and_respects_capacity():
    candidates = [
        {
            "frame_id": 1,
            "scores": {"edge": 0.0, "original": 0.8, "q90": 0.7, "current@640": 9.0},
            "payload_bytes": {"edge": 0, "original": 100, "q90": 50, "current@640": 1},
        },
        {
            "frame_id": 2,
            "scores": {"edge": 0.0, "original": 0.4, "q90": 0.5},
            "payload_bytes": {"edge": 0, "original": 100, "q90": 50},
        },
    ]

    plan = action_plan_for_budget(candidates, budget=0.5)

    assert set(plan["actions"].values()) <= {"edge", "original", "q90"}
    assert plan["request_bytes"] <= plan["capacity_bytes"]


def test_coco_prediction_conversion_uses_manifest_category_ids():
    rows = coco_predictions(
        7,
        [{"xyxy": [10.0, 20.0, 30.0, 50.0], "conf": 0.75, "cls": 1}],
        category_ids=[1, 3, 4],
    )

    assert rows == [
        {
            "image_id": 7,
            "category_id": 3,
            "bbox": [10.0, 20.0, 20.0, 30.0],
            "score": 0.75,
        }
    ]


def test_cloud_failure_preserves_edge_output():
    edge = [{"xyxy": [1, 2, 3, 4], "conf": 0.5, "cls": 0}]

    boxes, status = resolve_output_boxes("q90", {"status": "error"}, edge)

    assert boxes == edge
    assert boxes is not edge
    assert status == "fallback"
