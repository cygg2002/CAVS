from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("deployment_gap_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_normalize_plan_fills_unselected_images_with_edge() -> None:
    assert MODULE.normalize_plan({2: "q90"}, [1, 2, 3]) == {
        1: "edge",
        2: "q90",
        3: "edge",
    }


def test_hybrid_predictions_replace_only_requested_action_groups() -> None:
    plan = {1: "edge", 2: "q90", 3: "current@640"}
    real = {1: [{"tag": "real-edge"}], 2: [{"tag": "real-q"}], 3: [{"tag": "real-r"}]}
    cached = {
        "edge": {1: [{"tag": "cache-edge"}]},
        "q90": {2: [{"tag": "cache-q"}]},
        "current@640": {3: [{"tag": "cache-r"}]},
        "original": {},
    }

    rows = MODULE.hybrid_predictions(plan, real, cached, frozenset({"q90"}))

    assert [row["tag"] for row in rows] == ["real-edge", "cache-q", "real-r"]


def test_shapley_values_reconcile_the_full_difference() -> None:
    players = ("edge", "original", "q90", "current@640")
    effects = {"edge": 1.0, "original": -2.0, "q90": 3.5, "current@640": 0.25}
    values = {}
    for subset in MODULE.powerset(players):
        values[subset] = 10.0 + sum(effects[player] for player in subset)

    result = MODULE.shapley_values(values, players)

    assert result == pytest.approx(effects)
    assert sum(result.values()) == pytest.approx(values[frozenset(players)] - values[frozenset()])


def test_apply_paper_costs_preserves_scores_and_replaces_available_payloads() -> None:
    candidate = {
        "frame_id": 7,
        "scores": {"edge": 0.0, "original": 0.4, "q90": 0.5, "current@640": 0.6},
        "payload_bytes": {"edge": 0, "original": 80, "q90": 50, "current@640": 20},
    }
    paper = {"payload_bytes": {"original": 100, "q90": 60, "current@640": 25}}

    result = MODULE.apply_paper_costs(candidate, paper)

    assert result["scores"] == candidate["scores"]
    assert result["payload_bytes"] == {
        "edge": 0,
        "original": 100,
        "q90": 60,
        "current@640": 25,
    }


def test_apply_paper_costs_drops_action_missing_from_paper_cache() -> None:
    candidate = {
        "frame_id": 7,
        "scores": {"edge": 0.0, "original": 0.4, "q90": 0.5, "current@640": 0.6},
        "payload_bytes": {"edge": 0, "original": 80, "q90": 50, "current@640": 20},
    }
    paper = {"payload_bytes": {"original": 100, "q90": 60}}

    result = MODULE.apply_paper_costs(candidate, paper)

    assert "current@640" not in result["scores"]
    assert "current@640" not in result["payload_bytes"]


def test_resume_report_reuses_only_matching_byte_anchors() -> None:
    report = {
        "paper_original_total_bytes": 100,
        "real_q95_original_total_bytes": 80,
        "budgets": {"10": {"complete": True}},
    }

    assert MODULE.validate_resume_report(
        report, paper_total_bytes=100, real_total_bytes=80
    ) == {"10"}
    with pytest.raises(RuntimeError, match="paper-byte anchor changed"):
        MODULE.validate_resume_report(report, paper_total_bytes=101, real_total_bytes=80)


def test_q90_path_decomposition_reconciles_total_effect() -> None:
    result = MODULE.q90_path_decomposition(
        cached_paper_codec=0.55,
        opencv_codec_paper_inference=0.54,
        opencv_codec_real_inference=0.52,
    )

    assert result == pytest.approx(
        {
            "total_q90_real_minus_cached": -0.03,
            "codec_opencv_minus_pillow_under_paper_inference": -0.01,
            "realtime_inference_minus_paper_inference_under_opencv": -0.02,
        }
    )
