# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Contracts for the paper-faithful Current+MAUR fixed-byte scheduler."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO_DIR))

from paper_budget_scheduler import schedule_fixed_byte, summarize_fixed_byte_plan


def test_fixed_byte_scheduler_never_exceeds_capacity_and_selects_one_action_per_frame():
    """The deployment scheduler preserves the paper multiple-choice capacity contract."""
    candidates = [
        {
            "frame_id": 1,
            "scores": {"edge": 0.0, "original": 0.9, "q90": 0.7, "current@640": 0.6},
            "payload_bytes": {"edge": 0, "original": 1000, "q90": 500, "current@640": 200},
        },
        {
            "frame_id": 2,
            "scores": {"edge": 0.0, "original": 0.8, "q90": 0.65, "current@640": 0.1},
            "payload_bytes": {"edge": 0, "original": 1000, "q90": 500, "current@640": 250},
        },
        {
            "frame_id": 3,
            "scores": {"edge": 0.0, "original": 0.2, "q90": 0.15},
            "payload_bytes": {"edge": 0, "original": 1000, "q90": 500},
        },
    ]

    plan = schedule_fixed_byte(candidates, budget=0.25)

    assert plan["capacity_bytes"] == pytest.approx(750.0)
    assert plan["request_bytes"] <= plan["capacity_bytes"]
    assert set(plan["actions"]) == {1, 2, 3}
    assert set(plan["actions"].values()) <= {"edge", "original", "q90", "current@640"}


def test_fixed_byte_scheduler_is_deterministic_for_equal_value_actions():
    """Paper tie-breaking prefers lower cost before the stable action identifier."""
    candidates = [
        {
            "frame_id": 7,
            "scores": {"edge": 0.0, "original": 0.5, "q90": 0.5, "current@640": 0.5},
            "payload_bytes": {"edge": 0, "original": 1000, "q90": 500, "current@640": 200},
        }
    ]

    plan = schedule_fixed_byte(candidates, budget=1.0)

    assert plan["actions"] == {7: "current@640"}
    assert plan["request_bytes"] == 200


def test_fixed_byte_summary_reports_byte_and_frame_ratios_separately():
    """A 10% byte budget may legitimately upload more than 10% of the frames."""
    summary = summarize_fixed_byte_plan(
        actions={1: "q90", 2: "current@640", 3: "edge", 4: "edge"},
        request_bytes=100,
        original_total_bytes=1000,
        capacity_bytes=100,
    )

    assert summary["byte_ratio"] == pytest.approx(0.1)
    assert summary["frame_upload_ratio"] == pytest.approx(0.5)
    assert summary["capacity_utilization"] == pytest.approx(1.0)
    assert summary["action_counts"] == {"edge": 2, "original": 0, "q90": 1, "current@640": 1}


@pytest.mark.parametrize("budget", [-0.1, 0.0, 1.1])
def test_fixed_byte_scheduler_rejects_non_paper_budget_values(budget):
    """The UI paper mode accepts only a positive fraction no larger than the full-upload baseline."""
    candidates = [
        {
            "frame_id": 1,
            "scores": {"edge": 0.0, "original": 0.2},
            "payload_bytes": {"edge": 0, "original": 100},
        }
    ]

    with pytest.raises(ValueError, match="budget"):
        schedule_fixed_byte(candidates, budget=budget)
