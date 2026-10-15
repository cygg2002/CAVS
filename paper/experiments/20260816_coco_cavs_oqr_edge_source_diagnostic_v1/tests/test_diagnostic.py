"""Regression tests for the cached edge-source diagnosis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("edge_source_diagnostic", ROOT / "scripts/run_diagnostic.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_normalize_plan_defaults_missing_images_to_edge() -> None:
    plan = MODULE.normalize_plan({"2": "q90", "3": "current@640"}, [1, 2, 3])

    assert plan == {1: "edge", 2: "q90", 3: "current@640"}


def test_repository_root_is_the_current_ultralytics_workspace() -> None:
    assert MODULE.REPO.name == "ultralytics-main"


def test_plan_comparison_keeps_membership_and_action_agreement_separate() -> None:
    reference = {1: "edge", 2: "q90", 3: "current@640", 4: "q90"}
    candidate = {1: "edge", 2: "original", 3: "current@640", 4: "edge"}

    result = MODULE.compare_plans(reference, candidate)

    assert result["exact_action_agreement"] == 0.5
    assert result["upload_membership_jaccard"] == 2 / 3
    assert result["shared_upload_same_action_rate"] == 0.5
    assert result["action_transition_counts"]["q90->original"] == 1
    assert result["action_transition_counts"]["q90->edge"] == 1
