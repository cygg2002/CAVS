"""Pure contracts for AFPD same-run attribution bootstrap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_COMPARATORS = ("Current-OQ", "Current-OQR")
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
ALLOWED_ACTIONS = {"edge", "original", "q90", "roi"}


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "posthoc_same_run_attribution_bootstrap":
        raise ValueError("Post-hoc status changed")
    for key in (
        "confirmatory",
        "final_is_hidden",
        "final_used_for_selection",
        "training_allowed",
        "tuning_allowed",
        "detector_inference_allowed",
    ):
        if protocol.get(key) is not False:
            raise ValueError(f"Forbidden protocol flag: {key}")
    if protocol.get("candidate") != "AFPD-OQR":
        raise ValueError("Candidate changed")
    if tuple(protocol.get("comparators", ())) != EXPECTED_COMPARATORS:
        raise ValueError("Same-run comparators changed")
    if protocol.get("mode") != "fixed_byte":
        raise ValueError("Only Fixed-byte is in scope")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("Budgets changed")
    if int(protocol.get("bootstrap_replicates", -1)) != 1000:
        raise ValueError("Bootstrap replicate count changed")
    if int(protocol.get("expected_comparisons", -1)) != 6:
        raise ValueError("Expected comparison count changed")
    if int(protocol.get("bootstrap_workers", 0)) < 1:
        raise ValueError("At least one bootstrap worker is required")


def comparison_specs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    rows = []
    for comparator in protocol["comparators"]:
        for budget in map(float, protocol["budgets"]):
            rows.append(
                {
                    "key": f"AFPD-OQR_vs_{comparator}|fixed_byte|{budget:g}",
                    "candidate": "AFPD-OQR",
                    "candidate_key": f"AFPD-OQR|fixed_byte|{budget:g}",
                    "comparator": comparator,
                    "comparator_key": f"{comparator}|fixed_byte|{budget:g}",
                    "mode": "fixed_byte",
                    "budget": budget,
                }
            )
    return rows


def parent_ready(marker: Path) -> bool:
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("complete") is True
    except (json.JSONDecodeError, OSError):
        return False


def validate_selection(
    selection: Mapping[int, str], expected_image_ids: Sequence[int]
) -> None:
    expected = set(map(int, expected_image_ids))
    selected = set(map(int, selection))
    unknown = selected - expected
    if unknown:
        raise ValueError(f"Selection contains unknown image IDs: {sorted(unknown)[:5]}")
    invalid_actions = {str(action) for action in selection.values()} - ALLOWED_ACTIONS
    if invalid_actions:
        raise ValueError(f"Selection contains invalid actions: {sorted(invalid_actions)}")
