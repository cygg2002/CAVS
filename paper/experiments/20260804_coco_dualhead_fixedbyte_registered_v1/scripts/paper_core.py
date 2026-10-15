"""Pure contracts for the registered DualHead fixed-byte experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[4]
SOURCE_CORE_PATH = (
    REPO
    / "paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/paper_core.py"
)
_SPEC = importlib.util.spec_from_file_location("budgetset_registered_core", SOURCE_CORE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(SOURCE_CORE_PATH)
_SOURCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SOURCE)

batches = _SOURCE.batches
bootstrap_remap = _SOURCE.bootstrap_remap
canonical_hash = _SOURCE.canonical_hash
category_index_to_id = _SOURCE.category_index_to_id
chunk_signature = _SOURCE.chunk_signature
original_e2e_ms = _SOURCE.original_e2e_ms
resolve_result_image_id = _SOURCE.resolve_result_image_id
select_fixed_call = _SOURCE.select_fixed_call
summarize_bootstrap = _SOURCE.summarize_bootstrap
validate_prediction_population = _SOURCE.validate_prediction_population

EXTERNAL_METHODS = {
    "ORIC-AP50",
    "MORIC-AP50",
    "ORIC-AP5095",
    "MORIC-AP5095",
}


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco_train2017_fresh_registered_evaluation":
        raise ValueError("Unexpected dataset contract")
    if protocol.get("roles") != {"audit": 1500, "registered_final": 3000}:
        raise ValueError("Registered evaluation role counts changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("Registered budgets changed")
    if tuple(protocol.get("modes", ())) != ("fixed_call", "fixed_byte"):
        raise ValueError("Registered modes changed")
    if protocol.get("primary_mode") != "fixed_byte":
        raise ValueError("Fixed-byte must remain the primary mode")
    if protocol.get("primary_candidate") != "DualHead":
        raise ValueError("DualHead must remain the primary candidate")
    if protocol.get("development_source_frozen") is not True:
        raise ValueError("Development source must remain frozen")
    if protocol.get("audit_used_for_training") is not False:
        raise ValueError("Audit cannot train models")
    if protocol.get("registered_final_used_for_selection") is not False:
        raise ValueError("Final cannot select models or comparators")
    if protocol.get("registered_final_access_once") is not True:
        raise ValueError("Final must be one-time")
    if protocol.get("registered_final_inference_after_audit_only") is not True:
        raise ValueError("Final inference must remain audit-gated")


def deterministic_evaluation_roles(
    population_ids: Iterable[int],
    excluded_ids: Iterable[int],
    role_counts: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, list[int]]:
    if dict(role_counts) != {
        "audit": int(role_counts.get("audit", -1)),
        "registered_final": int(role_counts.get("registered_final", -1)),
    }:
        raise ValueError("Only audit and registered-final roles are allowed")
    return _SOURCE.deterministic_roles(
        population_ids, excluded_ids, role_counts, seed=int(seed)
    )


def freeze_primary_comparators(
    rows: Sequence[Mapping[str, Any]], *, mode: str
) -> dict[str, str]:
    if str(mode) != "fixed_byte":
        raise ValueError("Primary comparators must be frozen on fixed-byte")
    output = {}
    for budget in (0.1, 0.25, 0.4):
        candidates = [
            row
            for row in rows
            if str(row.get("method")) in EXTERNAL_METHODS
            and str(row.get("mode")) == mode
            and abs(float(row.get("budget", -1.0)) - budget) < 1e-12
        ]
        if not candidates:
            raise ValueError(f"Missing external fixed-byte rows at {budget:g}")
        winner = max(
            candidates,
            key=lambda row: (
                float(row["map50_95"]),
                -float(row.get("upload_kb_per_image", 0.0)),
                str(row["method"]),
            ),
        )
        output[f"{mode}:{budget:g}"] = str(winner["method"])
    return output


def _capacity_safe(capacities: Sequence[Mapping[str, Any]]) -> bool:
    return bool(capacities) and all(
        float(row["used"]) <= float(row["capacity"]) + 1e-6
        for row in capacities
    )


def fixed_byte_audit_gate(
    comparisons: Sequence[Mapping[str, Any]],
    capacities: Sequence[Mapping[str, Any]],
    *,
    minimum_gain: float,
    required_gain_points: int,
    maximum_other_loss: float,
) -> dict[str, Any]:
    gains = [float(row["ap_gain"]) for row in comparisons]
    if len(gains) != 3 or len(capacities) != 3:
        raise ValueError("Audit gate requires three fixed-byte work points")
    gain_points = sum(value >= float(minimum_gain) for value in gains)
    noninferior = min(gains) >= -float(maximum_other_loss)
    capacity_safe = _capacity_safe(capacities)
    return {
        "passed": bool(
            gain_points >= int(required_gain_points)
            and noninferior
            and capacity_safe
        ),
        "gain_points": int(gain_points),
        "minimum_gain": float(min(gains)),
        "noninferior_all": bool(noninferior),
        "fixed_byte_capacity_safe": bool(capacity_safe),
    }


def fixed_byte_final_gate(
    intervals: Sequence[Mapping[str, Any]],
    capacities: Sequence[Mapping[str, Any]],
    *,
    required_superior_points: int,
    noninferiority_margin: float,
) -> dict[str, Any]:
    if len(intervals) != 3:
        raise ValueError("Final gate requires three fixed-byte intervals")
    superior = sum(float(row["ci_low"]) > 0.0 for row in intervals)
    noninferior = all(
        float(row["ci_low"]) >= -float(noninferiority_margin)
        for row in intervals
    )
    capacity_safe = _capacity_safe(capacities)
    return {
        "passed": bool(
            superior >= int(required_superior_points)
            and noninferior
            and capacity_safe
        ),
        "superior_points": int(superior),
        "noninferior_all": bool(noninferior),
        "fixed_byte_capacity_safe": bool(capacity_safe),
    }


def assert_final_allowed(
    audit_completion: Mapping[str, Any], *, final_exists: bool
) -> None:
    if audit_completion.get("complete") is not True:
        raise RuntimeError("Audit must complete before registered final")
    if audit_completion.get("passed") is not True:
        raise RuntimeError("Audit gate failed; registered final remains sealed")
    if final_exists:
        raise RuntimeError("Registered final has already completed")
