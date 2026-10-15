"""Pure contracts for the COCO/Oregon frozen full benchmark."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


PRIMARY_ACTIONS = {"edge", "original", "q90"}


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != "coco_oregon_full_scheduler_benchmark_v1":
        raise ValueError("Protocol ID changed")
    if protocol.get("result_status") != "posthoc_final_reuse_full_benchmark":
        raise ValueError("Result status changed")
    datasets = protocol.get("datasets", {})
    expected = {
        "coco": ("canonical_v1", 9678, 571, 5000),
        "oregon": ("canonical_oregon_event_v1", 9923, 4961, 4962),
    }
    if set(datasets) != set(expected):
        raise ValueError("Dataset set changed")
    for dataset, (run, fit, calibration, final) in expected.items():
        item = datasets[dataset]
        if item.get("canonical_run") != run:
            raise ValueError(f"Canonical run changed for {dataset}")
        roles = item.get("roles", {})
        if (int(roles.get("fit", -1)), int(roles.get("calibration", -1)), int(roles.get("final", -1))) != (fit, calibration, final):
            raise ValueError(f"Role counts changed for {dataset}")
        if item.get("detector_protocol") != "cloud_v1":
            raise ValueError(f"Primary detector protocol changed for {dataset}")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("Budget grid changed")
    if tuple(map(str, protocol.get("modes", ()))) != ("fixed_byte", "fixed_call"):
        raise ValueError("Mode grid changed")
    if set(map(str, protocol.get("primary_actions", ()))) != PRIMARY_ACTIONS:
        raise ValueError("Primary action space changed")
    if int(protocol.get("random", {}).get("seeds", -1)) != 50:
        raise ValueError("Random seed count changed")
    if int(protocol.get("bootstrap", {}).get("replicates", -1)) != 1000:
        raise ValueError("Bootstrap count changed")
    for flag in ("fit_only_training", "calibration_only_controls"):
        if protocol.get(flag) is not True:
            raise ValueError(f"{flag} must remain true")
    for flag in ("final_used_for_training_or_selection", "detector_inference", "gpu_required", "dualhead_v3_defined"):
        if protocol.get(flag) is not False:
            raise ValueError(f"{flag} must remain false")
    sources = protocol.get("source_experiments", {})
    primary = "\n".join(
        str(value) for key, value in sources.items() if not key.startswith("cloudv2")
    ).lower()
    for forbidden in protocol.get("forbidden_primary_source_substrings", ()):
        if str(forbidden).lower() in primary:
            raise ValueError(f"Cloud-V2 leaked into primary sources: {forbidden}")


def expected_primary_rows(protocol: Mapping[str, Any]) -> int:
    scalar = (
        len(protocol["internal_rankers"])
        + len(protocol["historical_rankers"])
        + len(protocol["external_methods"]) - 1
    )
    grid = len(protocol["scalar_payload_policies"]) * len(protocol["modes"]) * len(protocol["budgets"])
    candidates = len(protocol["primary_candidates"]) * len(protocol["modes"]) * len(protocol["budgets"])
    native = len(protocol["scalar_payload_policies"])
    systems = 3
    return scalar * grid + candidates + native + systems


def assert_primary_actions(actions: Mapping[int, str]) -> None:
    invalid = sorted(
        (int(image_id), str(action))
        for image_id, action in actions.items()
        if str(action) not in PRIMARY_ACTIONS - {"edge"}
    )
    if invalid:
        raise RuntimeError(f"Invalid primary action(s): {invalid[:3]}")


def assert_fixed_byte_capacity(
    samples: Sequence[Mapping[str, Any]],
    actions: Mapping[int, str],
    *,
    capacity: float,
    tolerance: float = 1e-6,
) -> float:
    assert_primary_actions(actions)
    by_id = {int(sample["image_id"]): sample for sample in samples}
    used = sum(
        float(by_id[int(image_id)]["payload_bytes"][str(action)])
        for image_id, action in actions.items()
    )
    if used > float(capacity) + float(tolerance):
        raise RuntimeError(f"Fixed-byte capacity exceeded: {used} > {capacity}")
    return used / max(float(capacity), 1e-12)


def source_tier(path: str) -> str:
    lowered = str(path).lower()
    return "cloudv2_reference" if ("cloudv2" in lowered or "cloud-v2" in lowered) else "canonical_primary"


def row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["dataset"]), str(row["method"]), str(row["policy"]),
        str(row["mode"]), float(row["budget"]), str(row.get("seed", "aggregate")),
    )


def validate_unique_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    keys = [row_key(row) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate result rows: {duplicates[:3]}")

