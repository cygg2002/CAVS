"""Pure validation helpers for the frozen Oregon Cloud-V4 point protocol."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_ROLES = {"fit": 9923, "calibration": 4961, "final": 4962}
EXPECTED_PROFILES = ("current",)
EXPECTED_IMGSZ = (640,)
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
EXPECTED_CLOUD_SHA = "8beb7c7e49c5a0ec60222723816f9c943f59a24d902c451be3e5c6910bffbacb"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, int]:
    if protocol.get("protocol_id") != "oregon_cloudv4_current640_maur_strict_point_v1":
        raise ValueError("unexpected protocol id")
    if protocol.get("status") != "posthoc_exploratory_final_reuse":
        raise ValueError("Oregon final must remain explicitly post-hoc")
    if bool(protocol.get("confirmatory", True)):
        raise ValueError("this reused final cannot be confirmatory")
    if protocol.get("roles") != EXPECTED_ROLES:
        raise ValueError("canonical Oregon role counts changed")
    cloud = protocol.get("cloud_model", {})
    if int(cloud.get("epoch", -1)) != 29 or cloud.get("sha256") != EXPECTED_CLOUD_SHA:
        raise ValueError("Cloud-V4 epoch/hash changed")
    if tuple(protocol.get("profiles", ())) != EXPECTED_PROFILES:
        raise ValueError("profile set changed")
    if tuple(map(int, protocol.get("imgsz", ()))) != EXPECTED_IMGSZ:
        raise ValueError("ROI scale set changed")
    expected_actions = {
        f"{profile}@{imgsz}" for profile in EXPECTED_PROFILES for imgsz in EXPECTED_IMGSZ
    }
    if set(map(str, protocol.get("roi_actions", ()))) != expected_actions:
        raise ValueError("Current@640 action set changed")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("primary budgets changed")
    if int(protocol.get("fold_count", -1)) != 5:
        raise ValueError("five event-group folds are required")
    if int(protocol.get("random_seeds", -1)) != 0:
        raise ValueError("Random is outside this point-only experiment")
    if int(protocol.get("bootstrap", {}).get("replicates", -1)) != 0:
        raise ValueError("bootstrap is outside this point-only experiment")
    if bool(protocol.get("final_used_for_training_or_selection", True)):
        raise ValueError("final cannot train or select")
    if bool(protocol.get("old_cloud_predictions_allowed", True)):
        raise ValueError("old cloud predictions must be forbidden")
    if bool(protocol.get("old_learner_checkpoints_allowed", True)):
        raise ValueError("old learner checkpoints must be forbidden")

    method_groups = (
        tuple(protocol.get("historical_rankers", ())),
        tuple(protocol.get("external_rankers", ())),
        tuple(protocol.get("internal_rankers", ())),
    )
    methods = [str(method) for group in method_groups for method in group]
    if methods != ["CurrentActionValue-OQR12"]:
        raise ValueError("only the strict CurrentActionValue-OQR learner is allowed")
    if protocol.get("scalar_action_policies") != []:
        raise ValueError("scalar policies are outside this point-only experiment")
    return {"roi_actions": len(expected_actions), "learned_methods": len(methods)}


def validate_role_isolation(roles: Mapping[str, Sequence[Mapping[str, Any]]]) -> int:
    if set(roles) != set(EXPECTED_ROLES):
        raise ValueError("fit/calibration/final roles are required")
    identities: dict[str, str] = {}
    for role, samples in roles.items():
        local: set[str] = set()
        for sample in samples:
            identity = str(sample["identity"])
            if identity in local:
                raise ValueError(f"duplicate identity within {role}: {identity}")
            local.add(identity)
            previous = identities.get(identity)
            if previous is not None:
                raise ValueError(
                    f"identity leakage between {previous} and {role}: {identity}"
                )
            identities[identity] = role
    return 0


def event_group_folds(
    samples: Sequence[Mapping[str, Any]], *, fold_count: int, seed: int
) -> list[list[str]]:
    if not 2 <= int(fold_count) <= len(samples):
        raise ValueError("fold count must create non-empty folds")
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        identity = str(sample["identity"])
        event_id = str(sample["event_id"])
        grouped.setdefault(event_id, []).append(identity)
    events = sorted(grouped)
    random.Random(int(seed)).shuffle(events)
    # Greedy balancing by image count preserves complete event groups.
    folds: list[list[str]] = [[] for _ in range(int(fold_count))]
    sizes = [0] * int(fold_count)
    for event_id in sorted(events, key=lambda item: (-len(grouped[item]), item)):
        target = min(range(int(fold_count)), key=lambda index: (sizes[index], index))
        values = sorted(grouped[event_id])
        folds[target].extend(values)
        sizes[target] += len(values)
    if any(not fold for fold in folds):
        raise ValueError("event grouping produced an empty fold")
    return [sorted(fold) for fold in folds]


def assert_capacity(request_bytes: float, capacity_bytes: float) -> None:
    request = float(request_bytes)
    capacity = float(capacity_bytes)
    tolerance = max(1e-6, abs(capacity) * 1e-12)
    if request > capacity + tolerance:
        raise ValueError(f"capacity exceeded: request={request} capacity={capacity}")


def assert_source_boundary(paths: Iterable[str]) -> None:
    for raw in paths:
        value = str(raw).lower()
        if "cloudv2" in value or "cloud-v2" in value:
            raise ValueError(f"old cloud artifact is forbidden: {raw}")
        if any(token in value for token in ("/actions/original/", "/actions/q90_predictions/", "/actions/roi/")):
            raise ValueError(f"old cloud artifact is forbidden: {raw}")
        if any(token in value for token in ("checkpoint.pt", "models.json", "action_gate.json", "threshold_choices.json")):
            raise ValueError(f"old learner artifact is forbidden: {raw}")


def bootstrap_comparisons(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    budgets = tuple(map(float, protocol["budgets"]))
    pairs = (
        ("AFPD-OQR12", "Raw-ActionValue-OQ", "core"),
        ("AFPD-OQR12", "ORIC-AP5095", "external"),
        ("AFPD-OQR12", "MORIC-AP5095", "external"),
        ("OccupancyFree-AFPD-OQR12", "AFPD-OQR12", "ablation"),
        ("OccupancyAware-AFPD-OQR12", "AFPD-OQR12", "ablation"),
        ("AFPD-OQR12", "CurrentActionValue-OQ", "same_run"),
        ("AFPD-OQR12", "CurrentActionValue-OQR12", "same_run"),
        ("AFPD-OQR12", "Confidence", "control"),
        ("AFPD-OQR12", "Random", "control"),
    )
    return [
        {
            "id": f"{candidate}__vs__{comparator}__b{budget:g}",
            "candidate": candidate,
            "comparator": comparator,
            "budget": budget,
            "mode": "fixed_byte",
            "family": family,
        }
        for candidate, comparator, family in pairs
        for budget in budgets
    ]


def bootstrap_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {**comparison, "unit": str(unit)}
        for unit in protocol["bootstrap"]["units"]
        for comparison in bootstrap_comparisons(protocol)
    ]
