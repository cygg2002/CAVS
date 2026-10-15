"""Pure contracts for calibration-only cost-aware policy selection."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_MODES = ("fixed_call", "online", "fixed_byte")


def connected_hash_components(
    hashes_by_image: Mapping[int, int], *, max_hamming: int
) -> list[list[int]]:
    """Build deterministic transitive near-duplicate components."""
    image_ids = sorted(map(int, hashes_by_image))
    parent = {image_id: image_id for image_id in image_ids}

    def find(image_id: int) -> int:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    for index, left in enumerate(image_ids):
        for right in image_ids[index + 1 :]:
            distance = bin(
                int(hashes_by_image[left]) ^ int(hashes_by_image[right])
            ).count("1")
            if distance <= int(max_hamming):
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for image_id in image_ids:
        grouped.setdefault(find(image_id), []).append(image_id)
    return sorted((sorted(values) for values in grouped.values()), key=lambda x: x[0])


def assign_group_disjoint_split(
    components: Iterable[Iterable[int]],
    *,
    class_ids_by_image: Mapping[int, Iterable[int]],
    target_selector_images: int,
    seed: int,
) -> dict[str, Any]:
    """Assign whole components while balancing size and rare class counts."""
    groups = [sorted(map(int, component)) for component in components]
    all_ids = sorted(image_id for group in groups for image_id in group)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Near-duplicate components overlap")
    if not 0 < int(target_selector_images) < len(all_ids):
        raise ValueError("Selector target must split the image set")

    classes = {
        image_id: set(map(int, class_ids_by_image.get(image_id, ())))
        for image_id in all_ids
    }
    totals = Counter(
        class_id for image_id in all_ids for class_id in classes[image_id]
    )
    target_gate = len(all_ids) - int(target_selector_images)

    def group_counts(group: list[int]) -> Counter[int]:
        return Counter(
            class_id for image_id in group for class_id in classes[image_id]
        )

    def seeded_tie(group: list[int]) -> str:
        value = f"{int(seed)}:" + ",".join(map(str, group))
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    ordered = sorted(
        groups,
        key=lambda group: (
            -sum(
                count / max(1, totals[class_id])
                for class_id, count in group_counts(group).items()
            ),
            -len(group),
            seeded_tie(group),
        ),
    )
    assigned = {"selector": [], "gate": []}
    sizes = {"selector": 0, "gate": 0}
    class_counts = {"selector": Counter(), "gate": Counter()}

    def assignment_cost(side: str, group: list[int]) -> tuple[float, str]:
        projected_sizes = dict(sizes)
        projected_sizes[side] += len(group)
        counts = group_counts(group)
        projected_classes = {
            name: Counter(class_counts[name]) for name in ("selector", "gate")
        }
        projected_classes[side].update(counts)
        size_cost = (
            abs(projected_sizes["selector"] - int(target_selector_images))
            / max(1, int(target_selector_images))
            + abs(projected_sizes["gate"] - target_gate) / max(1, target_gate)
        )
        class_cost = 0.0
        selector_fraction = int(target_selector_images) / len(all_ids)
        for class_id, total in totals.items():
            selector_target = total * selector_fraction
            gate_target = total - selector_target
            class_cost += abs(
                projected_classes["selector"][class_id] - selector_target
            ) / max(1, total)
            class_cost += abs(
                projected_classes["gate"][class_id] - gate_target
            ) / max(1, total)
        overflow = max(
            0, projected_sizes["selector"] - int(target_selector_images)
        ) + max(0, projected_sizes["gate"] - target_gate)
        return size_cost + class_cost + 10.0 * overflow, side

    for group in ordered:
        side = min(
            ("selector", "gate"),
            key=lambda name: assignment_cost(name, group),
        )
        assigned[side].append(group)
        sizes[side] += len(group)
        class_counts[side].update(group_counts(group))

    selector_ids = sorted(image_id for group in assigned["selector"] for image_id in group)
    gate_ids = sorted(image_id for group in assigned["gate"] for image_id in group)
    return {
        "selector_image_ids": selector_ids,
        "gate_image_ids": gate_ids,
        "selector_components": sorted(assigned["selector"], key=lambda x: x[0]),
        "gate_components": sorted(assigned["gate"], key=lambda x: x[0]),
        "selector_images": len(selector_ids),
        "gate_images": len(gate_ids),
        "selector_class_counts": dict(sorted(class_counts["selector"].items())),
        "gate_class_counts": dict(sorted(class_counts["gate"].items())),
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject selection, fallback, or candidate-set drift."""
    rankers = tuple(map(str, protocol["selectable_rankers"]))
    policies = tuple(map(str, protocol["policies"]))
    if len(rankers) != 8 or len(set(rankers)) != 8 or "Random" in rankers:
        raise ValueError("Exactly eight deterministic non-random Rankers are selectable")
    if len(policies) != 3 or len(set(policies)) != 3:
        raise ValueError("Exactly three unique action policies are required")
    if len(rankers) * len(policies) != int(
        protocol["expected_candidates_per_work_point"]
    ):
        raise ValueError("Candidate grid size changed")
    if protocol.get("random_selectable") is not False:
        raise ValueError("Random must remain reference-only")
    if protocol.get("final_used_for_selection") is not False:
        raise ValueError("Final must not participate in selection")
    if protocol.get("try_second_candidate_after_gate_failure") is not False:
        raise ValueError("A failed gate must fall back directly to the reference")
    if protocol.get("gate_failure_action") != (
        "reference_fallback_without_second_candidate"
    ):
        raise ValueError("Gate fallback contract changed")


def validate_candidate_grid(
    rows: Iterable[Mapping[str, Any]],
    *,
    rankers: Sequence[str],
    policies: Sequence[str],
) -> None:
    """Require exactly one row for every deterministic Ranker/policy pair."""
    keys = [(str(row["ranker"]), str(row["policy"])) for row in rows]
    counts = Counter(keys)
    expected = {
        (str(ranker), str(policy)) for ranker in rankers for policy in policies
    }
    if set(keys) != expected or any(count != 1 for count in counts.values()):
        raise ValueError(
            "Invalid candidate grid: "
            f"missing={sorted(expected - set(keys))} "
            f"extra={sorted(set(keys) - expected)} "
            f"duplicates={sorted(key for key, count in counts.items() if count != 1)}"
        )


def extract_frozen_policy_candidates(
    calibration_rows: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    budget: float,
    rankers: Sequence[str],
    gbdt_threshold: float,
) -> list[dict[str, Any]]:
    """Extract the three frozen policies from a threshold calibration curve."""
    relevant = [
        dict(row)
        for row in calibration_rows
        if str(row["mode"]) == str(mode)
        and math.isclose(float(row["budget"]), float(budget), abs_tol=1e-12)
        and str(row["ranker"]) in set(map(str, rankers))
    ]
    if not relevant:
        raise ValueError(f"No calibration rows for {mode} budget={budget}")
    minimum_threshold = min(float(row["threshold"]) for row in relevant)

    def matches(left: float, right: float) -> bool:
        if math.isinf(left) or math.isinf(right):
            return left == right
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)

    policies = (
        ("SharedAction", math.inf),
        ("CostEligible-AllROI", minimum_threshold),
        ("GBDT-Hierarchical", float(gbdt_threshold)),
    )
    candidates: list[dict[str, Any]] = []
    for ranker in map(str, rankers):
        ranker_rows = [row for row in relevant if str(row["ranker"]) == ranker]
        for policy, threshold in policies:
            matches_rows = [
                row
                for row in ranker_rows
                if matches(float(row["threshold"]), threshold)
            ]
            if len(matches_rows) != 1:
                raise ValueError(
                    f"Expected one {policy} row for {ranker}, got {len(matches_rows)}"
                )
            candidate = dict(matches_rows[0])
            candidate["policy"] = policy
            candidates.append(candidate)
    validate_candidate_grid(candidates, rankers=rankers, policies=[p[0] for p in policies])
    return candidates


def _order_index(value: str, order: Sequence[str]) -> int:
    try:
        return tuple(map(str, order)).index(str(value))
    except ValueError as error:
        raise ValueError(f"Unknown ordered value: {value}") from error


def choose_reference(
    rows: Iterable[Mapping[str, Any]], *, ranker_order: Sequence[str]
) -> dict[str, Any]:
    """Choose the maximum-AP SharedAction Ranker without using final."""
    candidates = [
        dict(row) for row in rows if str(row["policy"]) == "SharedAction"
    ]
    if not candidates:
        raise ValueError("No SharedAction reference candidate")
    return min(
        candidates,
        key=lambda row: (
            -float(row["ap"]),
            float(row["kb"]),
            _order_index(str(row["ranker"]), ranker_order),
        ),
    )


def choose_provisional(
    rows: Iterable[Mapping[str, Any]],
    *,
    reference: Mapping[str, Any],
    mode: str,
    ap_tolerance: float,
    ranker_order: Sequence[str],
    policy_order: Sequence[str],
) -> dict[str, Any]:
    """Select one provisional candidate under the frozen mode objective."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    minimum_ap = float(reference["ap"]) - float(ap_tolerance)
    eligible = [dict(row) for row in rows if float(row["ap"]) >= minimum_ap]
    if not eligible:
        return dict(reference)

    common_ties = lambda row: (  # noqa: E731
        _order_index(str(row["policy"]), policy_order),
        _order_index(str(row["ranker"]), ranker_order),
    )
    if mode in {"fixed_call", "online"}:
        return min(
            eligible,
            key=lambda row: (
                float(row["kb"]),
                -float(row["ap"]),
                *common_ties(row),
            ),
        )
    return min(
        eligible,
        key=lambda row: (
            -float(row["ap"]),
            float(row["kb"]),
            *common_ties(row),
        ),
    )


def same_configuration(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return (
        str(left["ranker"]) == str(right["ranker"])
        and str(left["policy"]) == str(right["policy"])
    )


def apply_risk_gate(
    *,
    reference: Mapping[str, Any],
    provisional: Mapping[str, Any],
    ap_delta_ci_low: float,
    ap_tolerance: float,
) -> dict[str, Any]:
    """Freeze the provisional choice or fall back directly to the reference."""
    identical = same_configuration(reference, provisional)
    passed = float(ap_delta_ci_low) >= -float(ap_tolerance)
    frozen = dict(provisional) if passed else dict(reference)
    return {
        "reference": dict(reference),
        "provisional": dict(provisional),
        "frozen": frozen,
        "ap_delta_ci_low": float(ap_delta_ci_low),
        "gate_passed": bool(passed),
        "used_reference_fallback": bool(not passed),
        "identical_configuration": bool(identical),
    }


def validate_measured_e2e_claim(metadata: Mapping[str, Any]) -> None:
    """Block estimated or incomplete measurements from being called measured."""
    if metadata.get("latency_source") != "measured":
        raise ValueError("An estimated latency cannot be labeled measured E2E")
    required = (
        "edge_host",
        "cloud_host",
        "edge_model_sha256",
        "cloud_model_sha256",
        "network_path",
        "sample_count",
        "raw_log_sha256",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        label = "raw logs" if "raw_log_sha256" in missing else "metadata"
        raise ValueError(f"Measured E2E is missing {label}: {missing}")
    if str(metadata["edge_host"]) == str(metadata["cloud_host"]):
        raise ValueError("Measured E2E requires distinct edge and cloud hosts")
    if int(metadata["sample_count"]) <= 0:
        raise ValueError("Measured E2E requires a positive sample count")
