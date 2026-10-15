#!/usr/bin/env python3
"""Measure how much AFPD supervision is allocated to ROI transitions."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
V1 = REPO / "paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


CORE = _load("transition_mass_core", ROOT / "scripts/core.py")
RUNNER = _load("transition_mass_v1_runner", V1 / "scripts/run_experiment.py")


def _weighted_counts(
    targets: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    action_names: list[list[str]],
) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    mass: Counter[str] = Counter()
    for row in range(targets.shape[0]):
        for probe in torch.nonzero(valid[row], as_tuple=False).flatten().tolist():
            column = int(targets[row, probe])
            name = action_names[row][column]
            counts[name] += 1
            mass[name] += float(weights[row, probe])
    return counts, mass


def run() -> dict:
    protocol = json.loads((ROOT / "configs/protocol.json").read_text())
    records, tensors = RUNNER._screen_tensor_data("fit", "OQR")
    if len(records) != int(protocol["expected_images"]):
        raise RuntimeError("Canonical fit population changed")
    names = [[str(action["action_id"]) for action in row["actions"]] for row in records]
    allowed_ids = frozenset(("edge", "original", "q90", "roi"))
    if any(
        not row
        or row[0] != "edge"
        or len(row) != len(set(row))
        or not frozenset(row).issubset(allowed_ids)
        for row in names
    ):
        raise RuntimeError("Unexpected action IDs or action space")
    action_mask = tensors["action_mask"].to(torch.bool)
    roi_columns = CORE.action_column_mask(names, action_mask, "roi")
    roi_available = roi_columns.any(dim=1)

    prices, price_mask = RUNNER.CORE.frontier_probe_prices(
        tensors["raw_target"],
        tensors["cost_ratio"],
        action_mask,
        relative_offset=float(protocol["relative_offset"]),
        maximum_price=float(protocol["maximum_price"]),
    )
    teacher_utility = RUNNER.CORE.frontier_utilities(
        tensors["raw_target"], tensors["cost_ratio"], action_mask, prices
    )
    targets = teacher_utility.argmax(dim=-1)
    top_two = teacher_utility.topk(k=2, dim=-1).values
    weights = RUNNER.CORE.boundary_weights(
        top_two[..., 0] - top_two[..., 1],
        scale=float(protocol["boundary_scale"]),
        boost=float(protocol["boundary_boost"]),
    ) * price_mask.to(torch.float32)

    valid_pairs, roi_pairs = CORE.crossing_pair_masks(
        tensors["raw_target"],
        tensors["cost_ratio"],
        action_mask,
        roi_columns,
        maximum_price=float(protocol["maximum_price"]),
    )
    # Probe columns: two endpoints, all low pair probes, then all high pair probes.
    pair_count = valid_pairs.shape[1]
    roi_probe = torch.zeros_like(price_mask, dtype=torch.bool)
    roi_probe[:, 2 : 2 + pair_count] = roi_pairs
    roi_probe[:, 2 + pair_count : 2 + 2 * pair_count] = roi_pairs
    all_crossing_probe = torch.zeros_like(price_mask, dtype=torch.bool)
    all_crossing_probe[:, 2 : 2 + pair_count] = valid_pairs
    all_crossing_probe[:, 2 + pair_count : 2 + 2 * pair_count] = valid_pairs
    if not torch.equal(all_crossing_probe, price_mask & all_crossing_probe):
        raise RuntimeError("Crossing-probe accounting disagrees with v1")

    target_counts, target_mass = _weighted_counts(targets, weights, price_mask, names)
    conditional_counts, conditional_mass = _weighted_counts(
        targets[roi_available],
        weights[roi_available],
        price_mask[roi_available],
        [row for row, keep in zip(names, roi_available.tolist()) if keep],
    )
    total_mass = float(weights.sum())
    conditional_total_mass = float(weights[roi_available].sum())
    crossing_mass = float((weights * all_crossing_probe).sum())
    roi_crossing_mass = float((weights * roi_probe).sum())
    conditional_crossing_mass = float(
        (weights[roi_available] * all_crossing_probe[roi_available]).sum()
    )
    conditional_roi_crossing_mass = float(
        (weights[roi_available] * roi_probe[roi_available]).sum()
    )
    roi_frontier = 0
    for row in torch.nonzero(roi_available, as_tuple=False).flatten().tolist():
        active = set(targets[row][price_mask[row]].tolist())
        roi_column = int(torch.nonzero(roi_columns[row], as_tuple=False)[0])
        roi_frontier += roi_column in active

    metrics = {
        "images": len(records),
        "action_pattern_counts": {
            "|".join(pattern): count
            for pattern, count in Counter(map(tuple, names)).items()
        },
        "roi_available_images": int(roi_available.sum()),
        "roi_available_rate": CORE.safe_fraction(int(roi_available.sum()), len(records)),
        "roi_frontier_images": int(roi_frontier),
        "roi_frontier_rate_given_available": CORE.safe_fraction(
            roi_frontier, int(roi_available.sum())
        ),
        "valid_crossing_pairs": int(valid_pairs.sum()),
        "roi_involved_valid_crossing_pairs": int(roi_pairs.sum()),
        "roi_pair_count_rate_global": CORE.safe_fraction(
            int(roi_pairs.sum()), int(valid_pairs.sum())
        ),
        "roi_pair_weight_rate_global": CORE.safe_fraction(
            roi_crossing_mass, crossing_mass
        ),
        "roi_pair_weight_rate_given_available": CORE.safe_fraction(
            conditional_roi_crossing_mass, conditional_crossing_mass
        ),
        "target_probe_counts": dict(target_counts),
        "target_boundary_weight": dict(target_mass),
        "target_boundary_weight_rate": {
            key: CORE.safe_fraction(value, total_mass) for key, value in target_mass.items()
        },
        "conditional_target_probe_counts": dict(conditional_counts),
        "conditional_target_boundary_weight": dict(conditional_mass),
        "conditional_target_boundary_weight_rate": {
            key: CORE.safe_fraction(value, conditional_total_mass)
            for key, value in conditional_mass.items()
        },
        "total_effective_weight": total_mass,
        "crossing_effective_weight": crossing_mass,
        "roi_crossing_effective_weight": roi_crossing_mass,
    }
    gate = protocol["go_no_go"]
    checks = {
        "roi_frontier_is_material": metrics["roi_frontier_rate_given_available"]
        >= float(gate["minimum_roi_frontier_rate_given_available"]),
        "global_roi_target_mass_is_low": metrics["target_boundary_weight_rate"].get("roi", 0.0)
        < float(gate["maximum_global_roi_target_weight_rate"]),
        "conditional_roi_crossing_mass_is_low": metrics[
            "roi_pair_weight_rate_given_available"
        ]
        < float(gate["maximum_conditional_roi_pair_weight_rate"]),
    }
    result = {
        "protocol_id": protocol["protocol_id"],
        "metrics": metrics,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "access": {
            "fit": True,
            "calibration": False,
            "audit": False,
            "val2017": False,
            "final": False,
        },
    }
    output = ROOT / "runs/diagnostic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (ROOT / "runs/completion.json").write_text(
        json.dumps({"complete": True, "gate_passed": result["gate"]["passed"]}, indent=2)
        + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
