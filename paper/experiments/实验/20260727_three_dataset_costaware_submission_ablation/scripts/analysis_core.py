"""Pure contracts for frozen cross-dataset diagnostic analyses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _order_index(value: str, order: Sequence[str]) -> int:
    try:
        return tuple(map(str, order)).index(str(value))
    except ValueError as error:
        raise ValueError(f"Unknown ordered value: {value}") from error


def index_unique_rows(
    rows: Iterable[Mapping[str, Any]], *, key_fields: Sequence[str]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Index rows and reject ambiguous joins."""
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = tuple(row[field] for field in key_fields)
        if key in indexed:
            raise ValueError(f"Duplicate row key: {key}")
        indexed[key] = row
    return indexed


def choose_reference_candidate(
    rows: Iterable[Mapping[str, Any]], *, ranker_order: Sequence[str]
) -> dict[str, Any]:
    """Choose the selector-only best SharedAction reference."""
    candidates = [dict(row) for row in rows if row["policy"] == "SharedAction"]
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


def choose_no_cost_candidate(
    rows: Iterable[Mapping[str, Any]],
    *,
    ranker_order: Sequence[str],
    policy_order: Sequence[str],
) -> dict[str, Any]:
    """Choose selector AP only; bytes do not participate in selection."""
    candidates = list(map(dict, rows))
    if not candidates:
        raise ValueError("No candidate rows")
    return min(
        candidates,
        key=lambda row: (
            -float(row["ap"]),
            _order_index(str(row["ranker"]), ranker_order),
            _order_index(str(row["policy"]), policy_order),
        ),
    )


def choose_final_oracle(
    rows: Iterable[Mapping[str, Any]],
    *,
    ranker_order: Sequence[str],
    policy_order: Sequence[str],
) -> dict[str, Any]:
    """Choose a final-only AP oracle for diagnostic regret calculation."""
    candidates = list(map(dict, rows))
    if not candidates:
        raise ValueError("No final candidate rows")
    return min(
        candidates,
        key=lambda row: (
            -float(row["ap"]),
            float(row["kb"]),
            _order_index(str(row["ranker"]), ranker_order),
            _order_index(str(row["policy"]), policy_order),
        ),
    )


def select_shared_pair(
    rows_by_dataset: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    mode: str,
    ap_tolerance: float,
    ranker_order: Sequence[str],
    policy_order: Sequence[str],
) -> dict[str, Any]:
    """Select one Ranker-family/policy pair without reading final results."""
    if mode not in {"fixed_call", "online", "fixed_byte"}:
        raise ValueError(f"Unsupported mode: {mode}")
    datasets = tuple(sorted(map(str, rows_by_dataset)))
    if not datasets:
        raise ValueError("No datasets")

    rows = {dataset: list(map(dict, rows_by_dataset[dataset])) for dataset in datasets}
    references = {
        dataset: choose_reference_candidate(
            rows[dataset], ranker_order=ranker_order
        )
        for dataset in datasets
    }
    indexes = {
        dataset: index_unique_rows(
            rows[dataset], key_fields=("ranker", "policy")
        )
        for dataset in datasets
    }
    common_pairs = set(indexes[datasets[0]])
    for dataset in datasets[1:]:
        common_pairs &= set(indexes[dataset])
    if not common_pairs:
        raise ValueError("No shared Ranker/policy pairs")

    summaries: list[dict[str, Any]] = []
    epsilon = float(ap_tolerance)
    for ranker, policy in sorted(common_pairs):
        deltas = []
        cost_ratios = []
        per_dataset = {}
        for dataset in datasets:
            candidate = indexes[dataset][(ranker, policy)]
            reference = references[dataset]
            delta = float(candidate["ap"]) - float(reference["ap"])
            reference_kb = float(reference["kb"])
            if reference_kb <= 0:
                raise ValueError(f"Non-positive reference KB for {dataset}")
            ratio = float(candidate["kb"]) / reference_kb
            deltas.append(delta)
            cost_ratios.append(ratio)
            per_dataset[dataset] = {
                "ap": float(candidate["ap"]),
                "kb": float(candidate["kb"]),
                "ap_delta": delta,
                "normalized_kb": ratio,
                "reference_ranker": str(reference["ranker"]),
                "reference_ap": float(reference["ap"]),
                "reference_kb": reference_kb,
                "passes_epsilon": delta >= -epsilon - 1e-12,
            }
        summaries.append(
            {
                "ranker": str(ranker),
                "policy": str(policy),
                "violating_datasets": sum(delta < -epsilon - 1e-12 for delta in deltas),
                "worst_ap_delta": min(deltas),
                "mean_ap_delta": sum(deltas) / len(deltas),
                "mean_normalized_kb": sum(cost_ratios) / len(cost_ratios),
                "per_dataset": per_dataset,
            }
        )

    feasible = [row for row in summaries if row["violating_datasets"] == 0]
    candidates = feasible or summaries

    def frozen_tie(row: Mapping[str, Any]) -> tuple[int, int]:
        return (
            _order_index(str(row["ranker"]), ranker_order),
            _order_index(str(row["policy"]), policy_order),
        )

    if feasible and mode in {"fixed_call", "online"}:
        selected = min(
            candidates,
            key=lambda row: (
                float(row["mean_normalized_kb"]),
                -float(row["worst_ap_delta"]),
                -float(row["mean_ap_delta"]),
                *frozen_tie(row),
            ),
        )
    elif feasible:
        selected = min(
            candidates,
            key=lambda row: (
                -float(row["mean_ap_delta"]),
                -float(row["worst_ap_delta"]),
                float(row["mean_normalized_kb"]),
                *frozen_tie(row),
            ),
        )
    else:
        selected = min(
            candidates,
            key=lambda row: (
                int(row["violating_datasets"]),
                -float(row["worst_ap_delta"]),
                -float(row["mean_ap_delta"]),
                float(row["mean_normalized_kb"]),
                *frozen_tie(row),
            ),
        )

    return {
        "strict_feasible": bool(feasible),
        "strict_feasible_pairs": len(feasible),
        "selection_status": (
            "strict_feasible" if feasible else "least_violating_diagnostic"
        ),
        "selected_ranker": selected["ranker"],
        "selected_policy": selected["policy"],
        "violating_datasets": selected["violating_datasets"],
        "worst_ap_delta": selected["worst_ap_delta"],
        "mean_ap_delta": selected["mean_ap_delta"],
        "mean_normalized_kb": selected["mean_normalized_kb"],
        "per_dataset": selected["per_dataset"],
        "all_pair_summaries": summaries,
    }
