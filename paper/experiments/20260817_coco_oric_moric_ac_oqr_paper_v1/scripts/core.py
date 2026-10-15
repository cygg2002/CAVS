"""Pure contracts for the paper-domain ORIC/MORIC action-conditioned run."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


METHODS = ("ORIC-AC-OQR", "MORIC-AC-OQR")
CONTRASTS = (
    ("CAVS-OQR", "ORIC-AC-OQR"),
    ("CAVS-OQR", "MORIC-AC-OQR"),
    ("ORIC-AC-OQR", "ORIC-SharedAction-OQR-MAUR"),
    ("MORIC-AC-OQR", "MORIC-SharedAction-OQR-MAUR"),
    ("ORIC-AC-OQR", "MORIC-AC-OQR"),
)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("dataset must be COCO")
    if protocol.get("runtime_domain") != "pytorch_paper_cache":
        raise ValueError("runtime domain must be the PyTorch paper cache")
    if protocol.get("roles") != {"fit": 9678, "calibration": 571, "final": 5000}:
        raise ValueError("canonical COCO population changed")
    if tuple(protocol.get("methods", {})) != METHODS:
        raise ValueError("method matrix changed")
    if list(protocol.get("action_space", ())) != [
        "edge",
        "original",
        "q90",
        "current@640",
    ]:
        raise ValueError("OQR action space changed")
    if list(map(float, protocol.get("budgets", ()))) != [0.1, 0.25, 0.4]:
        raise ValueError("budget grid changed")
    if list(protocol.get("modes", ())) != ["fixed_byte", "fixed_call"]:
        raise ValueError("evaluation modes changed")
    if int(protocol.get("fold_count", 0)) != 5:
        raise ValueError("fold count changed")
    if list(map(int, protocol.get("model_seeds", ()))) != [
        20260809,
        20260810,
        20260811,
    ]:
        raise ValueError("model seeds changed")
    if int(protocol["external_mlp"]["top_k"]) != 25:
        raise ValueError("ORIC Top-K changed")
    if list(map(int, protocol["external_mlp"]["hidden_dims"])) != [16, 16, 16, 16]:
        raise ValueError("ORIC image trunk changed")
    if protocol.get("training_is_fit_only") is not True:
        raise ValueError("training must be fit-only")
    if protocol.get("detector_inference_required") is not False:
        raise ValueError("detector inference is forbidden")
    if protocol.get("final_used_for_training_gate_or_selection") is not False:
        raise ValueError("final cannot train, gate, or select the method")
    if protocol.get("source_experiments_modified") is not False:
        raise ValueError("source experiments must remain read-only")
    if max(
        int(protocol["cpu"][key])
        for key in ("training_workers", "evaluation_workers", "bootstrap_workers")
    ) > 8:
        raise ValueError("worker count exceeds the isolation limit")
    forbidden = tuple(str(value).lower() for value in protocol["forbidden_path_tokens"])
    for key in ("source_factorial_experiment", "source_completion_experiment"):
        value = str(protocol[key]).lower()
        if any(token in value for token in forbidden):
            raise ValueError(f"forbidden deployment-domain source: {key}")


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    folds: list[int | str] = [*range(int(protocol["fold_count"])), "full"]
    jobs = []
    for method in METHODS:
        for seed in map(int, protocol["model_seeds"]):
            for fold in folds:
                token = "full" if fold == "full" else f"fold_{fold}"
                jobs.append(
                    {
                        "id": f"{method}|seed={seed}|{token}",
                        "method": method,
                        "target": str(protocol["methods"][method]["target"]),
                        "action_space": list(map(str, protocol["action_space"])),
                        "seed": seed,
                        "fold": fold,
                    }
                )
    return jobs


def bootstrap_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    return [
        {
            "id": f"{left}-vs-{right}|fixed_byte|{float(budget):g}",
            "left": left,
            "right": right,
            "mode": "fixed_byte",
            "budget": float(budget),
        }
        for left, right in CONTRASTS
        for budget in protocol["budgets"]
    ]


def unified_cloud_cdf_targets(
    rewards: np.ndarray,
    action_type: np.ndarray,
    action_mask: np.ndarray,
    *,
    reference_rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform all cloud actions with one training-only empirical CDF."""

    values = np.asarray(rewards, dtype=np.float64)
    types = np.asarray(action_type, dtype=np.int64)
    mask = np.asarray(action_mask, dtype=bool)
    if values.shape != types.shape or values.shape != mask.shape or values.ndim != 2:
        raise ValueError("reward, action type, and mask tensors must match")
    rows = np.asarray(list(reference_rows), dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or rows.min() < 0 or rows.max() >= len(values):
        raise ValueError("CDF reference rows are invalid")
    valid_cloud = mask & (types != 0)
    reference = np.sort(values[rows][valid_cloud[rows]])
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("CDF reference must contain finite cloud rewards")
    output = np.zeros_like(values, dtype=np.float64)
    output[valid_cloud] = (
        np.searchsorted(reference, values[valid_cloud], side="right") / len(reference)
    )
    return output, reference
