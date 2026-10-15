"""Pure protocol contracts for the VOC community-YOLO26 point benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


TRAINED_METHODS = (
    "CAVS-OQR-MAUR",
    "ORIC-AP5095",
    "MORIC-AP5095",
    "ORIC-AP50",
    "MORIC-AP50",
)

ENSEMBLED_METHODS = (
    "CAVS-OQR-MAUR",
    "CAVS-OQ-from-OQR",
    "ORIC-AP5095",
    "MORIC-AP5095",
    "ORIC-AP50",
    "MORIC-AP50",
)

FIXED_METHODS = ("DCSB-Score-Adapted", "Confidence", "Random")


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]


def _ordered(values: Sequence[str], seed: int) -> list[str]:
    return sorted(
        map(str, values),
        key=lambda value: (
            hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).digest(),
            value,
        ),
    )


def deterministic_fit_calibration_split(
    identities: Sequence[str],
    *,
    fit_count: int,
    calibration_count: int,
    seed: int,
) -> dict[str, list[str]]:
    values = list(map(str, identities))
    if len(values) != len(set(values)):
        raise ValueError("dataset identities must be unique")
    if int(fit_count) + int(calibration_count) != len(values):
        raise ValueError("fit and calibration counts must cover the population")
    ordered = _ordered(values, seed)
    return {
        "fit": sorted(ordered[: int(fit_count)]),
        "calibration": sorted(ordered[int(fit_count) :]),
    }


def deterministic_folds(
    identities: Sequence[str], *, fold_count: int, seed: int
) -> list[list[str]]:
    values = list(map(str, identities))
    if len(values) != len(set(values)):
        raise ValueError("fold identities must be unique")
    if int(fold_count) < 2 or int(fold_count) > len(values):
        raise ValueError("invalid fold count")
    folds = [[] for _ in range(int(fold_count))]
    for index, value in enumerate(_ordered(values, seed)):
        folds[index % int(fold_count)].append(value)
    return [sorted(fold) for fold in folds]


def validate_category_names(
    actual: Mapping[int, str] | Sequence[str], expected: Sequence[str]
) -> None:
    if isinstance(actual, Mapping):
        observed = [str(actual[index]) for index in sorted(map(int, actual))]
    else:
        observed = list(map(str, actual))
    wanted = list(map(str, expected))
    if observed != wanted:
        raise ValueError(f"VOC category schema changed: {observed} != {wanted}")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    roles = {str(key): int(value) for key, value in protocol["roles"].items()}
    if roles != {"fit": 15629, "calibration": 922, "final": 4952}:
        raise ValueError(f"role counts changed: {roles}")
    if roles["fit"] + roles["calibration"] != 16551:
        raise ValueError("VOC trainval population changed")
    if int(protocol["bootstrap"]["replicates"]) != 0:
        raise ValueError("this experiment is point-only")
    if list(map(float, protocol["budgets"])) != [0.1, 0.25, 0.4]:
        raise ValueError("paper budgets changed")
    if list(map(str, protocol["actions"])) != [
        "edge",
        "original",
        "q90",
        "current@640",
    ]:
        raise ValueError("OQR action space changed")
    if len(protocol["data"]["category_names"]) != 20:
        raise ValueError("VOC must have 20 categories")
    expected_archives = {
        "VOCtrainval_06-Nov-2007.tar": 460032000,
        "VOCtest_06-Nov-2007.tar": 451020800,
        "VOCtrainval_11-May-2012.tar": 1999639040,
    }
    if {str(key): int(value) for key, value in protocol["data"]["archive_sizes"].items()} != expected_archives:
        raise ValueError("VOC archive sizes changed")
    if protocol.get("final_used_for_training_or_selection") is not False:
        raise ValueError("final must remain evaluation-only")
    if protocol.get("old_learner_checkpoints_allowed") is not False:
        raise ValueError("old learner reuse is forbidden")


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    return [
        {
            "id": f"{method}|seed={int(seed)}|full",
            "method": method,
            "seed": int(seed),
            "role": "fit",
        }
        for method in TRAINED_METHODS
        for seed in protocol["model_seeds"]
    ]


def expected_point_keys(
    protocol: Mapping[str, Any],
) -> list[tuple[str, str, str, str, float]]:
    validate_protocol(protocol)
    variants = [
        (method, str(seed))
        for method in ENSEMBLED_METHODS
        for seed in (*map(int, protocol["model_seeds"]), "ensemble")
    ]
    variants.extend((method, "fixed") for method in FIXED_METHODS)
    return [
        (role, method, seed, mode, float(budget))
        for role in ("calibration", "final")
        for method, seed in variants
        for mode in protocol["modes"]
        for budget in protocol["budgets"]
    ]


def validate_point_rows(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    expected = set(expected_point_keys(protocol))
    observed = [
        (
            str(row["role"]),
            str(row["method"]),
            str(row["seed"]),
            str(row["mode"]),
            float(row["budget"]),
        )
        for row in rows
    ]
    if len(observed) != len(set(observed)):
        raise RuntimeError("point result matrix contains duplicate rows")
    missing = expected - set(observed)
    unexpected = set(observed) - expected
    if missing or unexpected:
        raise RuntimeError(
            f"point result matrix is incomplete: missing={len(missing)}, unexpected={len(unexpected)}"
        )


def _selector_module() -> Any:
    path = REPO / "paper/experiments/GAIR_v5/gair/selector.py"
    spec = importlib.util.spec_from_file_location("voc_point_capacity_selector", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_actions_for_capacity(
    groups: Sequence[Sequence[Mapping[str, Any]]], capacity: float
) -> list[dict[str, Any]]:
    result = _selector_module().select_actions_for_budget(
        [[dict(action) for action in group] for group in groups], float(capacity)
    )
    selected = [dict(row) for row in result["selected"]]
    if sum(float(row["cost"]) for row in selected) > float(capacity) + 1e-9:
        raise RuntimeError("capacity solver exceeded the byte budget")
    if len({int(row["image_id"]) for row in selected}) != len(selected):
        raise RuntimeError("capacity solver selected multiple actions for one image")
    return selected
