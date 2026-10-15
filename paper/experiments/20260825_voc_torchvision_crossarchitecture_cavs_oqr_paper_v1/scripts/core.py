"""Pure protocol contracts for the frozen VOC cross-architecture experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


CAVS_METHODS = ("CAVS-O-Core", "CAVS-OQ-Core", "CAVS-OQR-Core")
EXTERNAL_METHODS = (
    "ORIC-AP5095",
    "MORIC-AP5095",
    "ORIC-AP50",
    "MORIC-AP50",
)
FIXED_METHODS = ("DCSB-Score-Adapted", "Confidence", "Random")
ROW_KEYS = ("role", "method", "policy", "seed", "mode", "budget")


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
    for index, identity in enumerate(_ordered(values, seed)):
        folds[index % int(fold_count)].append(identity)
    return [sorted(fold) for fold in folds]


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "preregistered_torchvision_coco_crossarchitecture_cavs_oqr_benchmark":
        raise ValueError("experiment status changed")
    if {str(key): int(value) for key, value in protocol.get("roles", {}).items()} != {
        "fit": 15629,
        "calibration": 922,
        "final": 4952,
    }:
        raise ValueError("VOC role counts changed")
    contracts = protocol.get("role_contracts", {})
    if set(contracts) != {"fit", "calibration", "final"}:
        raise ValueError("frozen role contracts changed")
    for role, contract in contracts.items():
        for key in ("gt_sha256", "images_sha256", "identity_sha256"):
            if len(str(contract.get(key, ""))) != 64:
                raise ValueError(f"incomplete role hash: {role}/{key}")
    if list(map(float, protocol.get("budgets", ()))) != [0.1, 0.25, 0.4]:
        raise ValueError("paper budget grid changed")
    if list(map(str, protocol.get("modes", ()))) != ["fixed_byte", "fixed_call"]:
        raise ValueError("evaluation modes changed")
    if protocol.get("action_spaces") != {
        "O": ["edge", "original"],
        "OQ": ["edge", "original", "q90"],
        "OQR": ["edge", "original", "q90", "current@official"],
    }:
        raise ValueError("O/OQ/OQR action spaces changed")
    if len(protocol.get("model_seeds", ())) != 3 or len(set(protocol["model_seeds"])) != 3:
        raise ValueError("three unique model seeds are required")
    if int(protocol.get("fold_count", 0)) != 5:
        raise ValueError("five folds are required")
    if int(protocol.get("random", {}).get("seeds", 0)) != 50:
        raise ValueError("Random-50 contract changed")
    if int(protocol.get("bootstrap", {}).get("replicates", 0)) != 1000:
        raise ValueError("bootstrap replicate count changed")
    if protocol.get("bootstrap", {}).get("nested_random") is not True:
        raise ValueError("Random bootstrap must be nested")
    if protocol.get("metric", {}).get("ignore_difficult") is not True:
        raise ValueError("VOC difficult objects must be ignored during matching")
    if protocol.get("final_used_for_training_or_selection") is not False:
        raise ValueError("final cannot train or select the router")
    if protocol.get("old_learner_checkpoints_allowed") is not False:
        raise ValueError("old learner reuse is forbidden")
    expected_weights = {
        "edge": "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2",
        "cloud": "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf",
    }
    observed = {
        role: str(protocol.get("detectors", {}).get(role, {}).get("weights_sha256", ""))
        for role in expected_weights
    }
    if observed != expected_weights:
        raise ValueError("official Torchvision weight identities changed")
    mapping = protocol.get("category_mapping", {})
    if len(mapping) != 20 or len(set(map(str, mapping.values()))) != 20:
        raise ValueError("COCO-to-VOC mapping must be one-to-one over 20 classes")
    if int(protocol.get("model", {}).get("max_actions", 0)) != 4:
        raise ValueError("OQR requires exactly four action slots")
    roi = protocol.get("roi", {})
    if (str(roi.get("action_id")), str(roi.get("profile"))) != (
        "current@official",
        "current",
    ):
        raise ValueError("registered ROI action changed")
    if protocol.get("fusion") != {"name": "MAUR", "duplicate_iou": 0.5, "nms_iou": 0.7}:
        raise ValueError("registered MAUR fusion changed")


def reward_roles(role: str) -> tuple[str, ...]:
    value = str(role)
    if value == "all":
        return ("fit", "calibration")
    if value in {"fit", "calibration"}:
        return (value,)
    raise ValueError("final is forbidden for contextual rewards")


def validate_training_roles(roles: Sequence[str]) -> None:
    values = set(map(str, roles))
    if "final" in values:
        raise ValueError("final is forbidden for training")
    if values - {"fit", "fit_oof"}:
        raise ValueError(f"unknown training role: {sorted(values)}")


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    jobs: list[dict[str, Any]] = []
    folds: list[int | str] = [*range(int(protocol["fold_count"])), "full"]
    for method in CAVS_METHODS:
        for seed in map(int, protocol["model_seeds"]):
            for fold in folds:
                job = {
                    "id": f"{method}|seed={seed}|fold={fold}",
                    "method": method,
                    "seed": seed,
                    "fold": fold,
                    "train_roles": ["fit"],
                    "predict_roles": ["fit_oof"] if fold != "full" else ["calibration", "final"],
                }
                validate_training_roles(job["train_roles"])
                jobs.append(job)
    for method in EXTERNAL_METHODS:
        for seed in map(int, protocol["model_seeds"]):
            jobs.append(
                {
                    "id": f"{method}|seed={seed}|fold=full",
                    "method": method,
                    "seed": seed,
                    "fold": "full",
                    "train_roles": ["fit"],
                    "predict_roles": ["calibration", "final"],
                }
            )
    if len(jobs) != 66 or len({job["id"] for job in jobs}) != 66:
        raise ValueError(f"training matrix changed: {len(jobs)}")
    return jobs


def expected_point_keys(protocol: Mapping[str, Any]) -> list[tuple[str, str, str, str, str, float]]:
    validate_protocol(protocol)
    seeds = [*map(str, protocol["model_seeds"]), "ensemble"]
    variants: list[tuple[str, str, str]] = []
    variants.extend(("CAVS-O-Core", "Joint-O", seed) for seed in seeds)
    variants.extend(("CAVS-OQ-Core", "Joint-OQ", seed) for seed in seeds)
    variants.extend(("CAVS-OQR-Core", "Joint-OQR", seed) for seed in seeds)
    for method in EXTERNAL_METHODS:
        for policy in ("Original-only", "SharedAction-OQ"):
            variants.extend((method, policy, seed) for seed in seeds)
    for method in ("DCSB-Score-Adapted", "Confidence"):
        variants.extend((method, policy, "fixed") for policy in ("Original-only", "SharedAction-OQ"))
    variants.extend(("Random", policy, "mean50") for policy in ("Original-only", "SharedAction-OQ"))
    return [
        (role, method, policy, seed, mode, float(budget))
        for role in ("calibration", "final")
        for method, policy, seed in variants
        for mode in protocol["modes"]
        for budget in protocol["budgets"]
    ]


def validate_point_rows(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> None:
    expected = set(expected_point_keys(protocol))
    observed = [
        (
            str(row["role"]),
            str(row["method"]),
            str(row["policy"]),
            str(row["seed"]),
            str(row["mode"]),
            float(row["budget"]),
        )
        for row in rows
    ]
    if len(observed) != len(set(observed)):
        raise ValueError("point matrix contains duplicate keys")
    if set(observed) != expected:
        raise ValueError(
            f"point matrix incomplete: missing={len(expected - set(observed))}, "
            f"unexpected={len(set(observed) - expected)}"
        )


def fixed_call_count(population: int, budget: float) -> int:
    return int(math.floor(int(population) * float(budget) + 0.5))


def nested_random_delta(
    candidate_metrics: Sequence[float], random_metrics: Sequence[Sequence[float]]
) -> np.ndarray:
    candidate = np.asarray(candidate_metrics, dtype=np.float64)
    random = np.asarray(random_metrics, dtype=np.float64)
    if candidate.ndim != 1 or random.ndim != 2 or random.shape[1] != candidate.shape[0]:
        raise ValueError("random metrics must have shape [seed, bootstrap]")
    return candidate - random.mean(axis=0)


def validate_unique_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [tuple(row.get(key) for key in ROW_KEYS) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result rows")


def _selector_module() -> Any:
    repo = Path(__file__).resolve().parents[4]
    path = repo / "paper/experiments/GAIR_v5/gair/selector.py"
    spec = importlib.util.spec_from_file_location("voc_gluoncv_capacity_selector", path)
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
