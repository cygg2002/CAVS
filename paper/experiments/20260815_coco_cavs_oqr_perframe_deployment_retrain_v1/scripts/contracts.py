"""Pure contracts for the deployment-consistent CAVS experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


ACTIONS = ("edge", "original", "q90", "current@640")
ROLES = {"fit": 9678, "calibration": 571, "final": 5000}
SEEDS = (20260809, 20260810, 20260811)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("dataset must be COCO")
    if protocol.get("roles") != ROLES:
        raise ValueError("canonical role counts changed")
    if tuple(protocol.get("actions", ())) != ACTIONS:
        raise ValueError("action space changed")
    if tuple(map(int, protocol.get("model_seeds", ()))) != SEEDS:
        raise ValueError("model seeds changed")
    if list(map(float, protocol.get("budgets", ()))) != [0.1, 0.25, 0.4]:
        raise ValueError("budget grid changed")
    inference = protocol.get("inference", {})
    expected = {
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "rect": True,
        "end2end": False,
        "single_image": True,
        "scaleup_full": False,
        "scaleup_roi": True,
    }
    for key, value in expected.items():
        if inference.get(key) != value:
            raise ValueError(f"inference contract changed: {key}")
    if protocol.get("source_experiments_modified") is not False:
        raise ValueError("source experiments must remain read-only")
    if protocol.get("old_learner_checkpoints_reused") is not False:
        raise ValueError("old learner checkpoints cannot be reused")
    if protocol.get("final_used_for_training_or_selection") is not False:
        raise ValueError("final cannot influence training or model selection")


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    jobs = []
    for seed in map(int, protocol["model_seeds"]):
        for fold in (*range(int(protocol["fold_count"])), "full"):
            token = "full" if fold == "full" else f"fold_{fold}"
            jobs.append({"id": f"seed={seed}|{token}", "seed": seed, "fold": fold})
    return jobs


def require_single_image(source: Any) -> Any:
    """Reject list/tuple sources so neighbour images can never affect shape."""
    if isinstance(source, (list, tuple)):
        raise TypeError("deployment-consistent inference accepts exactly one image")
    return source


def action_predict_kwargs(protocol: Mapping[str, Any], action: str) -> dict[str, Any]:
    validate_protocol(protocol)
    if action not in ACTIONS:
        raise ValueError(action)
    inference = protocol["inference"]
    if action == "edge":
        half, scaleup = bool(inference["edge_half"]), False
    elif action in {"original", "q90"}:
        half, scaleup = bool(inference["full_half"]), False
    else:
        half, scaleup = bool(inference["roi_half"]), True
    return {
        "imgsz": int(inference["imgsz"]),
        "conf": float(inference["conf"]),
        "iou": float(inference["iou"]),
        "max_det": int(inference["max_det"]),
        "device": inference["device"],
        "rect": True,
        "end2end": False,
        "half": half,
        "scaleup": scaleup,
        "verbose": False,
    }


def role_manifest(canonical_root: str | Path, role: str) -> Path:
    if role not in ROLES:
        raise ValueError(role)
    return Path(canonical_root) / f"common/coco/{role}/source_manifest.json"


def validate_population(values: Sequence[int], *, role: str, expected: int | None = None) -> None:
    count = ROLES[role] if expected is None else int(expected)
    ids = list(map(int, values))
    if len(ids) != count or len(ids) != len(set(ids)):
        raise RuntimeError(f"role population changed: {role} {len(ids)} != {count}")

