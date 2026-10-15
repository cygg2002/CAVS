#!/usr/bin/env python3
"""Generate one shared-context reward matrix for every factorial action."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
SOURCE_ROOT = REPO / PROTOCOL["source_current_experiment"]
V1_ROOT = REPO / PROTOCOL["source_factorial_v1"]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("factorial_reward_contract", ROOT / "scripts/core.py")
CORE.validate_protocol(PROTOCOL)
SOURCE = load_module(
    "factorial_reward_source",
    REPO
    / "paper/experiments/20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1/scripts/build_rewards.py",
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_role(role: str) -> dict[str, Any]:
    if role not in ("fit", "calibration"):
        raise ValueError("rewards are forbidden for final")
    root = SOURCE_ROOT / f"runs/common/coco/{role}"
    manifest = read_json(root / "source_manifest.json")
    if manifest.get("cloud_model_sha256") != PROTOCOL["cloud_model_sha256"]:
        raise RuntimeError(f"wrong cloud cache: {role}")
    samples = read_json(root / "samples.json")
    if len(samples) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"role population changed: {role}")
    actions = {
        "edge": SOURCE.group_predictions(Path(manifest["action_sources"]["edge"])),
        "original": SOURCE.group_predictions(Path(manifest["action_sources"]["original"])),
        "q90": SOURCE.group_predictions(Path(manifest["action_sources"]["q90"])),
    }
    for profile in PROTOCOL["profiles"]:
        canonical = f"{profile}@640"
        actions[f"{canonical}__reversible"] = SOURCE.group_predictions(
            Path(manifest["action_sources"][canonical])
        )
        actions[f"{canonical}__maur"] = SOURCE.group_predictions(
            V1_ROOT / f"runs/actions/{role}/{profile}__640_maur.json"
        )
    return {
        "root": root,
        "gt": root / "gt.json",
        "samples": samples,
        "actions": actions,
    }


def factorial_reward_actions() -> tuple[str, ...]:
    return (
        "original",
        "q90",
        *(
            f"{profile}@640__{fusion.lower()}"
            for profile in PROTOCOL["profiles"]
            for fusion in PROTOCOL["fusions"]
        ),
    )


def reward_folds(role: str, samples: list[dict[str, Any]]) -> list[list[int]]:
    expected = {int(sample["image_id"]) for sample in samples}
    if role == "calibration":
        return [sorted(expected)]
    folds = [
        list(map(int, fold))
        for fold in read_json(SOURCE_ROOT / "runs/rewards/fit/folds.json")["folds"]
    ]
    observed = [image_id for fold in folds for image_id in fold]
    if len(folds) != int(PROTOCOL["fold_count"]):
        raise RuntimeError("fold count changed")
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise RuntimeError("fold population changed")
    return folds


def bind_source() -> None:
    adapted = dict(PROTOCOL)
    adapted["cloud_model"] = {"sha256": PROTOCOL["cloud_model_sha256"]}
    adapted["reward"] = dict(PROTOCOL["reward"])
    adapted["reward"]["workers"] = int(PROTOCOL["cpu"]["reward_workers"])
    SOURCE.ROOT = ROOT
    SOURCE.RUNS = RUNS
    SOURCE.PROTOCOL = adapted
    SOURCE.load_role = load_role
    SOURCE.reward_actions = factorial_reward_actions
    SOURCE.reward_folds = reward_folds


bind_source()


def validate_shared_context_artifacts(role: str) -> None:
    """Require the one context manifest consumed by every action shard."""
    contexts = RUNS / f"rewards/{role}/contexts.json"
    reward_map = RUNS / f"rewards/{role}/reward_map.pkl"
    if not contexts.is_file() or not reward_map.is_file():
        raise FileNotFoundError(f"missing shared reward artifacts: {role}")
    expected = int(PROTOCOL["roles"][role])
    if len(read_json(contexts)) != expected:
        raise RuntimeError(f"shared context population changed: {role}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("fit", "calibration", "all"), default="all")
    args = parser.parse_args()
    if args.role == "all":
        roles = {role: SOURCE.run_role(role) for role in ("fit", "calibration")}
        for role in roles:
            validate_shared_context_artifacts(role)
        result = {
            "complete": True,
            "roles": {role: int(report["tasks"]) for role, report in roles.items()},
            "final_accessed": False,
            "cloud_model_sha256": PROTOCOL["cloud_model_sha256"],
        }
        SOURCE.atomic_json(RUNS / "rewards/report.json", result)
        SOURCE.atomic_json(RUNS / "rewards/completion.json", result)
    else:
        result = SOURCE.run_role(args.role)
        validate_shared_context_artifacts(args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
