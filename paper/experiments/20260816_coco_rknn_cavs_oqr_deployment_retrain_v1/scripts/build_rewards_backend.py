#!/usr/bin/env python3
"""Build frozen-fold contextual rewards for one RKNN backend."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import backend_runtime
import contracts


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reward_actions() -> tuple[str, ...]:
    return ("original", "q90", "current@640")


def role_root(root: str | Path, backend: str, role: str) -> Path:
    if role not in contracts.ROLES:
        raise ValueError(role)
    return backend_runtime.backend_runs(root, backend) / f"common/coco/{role}"


def reward_folds(role: str, samples: list[dict[str, Any]], protocol: dict[str, Any]) -> list[list[int]]:
    expected = {int(sample["image_id"]) for sample in samples}
    if role == "calibration":
        return [sorted(expected)]
    if role != "fit":
        raise ValueError("contextual rewards are forbidden for final")
    folds = [
        list(map(int, fold))
        for fold in read_json(Path(protocol["frozen_folds"]))["folds"]
    ]
    observed = [image_id for fold in folds for image_id in fold]
    if len(folds) != int(protocol["fold_count"]):
        raise RuntimeError("frozen fold count changed")
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise RuntimeError("frozen fold population changed")
    return folds


def run_backend(backend: str, role: str) -> dict[str, Any]:
    adapted = backend_runtime.adapt_protocol(PROTOCOL, backend)
    runs = backend_runtime.backend_runs(ROOT, backend)
    source_root = Path(adapted["source_training_experiment"])
    source = load_module(
        f"rknn_reward_source_{backend.replace('-', '_')}",
        source_root / "scripts/build_rewards.py",
    )

    def load_role(value: str) -> dict[str, Any]:
        root = role_root(ROOT, backend, value)
        manifest = read_json(root / "source_manifest.json")
        if manifest.get("backend") != backend:
            raise RuntimeError(f"cross-backend common bundle: {backend}/{value}")
        if manifest.get("cloud_model_sha256") != adapted["cloud_model"]["sha256"]:
            raise RuntimeError(f"wrong cloud model in common role: {backend}/{value}")
        return {
            "root": root,
            "gt": root / "gt.json",
            "samples": read_json(root / "samples.json"),
            "actions": {
                action_id: source.group_predictions(Path(manifest["action_sources"][action_id]))
                for action_id in ("edge", *reward_actions())
            },
        }

    source.ROOT = ROOT
    source.RUNS = runs
    source.PROTOCOL = adapted
    source.load_role = load_role
    source.reward_actions = reward_actions
    source.reward_folds = lambda value, samples: reward_folds(value, samples, adapted)
    if role == "all":
        return source.run_all()
    return source.run_role(role)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=contracts.BACKENDS, required=True)
    parser.add_argument("--role", choices=("fit", "calibration", "all"), default="all")
    args = parser.parse_args()
    print(json.dumps(run_backend(args.backend, args.role), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
