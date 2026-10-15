#!/usr/bin/env python3
"""Build frozen-fold contextual AP50-95 rewards for COCO O/Q/R12."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE_ROOT = Path(PROTOCOL["source_training_experiment"])
SOURCE = load_module(
    "coco_oqr12_reward_source", SOURCE_ROOT / "scripts/build_rewards.py"
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_role(role: str) -> dict[str, Any]:
    root = RUNS / f"common/coco/{role}"
    manifest = read_json(root / "source_manifest.json")
    if manifest.get("cloud_model_sha256") != PROTOCOL["cloud_model"]["sha256"]:
        raise RuntimeError(f"wrong cloud cache in COCO role: {role}")
    return {
        "root": root,
        "gt": root / "gt.json",
        "samples": read_json(root / "samples.json"),
        "actions": {
            action_id: SOURCE.group_predictions(Path(path))
            for action_id, path in manifest["action_sources"].items()
        },
    }


def reward_folds(role: str, samples: list[dict[str, Any]]) -> list[list[int]]:
    expected = {int(sample["image_id"]) for sample in samples}
    if role == "calibration":
        return [sorted(expected)]
    if role != "fit":
        raise ValueError(role)
    folds = [
        list(map(int, fold))
        for fold in read_json(Path(PROTOCOL["frozen_folds"]))["folds"]
    ]
    observed = [image_id for fold in folds for image_id in fold]
    if len(folds) != int(PROTOCOL["fold_count"]):
        raise RuntimeError("frozen fold count changed")
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise RuntimeError("frozen fold population changed")
    return folds


def bind_source() -> None:
    SOURCE.ROOT = ROOT
    SOURCE.RUNS = RUNS
    SOURCE.PROTOCOL = PROTOCOL
    SOURCE.load_role = load_role
    SOURCE.reward_folds = reward_folds


bind_source()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("fit", "calibration", "all"), default="all")
    args = parser.parse_args()
    result = SOURCE.run_all() if args.role == "all" else SOURCE.run_role(args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
