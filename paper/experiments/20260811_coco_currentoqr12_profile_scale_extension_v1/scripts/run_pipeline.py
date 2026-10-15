#!/usr/bin/env python3
"""Resumable orchestration for the isolated COCO OQR12 extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]


def command(script: str, *args: str) -> dict[str, Any]:
    path = ROOT / "scripts" / script
    process = subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO,
        text=True,
        check=True,
    )
    return {"complete": True, "script": script, "arguments": list(args), "returncode": process.returncode}


def preflight() -> dict[str, Any]:
    return command("cache_candidates.py", "--stage", "preflight")


def geometry() -> dict[str, Any]:
    return command("cache_candidates.py", "--stage", "geometry")


def cache() -> dict[str, Any]:
    return command("cache_candidates.py", "--stage", "inference")


def common() -> dict[str, Any]:
    return command("build_dataset.py", "--role", "all")


def rewards() -> dict[str, Any]:
    return command("build_rewards.py", "--role", "all")


def learners() -> dict[str, Any]:
    return command("run_experiment.py", "--stage", "all")


def run_all() -> dict[str, Any]:
    output = {}
    for name, function in (
        ("preflight", preflight),
        ("geometry", geometry),
        ("cache", cache),
        ("common", common),
        ("rewards", rewards),
        ("learners", learners),
    ):
        print(f"[stage] {name}", flush=True)
        output[name] = function()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "geometry", "cache", "common", "rewards", "learners", "all"),
        default="all",
    )
    args = parser.parse_args()
    result = {
        "preflight": preflight,
        "geometry": geometry,
        "cache": cache,
        "common": common,
        "rewards": rewards,
        "learners": learners,
        "all": run_all,
    }[args.stage]()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
