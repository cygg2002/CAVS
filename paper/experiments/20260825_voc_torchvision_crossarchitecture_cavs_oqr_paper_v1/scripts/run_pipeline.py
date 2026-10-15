#!/usr/bin/env python3
"""Stage runner for the frozen VOC cross-architecture paper experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def run_stage(stage: str) -> None:
    if stage == "tests":
        _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    elif stage == "data":
        _run([sys.executable, "scripts/prepare_data.py"])
    elif stage == "actions":
        _run([sys.executable, "scripts/cache_actions.py", "--role", "all"])
    elif stage == "endpoint":
        _run([sys.executable, "scripts/evaluate_endpoints.py"])
    elif stage == "common":
        _run([sys.executable, "scripts/build_common.py"])
    elif stage == "rewards":
        _run([sys.executable, "scripts/build_rewards.py"])
    elif stage == "train":
        _run([sys.executable, "scripts/train_learners.py"])
    elif stage == "point":
        _run([sys.executable, "scripts/evaluate_point.py"])
    elif stage == "bootstrap":
        _run([sys.executable, "scripts/bootstrap.py"])
    elif stage == "report":
        _run([sys.executable, "scripts/write_report.py"])
    else:
        raise ValueError(stage)


def main() -> None:
    stages = (
        "tests",
        "data",
        "actions",
        "endpoint",
        "common",
        "rewards",
        "train",
        "point",
        "bootstrap",
        "report",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*stages, "all"), default="all")
    args = parser.parse_args()
    selected = stages if args.stage == "all" else (args.stage,)
    for stage in selected:
        print(json.dumps({"stage": stage, "status": "starting"}), flush=True)
        run_stage(stage)
        print(json.dumps({"stage": stage, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
