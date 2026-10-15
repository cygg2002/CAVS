#!/usr/bin/env python3
"""Stage runner for the frozen VOC cross-architecture paper experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
DEFAULT_GLUON_PYTHON = Path("/home/111_wcy/anaconda3/envs/voc-gluoncv/bin/python")


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def run_stage(stage: str, gluon_python: Path) -> None:
    if stage == "tests":
        _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    elif stage == "data":
        _run([sys.executable, "scripts/prepare_data.py"])
    elif stage == "weights":
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = f"{gluon_python.parent.parent / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
        _run([str(gluon_python), "scripts/download_weights.py"], environment=environment)
    elif stage == "actions":
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = f"{gluon_python.parent.parent / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
        environment["MXNET_CUDNN_LIB_CHECKING"] = "0"
        environment["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
        _run([str(gluon_python), "scripts/cache_actions.py", "--role", "all", "--action", "all"], environment=environment)
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
        "weights",
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
    parser.add_argument("--gluon-python", default=str(DEFAULT_GLUON_PYTHON))
    args = parser.parse_args()
    selected = stages if args.stage == "all" else (args.stage,)
    for stage in selected:
        print(json.dumps({"stage": stage, "status": "starting"}), flush=True)
        run_stage(stage, Path(args.gluon_python))
        print(json.dumps({"stage": stage, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
