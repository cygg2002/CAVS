#!/usr/bin/env python3
"""Wait without load, then run formal action caches one at a time at the GPU gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
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


ACTIONS = load_module("voc_gluoncv_wait_actions", ROOT / "scripts/cache_actions.py")


def wait_for_memory(poll_seconds: int) -> int:
    minimum = int(PROTOCOL["inference"]["minimum_free_gpu_mib"])
    while True:
        free = ACTIONS.free_gpu_mib()
        if free >= minimum:
            print(
                json.dumps({"status": "gpu_gate_open", "free_mib": free, "minimum_mib": minimum}),
                flush=True,
            )
            return free
        print(
            json.dumps({"status": "waiting_for_gpu", "free_mib": free, "minimum_mib": minimum}),
            flush=True,
        )
        time.sleep(max(10, int(poll_seconds)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gluon-python",
        default="/home/111_wcy/anaconda3/envs/voc-gluoncv/bin/python",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--continue-paper-pipeline",
        action="store_true",
        help="After action caches pass, run common/reward/train/point/bootstrap/report.",
    )
    args = parser.parse_args()
    gluon_python = Path(args.gluon_python)
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = (
        f"{gluon_python.parent.parent / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    )
    environment["MXNET_CUDNN_LIB_CHECKING"] = "0"
    environment["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    ordered = [
        (role, action)
        for role in ("final", "fit", "calibration")
        for action in ("edge", "original", "q90")
    ]
    for role, action in ordered:
        completion = RUNS / f"actions/{role}/{action}/completion.json"
        if completion.is_file():
            print(json.dumps({"status": "cached", "role": role, "action": action}), flush=True)
            continue
        wait_for_memory(int(args.poll_seconds))
        subprocess.run(
            [
                str(gluon_python),
                "scripts/cache_actions.py",
                "--role",
                role,
                "--action",
                action,
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        if role == "final" and action == "q90":
            subprocess.run(
                [sys.executable, "scripts/evaluate_endpoints.py"],
                cwd=ROOT,
                check=True,
            )
    # All artifacts are now complete; this pass only validates hashes/populations
    # and writes the aggregate completion contract.
    subprocess.run(
        [
            str(gluon_python),
            "scripts/cache_actions.py",
            "--role",
            "all",
            "--action",
            "all",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    if args.continue_paper_pipeline:
        for script in (
            "build_common.py",
            "build_rewards.py",
            "train_learners.py",
            "evaluate_point.py",
            "bootstrap.py",
            "write_report.py",
        ):
            print(json.dumps({"status": "starting_downstream", "script": script}), flush=True)
            subprocess.run(
                [sys.executable, f"scripts/{script}"],
                cwd=ROOT,
                check=True,
            )
    print(json.dumps({"complete": True, "formal_actions": len(ordered)}), flush=True)


if __name__ == "__main__":
    main()
