#!/usr/bin/env python3
"""Launch the resumable ORIC/MORIC AC-OQR pipeline at low priority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def foreground() -> None:
    runner = ROOT / "scripts/run_experiment.py"
    for stage in ("preflight", "train", "evaluate", "bootstrap", "report"):
        subprocess.run([sys.executable, str(runner), "--stage", stage], check=True)


def start_now() -> None:
    launcher = RUNS / "launcher"
    launcher.mkdir(parents=True, exist_ok=True)
    log_path = launcher / "pipeline.log"
    command = [
        "nice",
        "-n",
        str(int(PROTOCOL["cpu"]["nice"])),
        sys.executable,
        str(SCRIPT),
        "--foreground",
    ]
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        )
    (launcher / "pid").write_text(f"{process.pid}\n", encoding="ascii")
    (launcher / "launch.json").write_text(
        json.dumps(
            {
                "pid": process.pid,
                "command": command,
                "log": str(log_path),
                "runtime_domain": PROTOCOL["runtime_domain"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pid": process.pid, "log": str(log_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start-now", action="store_true")
    group.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    start_now() if args.start_now else foreground()


if __name__ == "__main__":
    main()
