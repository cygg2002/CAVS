#!/usr/bin/env python3
"""Launch the independent COCO and Oregon CPU pipelines."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/launch"
PYTHON = sys.executable


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    processes = []
    for dataset in ("coco", "oregon"):
        log = (RUNS / f"{dataset}.log").open("a", encoding="utf-8")
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        process = subprocess.Popen(
            [PYTHON, str(ROOT / "scripts/run_benchmark.py"), "--dataset", dataset, "--stage", "all"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        (RUNS / f"{dataset}.pid").write_text(f"{process.pid}\n")
        processes.append((dataset, process, log))
    status = {}
    for dataset, process, log in processes:
        code = process.wait()
        log.close()
        status[dataset] = {"pid": process.pid, "exit_code": code, "complete": code == 0}
    (RUNS / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    if not all(item["complete"] for item in status.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

