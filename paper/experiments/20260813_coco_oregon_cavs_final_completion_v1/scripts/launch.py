#!/usr/bin/env python3
"""Launch the resumable final-completion pipeline at low CPU priority."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
STATUS = ROOT / "runs/launcher/status.json"
LOG = ROOT / "runs/launcher/pipeline.log"


def read_status() -> dict:
    return json.loads(STATUS.read_text()) if STATUS.is_file() else {}


def alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    current = read_status()
    if current.get("pid") and alive(int(current["pid"])):
        print(json.dumps({**current, "already_running": True}, indent=2))
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with LOG.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--stage", "report"],
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            preexec_fn=lambda: os.nice(19),
        )
    status = {
        "pid": process.pid,
        "started_unix": time.time(),
        "nice": 19,
        "runtime_workers": environment.get("CAVS_RUNTIME_WORKERS"),
        "log": str(LOG),
        "command": [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--stage", "report"],
        "resumable": True,
    }
    atomic_json(STATUS, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
