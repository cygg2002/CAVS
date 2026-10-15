#!/usr/bin/env python3
"""Wait for the immutable source bootstrap, then start the extension."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
SOURCE_COMPLETION = (
    REPO
    / "paper/experiments/20260804_coco_dualhead_canonical9678_val2017_paper_v1"
    / "runs/registered_final/completion.json"
)
OUTPUT_COMPLETION = ROOT / "runs/COMPLETED.json"
STATUS = ROOT / "runs/queue_status.json"
RUNNER = ROOT / "scripts/run_fullaction.py"


def atomic_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_name(f".{STATUS.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATUS)


def source_complete() -> bool:
    if not SOURCE_COMPLETION.is_file():
        return False
    try:
        return json.loads(SOURCE_COMPLETION.read_text(encoding="utf-8")).get(
            "complete"
        ) is True
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("Polling faster than 10 seconds is unnecessary")
    while not source_complete():
        atomic_status(
            {
                "state": "waiting_for_source_bootstrap",
                "pid": os.getpid(),
                "source_completion": str(SOURCE_COMPLETION),
                "output_completion": str(OUTPUT_COMPLETION),
            }
        )
        print("[queue] waiting for immutable source bootstrap", flush=True)
        time.sleep(args.poll_seconds)
    if OUTPUT_COMPLETION.is_file():
        atomic_status({"state": "already_complete", "pid": os.getpid()})
        return
    atomic_status(
        {
            "state": "running_extension",
            "pid": os.getpid(),
            "source_completion": str(SOURCE_COMPLETION),
        }
    )
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--stage", "all"],
        cwd=REPO,
        check=False,
    )
    atomic_status(
        {
            "state": "complete" if result.returncode == 0 else "failed",
            "pid": os.getpid(),
            "returncode": result.returncode,
            "output_completion_exists": OUTPUT_COMPLETION.is_file(),
        }
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
