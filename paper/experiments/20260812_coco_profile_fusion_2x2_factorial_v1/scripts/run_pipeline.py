#!/usr/bin/env python3
"""Wait for active work, then run the isolated 2×2 factorial pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("complete") is True
    except (json.JSONDecodeError, OSError):
        return False


def wait_for_active_experiments() -> None:
    paths = [REPO / value for value in PROTOCOL["wait_for_completion"]]
    interval = int(PROTOCOL["wait"]["poll_seconds"])
    announced = False
    while not all(complete(path) for path in paths):
        if not announced:
            pending = [str(path) for path in paths if not complete(path)]
            print("[wait] active source experiments are still running:", flush=True)
            for path in pending:
                print(f"[wait] {path}", flush=True)
            announced = True
        time.sleep(interval)
    print("[wait] active experiments completed; factorial pipeline may start", flush=True)


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / script), *arguments],
        cwd=REPO,
        check=True,
    )


def main(*, start_now: bool = False) -> None:
    if start_now:
        print("[start-now] prerequisite wait explicitly bypassed by user", flush=True)
    else:
        wait_for_active_experiments()
    run("build_actions.py", "--roles", "prefinal")
    run("build_rewards.py", "--role", "all")
    run("run_experiment.py", "--stage", "gate")
    gate = json.loads((ROOT / "runs/gate/report.json").read_text(encoding="utf-8"))
    if not gate["gate_passed"]:
        run("run_experiment.py", "--stage", "report")
        return
    run("build_actions.py", "--roles", "final")
    run("run_experiment.py", "--stage", "all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-now",
        action="store_true",
        help="bypass the conservative prerequisite wait and use currently idle CPUs",
    )
    main(start_now=parser.parse_args().start_now)
