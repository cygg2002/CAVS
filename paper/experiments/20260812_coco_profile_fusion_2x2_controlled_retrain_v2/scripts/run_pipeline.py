#!/usr/bin/env python3
"""Run the isolated controlled 2×2 pipeline without touching active work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def run(script: str, *arguments: str) -> None:
    nice = int(PROTOCOL["cpu"]["nice"])
    subprocess.run(
        ["nice", "-n", str(nice), sys.executable, "-u", str(ROOT / "scripts" / script), *arguments],
        cwd=REPO,
        check=True,
    )


def main() -> None:
    run("build_rewards.py", "--role", "all")
    run("run_experiment.py", "--stage", "preflight")
    run("run_experiment.py", "--stage", "gate")
    run("run_experiment.py", "--stage", "all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-now",
        action="store_true",
        help="accepted for launch compatibility; this isolated pipeline starts immediately",
    )
    parser.parse_args()
    main()
