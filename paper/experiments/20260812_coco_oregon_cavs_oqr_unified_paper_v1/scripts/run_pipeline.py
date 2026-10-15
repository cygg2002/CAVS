#!/usr/bin/env python3
"""Run both datasets sequentially while each stage parallelizes internally."""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNNER = ROOT / "scripts/run_experiment.py"


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-u", str(RUNNER), *arguments], check=True)


def main() -> None:
    for dataset in ("coco", "oregon"):
        run("--dataset", dataset, "--stage", "all")
    run("--stage", "report")


if __name__ == "__main__":
    main()
