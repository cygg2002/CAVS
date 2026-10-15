#!/usr/bin/env python3
"""Run the isolated same-selection fusion ablation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_experiment.py"


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-u", str(RUNNER), *arguments], check=True)


def main() -> None:
    for dataset in ("coco", "oregon"):
        run("--dataset", dataset, "--stage", "all")
    run("--stage", "report")


if __name__ == "__main__":
    main()
