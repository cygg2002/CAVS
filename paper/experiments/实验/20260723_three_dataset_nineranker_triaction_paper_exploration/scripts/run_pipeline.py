#!/usr/bin/env python3
"""Run one dataset core and bootstrap as a resumable remote pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_stage(dataset: str, stage: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT.parent / "run_dataset.py"),
            "--dataset",
            dataset,
            "--stage",
            stage,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "ena24", "oregon"), required=True)
    args = parser.parse_args()
    status_path = EXP_ROOT / "runs" / args.dataset / "pipeline_status.json"
    try:
        write_json(status_path, {"dataset": args.dataset, "status": "core_running"})
        run_stage(args.dataset, "core")
        write_json(status_path, {"dataset": args.dataset, "status": "bootstrap_running"})
        run_stage(args.dataset, "bootstrap")
    except Exception as error:
        write_json(
            status_path,
            {
                "dataset": args.dataset,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    write_json(status_path, {"dataset": args.dataset, "status": "complete"})


if __name__ == "__main__":
    main()
