#!/usr/bin/env python3
"""Resumable end-to-end Cloud-V3 Oregon experiment orchestration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PYTHON = sys.executable


STAGES = (
    ("cloud_cache", "cache_actions.py", ["--stage", "all"]),
    ("common", "build_common.py", ["--role", "all"]),
    ("rewards", "build_rewards.py", ["--role", "all"]),
    ("learners", "train_learners.py", ["--stage", "all"]),
    ("evaluation", "evaluate.py", ["--stage", "all"]),
    ("bootstrap", "bootstrap.py", ["--stage", "all"]),
    ("report", "report.py", []),
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def complete(stage: str) -> bool:
    if stage == "evaluation":
        return (
            RUNS / "evaluation/point/completion.json"
        ).is_file() and (RUNS / "evaluation/references/completion.json").is_file()
    return (RUNS / stage / "completion.json").is_file()


def wait_for_existing_cloud_launcher() -> None:
    if complete("cloud_cache"):
        return
    pid_path = RUNS / "cloud_cache_launcher.pid"
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        return
    if pid == os.getpid():
        return
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            break
        print(
            f"[pipeline] existing Cloud-V3 cache PID {pid} is active; waiting for handoff",
            flush=True,
        )
        time.sleep(45)
    if not complete("cloud_cache"):
        print("[pipeline] prior cache launcher exited incomplete; resuming it here", flush=True)


def run_stage(stage: str, script_name: str, arguments: list[str]) -> dict[str, Any]:
    if complete(stage):
        print(f"[pipeline] skip complete stage: {stage}", flush=True)
        return {"stage": stage, "status": "already_complete"}
    if stage == "cloud_cache":
        wait_for_existing_cloud_launcher()
        if complete(stage):
            return {"stage": stage, "status": "completed_by_existing_launcher"}
    command = [PYTHON, str(ROOT / "scripts" / script_name), *arguments]
    log_path = RUNS / f"logs/{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] start {stage}: {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[pipeline start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"stage {stage} failed with code {process.returncode}; see {log_path}"
        )
    if not complete(stage):
        raise RuntimeError(f"stage {stage} returned success without completion marker")
    print(f"[pipeline] complete {stage}", flush=True)
    return {"stage": stage, "status": "complete", "log": str(log_path)}


def run_all(start_stage: str | None = None) -> list[dict[str, Any]]:
    enabled = start_stage is None
    reports = []
    for stage, script, arguments in STAGES:
        if stage == start_stage:
            enabled = True
        if not enabled:
            continue
        reports.append(run_stage(stage, script, arguments))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-stage", choices=[stage for stage, _, _ in STAGES])
    args = parser.parse_args()
    print(json.dumps(run_all(args.start_stage), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
