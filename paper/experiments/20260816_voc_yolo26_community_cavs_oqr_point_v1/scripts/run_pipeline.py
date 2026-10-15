#!/usr/bin/env python3
"""Run the resumable VOC point benchmark from its first incomplete stage."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def wait_for_archives() -> None:
    root = Path(PROTOCOL["data"]["archive_root"])
    expected = {str(key): int(value) for key, value in PROTOCOL["data"]["archive_sizes"].items()}
    while True:
        sizes = {name: (root / name).stat().st_size if (root / name).is_file() else 0 for name in expected}
        oversized = {name: size for name, size in sizes.items() if size > expected[name]}
        if oversized:
            raise RuntimeError(f"VOC archive exceeds pinned size: {oversized}")
        if sizes == expected:
            print("[pipeline] all VOC archives reached their pinned sizes", flush=True)
            return
        pending = ", ".join(f"{name}={sizes[name]}/{expected[name]}" for name in expected if sizes[name] != expected[name])
        print(f"[pipeline] waiting for archives: {pending}", flush=True)
        time.sleep(30)


def run_stage(name: str, *arguments: str) -> None:
    print(f"[pipeline] start {name}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / name), *arguments], cwd=REPO, check=True)
    print(f"[pipeline] complete {name}", flush=True)


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    lock_path = RUNS / "pipeline.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another VOC pipeline is already running") from error
        lock.write(str(os.getpid()))
        lock.flush()
        wait_for_archives()
        run_stage("prepare_data.py")
        run_stage("cache_actions.py", "--stage", "all")
        run_stage("build_common.py")
        run_stage("build_rewards.py")
        run_stage("train_learners.py")
        run_stage("evaluate_point.py")
        run_stage("write_report.py")
        report = {
            "complete": True,
            "pid": os.getpid(),
            "bootstrap_performed": False,
            "final_used_for_training_or_selection": False,
        }
        (RUNS / "completion.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
