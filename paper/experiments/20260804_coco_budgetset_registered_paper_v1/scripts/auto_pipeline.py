#!/usr/bin/env python3
"""Wait for GPU memory, resume frozen inference, and run the gated paper pipeline."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ROLES = ("development", "audit", "registered_final")
EDGE_GPU_FREE_MIB = 4_000
CLOUD_GPU_FREE_MIB = 7_500
POLL_SECONDS = 60


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inference_complete(detector: str) -> bool:
    return all(
        (RUNS / "inference" / role / detector / "completion.json").is_file()
        for role in ROLES
    )


def gpu_free_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory values")
    return max(values)


def run_stage(stage: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_paper.py"), "--stage", stage],
        check=True,
    )


def wait_and_run_detector(detector: str, batch: int, minimum_gpu_free_mib: int) -> None:
    while not inference_complete(detector):
        try:
            free_mib = gpu_free_mib()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(f"[auto] GPU query failed: {error}", flush=True)
            time.sleep(POLL_SECONDS)
            continue
        if free_mib < int(minimum_gpu_free_mib):
            print(
                f"[auto] waiting for GPU: free={free_mib} MiB, "
                f"required={minimum_gpu_free_mib} MiB",
                flush=True,
            )
            time.sleep(POLL_SECONDS)
            continue
        print(f"[auto] starting {detector} inference with {free_mib} MiB free", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/infer_original.py"),
                "--role",
                "all",
                "--detector",
                detector,
                "--device",
                "0",
                "--batch",
                str(int(batch)),
            ],
            check=False,
        )
        if result.returncode and not inference_complete(detector):
            print(f"[auto] {detector} inference exited {result.returncode}; retrying", flush=True)
            time.sleep(POLL_SECONDS)


def main() -> None:
    lock_path = RUNS / "auto_pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another automatic paper pipeline is already running") from error
        wait_and_run_detector("cloud", 1, CLOUD_GPU_FREE_MIB)
        wait_and_run_detector("edge", 16, EDGE_GPU_FREE_MIB)
        for stage in ("rewards", "train", "audit"):
            run_stage(stage)
        audit_completion = read_json(RUNS / "audit/completion.json")
        if audit_completion.get("passed") is True:
            if not (RUNS / "registered_final/completion.json").is_file():
                run_stage("final")
        else:
            print("[auto] audit gate failed; registered final remains sealed", flush=True)


if __name__ == "__main__":
    main()
