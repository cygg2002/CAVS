#!/usr/bin/env python3
"""Bounded parallel launcher; never touches the running OQR12 experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/launch"
PYTHON = sys.executable
CONTROL_PYTHON = os.environ.get("CAVS_CONTROL_PYTHON", "/home/111_wcy/anaconda3/envs/cameratrapai/bin/python")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")


def launch(label: str, args: list[str]) -> tuple[str, subprocess.Popen, object]:
    RUNS.mkdir(parents=True, exist_ok=True)
    log = (RUNS / f"{label}.log").open("a", encoding="utf-8")
    process = subprocess.Popen([PYTHON, str(ROOT / "scripts/run_completeness.py"), *args], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=ENV)
    (RUNS / f"{label}.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return label, process, log


def wait_group(items: list[tuple[str, subprocess.Popen, object]], status: dict[str, object]) -> None:
    failed = []
    for label, process, log in items:
        code = process.wait()
        log.close()
        status[label] = {"pid": process.pid, "exit_code": code, "complete": code == 0}
        if code:
            failed.append(label)
    (RUNS / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(1)


def main() -> None:
    status: dict[str, object] = {"started_at": time.time(), "isolated_from": "20260811_coco_currentoqr12_profile_scale_extension_v1"}
    export_log = (RUNS / "export_oregon_controls.log").open("a", encoding="utf-8")
    export = subprocess.run([CONTROL_PYTHON, str(ROOT / "scripts/export_oregon_controls.py")], stdout=export_log, stderr=subprocess.STDOUT, env=ENV)
    export_log.close()
    status["export_oregon_controls"] = {"exit_code": export.returncode, "complete": export.returncode == 0}
    if export.returncode:
        (RUNS / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(export.returncode)
    # Bootstrap is the largest CPU phase. Two datasets run concurrently, each
    # with a bounded 12-process pool.
    if os.environ.get("CAVS_REUSE_RUNNING_OREGON") == "1":
        wait_group([launch("bootstrap_coco", ["--stage", "bootstrap", "--dataset", "coco"])], status)
        oregon_report = ROOT / "runs/bootstrap/oregon/report.json"
        while not oregon_report.is_file():
            running = subprocess.run(
                ["pgrep", "-f", "run_completeness.py --stage bootstrap --dataset oregon"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            if not running:
                status["bootstrap_oregon"] = {"complete": False, "error": "running Oregon bootstrap exited without report"}
                (RUNS / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                raise SystemExit(1)
            time.sleep(30)
        status["bootstrap_oregon"] = {"complete": True, "reused_running_process": True}
    else:
        wait_group([
            launch("bootstrap_coco", ["--stage", "bootstrap", "--dataset", "coco"]),
            launch("bootstrap_oregon", ["--stage", "bootstrap", "--dataset", "oregon"]),
        ], status)
    # Action masks and grouped/solver summaries are cache-only; run the two
    # dataset copies concurrently, then fit the isolated Oregon DCSB baseline.
    wait_group([
        launch("masks_coco", ["--stage", "masks", "--dataset", "coco"]),
        launch("masks_oregon", ["--stage", "masks", "--dataset", "oregon"]),
        launch("groups_coco", ["--stage", "groups", "--dataset", "coco"]),
        launch("groups_oregon", ["--stage", "groups", "--dataset", "oregon"]),
    ], status)
    wait_group([launch("dcsb_oregon", ["--stage", "dcsb"])], status)
    status["finished_at"] = time.time()
    (RUNS / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
