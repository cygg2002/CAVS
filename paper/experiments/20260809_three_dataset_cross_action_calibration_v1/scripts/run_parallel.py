#!/usr/bin/env python3
"""Launch the three independent CPU diagnostics concurrently and resumably."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import csv


sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_core import evaluate_gate, validate_protocol  # noqa: E402


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
DATASETS = ("coco", "ena24", "oregon")
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
validate_protocol(PROTOCOL)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def is_complete(dataset: str) -> bool:
    path = RUNS / dataset / "COMPLETED.json"
    return path.is_file() and bool(json.loads(path.read_text(encoding="utf-8")).get("complete"))


def write_aggregate() -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        with (RUNS / dataset / "evaluation/results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows.extend(csv.DictReader(handle))
    gate = evaluate_gate(rows, PROTOCOL)
    output = RUNS / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = output / f".results.csv.{os.getpid()}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output / "results.csv")
    atomic_json(output / "gate.json", gate)
    atomic_json(
        RUNS / "COMPLETED.json",
        {
            "complete": True,
            "datasets": list(DATASETS),
            "rows": len(rows),
            "gate_passed": bool(gate["passed"]),
            "final_accessed": False,
        },
    )
    return gate


def main() -> None:
    launch = RUNS / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    processes: dict[str, tuple[subprocess.Popen[bytes], object]] = {}
    state: dict[str, object] = {"launcher_pid": os.getpid(), "datasets": {}}
    for dataset in DATASETS:
        if is_complete(dataset):
            state["datasets"][dataset] = {"status": "complete", "cached": True}
            continue
        log = (launch / f"{dataset}.log").open("ab", buffering=0)
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/run_diagnostic.py"), "--dataset", dataset, "--stage", "all"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes[dataset] = (process, log)
        state["datasets"][dataset] = {"status": "running", "pid": process.pid, "cached": False}
    atomic_json(launch / "status.json", state)
    failed = False
    while processes:
        time.sleep(2.0)
        for dataset, (process, log) in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            state["datasets"][dataset] = {
                "status": "complete" if code == 0 and is_complete(dataset) else "failed",
                "exit_code": int(code),
                "cached": False,
            }
            failed |= state["datasets"][dataset]["status"] == "failed"
            del processes[dataset]
        atomic_json(launch / "status.json", state)
    state["complete"] = not failed
    if not failed:
        state["gate"] = write_aggregate()
    atomic_json(launch / "status.json", state)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
