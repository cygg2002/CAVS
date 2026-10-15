#!/usr/bin/env python3
"""Run the frozen 32-case causal-online main matrix serially on RK3588."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from main_matrix import file_sha256, gate_run, validate_main_matrix


SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _health_url(cloud_url: str) -> str:
    parsed = urlsplit(cloud_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("cloud URL must be HTTP(S)")
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def check_cloud(cloud_url: str, token: str, expected_sha256: str) -> dict[str, Any]:
    request = Request(
        _health_url(cloud_url),
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urlopen(request, timeout=15) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping) or value.get("status") != "ok":
        raise RuntimeError("cloud health check is not ok")
    if str(value.get("model_sha256", "")).lower() != expected_sha256.lower():
        raise RuntimeError("cloud health model hash differs from the frozen matrix")
    return dict(value)


def _command(
    *,
    bundle: Path,
    row: Mapping[str, Any],
    run_id: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        "bash",
        str(bundle / "run_online.sh"),
        "--run-id",
        run_id,
        "--method",
        str(row["method"]),
        "--budget",
        f"{int(row['budget_numerator'])}/{int(row['budget_denominator'])}",
        "--sequence",
        str(bundle / "sequences/coco_val2017_sequence.json"),
        "--images-dir",
        str(args.images_dir),
        "--policy",
        str(bundle / str(row["policy"])),
        "--control",
        str(bundle / str(row["control"])),
        "--rknn",
        str(args.rknn),
        "--rknn-helpers-dir",
        str(args.rknn_helpers_dir),
        "--cloud-url",
        args.cloud_url,
        "--cloud-sha256",
        args.cloud_sha256,
        "--token-env",
        args.token_env,
        "--runs-root",
        str(args.runs_root),
        "--core",
        args.core,
        "--progress-every",
        str(args.progress_every),
    ]
    if row.get("random_seed") is not None:
        command.extend(("--random-seed", str(int(row["random_seed"]))))
    return command


def _system_snapshot() -> dict[str, Any]:
    temperatures = {}
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            temperatures[path.parent.name] = int(path.read_text().strip())
        except (OSError, ValueError):
            pass
    return {
        "timestamp": _now(),
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "loadavg": list(os.getloadavg()),
        "temperatures_millicelsius": temperatures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--rknn-helpers-dir", type=Path, required=True)
    parser.add_argument("--cloud-url", required=True)
    parser.add_argument("--cloud-sha256", required=True)
    parser.add_argument("--token-env", default="CAVS_TOKEN")
    parser.add_argument("--core", choices=("auto", "0", "1", "2", "all"), default="0")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not SAFE_TAG.fullmatch(args.tag):
        raise ValueError("tag contains unsafe characters")
    if args.max_attempts <= 0 or args.progress_every <= 0:
        raise ValueError("attempt and progress counts must be positive")
    bundle = args.bundle.resolve()
    matrix_path = args.matrix.resolve()
    rows = validate_main_matrix(matrix_path, bundle, args.protocol)
    for field in ("images_dir", "rknn", "rknn_helpers_dir"):
        if not Path(getattr(args, field)).exists():
            raise FileNotFoundError(getattr(args, field))
    token = os.environ.get(args.token_env, "")
    if not token:
        raise RuntimeError(f"missing cloud token environment variable: {args.token_env}")
    health = check_cloud(args.cloud_url, token, args.cloud_sha256)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    state = args.runs_root / f"_{args.tag}_main32_state"
    if state.exists():
        raise FileExistsError(f"formal state already exists; use a new tag: {state}")
    state.mkdir(parents=True)
    shutil.copy2(matrix_path, state / "RUN_MATRIX.json")
    matrix_sha256 = file_sha256(matrix_path)
    _atomic_json(
        state / "execution_contract.json",
        {
            "tag": args.tag,
            "matrix_sha256": matrix_sha256,
            "main_runs": 32,
            "cloud_url": args.cloud_url,
            "cloud_sha256": args.cloud_sha256,
            "cloud_health": health,
            "started_at": _now(),
            "system_start": _system_snapshot(),
            "evaluation_during_execution": False,
        },
    )
    progress: dict[str, Any] = {
        "tag": args.tag,
        "matrix_sha256": matrix_sha256,
        "completed": [],
        "failed_attempts": [],
        "next_run_index": 1,
        "updated_at": _now(),
    }
    _atomic_json(state / "progress.json", progress)

    environment = dict(os.environ)
    environment["ONLINE_PYTHON"] = sys.executable
    for row in rows:
        index = int(row["run_index"])
        completed = None
        for attempt in range(1, args.max_attempts + 1):
            check_cloud(args.cloud_url, token, args.cloud_sha256)
            run_id = f"{args.tag}-r{index:02d}-{row['run_key']}-a{attempt}"
            run_dir = args.runs_root / run_id
            if run_dir.exists():
                raise FileExistsError(run_dir)
            log = state / f"r{index:02d}-{row['run_key']}-a{attempt}.log"
            print(f"START index={index}/32 attempt={attempt} run_id={run_id} time={_now()}", flush=True)
            with log.open("wb") as handle:
                result = subprocess.run(
                    _command(bundle=bundle, row=row, run_id=run_id, args=args),
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            error = None
            summary = None
            if result.returncode == 0:
                try:
                    summary = gate_run(
                        run_dir / "frames.jsonl",
                        run_dir / "run_contract.json",
                        row,
                    )
                except Exception as exc:  # retained as formal failed-attempt evidence
                    error = f"{type(exc).__name__}: {exc}"
            else:
                error = f"runner exit code {result.returncode}"
            if summary is not None:
                completed = {
                    "run_index": index,
                    "run_key": row["run_key"],
                    "run_id": run_id,
                    "attempt": attempt,
                    "summary": summary,
                    "completed_at": _now(),
                    "system_after": _system_snapshot(),
                }
                progress["completed"].append(completed)
                progress["next_run_index"] = index + 1
                progress["updated_at"] = _now()
                _atomic_json(state / f"r{index:02d}_PASS.json", completed)
                _atomic_json(state / "progress.json", progress)
                print(f"PASS index={index}/32 run_id={run_id} time={_now()}", flush=True)
                break
            failure = {
                "run_index": index,
                "run_key": row["run_key"],
                "run_id": run_id,
                "attempt": attempt,
                "error": error,
                "failed_at": _now(),
            }
            progress["failed_attempts"].append(failure)
            progress["updated_at"] = _now()
            _atomic_json(state / f"r{index:02d}_a{attempt}_FAILED.json", failure)
            _atomic_json(state / "progress.json", progress)
            print(f"FAILED index={index}/32 attempt={attempt} error={error}", flush=True)
            time.sleep(5)
        if completed is None:
            raise RuntimeError(f"formal row {index} failed {args.max_attempts} attempts")

    completion = {
        "complete": True,
        "tag": args.tag,
        "matrix_sha256": matrix_sha256,
        "main_runs": 32,
        "failed_attempts": len(progress["failed_attempts"]),
        "completed_at": _now(),
        "system_end": _system_snapshot(),
        "evaluation_during_execution": False,
    }
    _atomic_json(state / "MAIN32_COMPLETE.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
