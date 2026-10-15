#!/usr/bin/env python3
"""Run frozen rows 33..46 serially with root-managed Linux traffic shaping."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from main_matrix import file_sha256
from remaining_matrix import (
    gate_remaining_run,
    network_commands,
    validate_remaining_matrix,
)
from run_main_matrix import _command, check_cloud


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


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _clear_network(interface: str) -> dict[str, Any]:
    result = _run(["tc", "qdisc", "del", "dev", interface, "root"], check=False)
    return {
        "command": ["tc", "qdisc", "del", "dev", interface, "root"],
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _network_snapshot(interface: str) -> list[str]:
    result = _run(["tc", "-s", "qdisc", "show", "dev", interface])
    return [line for line in result.stdout.splitlines() if line.strip()]


def _apply_network(
    interface: str,
    profile_name: str,
    profile: Mapping[str, object],
    cloud_host: str,
) -> dict[str, Any]:
    cleared = _clear_network(interface)
    applied = []
    for command in network_commands(interface, profile_name, profile):
        result = _run(command)
        applied.append(
            {
                "command": command,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
    ping = _run(["ping", "-c", "5", "-W", "3", cloud_host], check=False)
    return {
        "profile": profile_name,
        "intended": dict(profile),
        "interface": interface,
        "cleared_before_apply": cleared,
        "applied": applied,
        "qdisc": _network_snapshot(interface),
        "ping_returncode": ping.returncode,
        "ping_output": ping.stdout.strip(),
        "measured_at": _now(),
    }


def _demote(user: pwd.struct_passwd):
    def callback() -> None:
        os.initgroups(user.pw_name, user.pw_gid)
        os.setgid(user.pw_gid)
        os.setuid(user.pw_uid)

    return callback


def _load_progress(state: Path, args: argparse.Namespace, matrix_sha256: str) -> dict[str, Any]:
    progress_path = state / "progress.json"
    if not args.resume:
        if state.exists():
            raise FileExistsError(f"formal state already exists; use a new tag: {state}")
        state.mkdir(parents=True)
        return {
            "tag": args.tag,
            "matrix_sha256": matrix_sha256,
            "completed": [],
            "failed_attempts": [],
            "next_run_index": 33,
            "updated_at": _now(),
        }
    if not progress_path.is_file():
        raise FileNotFoundError(f"cannot resume without progress file: {progress_path}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("tag") != args.tag or progress.get("matrix_sha256") != matrix_sha256:
        raise RuntimeError("resume state differs from the requested tag or matrix")
    return dict(progress)


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
    parser.add_argument("--online-python", type=Path, required=True)
    parser.add_argument("--run-user", default="orangepi")
    parser.add_argument("--network-interface", default="wlan0")
    parser.add_argument("--core", choices=("auto", "0", "1", "2", "all"), default="0")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.geteuid() != 0:
        raise PermissionError("remaining-matrix runner must be root to apply frozen tc profiles")
    if not SAFE_TAG.fullmatch(args.tag):
        raise ValueError("tag contains unsafe characters")
    if args.max_attempts <= 0 or args.progress_every <= 0:
        raise ValueError("attempt and progress counts must be positive")
    bundle = args.bundle.resolve()
    matrix_path = args.matrix.resolve()
    rows = validate_remaining_matrix(matrix_path, bundle, args.protocol)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    profiles = matrix["network_profiles"]
    for field in ("images_dir", "rknn", "rknn_helpers_dir", "online_python"):
        if not Path(getattr(args, field)).exists():
            raise FileNotFoundError(getattr(args, field))
    user = pwd.getpwnam(args.run_user)
    token = os.environ.get(args.token_env, "")
    if not token:
        raise RuntimeError(f"missing cloud token environment variable: {args.token_env}")
    parsed_cloud = urlsplit(args.cloud_url)
    if not parsed_cloud.hostname:
        raise ValueError("cloud URL has no hostname")
    health = check_cloud(args.cloud_url, token, args.cloud_sha256)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    state = args.runs_root / f"_{args.tag}_remaining14_state"
    matrix_sha256 = file_sha256(matrix_path)
    progress = _load_progress(state, args, matrix_sha256)
    completed_indices = {int(item["run_index"]) for item in progress["completed"]}

    if not args.resume:
        shutil.copy2(matrix_path, state / "RUN_MATRIX.json")
        _atomic_json(
            state / "execution_contract.json",
            {
                "tag": args.tag,
                "matrix_sha256": matrix_sha256,
                "remaining_runs": len(rows),
                "cloud_url": args.cloud_url,
                "cloud_sha256": args.cloud_sha256,
                "cloud_health": health,
                "started_at": _now(),
                "system_start": _system_snapshot(),
                "evaluation_during_execution": False,
                "run_user": args.run_user,
                "network_interface": args.network_interface,
                "network_profiles": profiles,
                "tc_parameters": {
                    "burst": "16kb",
                    "latency": "2000ms",
                    "netem_limit_packets": 10000,
                },
            },
        )
        _atomic_json(state / "progress.json", progress)

    environment = dict(os.environ)
    environment.update(
        {
            "ONLINE_PYTHON": str(args.online_python),
            "HOME": user.pw_dir,
            "USER": user.pw_name,
            "LOGNAME": user.pw_name,
        }
    )
    try:
        for row in rows:
            index = int(row["run_index"])
            if index in completed_indices:
                continue
            completed = None
            existing_attempts = {
                int(path.name.rsplit("-a", 1)[1])
                for path in args.runs_root.glob(f"{args.tag}-r{index:02d}-{row['run_key']}-a*")
                if path.name.rsplit("-a", 1)[-1].isdigit()
            }
            for attempt in range(1, args.max_attempts + 1):
                if attempt in existing_attempts:
                    failure = {
                        "run_index": index,
                        "run_key": row["run_key"],
                        "run_id": f"{args.tag}-r{index:02d}-{row['run_key']}-a{attempt}",
                        "attempt": attempt,
                        "error": "pre-existing incomplete attempt retained after interruption",
                        "failed_at": _now(),
                    }
                    if failure["run_id"] not in {item["run_id"] for item in progress["failed_attempts"]}:
                        progress["failed_attempts"].append(failure)
                    continue
                profile_name = str(row["network_profile"])
                network_audit = _apply_network(
                    args.network_interface,
                    profile_name,
                    profiles[profile_name],
                    parsed_cloud.hostname,
                )
                check_cloud(args.cloud_url, token, args.cloud_sha256)
                run_id = f"{args.tag}-r{index:02d}-{row['run_key']}-a{attempt}"
                run_dir = args.runs_root / run_id
                if run_dir.exists():
                    raise FileExistsError(run_dir)
                log = state / f"r{index:02d}-{row['run_key']}-a{attempt}.log"
                print(f"START index={index}/46 attempt={attempt} run_id={run_id} time={_now()}", flush=True)
                with log.open("wb") as handle:
                    result = subprocess.run(
                        _command(bundle=bundle, row=row, run_id=run_id, args=args),
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                        preexec_fn=_demote(user),
                    )
                network_audit["qdisc_after_run"] = _network_snapshot(args.network_interface)
                network_audit["cleared_after_run"] = _clear_network(args.network_interface)
                run_dir.mkdir(parents=True, exist_ok=True)
                _atomic_json(run_dir / "network_contract.json", network_audit)
                error = None
                summary = None
                if result.returncode == 0:
                    try:
                        summary = gate_remaining_run(
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
                        "family": row["family"],
                        "network_profile": profile_name,
                        "summary": summary,
                        "completed_at": _now(),
                        "system_after": _system_snapshot(),
                    }
                    progress["completed"].append(completed)
                    progress["next_run_index"] = index + 1
                    progress["updated_at"] = _now()
                    _atomic_json(state / f"r{index:02d}_PASS.json", completed)
                    _atomic_json(state / "progress.json", progress)
                    print(f"PASS index={index}/46 run_id={run_id} time={_now()}", flush=True)
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
                print(f"FAILED index={index}/46 attempt={attempt} error={error}", flush=True)
                time.sleep(5)
            if completed is None:
                raise RuntimeError(f"formal row {index} failed {args.max_attempts} attempts")
    finally:
        _clear_network(args.network_interface)

    completion = {
        "complete": True,
        "tag": args.tag,
        "matrix_sha256": matrix_sha256,
        "remaining_runs": len(rows),
        "failed_attempts": len(progress["failed_attempts"]),
        "completed_at": _now(),
        "system_end": _system_snapshot(),
        "evaluation_during_execution": False,
    }
    _atomic_json(state / "REMAINING14_COMPLETE.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
