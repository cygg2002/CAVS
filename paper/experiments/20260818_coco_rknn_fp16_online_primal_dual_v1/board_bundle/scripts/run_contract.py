"""Immutable online-run bindings and replay-validated committed-frame logs."""

from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from online_control import CausalByteController, decision_dict


CONTRACT_SCHEMA = 1
FRAME_SCHEMA = 1
PROTOCOL_ID = "coco_rknn_fp16_causal_online_hil_v1"
RKNN_HELPERS_SCHEMA = "cavs_rknn_helpers_manifest_v1"
REQUIRED_RKNN_HELPERS = frozenset(("coco_val_rknn.py", "yolo26_decode.py"))
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _valid_hash(value: str, label: str) -> str:
    digest = str(value).lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hex characters")
    return digest


def _artifact(path: str | Path) -> dict[str, str]:
    target = Path(path)
    return {"file": target.name, "sha256": file_sha256(target)}


def verify_rknn_helpers_manifest(
    manifest_path: str | Path, helpers_dir: str | Path
) -> dict[str, Any]:
    """Verify the exact RKNN preprocessing, decode, and NMS implementation."""
    manifest = Path(manifest_path)
    root = Path(helpers_dir).resolve()
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read RKNN helpers manifest") from exc
    if not isinstance(value, Mapping) or value.get("schema") != RKNN_HELPERS_SCHEMA:
        raise ValueError("unsupported RKNN helpers manifest")
    rows = value.get("files")
    if not root.is_dir() or not isinstance(rows, list):
        raise ValueError("RKNN helpers directory or manifest files are invalid")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("RKNN helper entry must be an object")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ValueError("RKNN helper paths must be unique")
        seen.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("RKNN helper escapes its root") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(row.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"RKNN helper size mismatch: {relative}")
        if str(row.get("sha256", "")).lower() != file_sha256(path):
            raise ValueError(f"RKNN helper SHA-256 mismatch: {relative}")
    if seen != REQUIRED_RKNN_HELPERS:
        raise ValueError("RKNN helpers manifest must bind both runtime modules")
    return {
        "schema": RKNN_HELPERS_SCHEMA,
        "files": len(rows),
        "manifest_sha256": file_sha256(manifest),
    }


def build_run_contract(
    *,
    run_id: str,
    method: str,
    budget: str | float | Fraction,
    sequence_path: str | Path,
    policy_path: str | Path,
    control_path: str | Path,
    rknn_path: str | Path,
    cloud_sha256: str,
    action_ids: Sequence[str],
    random_seed: int | None = None,
    bundle_manifest_path: str | Path | None = None,
    rknn_helpers_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a run ID to all assets that can affect an online decision."""
    if not _RUN_ID.fullmatch(str(run_id)):
        raise ValueError("run ID must be a safe 1-128 character identifier")
    name = str(method).lower()
    if not name:
        raise ValueError("method is required")
    fraction = budget if isinstance(budget, Fraction) else Fraction(str(budget))
    if fraction <= 0 or fraction > 1:
        raise ValueError("budget must be in (0, 1]")
    actions = tuple(map(str, action_ids))
    if not actions or actions[0] != "edge" or "original" not in actions:
        raise ValueError("action IDs must begin with edge and include original")
    if len(actions) != len(set(actions)):
        raise ValueError("action IDs must be unique")
    if random_seed is not None:
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ValueError("random seed must be an integer or null")
        if name != "random":
            raise ValueError("random seed is only valid for the random method")

    artifacts = {
        "sequence": _artifact(sequence_path),
        "policy": _artifact(policy_path),
        "control": _artifact(control_path),
        "rknn": _artifact(rknn_path),
    }
    if bundle_manifest_path is not None:
        artifacts["bundle"] = _artifact(bundle_manifest_path)
    if rknn_helpers_manifest_path is not None:
        artifacts["rknn_helpers"] = _artifact(rknn_helpers_manifest_path)
    cloud = _valid_hash(cloud_sha256, "cloud")
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "run_id": str(run_id),
        "method": name,
        "budget_numerator": int(fraction.numerator),
        "budget_denominator": int(fraction.denominator),
        "action_ids": list(actions),
        "sequence_sha256": artifacts["sequence"]["sha256"],
        "policy_sha256": artifacts["policy"]["sha256"],
        "control_sha256": artifacts["control"]["sha256"],
        "rknn_sha256": artifacts["rknn"]["sha256"],
        "cloud_sha256": cloud,
        "random_seed": int(random_seed) if random_seed is not None else None,
        "zero_lookahead": True,
        "global_plan": False,
        "buffer_frames": 0,
        "artifacts": artifacts,
    }
    if bundle_manifest_path is not None:
        contract["bundle_sha256"] = artifacts["bundle"]["sha256"]
    if rknn_helpers_manifest_path is not None:
        contract["rknn_helpers_manifest_sha256"] = artifacts["rknn_helpers"]["sha256"]
    return contract


@contextmanager
def exclusive_run_lock(run_dir: str | Path):
    """Hold a non-blocking single-writer lock for one run directory."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".run.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("run ID is already active") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ensure_run_contract(run_dir: str | Path, contract: Mapping[str, Any]) -> str:
    """Create an immutable contract, or verify an exact resume contract."""
    directory = Path(run_dir)
    expected = dict(contract)
    if int(expected.get("schema_version", -1)) != CONTRACT_SCHEMA:
        raise ValueError("unsupported run contract schema")
    if expected.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected online protocol")
    if not _RUN_ID.fullmatch(str(expected.get("run_id", ""))):
        raise ValueError("invalid run ID in contract")
    for name in ("sequence", "policy", "control", "rknn", "cloud"):
        _valid_hash(str(expected.get(f"{name}_sha256", "")), name)
    if "bundle_sha256" in expected:
        _valid_hash(str(expected["bundle_sha256"]), "bundle")
    if "rknn_helpers_manifest_sha256" in expected:
        _valid_hash(
            str(expected["rknn_helpers_manifest_sha256"]), "RKNN helpers manifest"
        )
    artifacts = expected.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("run contract artifacts are required")
    for name in ("sequence", "policy", "control", "rknn"):
        artifact = artifacts.get(name)
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("sha256") != expected[f"{name}_sha256"]
        ):
            raise ValueError(f"run contract artifact hash disagrees for {name}")
    bundle_artifact = artifacts.get("bundle")
    if "bundle_sha256" in expected and (
        not isinstance(bundle_artifact, Mapping)
        or bundle_artifact.get("sha256") != expected["bundle_sha256"]
    ):
        raise ValueError("run contract artifact hash disagrees for bundle")
    helpers_artifact = artifacts.get("rknn_helpers")
    if "rknn_helpers_manifest_sha256" in expected and (
        not isinstance(helpers_artifact, Mapping)
        or helpers_artifact.get("sha256")
        != expected["rknn_helpers_manifest_sha256"]
    ):
        raise ValueError("run contract artifact hash disagrees for RKNN helpers")
    if "token" in json.dumps(expected, sort_keys=True).lower():
        raise ValueError("run contracts must not contain token material")

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_contract.json"
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid existing run contract") from exc
        if current != expected:
            raise RuntimeError("run ID contract mismatch; use a fresh run ID")
        return canonical_sha256(expected)
    unexpected = [entry.name for entry in directory.iterdir() if entry.name != path.name]
    if unexpected:
        raise RuntimeError("run ID has artifacts but is missing its run contract")
    _atomic_json(path, expected)
    return canonical_sha256(expected)


def append_committed_frame(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    run_id: str,
    contract_sha256: str,
) -> dict[str, Any]:
    """Append and fsync one fully resolved frame; no provisional rows are logged."""
    if not _RUN_ID.fullmatch(str(run_id)):
        raise ValueError("invalid run ID")
    digest = _valid_hash(contract_sha256, "contract")
    row = {
        **dict(record),
        "schema_version": FRAME_SCHEMA,
        "status": "committed",
        "run_id": str(run_id),
        "contract_sha256": digest,
    }
    required = (
        "position", "image_id", "file_name", "decision", "controller_state",
        "payload_bytes", "scores", "resolved_boxes",
    )
    if any(name not in row for name in required):
        raise ValueError("committed frame is missing required fields")
    encoded = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("could not append committed frame")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return row


def _read_frames(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"invalid frames.jsonl at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise RuntimeError(f"invalid frames.jsonl at line {line_number}")
            rows.append(dict(row))
    return rows


def restore_committed_prefix(
    frames_path: str | Path,
    sequence: Sequence[Mapping[str, Any]],
    *,
    contract_sha256: str,
    controller: CausalByteController,
    run_id: str | None = None,
) -> tuple[CausalByteController, int, list[dict[str, Any]]]:
    """Replay committed observations so resume never trusts a forged snapshot."""
    digest = _valid_hash(contract_sha256, "contract")
    rows = _read_frames(Path(frames_path))
    if len(rows) > len(sequence):
        raise RuntimeError("committed log is longer than the image sequence")
    expected_run_id = str(run_id) if run_id is not None else None
    if expected_run_id is not None and not _RUN_ID.fullmatch(expected_run_id):
        raise ValueError("invalid run ID")
    for position, row in enumerate(rows):
        item = sequence[position]
        if (
            row.get("status") != "committed"
            or int(row.get("schema_version", -1)) != FRAME_SCHEMA
        ):
            raise RuntimeError(f"frame {position} is not a committed record")
        if row.get("contract_sha256") != digest:
            raise RuntimeError("frame contract hash mismatch")
        frame_run_id = str(row.get("run_id", ""))
        expected_run_id = expected_run_id or frame_run_id
        if not _RUN_ID.fullmatch(frame_run_id) or frame_run_id != expected_run_id:
            raise RuntimeError("frame run ID mismatch")
        try:
            identity_matches = (
                int(row.get("position", -1)) == position
                and int(row.get("image_id", -1)) == int(item["image_id"])
                and str(row.get("file_name", "")) == str(item["file_name"])
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("sequence prefix mismatch") from exc
        if not identity_matches:
            raise RuntimeError("sequence prefix mismatch")
        try:
            replayed = controller.decide(
                int(item["image_id"]), row["payload_bytes"], row["scores"]
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"cannot replay committed frame {position}") from exc
        if decision_dict(replayed) != row.get("decision"):
            raise RuntimeError(f"decision replay mismatch at frame {position}")
        if controller.snapshot() != row.get("controller_state"):
            raise RuntimeError(f"controller state mismatch at frame {position}")
        if not isinstance(row.get("resolved_boxes"), list):
            raise RuntimeError(f"resolved boxes missing at frame {position}")
    return controller, len(rows), rows
