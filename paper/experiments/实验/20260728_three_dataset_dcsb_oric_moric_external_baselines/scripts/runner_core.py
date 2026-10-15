"""Safety contracts shared by the external-baseline runners."""

from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
import pickle
import tempfile
from typing import Any, Iterable

import numpy as np


def assert_output_path(path: Path, output_root: Path) -> Path:
    resolved = Path(path).resolve()
    root = Path(output_root).resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing to write outside experiment runs: {resolved}")
    return resolved


def atomic_write_json(path: Path, payload: Any, output_root: Path) -> None:
    target = assert_output_path(path, output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_npz(path: Path, output_root: Path, **arrays: Any) -> None:
    target = assert_output_path(path, output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
            np.savez_compressed(handle, **arrays)
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_pickle(path: Path, payload: Any, output_root: Path) -> None:
    target = assert_output_path(path, output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_rows(
    path: Path, rows: list[dict[str, Any]], output_root: Path
) -> None:
    target = assert_output_path(path, output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    fields = sorted({key for row in rows for key in row})
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def capture_hashes(paths: Iterable[Path]) -> dict[str, str]:
    resolved = sorted({Path(path).resolve() for path in paths})
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required source files are missing: {missing}")
    return {str(path): file_sha256(path) for path in resolved}


def assert_hashes_unchanged(expected: dict[str, str]) -> None:
    current = capture_hashes(Path(path) for path in expected)
    changed = {
        path: {"before": digest, "after": current.get(path)}
        for path, digest in expected.items()
        if current.get(path) != digest
    }
    if changed:
        raise RuntimeError(f"Read-only source files changed during the stage: {changed}")


def roles_for_stage(stage: str) -> tuple[str, ...]:
    roles = {
        "preflight": ("fit", "calibration", "final"),
        "rewards": ("fit",),
        "evaluate": ("calibration", "final"),
    }
    try:
        return roles[str(stage)]
    except KeyError as error:
        raise ValueError(f"Unknown stage: {stage}") from error


def role_source_paths(common_root: Path, dataset: str, role: str) -> list[Path]:
    root = Path(common_root) / str(dataset) / str(role)
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [manifest_path, root / "samples.json", root / "gt.json"]
    paths.extend(Path(path) for path in manifest["action_sources"].values())
    return paths


def validate_role_data(
    data: dict[str, Any], *, dataset: str, role: str, expected_count: int
) -> None:
    samples = data["samples"]
    sample_ids = [int(sample["image_id"]) for sample in samples]
    image_ids = list(map(int, data["image_ids"]))
    if len(samples) != int(expected_count):
        raise RuntimeError(
            f"{dataset}/{role} population mismatch: {len(samples)} != {expected_count}"
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"{dataset}/{role} contains duplicate sample image_ids")
    if sorted(sample_ids) != sorted(image_ids):
        raise RuntimeError(f"{dataset}/{role} sample and loaded image_ids differ")
