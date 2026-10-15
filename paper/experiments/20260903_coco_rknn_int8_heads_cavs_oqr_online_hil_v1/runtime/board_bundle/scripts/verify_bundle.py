#!/usr/bin/env python3
"""Create or verify the hash-pinned files used by the causal board runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cavs_online_runtime_bundle_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"bundle asset escapes root: {path}")
    return relative.as_posix()


def _asset_paths(root: Path, manifest: Path) -> list[Path]:
    excluded = {manifest.resolve()}
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() in excluded:
            continue
        relative = path.resolve().relative_to(root.resolve()).parts
        if any(part in {"__pycache__", "tests", "runs"} for part in relative):
            continue
        paths.append(path)
    return sorted(paths, key=lambda value: _safe_relative(root, value))


def build_manifest(root: str | Path, manifest: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest_path = Path(manifest).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    assets = []
    for path in _asset_paths(root_path, manifest_path):
        assets.append({
            "path": _safe_relative(root_path, path),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return {"schema": SCHEMA, "root": root_path.name, "assets": assets}


def verify_manifest(root: str | Path, manifest: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest_path = Path(manifest).resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read runtime bundle manifest") from exc
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported runtime bundle manifest")
    assets = value.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("runtime bundle manifest has no assets")
    seen: set[str] = set()
    for row in assets:
        if not isinstance(row, Mapping):
            raise ValueError("runtime bundle asset must be an object")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ValueError("runtime bundle asset paths must be unique")
        seen.add(relative)
        path = (root_path / relative).resolve()
        if path.parent != root_path and root_path not in path.parents:
            raise ValueError("runtime bundle asset escapes root")
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(row.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"runtime bundle size mismatch: {relative}")
        if str(row.get("sha256", "")).lower() != _sha256(path):
            raise ValueError(f"runtime bundle SHA-256 mismatch: {relative}")
    return {"schema": SCHEMA, "assets": len(assets), "manifest": str(manifest_path)}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        _write(args.manifest, build_manifest(args.root, args.manifest))
    result = verify_manifest(args.root, args.manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
