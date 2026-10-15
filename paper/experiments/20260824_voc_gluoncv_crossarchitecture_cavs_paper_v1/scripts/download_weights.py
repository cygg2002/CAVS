#!/usr/bin/env python3
"""Download and freeze the two official GluonCV detector parameter files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import urllib.request
import zipfile


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_gluoncv_weight_data", ROOT / "scripts/data_core.py")


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_archive(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return target
    temporary = target.with_name(f".{target.name}.tmp")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as handle:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            handle.write(block)
    temporary.replace(target)
    return target


def _archive_parameter_hashes(path: Path) -> tuple[str, str, str]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".params")]
        if len(members) != 1:
            raise RuntimeError(f"official model archive parameter members changed: {path}/{members}")
        sha1 = hashlib.sha1()
        sha256 = hashlib.sha256()
        with archive.open(members[0]) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                sha1.update(block)
                sha256.update(block)
    return members[0], sha1.hexdigest(), sha256.hexdigest()


def run(model_root: Path) -> dict[str, Any]:
    import gluoncv
    import mxnet
    from gluoncv.model_zoo.model_store import get_model_file

    model_root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for role, contract in PROTOCOL["detectors"].items():
        path = Path(get_model_file(str(contract["model"]), root=str(model_root))).resolve()
        observed_sha1 = file_sha1(path)
        if observed_sha1 != str(contract["official_sha1"]):
            raise RuntimeError(f"official model SHA-1 changed: {role}/{observed_sha1}")
        archive_name = Path(str(contract["url"])).name
        archive = _download_archive(
            str(contract["url"]), ROOT / "assets/downloads" / archive_name
        ).resolve()
        member, member_sha1, member_sha256 = _archive_parameter_hashes(archive)
        if member_sha1 != observed_sha1 or member_sha256 != DATA.file_sha256(path):
            raise RuntimeError(f"official archive content differs from parameters: {role}")
        rows[role] = {
            "model": str(contract["model"]),
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha1": observed_sha1,
            "sha256": DATA.file_sha256(path),
            "source_url": str(contract["url"]),
            "archive_path": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": DATA.file_sha256(archive),
            "archive_parameter_member": member,
        }
    manifest = {
        "complete": True,
        "protocol_id": PROTOCOL["protocol_id"],
        "weights": rows,
        "detector_training_performed": False,
        "runtime": {
            "python": sys.version,
            "mxnet_distribution": importlib.metadata.version("mxnet-cu112"),
            "mxnet_module": mxnet.__version__,
            "gluoncv_distribution": importlib.metadata.version("gluoncv"),
            "gluoncv_module": gluoncv.__version__,
        },
    }
    DATA.atomic_json(ROOT / "assets/weights_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default=str(ROOT / "assets/models"))
    args = parser.parse_args()
    print(json.dumps(run(Path(args.model_root)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
