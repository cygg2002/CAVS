#!/usr/bin/env python3
"""Relocate frozen policy-training inputs into an isolated alternate-host mirror."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


DEFAULT_OLD_ROOT = "/home/111_wcy/work/yolo26/ultralytics-main"


def _inside(root: Path, path: Path, *, field: str) -> Path:
    root = root.resolve()
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes mirror root: {path}") from exc
    return path


def _copy_or_link(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)

    def link_with_fallback(src: str, dst: str) -> str:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return dst

    shutil.copytree(source, destination, copy_function=link_with_fallback)


def _replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_strings(item, old, new) for item in value]
    if isinstance(value, Mapping):
        return {
            str(_replace_strings(key, old, new)): _replace_strings(item, old, new)
            for key, item in value.items()
        }
    return value


def _rewrite_json(path: Path, old: str, new: str) -> bool:
    raw = path.read_text(encoding="utf-8")
    if old not in raw:
        return False
    value = json.loads(raw)
    updated = _replace_strings(value, old, new)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(updated, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return True


def relocate_experiment(
    *,
    mirror_root: str | Path,
    experiment: str | Path,
    source_backend: str | Path,
    source_current: str | Path,
    source_training: str | Path,
    folds: str | Path,
    cloud_model: str | Path,
    old_root: str = DEFAULT_OLD_ROOT,
) -> dict[str, Any]:
    mirror = Path(mirror_root).resolve()
    experiment = _inside(mirror, Path(experiment), field="experiment")
    source_backend = _inside(mirror, Path(source_backend), field="source_backend")
    source_current = _inside(mirror, Path(source_current), field="source_current")
    source_training = _inside(mirror, Path(source_training), field="source_training")
    folds = _inside(mirror, Path(folds), field="folds")
    cloud_model = _inside(mirror, Path(cloud_model), field="cloud_model")
    if not old_root.startswith("/") or old_root == "/":
        raise ValueError("old_root must be a specific absolute path")
    for field, path in (("folds", folds), ("cloud_model", cloud_model)):
        if not path.is_file():
            raise FileNotFoundError(f"{field}: {path}")
    for field, path in (
        ("source_backend", source_backend),
        ("source_current", source_current),
        ("source_training", source_training),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{field}: {path}")

    destination_backend = experiment / "runs/backends/fp16"
    destination_backend.mkdir(parents=True, exist_ok=True)
    for name in ("common", "rewards"):
        source = source_backend / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        _copy_or_link(source, destination_backend / name)

    protocol_path = experiment / "configs/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["canonical_root"] = str(
        mirror / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/runs/canonical_v1"
    )
    protocol["source_current_experiment"] = str(source_current)
    protocol["source_training_experiment"] = str(source_training)
    protocol["source_backend_root"] = str(destination_backend)
    protocol["frozen_folds"] = str(folds)
    protocol["cloud_model"] = {
        **dict(protocol.get("cloud_model", {})),
        "path": str(cloud_model),
    }
    temporary = protocol_path.with_name(f".{protocol_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, protocol_path)

    rewritten = 0
    roots = (destination_backend / "common", source_current / "runs/common/coco")
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            rewritten += int(_rewrite_json(path, old_root, str(mirror)))

    result = {
        "complete": True,
        "mirror_root": str(mirror),
        "experiment": str(experiment),
        "source_backend": str(source_backend),
        "source_current": str(source_current),
        "source_training": str(source_training),
        "destination_backend": str(destination_backend),
        "rewritten_json_files": rewritten,
        "frozen_common_and_rewards_reused": True,
        "learner_checkpoints_reused": False,
    }
    report = experiment / "runs/alternate_relocation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--source-backend", type=Path, required=True)
    parser.add_argument("--source-current", type=Path, required=True)
    parser.add_argument("--source-training", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--cloud-model", type=Path, required=True)
    parser.add_argument("--old-root", default=DEFAULT_OLD_ROOT)
    args = parser.parse_args()
    result = relocate_experiment(
        mirror_root=args.mirror_root,
        experiment=args.experiment,
        source_backend=args.source_backend,
        source_current=args.source_current,
        source_training=args.source_training,
        folds=args.folds,
        cloud_model=args.cloud_model,
        old_root=args.old_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
