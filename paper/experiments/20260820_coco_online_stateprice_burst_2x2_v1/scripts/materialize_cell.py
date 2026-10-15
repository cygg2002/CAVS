#!/usr/bin/env python3
"""Materialize one frozen factorial cell using a matching online source run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "20260818_coco_rknn_fp16_online_primal_dual_v1"
for path in (PRIOR / "host", PRIOR / "board_bundle" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from materialize_adaptive_frames import HTTPActionExecutor, materialize_frames


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cell_selections(path: Path, cell: str) -> dict[str, str]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    selections = bundle.get(cell) if isinstance(bundle, Mapping) else None
    if not isinstance(selections, Mapping):
        raise ValueError(f"selection bundle lacks cell: {cell}")
    return {str(key): str(value) for key, value in selections.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--cloud-url", required=True)
    parser.add_argument("--cloud-sha256", required=True)
    parser.add_argument("--token-env", default="CAVS_TOKEN")
    parser.add_argument("--output-frames", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    source = _read_jsonl(args.source_frames)
    selections = load_cell_selections(args.selections, args.cell)
    executor = HTTPActionExecutor(
        images_dir=args.images_dir,
        policy_path=args.policy,
        cloud_url=args.cloud_url,
        cloud_sha256=args.cloud_sha256,
        token_env=args.token_env,
    )
    frames, report = materialize_frames(
        source,
        adaptive_selections=selections,
        execute_cloud=executor,
    )
    _atomic_jsonl(args.output_frames, frames)
    report.update({
        "cell": args.cell,
        "source_hashes": {
            "source_frames": file_sha256(args.source_frames),
            "selections": file_sha256(args.selections),
            "policy": file_sha256(args.policy),
        },
        "cloud_model_sha256": args.cloud_sha256,
        "output_frames_sha256": file_sha256(args.output_frames),
    })
    _atomic_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
