#!/usr/bin/env python3
"""Materialize changed adaptive actions for cache-domain AP diagnosis."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BOARD_SCRIPTS = ROOT / "board_bundle" / "scripts"
if str(BOARD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BOARD_SCRIPTS))

from online_methods import build_method_runtime
from online_runner import CloudClient, _execute_current_frame
from protocol import native_jpeg, q90_jpeg


def _valid_evaluation_box(box: Mapping[str, Any]) -> bool:
    try:
        x1, y1, x2, y2 = map(float, box["xyxy"])
        confidence = float(box.get("conf", box.get("score")))
        category = box["cls"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence))
        and x2 > x1
        and y2 > y1
        and isinstance(category, int)
        and not isinstance(category, bool)
        and 0 <= category < 80
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_frames(
    source_frames: Sequence[Mapping[str, Any]],
    *,
    adaptive_selections: Mapping[str, str],
    execute_cloud: Callable[[Mapping[str, Any], str], Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {str(int(row["image_id"])) for row in source_frames}
    if set(adaptive_selections) != expected:
        raise ValueError("adaptive selection population differs from source frames")
    output = []
    counts: Counter[str] = Counter()
    for row in source_frames:
        image_id = str(int(row["image_id"]))
        action = str(adaptive_selections[image_id])
        source_action = str(row["decision"]["action"])
        if action == source_action:
            resolved = list(row["resolved_boxes"])
            origin = "reused_same_action"
        elif action == "edge":
            resolved = list(row["edge_boxes"])
            origin = "edge_replacement"
        else:
            resolved = list(execute_cloud(row, action))
            origin = "new_cloud_execution"
        filtered = [dict(box) for box in resolved if _valid_evaluation_box(box)]
        materialized = dict(row)
        materialized["resolved_boxes"] = filtered
        materialized["decision"] = {**dict(row["decision"]), "action": action}
        materialized["materialization"] = {
            "source_action": source_action,
            "adaptive_action": action,
            "origin": origin,
            "diagnostic_cache_execution": True,
            "real_time_hil": False,
        }
        output.append(materialized)
        counts[origin] += 1
        counts["degenerate_boxes_filtered"] += len(resolved) - len(filtered)
    return output, {
        "frames": len(output),
        "reused_frames": int(counts["reused_same_action"]),
        "edge_replacements": int(counts["edge_replacement"]),
        "cloud_executions": int(counts["new_cloud_execution"]),
        "degenerate_boxes_filtered": int(counts["degenerate_boxes_filtered"]),
        "diagnostic_cache_execution": True,
        "real_time_hil": False,
    }


class HTTPActionExecutor:
    def __init__(
        self,
        *,
        images_dir: Path,
        policy_path: Path,
        cloud_url: str,
        cloud_sha256: str,
        token_env: str,
    ) -> None:
        self.images_dir = images_dir.resolve()
        self.method = build_method_runtime("cavs_oqr", policy_path=policy_path)
        self.cloud = CloudClient(
            cloud_url,
            token_env=token_env,
            cloud_sha256=cloud_sha256,
        )
        self.cloud.validate_health()

    def __call__(self, row: Mapping[str, Any], action: str) -> Sequence[Mapping[str, Any]]:
        if action not in {"original", "q90", "current@640"}:
            raise ValueError(f"unsupported cloud action: {action}")
        path = (self.images_dir / str(row["file_name"])).resolve()
        try:
            path.relative_to(self.images_dir)
        except ValueError as error:
            raise RuntimeError("frame path escapes image root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        roi = row.get("roi_xyxy")
        if action == "original":
            payload = native_jpeg(path)
        elif action == "q90":
            payload = q90_jpeg(path, long_side=640)
        else:
            if not isinstance(roi, list) or len(roi) != 4:
                raise RuntimeError("adaptive ROI action is unavailable")
            payload = q90_jpeg(path, crop_xyxy=roi, long_side=640)
        expected = int(row["payload_bytes"].get(action, -1))
        if len(payload) != expected:
            raise RuntimeError(
                f"regenerated payload differs for image {row['image_id']} action {action}: "
                f"{len(payload)} != {expected}"
            )
        observation = {
            "frame_id": int(row["image_id"]),
            "shape": list(row["shape"]),
            "edge_boxes": list(row["edge_boxes"]),
            "roi_xyxy": roi,
            "payload_bytes": dict(row["payload_bytes"]),
            "payloads": {action: payload},
            "scores": dict(row["scores"]),
            "timings_ms": {},
        }
        result = _execute_current_frame(
            self.method,
            self.cloud,
            observation,
            SimpleNamespace(action=action, payload_bytes=len(payload)),
        )
        if result.get("execution_status") != "ok":
            raise RuntimeError(
                f"cloud action failed for image {row['image_id']}: "
                f"{result.get('cloud', {}).get('error', 'unknown')}"
            )
        return list(result["resolved_boxes"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--cloud-url", required=True)
    parser.add_argument("--cloud-sha256", required=True)
    parser.add_argument("--token-env", default="CAVS_TOKEN")
    parser.add_argument("--output-frames", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    source = _read_jsonl(args.source_frames)
    selections_bundle = json.loads(args.selections.read_text(encoding="utf-8"))
    selections = selections_bundle.get("adaptive")
    if not isinstance(selections, Mapping):
        raise ValueError("selection bundle lacks adaptive selections")
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
    report["source_hashes"] = {
        "source_frames": _sha256(args.source_frames),
        "selections": _sha256(args.selections),
        "policy": _sha256(args.policy),
    }
    report["output_frames_sha256"] = _sha256(args.output_frames)
    _atomic_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
