#!/usr/bin/env python3
"""Render a compact FP16 vs INT8-heads comparison from completed summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise ValueError(f"incomplete run: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--int8-heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {"FP16": load(args.fp16), "INT8-heads": load(args.int8_heads)}
    output = {"protocol_id": rows["FP16"]["protocol_id"], "models": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("| 模型 | 10% AP50-95 | 25% AP50-95 | 40% AP50-95 | Edge AP50-95 |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for label, summary in rows.items():
        methods = summary["methods"]
        print(f"| {label} | {methods['cavs_oqr_10']['metrics']['AP50_95']:.6f} | {methods['cavs_oqr_25']['metrics']['AP50_95']:.6f} | {methods['cavs_oqr_40']['metrics']['AP50_95']:.6f} | {methods['edge']['metrics']['AP50_95']:.6f} |", flush=True)


if __name__ == "__main__":
    main()

