#!/usr/bin/env python3
"""Validate synchronized artifacts and adapt selection keys for the live runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"


def main() -> None:
    samples = json.loads((INPUTS / "paper_samples.json").read_text(encoding="utf-8"))
    if len(samples) != 5000 or len({int(sample["image_id"]) for sample in samples}) != 5000:
        raise RuntimeError("synchronized final sample population changed")
    raw = json.loads((INPUTS / "remote_selections.json").read_text(encoding="utf-8"))
    selections = {}
    theoretical = {}
    for budget in (0.1, 0.25, 0.4):
        source = f"final|CAVS-Current-MAUR-PerFrame|fixed_byte|{budget:g}"
        target = f"final|CAVS-Current-MAUR|fixed_byte|{budget:g}"
        if source not in raw:
            raise RuntimeError(f"missing synchronized selection: {source}")
        selections[target] = raw[source]
    with (INPUTS / "paper_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for budget in (0.1, 0.25, 0.4):
        found = [
            row
            for row in rows
            if row["role"] == "final"
            and row["mode"] == "fixed_byte"
            and abs(float(row["budget"]) - budget) < 1e-12
        ]
        if len(found) != 1:
            raise RuntimeError(f"missing cached point result: {budget}")
        theoretical[str(budget)] = float(found[0]["official_ap50_95"])
    (INPUTS / "paper_selections.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (INPUTS / "theoretical.json").write_text(
        json.dumps(theoretical, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"samples": len(samples), "theoretical": theoretical}, indent=2))


if __name__ == "__main__":
    main()

