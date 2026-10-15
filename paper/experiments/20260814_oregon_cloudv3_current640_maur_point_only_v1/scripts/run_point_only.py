#!/usr/bin/env python3
"""Isolated point-only evaluation for Oregon Cloud-V3 Current@640+MAUR."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
OUTPUT = ROOT / "runs/point"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
SOURCE_ROOT = REPO / PROTOCOL["source_experiment"]
SOURCE_SCRIPT = SOURCE_ROOT / "scripts/run_experiment.py"
sys.dont_write_bytecode = True


def load_source() -> Any:
    spec = importlib.util.spec_from_file_location("oregon_maur_point_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty results")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def expected_score_paths() -> list[Path]:
    return [
        SOURCE_ROOT
        / "runs/oregon/training/jobs/CAVS-OQR-MAUR"
        / f"seed_{seed}/full/scores.npz"
        for seed in PROTOCOL["model_seeds"]
    ]


def preflight(source: Any) -> dict[str, Any]:
    source_protocol = source.PROTOCOL
    if source_protocol["datasets"]["oregon"]["cloud_model_sha256"] != PROTOCOL["cloud_model_sha256"]:
        raise RuntimeError("Cloud-V3 model hash changed")
    if source_protocol["datasets"]["oregon"]["model_seeds"] != PROTOCOL["model_seeds"]:
        raise RuntimeError("model seeds changed")
    completion = SOURCE_ROOT / "runs/training/completion.json"
    if not completion.is_file() or not json.loads(completion.read_text())["complete"]:
        raise RuntimeError("strict full-fit training is incomplete")
    scores = expected_score_paths()
    missing = [str(path) for path in scores if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen scores: {missing}")
    return {
        "source_script_sha256": sha256(SOURCE_SCRIPT),
        "source_training_completion_sha256": sha256(completion),
        "score_sha256": {str(path.relative_to(REPO)): sha256(path) for path in scores},
    }


def run() -> dict[str, Any]:
    completion = OUTPUT / "completion.json"
    if completion.is_file():
        return json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    source = load_source()
    provenance = preflight(source)
    dataset = PROTOCOL["dataset"]
    role = PROTOCOL["role"]
    records = source.role_records(dataset, role, "OQR")
    samples = source.adapter(dataset).load_role(role)["samples"]
    single, ensemble = source.ensemble_action_scores(dataset, PROTOCOL["method"], role)
    if sorted(map(int, single)) != sorted(PROTOCOL["model_seeds"]):
        raise RuntimeError("three-seed ensemble changed")

    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        score_map = source.action_score_map(records, ensemble[budget_index])
        selection = source._joint_selection(
            dataset, samples, score_map, mode=PROTOCOL["mode"], budget=budget
        )
        key = f"final|CAVS-OQR-MAUR|Joint-OQR|fixed_byte|{budget:g}|seed=ensemble"
        selections[key] = {str(image_id): str(action) for image_id, action in selection.items()}
        row = source.evaluate_selection(
            dataset,
            role,
            selection,
            method=PROTOCOL["method"],
            policy=PROTOCOL["policy"],
            mode=PROTOCOL["mode"],
            budget=budget,
            seed="ensemble",
            fusion=PROTOCOL["fusion"],
        )
        if float(row["capacity_utilization"]) < 0.98:
            raise RuntimeError(f"capacity utilization below 98% at budget {budget}")
        rows.append(row)

    if len(rows) != int(PROTOCOL["expected_rows"]):
        raise RuntimeError(f"result row count changed: {len(rows)}")
    atomic_csv(OUTPUT / "results.csv", rows)
    atomic_json(OUTPUT / "selections.json", selections)
    report = {
        "complete": True,
        "dataset": dataset,
        "method": PROTOCOL["method"],
        "profile": "Current@640",
        "fusion": PROTOCOL["fusion"],
        "training": "strict three-seed full-fit ensemble reused",
        "rows": rows,
        "provenance": provenance,
        "bootstrap_performed": False,
        "training_performed": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(OUTPUT / "report.json", report)
    summary = [
        "# Oregon Cloud-V3 Current@640+MAUR 严格全重训点评测",
        "",
        "| Fixed-byte预算 | Official AP50-95 | AP50 | AP75 | KB/图 | Original | Q90 | ROI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary.append(
            f"| {float(row['budget']):.0%} | {float(row['official_ap50_95']):.6f} | "
            f"{float(row['official_ap50']):.6f} | {float(row['official_ap75']):.6f} | "
            f"{float(row['kb_per_image']):.2f} | {int(row['original_count'])} | "
            f"{int(row['q90_count'])} | {int(row['roi_count'])} |"
        )
    summary.extend([
        "",
        "本实验只复用已完成的三种子严格全重训 checkpoint；未重训、未调参、未运行 bootstrap。",
    ])
    (OUTPUT / "RESULTS_ZH.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    atomic_json(completion, {"complete": True, "results_sha256": sha256(OUTPUT / "results.csv")})
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))

