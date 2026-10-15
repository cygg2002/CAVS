#!/usr/bin/env python3
"""Evaluate the frozen Cloud-V4 CurrentActionValue-OQR ensemble at three points."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
OUTPUT = RUNS / "point"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
sys.dont_write_bytecode = True


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL = load_module("cloudv4_point_evaluator", ROOT / "scripts/evaluate.py")


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
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run() -> dict[str, Any]:
    completion = OUTPUT / "completion.json"
    if completion.is_file():
        return json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    learner_completion = RUNS / "learners/completion.json"
    if not learner_completion.is_file():
        raise RuntimeError("strict three-seed learner training is incomplete")

    role = "final"
    data = EVAL.load_role(role, fusion="MAUR")
    records = EVAL.role_records(role, "OQR12")
    score = EVAL._budget_matrix(EVAL.full_job_score("CurrentActionValue-OQR12", role))
    EVAL._EVAL_STATE.clear()
    EVAL._EVAL_STATE[role] = data
    rows = []
    selections = {}
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        score_map = EVAL._score_map(records, score[budget_index])
        selection = EVAL.schedule_joint(
            data["samples"], score_map, mode="fixed_byte", budget=budget
        )
        task = {
            "role": role,
            "method": "CAVS-OQR-MAUR",
            "policy": "Joint-OQR",
            "mode": "fixed_byte",
            "budget": budget,
            "selection": selection,
        }
        row = EVAL._evaluate(task)
        row["seed"] = "ensemble"
        row["cloud_model"] = "Cloud-V4-epoch29-best"
        row["primary_roi_fusion"] = "MAUR"
        if float(row["capacity_utilization"]) < 0.98:
            raise RuntimeError(f"capacity utilization below 98% at budget {budget}")
        rows.append(row)
        selections[f"final|fixed_byte|{budget:g}"] = {
            str(image_id): str(action) for image_id, action in selection.items()
        }
    EVAL._EVAL_STATE.clear()
    if len(rows) != 3:
        raise RuntimeError(f"point row count changed: {len(rows)}")

    atomic_csv(OUTPUT / "results.csv", rows)
    atomic_json(OUTPUT / "selections.json", selections)
    report = {
        "complete": True,
        "method": "CAVS-OQR-MAUR",
        "cloud_model": PROTOCOL["cloud_model"],
        "rows": rows,
        "training_jobs": 18,
        "training_performed_in_this_experiment": True,
        "bootstrap_performed": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(OUTPUT / "report.json", report)
    lines = [
        "# Oregon Cloud-V4 Current@640+MAUR 严格全重训点评测",
        "",
        "| Fixed-byte预算 | AP50-95 | AP50 | AP75 | KB/图 | Original | Q90 | ROI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['budget']):.0%} | {float(row['official_ap50_95']):.6f} | "
            f"{float(row['official_ap50']):.6f} | {float(row['official_ap75']):.6f} | "
            f"{float(row['kb_per_image']):.2f} | {int(row['original_count'])} | "
            f"{int(row['q90_count'])} | {int(row['roi_count'])} |"
        )
    lines.extend([
        "",
        "Cloud-V4动作缓存、MAUR reward及三种子学习器均在本实验中重新生成；未运行bootstrap。",
    ])
    (OUTPUT / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(completion, {"complete": True, "results_sha256": sha256(OUTPUT / "results.csv")})
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))

