#!/usr/bin/env python3
"""Independently retrain AFPD/Occupancy and evaluate complete COCO val2017."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads(
    (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
)
SOURCE_ROOT = REPO / PROTOCOL["source_experiment"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("afpd_fullval_core", ROOT / "scripts/core.py")
SOURCE = load_module(
    "afpd_fullval_source", SOURCE_ROOT / "scripts/run_experiment.py"
)
CORE.validate_protocol(PROTOCOL)
SOURCE.RUNS = RUNS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_preflight() -> dict[str, Any]:
    output = RUNS / "paper_preflight/report.json"
    completion = RUNS / "paper_preflight/completion.json"
    if completion.is_file():
        return read_json(output)
    source_calibration = read_json(SOURCE_ROOT / "runs/calibration/report.json")
    source_final = read_json(SOURCE_ROOT / "runs/final/report.json")
    if bool(source_calibration["primary_gate"]["passed"]):
        raise RuntimeError("Expected the prior AFPD calibration gate to fail")
    if bool(source_final.get("evaluated", False)) or bool(
        source_final.get("final_accessed", False)
    ):
        raise RuntimeError("Prior AFPD protocol unexpectedly accessed final")
    source_files = (
        ROOT / "configs/protocol.json",
        ROOT / "scripts/core.py",
        SCRIPT,
        SOURCE_ROOT / "configs/protocol.json",
        SOURCE_ROOT / "scripts/core.py",
        SOURCE_ROOT / "scripts/run_experiment.py",
        SOURCE.SOURCES["afpd_v1"] / "scripts/training.py",
        SOURCE.SOURCES["occupancy_v2"] / "scripts/training.py",
        SOURCE.SOURCES["afpd_v1"] / "runs/rewards/fit/roi_reward_map.pkl",
        SOURCE.SOURCES["full_benchmark"] / "runs/coco/point/selections.json",
    )
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    frozen_selections = read_json(source_files[-1])
    for budget in map(float, PROTOCOL["budgets"]):
        for prefix in (
            "Raw-ActionValue-OQ|Joint-OQ",
            "ORIC-MLP|SharedAction-NoROI",
            "MORIC-MLP|SharedAction-NoROI",
        ):
            key = f"{prefix}|fixed_byte|{budget:g}"
            if key not in frozen_selections:
                raise RuntimeError(f"Missing frozen comparator selection: {key}")
    marker = RUNS / "independent_retrain_started.json"
    if not marker.is_file():
        existing = [
            path
            for path in (RUNS / "calibration").glob("**/checkpoint.pt")
            if path.is_file()
        ]
        if existing:
            raise RuntimeError("Target checkpoints predate independent-run marker")
        atomic_json(
            marker,
            {
                "protocol_id": PROTOCOL["protocol_id"],
                "independent_retrain": True,
                "source_checkpoint_reuse": False,
            },
        )
    inherited = SOURCE.run_preflight()
    hashes = {str(path): file_sha256(path) for path in source_files}
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "status": PROTOCOL["status"],
        "confirmatory": False,
        "prior_calibration_gate_failed": True,
        "source_checkpoint_reuse": False,
        "full_val_unconditional": True,
        "final_is_hidden": False,
        "final_accessed": False,
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "inherited_preflight": inherited,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output)})
    return report


def run_training() -> dict[str, Any]:
    report_path = RUNS / "independent_training/report.json"
    completion = RUNS / "independent_training/completion.json"
    if completion.is_file():
        return read_json(report_path)
    run_preflight()
    action_oof = SOURCE.run_action_oof_training()
    action_oof_evaluation = SOURCE.run_action_oof_evaluation()
    occupancy_oof = SOURCE.run_occupancy_oof_training()
    occupancy_oof_evaluation = SOURCE.run_occupancy_oof_evaluation()
    calibration_training = SOURCE.run_calibration_training()
    checkpoint_paths = [
        ROOT / relative for relative in CORE.checkpoint_relative_paths(PROTOCOL)
    ]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing independently trained checkpoints: {missing}")
    source_inodes = {
        (SOURCE_ROOT / relative).stat().st_ino
        for relative in CORE.checkpoint_relative_paths(PROTOCOL)
        if (SOURCE_ROOT / relative).is_file()
    }
    target_inodes = {path.stat().st_ino for path in checkpoint_paths}
    if source_inodes & target_inodes:
        raise RuntimeError("A source checkpoint was linked into the new run")
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "independent_retrain": True,
        "source_checkpoint_reuse": False,
        "action_oof_jobs": int(action_oof["jobs"]),
        "occupancy_oof_jobs": int(occupancy_oof["jobs"]),
        "fullfit_action_jobs": int(calibration_training["action_jobs"]),
        "fullfit_occupancy_jobs": int(calibration_training["occupancy_jobs"]),
        "action_oof_rows": int(action_oof_evaluation["rows"]),
        "occupancy_oof_rows": int(occupancy_oof_evaluation["rows"]),
        "checkpoint_hashes": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in checkpoint_paths
        },
        "final_accessed": False,
    }
    if report["action_oof_jobs"] != CORE.expected_oof_action_jobs(PROTOCOL):
        raise RuntimeError("Action OOF job count changed")
    if report["occupancy_oof_jobs"] != CORE.expected_oof_occupancy_jobs(PROTOCOL):
        raise RuntimeError("Occupancy OOF job count changed")
    if (
        report["fullfit_action_jobs"] + report["fullfit_occupancy_jobs"]
        != CORE.expected_fullfit_jobs(PROTOCOL)
    ):
        raise RuntimeError("Full-fit job count changed")
    atomic_json(report_path, report)
    atomic_json(
        completion, {"complete": True, "report_sha256": file_sha256(report_path)}
    )
    return report


def run_calibration() -> dict[str, Any]:
    run_training()
    return SOURCE.run_calibration_evaluation()


def run_full_val() -> dict[str, Any]:
    report_path = RUNS / "paper_full_val/report.json"
    completion = RUNS / "paper_full_val/completion.json"
    if completion.is_file():
        return read_json(report_path)
    calibration = run_calibration()
    original = SOURCE.run_calibration_evaluation
    forced_calibration = dict(calibration)
    forced_calibration.update(
        {
            "open_posthoc_final": True,
            "original_open_posthoc_final": bool(calibration["open_posthoc_final"]),
            "full_val_unconditional": True,
            "confirmatory": False,
        }
    )
    SOURCE.run_calibration_evaluation = lambda: forced_calibration
    try:
        prediction = SOURCE.run_final_prediction()
        evaluation = SOURCE.run_final_evaluation()
    finally:
        SOURCE.run_calibration_evaluation = original
    if not bool(evaluation.get("evaluated", False)):
        raise RuntimeError("Unconditional full-val evaluation did not run")
    rows = read_csv(RUNS / "final/evaluation/results.csv")
    if len(rows) != CORE.expected_result_rows(PROTOCOL):
        raise RuntimeError(f"Expected 36 final rows, got {len(rows)}")
    for row in rows:
        if row["mode"] == "fixed_byte":
            CORE.assert_capacity(
                float(row["request_bytes"]), float(row["capacity_bytes"])
            )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "status": PROTOCOL["status"],
        "confirmatory": False,
        "prior_calibration_gate_failed": True,
        "rerun_calibration_gate": calibration["primary_gate"],
        "full_val_unconditional": True,
        "final_is_hidden": False,
        "final_used_for_selection": False,
        "prediction": prediction,
        "evaluation": evaluation,
        "rows": len(rows),
    }
    atomic_json(report_path, report)
    atomic_json(
        completion, {"complete": True, "report_sha256": file_sha256(report_path)}
    )
    return report


def run_oric_bootstrap() -> dict[str, Any]:
    output = RUNS / "bootstrap_oric"
    report_path = output / "report.json"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(report_path)
    run_full_val()
    new_selections = read_json(RUNS / "final/evaluation/selections.json")
    frozen_selections = read_json(
        SOURCE.SOURCES["full_benchmark"] / "runs/coco/point/selections.json"
    )
    data = SOURCE.load_role("final")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    ground_truth = read_json(SOURCE.COMMON / "final/gt.json")
    comparisons = []
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        candidate_key = f"AFPD-OQR|fixed_byte|{budget:g}"
        comparator_key = f"ORIC-MLP|SharedAction-NoROI|fixed_byte|{budget:g}"
        result_path = output / f"{budget:g}_AFPD_vs_ORIC.json"
        if result_path.is_file():
            item = read_json(result_path)
            values = np.asarray(item["values"], dtype=np.float64)
        else:
            candidate = SOURCE._selection_from_json(new_selections, candidate_key)
            comparator = SOURCE._selection_from_json(
                frozen_selections, comparator_key
            )
            candidate_predictions, _ = SOURCE.V1.REL.canonical.mixed_predictions(
                data, candidate, 0.001
            )
            comparator_predictions, _ = SOURCE.V1.REL.canonical.mixed_predictions(
                data, comparator, 0.001
            )
            seed = int(PROTOCOL["bootstrap_seed"]) + budget_index * 101
            generator = np.random.default_rng(seed)
            indices = generator.integers(
                0,
                len(image_ids),
                size=(int(PROTOCOL["bootstrap_replicates"]), len(image_ids)),
                dtype=np.int32,
            )
            SOURCE.BOOTSTRAP_CORE.configure_bootstrap(
                ground_truth,
                image_ids,
                candidate_predictions,
                comparator_predictions,
            )
            values = np.empty(len(indices), dtype=np.float64)
            workers = min(
                int(SOURCE.PROTOCOL["cpu"]["bootstrap_workers"]),
                max(1, (os.cpu_count() or 1) - 2),
            )
            context = mp.get_context("fork")
            try:
                with context.Pool(processes=workers) as pool:
                    tasks = ((index, indices[index]) for index in range(len(indices)))
                    for count, (replicate, delta) in enumerate(
                        pool.imap_unordered(
                            SOURCE.BOOTSTRAP_CORE.bootstrap_worker,
                            tasks,
                            chunksize=1,
                        ),
                        start=1,
                    ):
                        values[int(replicate)] = float(delta)
                        if count % 100 == 0 or count == len(indices):
                            print(
                                f"[bootstrap:AFPD_vs_ORIC:{budget:g}] "
                                f"{count}/{len(indices)}",
                                flush=True,
                            )
            finally:
                SOURCE.BOOTSTRAP_CORE.clear_bootstrap()
            item = {
                "protocol_id": PROTOCOL["protocol_id"],
                "budget": budget,
                "candidate": "AFPD-OQR",
                "candidate_key": candidate_key,
                "comparator": "ORIC-MLP",
                "comparator_key": comparator_key,
                "comparison_type": "AFPD_vs_ORIC",
                "replicates": len(values),
                "seed": seed,
                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "resampling_unit": "image",
                "values": values.tolist(),
            }
            atomic_json(result_path, item)
        comparisons.append(
            {key: value for key, value in item.items() if key != "values"}
            | SOURCE._bootstrap_summary(values)
        )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "comparison_type": "AFPD_vs_ORIC",
        "comparisons": comparisons,
        "replicates_per_comparison": int(PROTOCOL["bootstrap_replicates"]),
        "confirmatory": False,
        "final_is_hidden": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion, {"complete": True, "report_sha256": file_sha256(report_path)}
    )
    return report


def run_bootstrap() -> dict[str, Any]:
    run_full_val()
    source_report = SOURCE.run_bootstrap()
    oric_report = run_oric_bootstrap()
    comparisons = list(source_report["comparisons"]) + list(
        oric_report["comparisons"]
    )
    if len(comparisons) != CORE.expected_bootstrap_rows(PROTOCOL):
        raise RuntimeError(f"Expected 15 bootstrap comparisons, got {len(comparisons)}")
    return {
        "protocol_id": PROTOCOL["protocol_id"],
        "comparisons": comparisons,
        "replicates_per_comparison": int(PROTOCOL["bootstrap_replicates"]),
        "confirmatory": False,
        "final_is_hidden": False,
    }


def _fixed_byte_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in read_csv(path) if row["mode"] == "fixed_byte"]


def run_report() -> dict[str, Any]:
    report_path = RUNS / "REPORT.json"
    completion = RUNS / "COMPLETED.json"
    if completion.is_file():
        return read_json(report_path)
    training = run_training()
    calibration = run_calibration()
    full_val = run_full_val()
    bootstrap = run_bootstrap()
    final_rows = _fixed_byte_rows(RUNS / "final/evaluation/results.csv")
    reference_rows = _fixed_byte_rows(
        RUNS / "final/evaluation/frozen_references.csv"
    )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "status": PROTOCOL["status"],
        "confirmatory": False,
        "prior_calibration_gate_failed": True,
        "rerun_calibration_gate": calibration["primary_gate"],
        "full_val_unconditional": True,
        "final_is_hidden": False,
        "final_used_for_selection": False,
        "training": training,
        "full_val": full_val,
        "bootstrap": bootstrap,
    }
    atomic_json(report_path, report)
    methods = list(PROTOCOL["methods"])
    lines = [
        "# COCO AFPD 独立重训练完整评测",
        "",
        "本实验从头训练全部 AFPD/Occupancy 模型，并完整评测 val2017=5,000。",
        "val2017 已被历史实验使用，因此该结果是 post-hoc 论文格式评测，不是新的隐藏 final。",
        "",
        f'- prior calibration gate failed: **true**',
        f'- rerun calibration gate passed: **{str(bool(calibration["primary_gate"]["passed"])).lower()}**',
        "- full val evaluated unconditionally: **true**",
        "",
        "## Full val Fixed-byte",
        "",
        "| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        cells = []
        for budget in map(float, PROTOCOL["budgets"]):
            row = next(
                row
                for row in final_rows
                if row["method"] == method
                and math.isclose(float(row["budget"]), budget)
            )
            cells.append(
                f'{float(row["official_ap50_95"]):.6f} / '
                f'{float(row["kb_per_image"]):.2f}'
            )
        lines.append(f'| {method} | ' + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Frozen external references",
            "",
            "| 方法 | 10% AP | 25% AP | 40% AP |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, policy in (
        ("Raw-ActionValue-OQ", "Joint-OQ"),
        ("ORIC-MLP", "SharedAction-NoROI"),
        ("MORIC-MLP", "SharedAction-NoROI"),
    ):
        cells = []
        for budget in map(float, PROTOCOL["budgets"]):
            row = next(
                row
                for row in reference_rows
                if row["method"] == method
                and row.get("policy") == policy
                and math.isclose(float(row["budget"]), budget)
            )
            cells.append(f'{float(row["official_map50_95"]):.6f}')
        lines.append(f'| {method} | ' + " | ".join(cells) + " |")
    notes = ROOT / "notes/RESULTS_ZH.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    temporary = notes.with_name(f".{notes.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, notes)
    atomic_json(
        completion,
        {
            "complete": True,
            "report_sha256": file_sha256(report_path),
            "notes_sha256": file_sha256(notes),
            "confirmatory": False,
            "final_is_hidden": False,
        },
    )
    return report


def run_all() -> dict[str, Any]:
    outputs = {}
    for name, function in (
        ("preflight", run_preflight),
        ("training", run_training),
        ("calibration", run_calibration),
        ("full_val", run_full_val),
        ("bootstrap", run_bootstrap),
        ("report", run_report),
    ):
        print(f"[stage] {name}", flush=True)
        outputs[name] = function()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "training", "calibration", "full_val", "bootstrap", "report", "all"),
        default="all",
    )
    args = parser.parse_args()
    functions = {
        "preflight": run_preflight,
        "training": run_training,
        "calibration": run_calibration,
        "full_val": run_full_val,
        "bootstrap": run_bootstrap,
        "report": run_report,
        "all": run_all,
    }
    result = functions[args.stage]()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
