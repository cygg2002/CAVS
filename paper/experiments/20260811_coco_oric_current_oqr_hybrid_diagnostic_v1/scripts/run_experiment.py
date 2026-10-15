#!/usr/bin/env python3
"""Run the frozen ORIC-anchor plus Current-action COCO diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(REPO))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("oric_current_oqr_hybrid_core", ROOT / "scripts/core.py")
SOURCES = {name: REPO / path for name, path in PROTOCOL["sources"].items()}
STRICT = load_module(
    "oric_current_oqr_hybrid_strict_source",
    SOURCES["strict_current"] / "scripts/run_experiment.py",
)
sys.path.insert(0, str(SOURCES["external"] / "scripts"))
EXTERNAL = load_module(
    "oric_current_oqr_hybrid_external_source",
    SOURCES["external"] / "scripts/run_dataset.py",
)
CURRENT = STRICT.SOURCE
CORE.validate_protocol(PROTOCOL)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def source_paths() -> list[Path]:
    return [
        ROOT / "configs/protocol.json",
        ROOT / "scripts/core.py",
        SCRIPT,
        SOURCES["strict_current"] / "configs/protocol.json",
        SOURCES["strict_current"] / "scripts/run_experiment.py",
        SOURCES["current_fullval"] / "runs/final/scores.pkl",
        SOURCES["current_fullval"] / "runs/final/evaluation/results.csv",
        SOURCES["external"] / "configs/protocol.json",
        SOURCES["external"] / "scripts/run_dataset.py",
        SOURCES["external"] / "runs/coco/evaluation/oric_mlp.pkl",
        SOURCES["external"] / "runs/coco/rewards/fit_artifacts.npz",
        SOURCES["full_benchmark"] / "runs/coco/point/final_results.csv",
    ]


_INPUT_CACHE: dict[str, Any] = {}


def frozen_inputs() -> dict[str, Any]:
    if _INPUT_CACHE:
        return _INPUT_CACHE
    data = CURRENT.load_role("final")
    samples = data["samples"]
    image_ids = [int(sample["image_id"]) for sample in samples]
    artifact = read_pickle(SOURCES["current_fullval"] / "runs/final/scores.pkl")
    current_maps: dict[str, dict[int, dict[str, float]]] = {}
    for method, action_space in (("Current-OQ", "OQ"), ("Current-OQR", "OQR")):
        records = CURRENT.records_for_role("final", action_space)
        record_ids = [int(record["image_id"]) for record in records]
        artifact_ids = list(map(int, artifact["image_ids"][action_space]))
        if record_ids != image_ids or artifact_ids != image_ids:
            raise RuntimeError(f"Current score/image order changed: {method}")
        matrix = STRICT.CORE.ensemble_scores(
            artifact["action_scores"][method], PROTOCOL["current_seeds"]
        )
        current_maps[method] = CURRENT.V1._score_map(records, matrix)

    external_protocol = read_json(SOURCES["external"] / "configs/protocol.json")
    with np.load(
        SOURCES["external"] / "runs/coco/rewards/fit_artifacts.npz",
        allow_pickle=False,
    ) as fit_artifact:
        category_ids = list(map(int, fit_artifact["category_ids"]))
    checkpoint = read_pickle(
        SOURCES["external"] / "runs/coco/evaluation/oric_mlp.pkl"
    )
    features = EXTERNAL._role_features(
        data, category_ids, int(external_protocol["oric"]["top_k"])
    )
    values = np.asarray(EXTERNAL.predict_reward_mlp(checkpoint, features), dtype=np.float64)
    if values.shape != (len(image_ids),) or not np.isfinite(values).all():
        raise RuntimeError("Frozen ORIC score vector is invalid")
    oric_scores = dict(zip(image_ids, map(float, values)))
    methods = {
        "Current-OQ": current_maps["Current-OQ"],
        "Current-OQR": current_maps["Current-OQR"],
        "ORIC-Current-OQ": CORE.compose_hybrid_scores(
            image_ids,
            oric_scores,
            current_maps["Current-OQ"],
            actions=PROTOCOL["action_spaces"]["ORIC-Current-OQ"],
        ),
        "ORIC-Current-OQR": CORE.compose_hybrid_scores(
            image_ids,
            oric_scores,
            current_maps["Current-OQR"],
            actions=PROTOCOL["action_spaces"]["ORIC-Current-OQR"],
        ),
    }
    _INPUT_CACHE.update(
        data=data,
        samples=samples,
        image_ids=image_ids,
        oric_scores=oric_scores,
        method_scores=methods,
    )
    return _INPUT_CACHE


def run_preflight() -> dict[str, Any]:
    output = RUNS / "preflight/report.json"
    completion = RUNS / "preflight/completion.json"
    if completion.is_file():
        return read_json(output)
    missing = [str(path) for path in source_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    inputs = frozen_inputs()
    if len(inputs["image_ids"]) != int(PROTOCOL["roles"]["final"]):
        raise RuntimeError("Canonical final population changed")
    hashes = {
        str(path.relative_to(REPO)): file_sha256(path) for path in source_paths()
    }
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "final_images": len(inputs["image_ids"]),
        "methods": list(PROTOCOL["methods"]),
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "detector_inference": False,
        "selector_retraining": False,
        "final_is_hidden": False,
        "final_used_for_selection": False,
        "confirmatory": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output)})
    return report


def evaluate_selection(
    data: Mapping[str, Any],
    *,
    method: str,
    mode: str,
    budget: float,
    selection: Mapping[int, str],
    capacity_bytes: float,
) -> dict[str, Any]:
    chosen = {int(image_id): str(action) for image_id, action in selection.items()}
    predictions, _ = CURRENT.V1.REL.canonical.mixed_predictions(data, chosen, 0.001)
    metric = data["evaluator"].evaluate(predictions)
    request_bytes = sum(
        int(data["sample_by_id"][image_id]["payload_bytes"][action])
        for image_id, action in chosen.items()
    )
    expected_calls = int(math.floor(len(data["samples"]) * float(budget) + 0.5))
    if mode == "fixed_byte":
        CORE.assert_capacity(request_bytes, capacity_bytes)
        utilization = request_bytes / max(1.0, capacity_bytes)
    else:
        if len(chosen) != expected_calls:
            raise RuntimeError("Fixed-call selection count changed")
        utilization = len(chosen) / max(1, expected_calls)
    allowed = set(PROTOCOL["action_spaces"][method])
    if not set(chosen.values()) <= allowed:
        raise RuntimeError(f"Forbidden action selected by {method}")
    counts = Counter(chosen.values())
    return {
        "method": method,
        "mode": mode,
        "budget": float(budget),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": int(request_bytes),
        "capacity_bytes": float(capacity_bytes),
        "capacity_utilization": float(utilization),
        "selected_images": len(chosen),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "final_is_hidden": False,
        "final_used_for_selection": False,
        "confirmatory": False,
    }


def _selection_from_json(values: Mapping[str, Any], key: str) -> dict[int, str]:
    return {int(image_id): str(action) for image_id, action in values[key].items()}


def run_evaluation() -> dict[str, Any]:
    output = RUNS / "evaluation/report.json"
    completion = RUNS / "evaluation/completion.json"
    if completion.is_file():
        return read_json(output)
    preflight = run_preflight()
    inputs = frozen_inputs()
    data, samples = inputs["data"], inputs["samples"]
    original_total = sum(int(row["payload_bytes"]["original"]) for row in samples)
    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    for method in PROTOCOL["methods"]:
        score_map = inputs["method_scores"][method]
        for mode in PROTOCOL["modes"]:
            for budget in map(float, PROTOCOL["budgets"]):
                capacity = budget * original_total
                selection = CURRENT._schedule(
                    samples,
                    score_map,
                    mode=mode,
                    budget=budget,
                    capacity_bytes=capacity,
                )
                key = f"{method}|{mode}|{budget:g}"
                selections[key] = {
                    str(image_id): action for image_id, action in selection.items()
                }
                rows.append(
                    evaluate_selection(
                        data,
                        method=method,
                        mode=mode,
                        budget=budget,
                        selection=selection,
                        capacity_bytes=capacity,
                    )
                )
    expected_rows = 24
    keys = {(row["method"], row["mode"], row["budget"]) for row in rows}
    if len(rows) != expected_rows or len(keys) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} unique rows, got {len(rows)}")

    frozen_current = read_csv(
        SOURCES["current_fullval"] / "runs/final/evaluation/results.csv"
    )
    reproduction = []
    for method in ("Current-OQ", "Current-OQR"):
        for budget in map(float, PROTOCOL["budgets"]):
            observed = next(
                row for row in rows
                if row["method"] == method
                and row["mode"] == "fixed_byte"
                and math.isclose(float(row["budget"]), budget)
            )
            expected = next(
                row for row in frozen_current
                if row["method"] == method
                and row["seed"] == "ensemble"
                and row["mode"] == "fixed_byte"
                and math.isclose(float(row["budget"]), budget)
            )
            delta = float(observed["official_ap50_95"]) - float(
                expected["official_ap50_95"]
            )
            if abs(delta) > 1e-9:
                raise RuntimeError(f"Current reproduction failed: {method}/{budget}")
            reproduction.append({"method": method, "budget": budget, "delta_ap": delta})

    benchmark = read_csv(
        SOURCES["full_benchmark"] / "runs/coco/point/final_results.csv"
    )
    controls = []
    for budget in map(float, PROTOCOL["budgets"]):
        count = int(math.floor(len(samples) * budget + 0.5))
        ordered = sorted(
            inputs["image_ids"],
            key=lambda image_id: (-inputs["oric_scores"][image_id], image_id),
        )
        selection = {image_id: "original" for image_id in ordered[:count]}
        control = evaluate_selection(
            data,
            method="Current-OQ",
            mode="fixed_call",
            budget=budget,
            selection=selection,
            capacity_bytes=budget * original_total,
        )
        expected = next(
            row for row in benchmark
            if row["method"] == "ORIC-MLP"
            and row["policy"] == "Original-only"
            and row["mode"] == "fixed_call"
            and math.isclose(float(row["budget"]), budget)
        )
        delta = float(control["official_ap50_95"]) - float(expected["official_map50_95"])
        if abs(delta) > 1e-9:
            raise RuntimeError(f"ORIC ranking reproduction failed: {budget}/{delta}")
        controls.append(
            {
                "method": "ORIC-Original-Control",
                "budget": budget,
                "official_ap50_95": control["official_ap50_95"],
                "delta_source": delta,
            }
        )

    atomic_csv(RUNS / "evaluation/results.csv", rows)
    atomic_csv(RUNS / "evaluation/controls.csv", controls)
    atomic_json(RUNS / "evaluation/selections.json", selections)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "rows": len(rows),
        "unique_rows": len(keys),
        "reproduction": reproduction,
        "oric_control": controls,
        "source_digest": preflight["source_digest"],
        "final_is_hidden": False,
        "final_used_for_selection": False,
        "confirmatory": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output)})
    return report


def bootstrap_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_delta_ap": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "probability_positive": float(np.mean(values > 0.0)),
    }


def run_bootstrap() -> dict[str, Any]:
    output = RUNS / "bootstrap/report.json"
    completion = RUNS / "bootstrap/completion.json"
    if completion.is_file():
        return read_json(output)
    run_evaluation()
    inputs = frozen_inputs()
    data, image_ids = inputs["data"], inputs["image_ids"]
    ground_truth = read_json(CURRENT.COMMON / "final/gt.json")
    selections = read_json(RUNS / "evaluation/selections.json")
    predictions = {}
    for method in PROTOCOL["methods"]:
        for budget in map(float, PROTOCOL["budgets"]):
            key = f"{method}|fixed_byte|{budget:g}"
            selection = _selection_from_json(selections, key)
            predictions[(method, budget)], _ = CURRENT.V1.REL.canonical.mixed_predictions(
                data, selection, 0.001
            )

    comparisons = []
    expected_comparisons = 12
    workers = min(
        int(PROTOCOL["bootstrap"]["workers"]),
        max(1, (os.cpu_count() or 1) - 2),
    )
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        seed = int(PROTOCOL["bootstrap"]["seed"]) + budget_index * 1009
        generator = np.random.default_rng(seed)
        indices = generator.integers(
            0,
            len(image_ids),
            size=(int(PROTOCOL["bootstrap"]["replicates"]), len(image_ids)),
            dtype=np.int32,
        )
        indices_sha = hashlib.sha256(indices.tobytes()).hexdigest()
        for candidate, baseline in PROTOCOL["comparisons"]:
            comparison_id = f"{candidate}_vs_{baseline}_{budget:g}".replace("/", "_")
            result_path = RUNS / f"bootstrap/{comparison_id}.json"
            if result_path.is_file():
                item = read_json(result_path)
                values = np.asarray(item["values"], dtype=np.float64)
            else:
                CURRENT.BOOTSTRAP_CORE.configure_bootstrap(
                    ground_truth,
                    image_ids,
                    predictions[(candidate, budget)],
                    predictions[(baseline, budget)],
                )
                values = np.empty(len(indices), dtype=np.float64)
                context = mp.get_context("fork")
                try:
                    with context.Pool(processes=workers) as pool:
                        tasks = ((index, indices[index]) for index in range(len(indices)))
                        for count, (replicate, delta) in enumerate(
                            pool.imap_unordered(
                                CURRENT.BOOTSTRAP_CORE.bootstrap_worker,
                                tasks,
                                chunksize=1,
                            ),
                            start=1,
                        ):
                            values[int(replicate)] = float(delta)
                            if count % 100 == 0 or count == len(indices):
                                print(
                                    f"[bootstrap:{candidate}_vs_{baseline}:{budget:g}] "
                                    f"{count}/{len(indices)}",
                                    flush=True,
                                )
                finally:
                    CURRENT.BOOTSTRAP_CORE.clear_bootstrap()
                item = {
                    "protocol_id": PROTOCOL["protocol_id"],
                    "candidate": candidate,
                    "baseline": baseline,
                    "budget": budget,
                    "replicates": len(values),
                    "seed": seed,
                    "indices_sha256": indices_sha,
                    "resampling_unit": "image",
                    "values": values.tolist(),
                }
                atomic_json(result_path, item)
            comparisons.append(
                {key: value for key, value in item.items() if key != "values"}
                | bootstrap_summary(values)
            )
    if len(comparisons) != expected_comparisons:
        raise RuntimeError(
            f"Expected {expected_comparisons} bootstrap comparisons, got {len(comparisons)}"
        )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "comparisons": comparisons,
        "replicates_per_comparison": int(PROTOCOL["bootstrap"]["replicates"]),
        "workers": workers,
        "final_is_hidden": False,
        "final_used_for_selection": False,
        "confirmatory": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output)})
    return report


def run_report() -> dict[str, Any]:
    output = RUNS / "report/report.json"
    completion = RUNS / "report/completion.json"
    if completion.is_file():
        return read_json(output)
    evaluation = run_evaluation()
    bootstrap = run_bootstrap()
    rows = read_csv(RUNS / "evaluation/results.csv")

    def row(method: str, mode: str, budget: float) -> dict[str, str]:
        return next(
            item for item in rows
            if item["method"] == method
            and item["mode"] == mode
            and math.isclose(float(item["budget"]), float(budget))
        )

    lines = [
        "# COCO ORIC–Current OQR 冻结混合诊断",
        "",
        "> 证据等级：COCO val2017 已消费后的 post-hoc final reuse；没有重新训练、调参或检测器推理。",
        "",
        "## Fixed-byte 主结果",
        "",
        "| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in PROTOCOL["methods"]:
        values = [row(method, "fixed_byte", budget) for budget in PROTOCOL["budgets"]]
        lines.append(
            f"| {method} | "
            + " | ".join(
                f"{float(item['official_ap50_95']):.6f} / {float(item['kb_per_image']):.2f}"
                for item in values
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Fixed-call 辅助结果",
            "",
            "| 方法 | 10% AP | 25% AP | 40% AP |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in PROTOCOL["methods"]:
        values = [row(method, "fixed_call", budget) for budget in PROTOCOL["budgets"]]
        lines.append(
            f"| {method} | "
            + " | ".join(f"{float(item['official_ap50_95']):.6f}" for item in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 关键归因",
            "",
            "| 预算 | ORIC混合OQR − Current-OQR | ORIC混合ROI增益 | Current ROI增益 |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    hybrid_deltas = []
    for budget in map(float, PROTOCOL["budgets"]):
        hybrid_oq = float(row("ORIC-Current-OQ", "fixed_byte", budget)["official_ap50_95"])
        hybrid_oqr = float(row("ORIC-Current-OQR", "fixed_byte", budget)["official_ap50_95"])
        current_oq = float(row("Current-OQ", "fixed_byte", budget)["official_ap50_95"])
        current_oqr = float(row("Current-OQR", "fixed_byte", budget)["official_ap50_95"])
        hybrid_deltas.append(hybrid_oqr - current_oqr)
        lines.append(
            f"| {budget:g} | {hybrid_oqr-current_oqr:+.6f} | "
            f"{hybrid_oqr-hybrid_oq:+.6f} | {current_oqr-current_oq:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap（Fixed-byte）",
            "",
            "| 候选 − 基线 | 预算 | mean ΔAP | 95% CI | Pr(Δ>0) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in bootstrap["comparisons"]:
        lines.append(
            f"| {item['candidate']} − {item['baseline']} | {float(item['budget']):g} | "
            f"{float(item['mean_delta_ap']):+.6f} | "
            f"[{float(item['ci_low']):+.6f}, {float(item['ci_high']):+.6f}] | "
            f"{float(item['probability_positive']):.3f} |"
        )
    positive = sum(delta > 0.0 for delta in hybrid_deltas)
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- ORIC锚定的OQR混合方法在 {positive}/3 个 Fixed-byte 工作点高于 Current-OQR。",
            "- 该混合严格保留 ORIC 的 Original 图片价值，并仅使用 Current 的图内动作优势。",
            "- 结果是机制归因，不是新的隐藏 final 或 confirmatory 证据。",
        ]
    )
    report_path = ROOT / "notes/RESULTS_ZH.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "evaluation": evaluation,
        "bootstrap_comparisons": len(bootstrap["comparisons"]),
        "hybrid_oqr_positive_points": positive,
        "results_path": str(report_path),
        "final_is_hidden": False,
        "final_used_for_selection": False,
        "confirmatory": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "evaluation", "bootstrap", "report", "all"),
        default="all",
    )
    args = parser.parse_args()
    stages = {
        "preflight": run_preflight,
        "evaluation": run_evaluation,
        "bootstrap": run_bootstrap,
        "report": run_report,
        "all": run_report,
    }
    print(json.dumps(stages[args.stage](), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
