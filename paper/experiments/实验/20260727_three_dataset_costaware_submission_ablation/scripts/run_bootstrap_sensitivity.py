#!/usr/bin/env python3
"""Replay frozen audit comparisons at 200/1000/2000 bootstrap replicates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

from submission_extensions_core import (
    runtime_manifest,
    summarize_bootstrap_prefixes,
    validated_parallel_workers,
)


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_20 = REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
RISK = REPO / "paper/experiments/20260725_three_dataset_costaware_selector_riskgate"
sys.path.insert(0, str(SOURCE_20 / "scripts"))
sys.path.insert(0, str(RISK / "scripts"))
import run_audit_gate as source_audit  # type: ignore  # noqa: E402
import run_ena24_strict as strict  # type: ignore  # noqa: E402
import run_final_paper_evaluation as final_eval  # type: ignore  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def write_checkpoint_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    write_json(temporary, value)
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(map(dict, rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in values for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    normalized = sorted({Path(path).resolve() for path in paths})
    missing = [str(path) for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing sensitivity inputs:\n" + "\n".join(missing))
    return {str(path): file_sha256(path) for path in normalized}


def source_paths(dataset: str) -> list[Path]:
    if dataset in {"coco", "oregon"}:
        paths = source_audit.audit_input_paths(dataset)
    else:
        paths = [
            *strict._calibration_manifest_inputs(),
            RISK / "runs/ena24/strict_split_manifest.json",
            RISK / "runs/ena24/strict_provisional_choices.json",
            RISK / "runs/ena24/strict_candidate_table.csv",
            RISK / "runs/ena24/strict_threshold_choices.json",
            RISK
            / "runs/ena24/strict_roles/near_duplicate_grouped_selector/gt.json",
            RISK / "runs/ena24/strict_roles/near_duplicate_grouped_gate/gt.json",
        ]
    return [
        *paths,
        EXP_ROOT / "configs/extensions_protocol.json",
        SCRIPT,
        EXP_ROOT / "scripts/submission_extensions_core.py",
    ]


def old_rows(dataset: str) -> dict[tuple[str, float], dict[str, Any]]:
    name = (
        "strict_risk_gate_checkpoint.json"
        if dataset == "ena24"
        else "audit_gate_checkpoint.json"
    )
    rows = read_json(RISK / "runs" / dataset / name)["rows"]
    return {(str(row["mode"]), float(row["budget"])): row for row in rows}


def source_comparisons(
    dataset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons = []
    if dataset in {"coco", "oregon"}:
        runtime = source_audit._runtime(dataset)
        choices = read_json(RISK / "runs" / dataset / "provisional_choices.json")
        for choice in choices:
            baseline = source_audit._configuration_actions(
                choice, prefix="reference", runtime=runtime
            )
            candidate = source_audit._configuration_actions(
                choice, prefix="provisional", runtime=runtime
            )
            comparisons.append(
                {
                    "mode": str(choice["mode"]),
                    "budget": float(choice["budget"]),
                    "reference_ranker": str(choice["reference_ranker"]),
                    "reference_policy": str(choice["reference_policy"]),
                    "provisional_ranker": str(choice["provisional_ranker"]),
                    "provisional_policy": str(choice["provisional_policy"]),
                    "baseline_actions": baseline,
                    "candidate_actions": candidate,
                }
            )
        return runtime["audit"], comparisons

    nine, tri = strict._load_engine()
    calibration = tri.current.load_role("ena24", "calibration")
    split = read_json(RISK / "runs/ena24/strict_split_manifest.json")
    selector = ena24_read_only_role(
        calibration,
        split["selector_image_ids"],
        role="near_duplicate_grouped_selector",
        tri=tri,
    )
    gate = ena24_read_only_role(
        calibration,
        split["gate_image_ids"],
        role="near_duplicate_grouped_gate",
        tri=tri,
    )
    state = strict._load_frozen_state(nine, tri, selector, gate)
    choices = read_json(RISK / "runs/ena24/strict_provisional_choices.json")
    candidates = strict._candidate_lookup()
    thresholds = strict._policy_thresholds()
    for choice in choices:
        mode = str(choice["mode"])
        budget = float(choice["budget"])
        selected_actions = []
        for key in ("reference", "provisional"):
            selected = dict(choice[key])
            candidate = candidates[
                (mode, budget, str(selected["ranker"]), str(selected["policy"]))
            ]
            selected_actions.append(
                strict._selected_actions(
                    tri=tri,
                    data=gate,
                    selector_scores=state["scores"]["selector"][selected["ranker"]],
                    gate_scores=state["scores"]["gate"][selected["ranker"]],
                    base_actions=state["base_actions"]["gate"],
                    roi_scores=state["roi_scores"]["gate"],
                    mode=mode,
                    budget=budget,
                    candidate=candidate,
                    threshold=thresholds[mode][selected["policy"]],
                )
            )
        comparisons.append(
            {
                "mode": mode,
                "budget": budget,
                "reference_ranker": str(choice["reference"]["ranker"]),
                "reference_policy": str(choice["reference"]["policy"]),
                "provisional_ranker": str(choice["provisional"]["ranker"]),
                "provisional_policy": str(choice["provisional"]["policy"]),
                "baseline_actions": selected_actions[0],
                "candidate_actions": selected_actions[1],
            }
        )
    return gate, comparisons


def ena24_read_only_role(
    calibration: Mapping[str, Any],
    image_ids: Iterable[int],
    *,
    role: str,
    tri: Any,
) -> dict[str, Any]:
    selected = set(map(int, image_ids))
    samples = [
        sample
        for sample in calibration["samples"]
        if int(sample["image_id"]) in selected
    ]
    if {int(sample["image_id"]) for sample in samples} != selected:
        raise RuntimeError(f"ENA24 {role} samples differ from the frozen split")
    gt_path = RISK / "runs/ena24/strict_roles" / role / "gt.json"
    return {
        "dataset": "ena24",
        "role": role,
        "root": gt_path.parent,
        "samples": samples,
        "sample_by_id": {
            int(sample["image_id"]): sample for sample in samples
        },
        "image_ids": sorted(selected),
        "gt_path": gt_path,
        "gt_map": {
            int(image_id): rows
            for image_id, rows in calibration["gt_map"].items()
            if int(image_id) in selected
        },
        "actions": {
            name: {
                int(image_id): predictions
                for image_id, predictions in action_rows.items()
                if int(image_id) in selected
            }
            for name, action_rows in calibration["actions"].items()
        },
        "evaluator": tri.current.CocoEvaluator(gt_path, sorted(selected)),
    }


def action_pair_digest(
    baseline: Mapping[int, str], candidate: Mapping[int, str]
) -> str:
    payload = json.dumps(
        [sorted(baseline.items()), sorted(candidate.items())], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cost_by_image(data: Mapping[str, Any], actions: Mapping[int, str]) -> dict[int, int]:
    return {
        image_id: (
            int(data["sample_by_id"][image_id]["payload_bytes"][actions[image_id]])
            if image_id in actions
            else 0
        )
        for image_id in map(int, data["image_ids"])
    }


_BOOTSTRAP_WORKER_STATE: dict[str, Any] | None = None


def _evaluate_bootstrap_replicate(replicate: int) -> tuple[float, float]:
    state = _BOOTSTRAP_WORKER_STATE
    if state is None:
        raise RuntimeError("Bootstrap worker state is not initialized")
    image_ids = state["image_ids"]
    source_ids = [image_ids[int(index)] for index in state["sampled"][replicate]]
    sampled_gt, prediction_sets = final_eval.build_bootstrap_coco_sample(
        state["ground_truth"],
        {
            "baseline": state["baseline_predictions"],
            "candidate": state["candidate_predictions"],
        },
        sampled_image_ids=source_ids,
    )
    gt_path = state["temporary_root"] / f"gt_{os.getpid()}_{replicate:04d}.json"
    write_json(gt_path, sampled_gt)
    evaluation_ids = list(range(1, len(source_ids) + 1))
    evaluator_type = state["evaluator_type"]
    try:
        baseline_ap = float(
            evaluator_type(gt_path, evaluation_ids).evaluate(
                prediction_sets["baseline"]
            )["map50_95"]
        )
        candidate_ap = float(
            evaluator_type(gt_path, evaluation_ids).evaluate(
                prediction_sets["candidate"]
            )["map50_95"]
        )
    finally:
        gt_path.unlink(missing_ok=True)
    kb_delta = (
        sum(state["candidate_cost"][image_id] for image_id in source_ids)
        - sum(state["baseline_cost"][image_id] for image_id in source_ids)
    ) / len(source_ids) / 1024.0
    return candidate_ap - baseline_ap, kb_delta


def raw_bootstrap(
    data: Mapping[str, Any],
    *,
    baseline_actions: Mapping[int, str],
    candidate_actions: Mapping[int, str],
    replicates: int,
    seed: int,
    checkpoint_path: Path,
    identity: Mapping[str, Any],
    label: str,
    workers: int = 1,
) -> tuple[list[float], list[float]]:
    workers = validated_parallel_workers(workers)
    if baseline_actions == candidate_actions:
        return [0.0] * replicates, [0.0] * replicates
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if {key: checkpoint.get(key) for key in identity} != dict(identity):
            raise RuntimeError(f"Raw bootstrap checkpoint identity changed: {label}")
        ap_deltas = list(map(float, checkpoint.get("ap_deltas", [])))
        kb_deltas = list(map(float, checkpoint.get("kb_deltas", [])))
    else:
        ap_deltas = []
        kb_deltas = []
    if len(ap_deltas) != len(kb_deltas) or len(ap_deltas) > replicates:
        raise RuntimeError(f"Invalid raw bootstrap checkpoint: {label}")

    baseline_predictions, _ = final_eval.mixed_predictions(
        data, dict(baseline_actions), 0.001
    )
    candidate_predictions, _ = final_eval.mixed_predictions(
        data, dict(candidate_actions), 0.001
    )
    ground_truth = final_eval.read_json(data["gt_path"])
    image_ids = list(map(int, data["image_ids"]))
    sampled = final_eval.deterministic_bootstrap_indices(
        len(image_ids), replicates=replicates, seed=seed
    )
    baseline_cost = cost_by_image(data, baseline_actions)
    candidate_cost = cost_by_image(data, candidate_actions)
    evaluator_type = type(data["evaluator"])
    remaining = range(len(ap_deltas), replicates)
    with tempfile.TemporaryDirectory(prefix=f"{data['dataset']}_sensitivity_") as tmp:
        global _BOOTSTRAP_WORKER_STATE
        _BOOTSTRAP_WORKER_STATE = {
            "baseline_predictions": baseline_predictions,
            "candidate_predictions": candidate_predictions,
            "ground_truth": ground_truth,
            "image_ids": image_ids,
            "sampled": sampled,
            "baseline_cost": baseline_cost,
            "candidate_cost": candidate_cost,
            "evaluator_type": evaluator_type,
            "temporary_root": Path(tmp),
        }
        if workers == 1:
            results = map(_evaluate_bootstrap_replicate, remaining)
            pool = None
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("Parallel bootstrap requires fork process support")
            pool = mp.get_context("fork").Pool(processes=workers)
            results = pool.imap(_evaluate_bootstrap_replicate, remaining, chunksize=1)
        completed_normally = False
        try:
            for replicate, (ap_delta, kb_delta) in zip(remaining, results):
                ap_deltas.append(ap_delta)
                kb_deltas.append(kb_delta)
                if (replicate + 1) % 10 == 0 or replicate + 1 == replicates:
                    write_checkpoint_json(
                        checkpoint_path,
                        {
                            **dict(identity),
                            "ap_deltas": ap_deltas,
                            "kb_deltas": kb_deltas,
                        },
                    )
                if (replicate + 1) % 25 == 0:
                    print(
                        f"[{data['dataset']}-sensitivity] {label} "
                        f"{replicate + 1}/{replicates}",
                        flush=True,
                    )
            completed_normally = True
        finally:
            if pool is not None:
                if completed_normally:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()
            _BOOTSTRAP_WORKER_STATE = None
    return ap_deltas, kb_deltas


def reconcile_200(
    expected_row: Mapping[str, Any], observed: Mapping[str, Any], *, tolerance: float = 1e-12
) -> None:
    expected = expected_row.get("bootstrap", expected_row)
    for field in (
        "official_map50_95_delta_mean",
        "official_map50_95_delta_ci_low",
        "official_map50_95_delta_ci_high",
        "official_map50_95_probability_gt_zero",
        "request_kb_per_image_delta_mean",
        "request_kb_per_image_delta_ci_low",
        "request_kb_per_image_delta_ci_high",
    ):
        if not math.isclose(
            float(expected[field]), float(observed[field]), abs_tol=tolerance
        ):
            raise RuntimeError(
                f"200-replicate sensitivity does not reproduce {field}: "
                f"{observed[field]} != {expected[field]}"
            )


def run(dataset: str, *, preflight_only: bool = False) -> list[dict[str, Any]]:
    protocol = read_json(EXP_ROOT / "configs/extensions_protocol.json")
    if dataset not in protocol["datasets"]:
        raise ValueError(f"Unsupported dataset: {dataset}")
    paths = source_paths(dataset)
    before = snapshot(paths)
    data, comparisons = source_comparisons(dataset)
    expected_rows = old_rows(dataset)
    if preflight_only:
        if len(comparisons) != 10:
            raise RuntimeError(f"Unexpected comparison count: {len(comparisons)}")
        for comparison in comparisons:
            key = (str(comparison["mode"]), float(comparison["budget"]))
            digest = action_pair_digest(
                comparison["baseline_actions"], comparison["candidate_actions"]
            )
            if digest != str(expected_rows[key]["action_pair_sha256"]):
                raise RuntimeError(f"Action reconstruction drift at {key}")
        return [
            {
                "dataset": dataset,
                "inputs": len(before),
                "audit_images": len(data["samples"]),
                "action_pairs": len(comparisons),
                "action_digests_reproduced": True,
                "preflight": True,
            }
        ]

    output = EXP_ROOT / "runs/bootstrap_sensitivity" / dataset
    raw_root = output / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    checkpoints = tuple(map(int, protocol["bootstrap_checkpoints"]))
    maximum = max(checkpoints)
    seed = int(protocol["bootstrap_seed"])
    summaries = []
    for comparison in comparisons:
        mode = str(comparison["mode"])
        budget = float(comparison["budget"])
        digest = action_pair_digest(
            comparison["baseline_actions"], comparison["candidate_actions"]
        )
        label = f"{mode}-{budget:.2f}"
        identity = {
            "dataset": dataset,
            "mode": mode,
            "budget": budget,
            "action_pair_sha256": digest,
            "bootstrap_seed": seed,
            "maximum_replicates": maximum,
            "protocol_sha256": file_sha256(
                EXP_ROOT / "configs/extensions_protocol.json"
            ),
        }
        ap_deltas, kb_deltas = raw_bootstrap(
            data,
            baseline_actions=comparison["baseline_actions"],
            candidate_actions=comparison["candidate_actions"],
            replicates=maximum,
            seed=seed,
            checkpoint_path=raw_root / f"{label}-{digest[:12]}.json",
            identity=identity,
            label=label,
        )
        rows = summarize_bootstrap_prefixes(
            ap_deltas=ap_deltas,
            kb_deltas=kb_deltas,
            checkpoints=checkpoints,
            mode=mode,
            ap_tolerance=float(protocol["ap_tolerance"]),
        )
        reconcile_200(expected_rows[(mode, budget)], rows[0])
        gate_200 = bool(rows[0]["gate_passed"])
        for row in rows:
            summaries.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "budget": budget,
                    "reference_ranker": comparison["reference_ranker"],
                    "reference_policy": comparison["reference_policy"],
                    "provisional_ranker": comparison["provisional_ranker"],
                    "provisional_policy": comparison["provisional_policy"],
                    "action_pair_sha256": digest,
                    "identical_action_shortcut": comparison["baseline_actions"]
                    == comparison["candidate_actions"],
                    "gate_flip_vs_200": bool(row["gate_passed"]) != gate_200,
                    **row,
                }
            )
        write_csv(output / "sensitivity.csv", summaries)
        print(f"[{dataset}-sensitivity] {label} complete", flush=True)

    expected_count = len(comparisons) * len(checkpoints)
    if len(summaries) != expected_count:
        raise RuntimeError(f"Sensitivity grid incomplete: {len(summaries)} != {expected_count}")
    after = snapshot(paths)
    if before != after:
        raise RuntimeError("A frozen input changed during bootstrap sensitivity")
    write_json(
        output / "provenance.json",
        {
            "protocol_sha256": file_sha256(
                EXP_ROOT / "configs/extensions_protocol.json"
            ),
            "input_sha256": before,
            "input_sha256_after": after,
            "final_artifacts_read": False,
            "pointwise_not_simultaneous": True,
            "percentile_interval_only": True,
            "rows": len(summaries),
            "runtime": runtime_manifest(),
        },
    )
    (output / "COMPLETED").write_text("ok\n", encoding="ascii")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "ena24", "oregon"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.dataset, preflight_only=args.preflight_only),
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
