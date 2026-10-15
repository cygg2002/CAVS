#!/usr/bin/env python3
"""Gate one frozen provisional selector on an independent audit role."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_20260720 = (
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
)
SOURCE_20260723 = (
    REPO
    / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
)
TRIACTION_ROOT = (
    REPO
    / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration"
)
EXPECTED_AUDIT_IMAGES = {"coco": 630, "oregon": 9923}
CONTROLLED_RANKER = "Current-Controlled-32D"
BOOTSTRAP_CONTEXT_FIELDS = {
    "dataset",
    "mode",
    "budget",
    "nominal_budget",
    "comparison",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in values for key in row})
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


def snapshot_sha256(paths: Iterable[Path]) -> dict[str, str]:
    normalized = sorted({Path(path).resolve() for path in paths})
    missing = [str(path) for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen audit inputs:\n" + "\n".join(missing))
    return {str(path): file_sha256(path) for path in normalized}


def canonical_root(dataset: str) -> Path:
    run = "canonical_oregon_event_v1" if dataset == "oregon" else "canonical_v1"
    return SOURCE_20260720 / "runs" / run


def role_input_paths(dataset: str, role: str) -> list[Path]:
    root = canonical_root(dataset) / "common" / dataset / role
    manifest_path = root / "source_manifest.json"
    manifest = read_json(manifest_path)
    return [
        manifest_path,
        Path(manifest["samples"]),
        Path(manifest["gt"]),
        *(Path(path) for path in manifest["action_sources"].values()),
    ]


def audit_input_paths(dataset: str) -> list[Path]:
    benchmark = canonical_root(dataset) / "benchmark" / dataset
    source_run = SOURCE_20260723 / "runs" / dataset
    output = EXP_ROOT / "runs" / dataset
    return [
        EXP_ROOT / "configs/protocol.json",
        output / "PROVISIONAL_FROZEN",
        output / "provisional_choices.json",
        benchmark / "models.json",
        benchmark / "action_gate.json",
        source_run / "CORE_COMPLETED",
        source_run / "models/current_controlled_32d.json",
        source_run / "models/saferoi_gbdt.pkl",
        *role_input_paths(dataset, "calibration"),
        *role_input_paths(dataset, "audit"),
    ]


def gate_passes(
    *,
    mode: str,
    ap_delta_ci_low: float,
    kb_delta_ci_high: float,
    ap_tolerance: float,
) -> bool:
    """Apply the frozen pointwise AP and mode-specific request-cost gates."""
    if mode not in {"fixed_call", "online", "fixed_byte"}:
        raise ValueError(f"Unsupported mode: {mode}")
    ap_low = float(ap_delta_ci_low)
    if math.isnan(ap_low) or ap_low < -float(ap_tolerance):
        return False
    if mode == "fixed_byte":
        return True
    kb_high = float(kb_delta_ci_high)
    return not math.isnan(kb_high) and kb_high <= 0.0


def freeze_choice(
    choice: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    ap_tolerance: float,
) -> dict[str, Any]:
    """Freeze the provisional configuration or fall back directly to reference."""
    mode = str(choice["mode"])
    passed = gate_passes(
        mode=mode,
        ap_delta_ci_low=float(bootstrap["official_map50_95_delta_ci_low"]),
        kb_delta_ci_high=float(
            bootstrap["request_kb_per_image_delta_ci_high"]
        ),
        ap_tolerance=float(ap_tolerance),
    )
    source = "provisional" if passed else "reference"
    result = {
        key: value
        for key, value in bootstrap.items()
        if key not in BOOTSTRAP_CONTEXT_FIELDS
    }
    result.update(dict(choice))
    for field in (
        "ranker",
        "policy",
        "threshold",
        "lambda",
        "median_original_bytes",
        "online_threshold",
    ):
        result[f"frozen_{field}"] = choice.get(f"{source}_{field}")
    result.update(
        {
            "gate_passed": passed,
            "used_reference_fallback": not passed,
            "frozen_source": source,
            "candidates_attempted": 1,
            "second_candidate_attempted": False,
            "cost_ci_gate_applied": mode in {"fixed_call", "online"},
        }
    )
    return result


def restore_checkpoint_identities(
    choices: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    ap_tolerance: float,
) -> list[dict[str, Any]]:
    """Reapply frozen work-point identity to completed bootstrap rows."""
    if len(rows) > len(choices):
        raise RuntimeError("Audit checkpoint has more rows than frozen choices")
    restored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        choice = choices[index]
        if not math.isclose(
            float(row["budget"]), float(choice["budget"]), abs_tol=1e-12
        ):
            raise RuntimeError(f"Audit checkpoint budget drift at row {index}")
        for field in (
            "reference_ranker",
            "reference_policy",
            "provisional_ranker",
            "provisional_policy",
        ):
            if str(row[field]) != str(choice[field]):
                raise RuntimeError(
                    f"Audit checkpoint {field} drift at row {index}"
                )
        restored.append(
            freeze_choice(choice, row, ap_tolerance=float(ap_tolerance))
        )
    return restored


def _action_map(
    data: Mapping[str, Any],
    base_actions: Mapping[int, str],
    roi_scores: Sequence[float],
    roi_threshold: float,
) -> dict[int, str]:
    samples = list(data["samples"])
    if len(samples) != len(roi_scores):
        raise ValueError("ROI score count differs from audit sample count")
    actions: dict[int, str] = {}
    for sample, roi_score in zip(samples, roi_scores):
        image_id = int(sample["image_id"])
        base = str(base_actions[image_id])
        base_bytes = sample["payload_bytes"].get(base)
        roi_bytes = sample["payload_bytes"].get("roi")
        eligible = (
            bool(sample.get("available", {}).get("roi", False))
            and base_bytes is not None
            and roi_bytes is not None
            and float(roi_bytes) < float(base_bytes)
        )
        actions[image_id] = (
            "roi"
            if eligible and float(roi_score) >= float(roi_threshold)
            else base
        )
    return actions


def reconstruct_actions(
    *,
    mode: str,
    budget: float,
    data: Mapping[str, Any],
    calibration_scores: Sequence[float],
    audit_scores: Sequence[float],
    base_actions: Mapping[int, str],
    roi_scores: Sequence[float],
    roi_threshold: float,
    online_threshold: float | None,
    lambda_value: float | None,
    median_original_bytes: float | None,
) -> dict[int, str]:
    """Reconstruct one audit selection from already frozen scheduler values."""
    samples = list(data["samples"])
    scores = list(map(float, audit_scores))
    if len(samples) != len(scores):
        raise ValueError("Ranker score count differs from audit sample count")
    actions = _action_map(data, base_actions, roi_scores, roi_threshold)

    if mode == "fixed_call":
        count = int(math.floor(len(scores) * float(budget) + 0.5))
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return {
            int(samples[index]["image_id"]): actions[int(samples[index]["image_id"])]
            for index in order[:count]
        }

    if mode == "online":
        if online_threshold is None:
            raise ValueError("Online reconstruction requires a frozen frame threshold")
        return {
            int(sample["image_id"]): actions[int(sample["image_id"])]
            for sample, score in zip(samples, scores)
            if score >= float(online_threshold)
        }

    if mode != "fixed_byte":
        raise ValueError(f"Unsupported mode: {mode}")
    if lambda_value is None or median_original_bytes is None:
        raise ValueError("Fixed-byte reconstruction requires frozen scheduler values")
    ordered_calibration = sorted(map(float, calibration_scores))
    if not ordered_calibration:
        raise ValueError("Fixed-byte reconstruction requires calibration scores")
    percentiles = [
        bisect_right(ordered_calibration, score) / len(ordered_calibration)
        for score in scores
    ]
    priorities = [
        percentile
        - float(lambda_value)
        * float(sample["payload_bytes"][actions[int(sample["image_id"])]])
        / max(1.0, float(median_original_bytes))
        for sample, percentile in zip(samples, percentiles)
    ]
    capacity = float(budget) * sum(
        float(sample["payload_bytes"]["original"]) for sample in samples
    )
    selected: dict[int, str] = {}
    used = 0.0
    for index in sorted(
        range(len(priorities)), key=lambda value: (-priorities[value], value)
    ):
        image_id = int(samples[index]["image_id"])
        action = actions[image_id]
        payload = float(samples[index]["payload_bytes"][action])
        if used + payload <= capacity + 1e-9:
            selected[image_id] = action
            used += payload
    return selected


def action_pair_digest(
    reference: Mapping[int, str], provisional: Mapping[int, str]
) -> str:
    payload = json.dumps(
        [sorted(reference.items()), sorted(provisional.items())],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _optional_number(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    return float(value)


def _runtime(dataset: str) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(TRIACTION_ROOT / "scripts"))
    import run_exploration as tri  # type: ignore

    tri.current.COMMON = canonical_root(dataset) / "common"
    tri.current.dataset_cost = tri.canonical_dataset_cost
    calibration = tri.current.load_role(dataset, "calibration")
    audit = tri.current.load_role(dataset, "audit")
    if len(audit["samples"]) != EXPECTED_AUDIT_IMAGES[dataset]:
        raise RuntimeError(
            f"{dataset} audit size drift: {len(audit['samples'])} "
            f"!= {EXPECTED_AUDIT_IMAGES[dataset]}"
        )

    benchmark = canonical_root(dataset) / "benchmark" / dataset
    models = tri.load_frozen_models(benchmark / "models.json")
    gate = read_json(benchmark / "action_gate.json")
    base_actions = tri.current.chosen_actions(models["action_heads"], gate, audit)

    source_run = SOURCE_20260723 / "runs" / dataset
    controlled_model = read_json(
        source_run / "models/current_controlled_32d.json"
    )

    def ranker_scores(data: Mapping[str, Any]) -> dict[str, Sequence[float]]:
        scores = tri.current.ranker_scores(models, data)
        pca32 = tri.np.asarray(
            [sample["pca32"] for sample in data["samples"]],
            dtype=tri.np.float64,
        )
        scores[CONTROLLED_RANKER] = tri.current.predict_pairwise_ranker(
            controlled_model, pca32
        )
        return scores

    with (source_run / "models/saferoi_gbdt.pkl").open("rb") as handle:
        roi_model = pickle.load(handle)
    roi_scores = tri.predict_gbdt(roi_model, tri.profile_features(audit))
    return {
        "tri": tri,
        "calibration": calibration,
        "audit": audit,
        "calibration_scores": ranker_scores(calibration),
        "audit_scores": ranker_scores(audit),
        "base_actions": base_actions,
        "roi_scores": roi_scores,
    }


def _configuration_actions(
    choice: Mapping[str, Any],
    *,
    prefix: str,
    runtime: Mapping[str, Any],
) -> dict[int, str]:
    ranker = str(choice[f"{prefix}_ranker"])
    return reconstruct_actions(
        mode=str(choice["mode"]),
        budget=float(choice["budget"]),
        data=runtime["audit"],
        calibration_scores=runtime["calibration_scores"][ranker],
        audit_scores=runtime["audit_scores"][ranker],
        base_actions=runtime["base_actions"],
        roi_scores=runtime["roi_scores"],
        roi_threshold=float(choice[f"{prefix}_threshold"]),
        online_threshold=_optional_number(
            choice.get(f"{prefix}_online_threshold")
        ),
        lambda_value=_optional_number(choice.get(f"{prefix}_lambda")),
        median_original_bytes=_optional_number(
            choice.get(f"{prefix}_median_original_bytes")
        ),
    )


def _identical_bootstrap(replicates: int) -> dict[str, Any]:
    return {
        "replicates": int(replicates),
        "official_map50_95_delta_mean": 0.0,
        "official_map50_95_delta_ci_low": 0.0,
        "official_map50_95_delta_ci_high": 0.0,
        "official_map50_95_probability_gt_zero": 0.0,
        "request_kb_per_image_delta_mean": 0.0,
        "request_kb_per_image_delta_ci_low": 0.0,
        "request_kb_per_image_delta_ci_high": 0.0,
        "bootstrap_unit": "image",
        "paired": True,
        "strategy_refit_in_bootstrap": False,
        "identical_action_shortcut": True,
    }


def run(dataset: str) -> list[dict[str, Any]]:
    if dataset not in EXPECTED_AUDIT_IMAGES:
        raise ValueError("This stage only supports COCO and Oregon independent audits")
    protocol = read_json(EXP_ROOT / "configs/protocol.json")
    output = EXP_ROOT / "runs" / dataset
    choices_path = output / "provisional_choices.json"
    choices = read_json(choices_path)
    expected = int(protocol["expected_work_points_per_dataset"])
    if len(choices) != expected:
        raise RuntimeError(f"Unexpected provisional choice count: {len(choices)}")
    keys = [
        (str(choice["mode"]), float(choice["budget"])) for choice in choices
    ]
    expected_keys = {
        (str(mode), float(budget))
        for mode, budgets in protocol["work_points"].items()
        for budget in budgets
    }
    if set(keys) != expected_keys or len(set(keys)) != len(keys):
        raise RuntimeError(
            "Provisional work points differ from the frozen protocol: "
            f"missing={sorted(expected_keys - set(keys))} "
            f"extra={sorted(set(keys) - expected_keys)}"
        )
    if {str(choice["dataset"]) for choice in choices} != {dataset}:
        raise RuntimeError("Provisional choices contain another dataset")
    replicates = int(protocol["bootstrap_replicates"])
    seed = int(protocol["bootstrap_seed"])
    tolerance = float(protocol["ap_tolerance"])

    inputs = audit_input_paths(dataset)
    before = snapshot_sha256(inputs)
    identity = {
        "dataset": dataset,
        "protocol_sha256": file_sha256(EXP_ROOT / "configs/protocol.json"),
        "provisional_choices_sha256": file_sha256(choices_path),
        "audit_input_sha256": before,
    }
    checkpoint_path = output / "audit_gate_checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        observed = {name: checkpoint.get(name) for name in identity}
        if observed != identity:
            raise RuntimeError("Audit-gate checkpoint differs from frozen inputs")
        rows = list(checkpoint.get("rows", []))
    else:
        rows = []
    restored_rows = restore_checkpoint_identities(
        choices, rows, ap_tolerance=tolerance
    )
    if restored_rows != rows:
        rows = restored_rows
        write_json(checkpoint_path, {**identity, "rows": rows})
        write_csv(output / "risk_gate.csv", rows)
    completed = {
        (str(row["mode"]), float(row["budget"])) for row in rows
    }
    runtime = _runtime(dataset) if len(completed) < expected else None
    tri = runtime["tri"] if runtime is not None else None
    cache = {
        str(row["action_pair_sha256"]): {
            key: row[key]
            for key in (
                "replicates",
                "official_map50_95_delta_mean",
                "official_map50_95_delta_ci_low",
                "official_map50_95_delta_ci_high",
                "official_map50_95_probability_gt_zero",
                "request_kb_per_image_delta_mean",
                "request_kb_per_image_delta_ci_low",
                "request_kb_per_image_delta_ci_high",
                "bootstrap_unit",
                "paired",
                "strategy_refit_in_bootstrap",
            )
            if key in row
        }
        for row in rows
    }

    for choice in choices:
        key = (str(choice["mode"]), float(choice["budget"]))
        if key in completed:
            continue
        if runtime is None or tri is None:
            raise RuntimeError("Audit runtime is unavailable for an incomplete gate")
        reference = _configuration_actions(
            choice, prefix="reference", runtime=runtime
        )
        provisional = _configuration_actions(
            choice, prefix="provisional", runtime=runtime
        )
        digest = action_pair_digest(reference, provisional)
        if reference == provisional:
            bootstrap = _identical_bootstrap(replicates)
        elif digest in cache:
            bootstrap = dict(cache[digest])
            bootstrap["reused_identical_action_pair"] = True
        else:
            bootstrap = tri.paired_bootstrap(
                runtime["audit"],
                baseline_actions=reference,
                candidate_actions=provisional,
                replicates=replicates,
                seed=seed,
                unit=str(protocol["bootstrap_unit"]),
            )
            cache[digest] = dict(bootstrap)
        bootstrap.update(
            {
                "audit_role": "audit",
                "audit_images": len(runtime["audit"]["samples"]),
                "action_pair_sha256": digest,
                "reference_uploaded_images": len(reference),
                "provisional_uploaded_images": len(provisional),
                "bootstrap_seed": seed,
            }
        )
        row = freeze_choice(choice, bootstrap, ap_tolerance=tolerance)
        rows.append(row)
        completed.add(key)
        write_json(checkpoint_path, {**identity, "rows": rows})
        write_csv(output / "risk_gate.csv", rows)
        print(
            f"[{dataset}-audit-gate] {key[0]} {key[1]:.2f} "
            f"{'passed' if row['gate_passed'] else 'reference fallback'}",
            flush=True,
        )

    if len(rows) != expected or len(completed) != expected:
        raise RuntimeError(f"Incomplete audit gate: {len(rows)} rows")
    after = snapshot_sha256(inputs)
    if before != after:
        raise RuntimeError("A frozen audit input changed during the gate")
    write_json(output / "frozen_choices.json", rows)
    provenance = {
        **identity,
        "audit_input_sha256_after": after,
        "gate_role": "independent_audit",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "final_artifacts_read": False,
        "frozen_choices_sha256": file_sha256(output / "frozen_choices.json"),
    }
    write_json(output / "audit_gate_provenance.json", provenance)
    (output / "RISK_GATE_COMPLETED").write_text("ok\n", encoding="ascii")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED_AUDIT_IMAGES), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args.dataset)
    print(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
