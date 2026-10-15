#!/usr/bin/env python3
"""Evaluate the frozen Oregon Cloud-V2 ROI profile/scale candidate space."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_SCRIPTS = (
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts"
)
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SOURCE_SCRIPTS))

from audit_ena24_roi_protocol import base_actions, q90_probabilities  # noqa: E402
from benchmark_core import image_quality  # noqa: E402
from protocol_core import class_aware_nms, maur_candidates  # noqa: E402
import run_multiranker_benchmark as benchmark  # noqa: E402
from scan_core import (  # noqa: E402
    best_candidate,
    candidate_matrix,
    ranked_candidate_actions,
    select_ap_safe_point,
    validate_roles,
    with_baseline_deltas,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in read_json(path):
        grouped[int(prediction["image_id"])].append(prediction)
    return dict(grouped)


def load_base(
    config: dict[str, Any], role: str
) -> tuple[dict[str, Any], dict[int, str], dict[str, Any]]:
    validate_roles(config["roles"])
    if role not in config["roles"]:
        raise ValueError(role)
    source = Path(config["source_cloudv2_root"])
    canonical = source / "runs/canonical_oregon_event_v1"
    benchmark.COMMON = canonical / "common"
    data = benchmark.load_role("oregon", role)
    model_root = canonical / "benchmark/oregon"
    models = read_json(model_root / "models.json")
    gate = read_json(model_root / "action_gate.json")
    q90_model = models["action_heads"]["q90"]
    threshold = float(gate["chosen"]["q90_threshold"])
    base = base_actions(data, q90_probabilities(q90_model, data), threshold)
    metadata = {
        "models_sha256": file_sha256(model_root / "models.json"),
        "action_gate_sha256": file_sha256(model_root / "action_gate.json"),
        "q90_threshold": threshold,
    }
    return data, base, metadata


def candidate_assets(
    config: dict[str, Any], role: str, candidate: dict[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    root = (
        Path(config["output_root"])
        / f"runs/actions/{role}/{candidate['profile']}"
    )
    geometry_path = root / "geometry/roi_manifest.json"
    prediction_root = root / f"imgsz_{candidate['imgsz']}"
    if not (prediction_root / "COMPLETED").is_file():
        raise RuntimeError(f"incomplete candidate: {candidate['id']}")
    geometry = read_json(geometry_path)
    summary = read_json(prediction_root / "summary.json")
    if summary["candidate_id"] != candidate["id"]:
        raise RuntimeError(f"candidate ID drift: {prediction_root}")
    if (
        summary["geometry_signature"]
        != geometry["signature"]["sha256"]
    ):
        raise RuntimeError(f"geometry signature drift: {prediction_root}")
    prediction_path = prediction_root / "roi_predictions_mapped.json"
    if (
        file_sha256(prediction_path)
        != summary["prediction_sha256"]
    ):
        raise RuntimeError(f"prediction hash mismatch: {prediction_root}")
    records = {
        int(image_id): value
        for image_id, value in geometry["records"].items()
    }
    predictions = grouped_predictions(prediction_path)
    return records, predictions, summary


def attach_candidates(
    config: dict[str, Any],
    role: str,
    data: dict[str, Any],
    base: dict[int, str],
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    details: dict[str, dict[int, dict[str, Any]]] = {}
    options: dict[int, list[dict[str, Any]]] = defaultdict(list)
    summaries = []
    for candidate in candidate_matrix(config["profiles"], config["imgsz"]):
        records, cloud_by_id, summary = candidate_assets(config, role, candidate)
        candidate_details = {}
        for sample in data["samples"]:
            image_id = int(sample["image_id"])
            record = records.get(image_id)
            if record is None:
                continue
            fused = class_aware_nms(
                maur_candidates(
                    data["actions"]["edge"].get(image_id, []),
                    cloud_by_id.get(image_id, []),
                    tuple(map(int, record["xyxy"])),
                ),
                iou_threshold=0.7,
            )
            base_action = base[image_id]
            payload = int(record["bytes"])
            base_payload = int(sample["payload_bytes"][base_action])
            available = payload < base_payload
            quality = (
                image_quality(data["gt_map"].get(image_id, []), fused)
                if available
                else None
            )
            full_scale = 640.0 / max(
                int(record["image_width"]), int(record["image_height"])
            )
            crop_scale = float(candidate["imgsz"]) / max(
                int(record["crop_width"]), int(record["crop_height"])
            )
            detail = {
                "candidate_id": candidate["id"],
                "profile": candidate["profile"],
                "imgsz": int(candidate["imgsz"]),
                "image_id": image_id,
                "quality": quality,
                "base_quality": float(sample["qualities"][base_action]),
                "payload_bytes": payload,
                "base_payload_bytes": base_payload,
                "available": available,
                "fused": fused,
                "record": record,
                "linear_magnification": crop_scale / full_scale,
            }
            candidate_details[image_id] = detail
            if available:
                options[image_id].append(detail)
        details[candidate["id"]] = candidate_details
        summaries.append(summary)
    return details, dict(options), summaries


def action_predictions(
    data: dict[str, Any],
    details: dict[str, dict[int, dict[str, Any]]],
    image_id: int,
    action: str,
) -> list[dict[str, Any]]:
    if action in {"original", "q90"}:
        return data["actions"][action].get(image_id, [])
    return details[action][image_id]["fused"]


def evaluate_actions(
    data: dict[str, Any],
    base: dict[int, str],
    details: dict[str, dict[int, dict[str, Any]]],
    selected: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    flat = []
    request_bytes = 0
    for sample in data["samples"]:
        image_id = int(sample["image_id"])
        option = selected.get(image_id)
        action = option["candidate_id"] if option else base[image_id]
        predictions = action_predictions(data, details, image_id, action)
        flat.extend(
            value for value in predictions if float(value["score"]) >= 0.001
        )
        request_bytes += (
            int(option["payload_bytes"])
            if option
            else int(sample["payload_bytes"][action])
        )
    metrics = data["evaluator"].evaluate(flat)
    count = len(data["samples"])
    return {
        "official_map50_95": float(metrics["map50_95"]),
        "official_map50": float(metrics["map50"]),
        "official_map75": float(metrics["map75"]),
        "request_kb_per_image": request_bytes / count / 1024.0,
    }


def options_for_candidates(
    options: dict[int, list[dict[str, Any]]],
    candidate_ids: set[str],
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            value
            for value in values
            if value["candidate_id"] in candidate_ids
        ]
        for image_id, values in options.items()
    }


def teacher_sweep(
    data: dict[str, Any],
    base: dict[int, str],
    details: dict[str, dict[int, dict[str, Any]]],
    options: dict[int, list[dict[str, Any]]],
    candidate_ids: Iterable[str],
    roi_ratios: Iterable[float],
    *,
    epsilon: float,
    source: str,
) -> list[dict[str, Any]]:
    allowed = set(candidate_ids)
    candidate_options = options_for_candidates(options, allowed)
    count = len(data["samples"])
    rows = []
    for requested_ratio in roi_ratios:
        requested_count = round(float(requested_ratio) * count)
        selected = ranked_candidate_actions(
            candidate_options, count=requested_count
        )
        metrics = evaluate_actions(data, base, details, selected)
        safe = [
            float(value["quality"])
            >= float(value["base_quality"]) - float(epsilon)
            for value in selected.values()
        ]
        action_counts = {
            candidate_id: sum(
                value["candidate_id"] == candidate_id
                for value in selected.values()
            )
            for candidate_id in sorted(allowed)
        }
        rows.append(
            {
                "source": source,
                "role": data["role"],
                "requested_roi_ratio": float(requested_ratio),
                "requested_roi": requested_count,
                "actual_roi": len(selected),
                "roi_ratio": len(selected) / count,
                "safe_precision": float(np.mean(safe)) if safe else 1.0,
                **metrics,
                "action_counts": action_counts,
            }
        )
    baseline = rows[0]
    for row in rows:
        row["map_delta_vs_base"] = (
            float(row["official_map50_95"])
            - float(baseline["official_map50_95"])
        )
        row["kb_delta_vs_base"] = (
            float(row["request_kb_per_image"])
            - float(baseline["request_kb_per_image"])
        )
    return rows


def candidate_stats(
    data: dict[str, Any],
    base: dict[int, str],
    details: dict[str, dict[int, dict[str, Any]]],
    *,
    epsilon: float,
) -> list[dict[str, Any]]:
    count = len(data["samples"])
    rows = []
    for candidate_id, by_image in details.items():
        values = list(by_image.values())
        available = [value for value in values if value["available"]]
        safe = [
            value
            for value in available
            if float(value["quality"])
            >= float(value["base_quality"]) - float(epsilon)
        ]
        strict = [
            value
            for value in available
            if float(value["quality"]) >= float(value["base_quality"])
        ]
        ratios = [
            float(value["payload_bytes"])
            / float(value["base_payload_bytes"])
            for value in available
        ]
        profile, imgsz = candidate_id.split("@", maxsplit=1)
        rows.append(
            {
                "role": data["role"],
                "candidate_id": candidate_id,
                "profile": profile,
                "imgsz": int(imgsz),
                "roi_generation_rate": len(values) / count,
                "payload_eligible_rate": len(available) / count,
                "proxy_safe_rate": len(safe) / count,
                "strict_safe_rate": len(strict) / count,
                "safe_rate_given_eligible": len(safe) / max(1, len(available)),
                "median_roi_base_payload_ratio": (
                    statistics.median(ratios) if ratios else None
                ),
            }
        )
    return rows


def base_summary(
    data: dict[str, Any],
    base: dict[int, str],
    details: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    result = evaluate_actions(data, base, details, {})
    result["action_counts"] = {
        action: list(base.values()).count(action)
        for action in ("original", "q90")
    }
    return result


def compact_sweep_rows(
    rows_by_candidate: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for candidate_id, rows in rows_by_candidate.items():
        for row in rows:
            output.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "action_counts"
                }
                | {"candidate_id": candidate_id}
            )
    return output


def selection_diagnostics(
    data: dict[str, Any],
    selected: dict[int, dict[str, Any]],
    *,
    epsilon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt = read_json(data["gt_path"])
    images = {int(image["id"]): image for image in gt["images"]}
    categories = {
        int(category["id"]): str(category["name"])
        for category in gt["categories"]
    }
    rows = []
    for image_id, option in selected.items():
        image = images[image_id]
        ground_truth = data["gt_map"].get(image_id, [])
        area_ratios = [
            float(value["bbox"][2])
            * float(value["bbox"][3])
            / (int(image["width"]) * int(image["height"]))
            for value in ground_truth
        ]
        minimum_area = min(area_ratios, default=0.0)
        if minimum_area < 0.02:
            size_bucket = "small_<2%"
        elif minimum_area < 0.08:
            size_bucket = "medium_2-8%"
        else:
            size_bucket = "large_>=8%"
        magnification = float(option["linear_magnification"])
        if magnification < 1.0:
            magnification_bucket = "<1x"
        elif magnification < 2.0:
            magnification_bucket = "1-2x"
        elif magnification < 4.0:
            magnification_bucket = "2-4x"
        else:
            magnification_bucket = ">=4x"
        class_names = sorted(
            {
                categories[int(value["category_id"])]
                for value in ground_truth
            }
        )
        rows.append(
            {
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "candidate_id": str(option["candidate_id"]),
                "profile": str(option["profile"]),
                "imgsz": int(option["imgsz"]),
                "gain": float(option["gain"]),
                "safe": float(option["quality"])
                >= float(option["base_quality"]) - float(epsilon),
                "payload_ratio": float(option["payload_bytes"])
                / float(option["base_payload_bytes"]),
                "linear_magnification": magnification,
                "magnification_bucket": magnification_bucket,
                "minimum_gt_area_ratio": minimum_area,
                "size_bucket": size_bucket,
                "classes": "|".join(class_names),
            }
        )

    aggregates = []
    for dimension in ("candidate_id", "size_bucket", "magnification_bucket"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[dimension])].append(row)
        for value, values in sorted(grouped.items()):
            aggregates.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "images": len(values),
                    "safe_rate": float(np.mean([row["safe"] for row in values])),
                    "mean_gain": float(np.mean([row["gain"] for row in values])),
                    "mean_payload_ratio": float(
                        np.mean([row["payload_ratio"] for row in values])
                    ),
                }
            )
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for class_name in str(row["classes"]).split("|"):
            if class_name:
                by_class[class_name].append(row)
    for class_name, values in sorted(by_class.items()):
        aggregates.append(
            {
                "dimension": "class",
                "value": class_name,
                "images": len(values),
                "safe_rate": float(np.mean([row["safe"] for row in values])),
                "mean_gain": float(np.mean([row["gain"] for row in values])),
                "mean_payload_ratio": float(
                    np.mean([row["payload_ratio"] for row in values])
                ),
            }
        )
    return rows, aggregates


def evaluate_role(
    config: dict[str, Any], role: str
) -> dict[str, Any]:
    data, base, base_metadata = load_base(config, role)
    details, options, cache_summaries = attach_candidates(
        config, role, data, base
    )
    epsilon = float(config["evaluation"]["epsilon"])
    tolerance = float(config["evaluation"]["ap_tolerance"])
    ratios = list(map(float, config["evaluation"]["roi_ratios"]))
    candidate_ids = [
        value["id"]
        for value in candidate_matrix(config["profiles"], config["imgsz"])
    ]
    sweeps = {
        candidate_id: teacher_sweep(
            data,
            base,
            details,
            options,
            [candidate_id],
            ratios,
            epsilon=epsilon,
            source=f"single_candidate:{candidate_id}",
        )
        for candidate_id in candidate_ids
    }
    multi_sweep = teacher_sweep(
        data,
        base,
        details,
        options,
        candidate_ids,
        ratios,
        epsilon=epsilon,
        source="multi_candidate_teacher_oracle",
    )
    output = Path(config["output_root"]) / f"runs/evaluation/{role}"
    write_csv(output / "candidate_stats.csv", candidate_stats(
        data, base, details, epsilon=epsilon
    ))
    write_csv(
        output / "single_candidate_sweeps.csv",
        compact_sweep_rows(sweeps),
    )
    write_csv(
        output / "multi_candidate_sweep.csv",
        [
            {key: value for key, value in row.items() if key != "action_counts"}
            for row in multi_sweep
        ],
    )
    report = {
        "protocol": "oregon_cloudv2_roi_profile_scale_scan_v1",
        "role": role,
        "base": base_summary(data, base, details),
        "base_metadata": base_metadata,
        "candidate_ids": candidate_ids,
        "candidate_stats": candidate_stats(
            data, base, details, epsilon=epsilon
        ),
        "single_candidates": {
            candidate_id: {
                "max_ap_point": max(
                    rows, key=lambda row: row["official_map50_95"]
                ),
                "lowest_kb_ap_safe_point": select_ap_safe_point(
                    rows, tolerance=tolerance
                ),
            }
            for candidate_id, rows in sweeps.items()
        },
        "multi_candidate": {
            "sweep": multi_sweep,
            "max_ap_point": max(
                multi_sweep, key=lambda row: row["official_map50_95"]
            ),
            "lowest_kb_ap_safe_point": select_ap_safe_point(
                multi_sweep, tolerance=tolerance
            ),
        },
        "cache_summaries": cache_summaries,
        "registered_final_test_accessed": False,
    }
    write_json(output / "report.json", report)
    return report


def freeze_calibration(
    config: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    if report["role"] != "calibration":
        raise ValueError("only calibration can freeze")
    candidates = report["single_candidates"]
    best_single_id, best_single = min(
        candidates.items(),
        key=lambda item: (
            float(item[1]["lowest_kb_ap_safe_point"]["request_kb_per_image"]),
            -float(item[1]["lowest_kb_ap_safe_point"]["official_map50_95"]),
            item[0],
        ),
    )
    multi = report["multi_candidate"]["lowest_kb_ap_safe_point"]
    freeze = {
        "protocol": report["protocol"],
        "candidate_ids": report["candidate_ids"],
        "best_single_candidate": best_single_id,
        "best_single_roi_ratio": float(
            best_single["lowest_kb_ap_safe_point"]["requested_roi_ratio"]
        ),
        "multi_candidate_roi_ratio": float(multi["requested_roi_ratio"]),
        "base_metadata": report["base_metadata"],
        "calibration_report_sha256": file_sha256(
            Path(config["output_root"])
            / "runs/evaluation/calibration/report.json"
        ),
        "audit_used_for_selection": False,
        "registered_final_test_accessed": False,
    }
    write_json(
        Path(config["output_root"]) / "runs/CALIBRATION_FROZEN.json",
        freeze,
    )
    return freeze


def finalize_audit(
    config: dict[str, Any],
    report: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    if report["role"] != "audit":
        raise ValueError("only audit can be finalized")
    current_ids = [
        value["id"]
        for value in candidate_matrix(config["profiles"], config["imgsz"])
    ]
    if freeze["candidate_ids"] != current_ids:
        raise RuntimeError("candidate set changed after calibration")
    data, base, _ = load_base(config, "audit")
    details, options, _ = attach_candidates(config, "audit", data, base)
    epsilon = float(config["evaluation"]["epsilon"])
    audit_base = report["base"]
    single = with_baseline_deltas(
        teacher_sweep(
            data,
            base,
            details,
            options,
            [freeze["best_single_candidate"]],
            [freeze["best_single_roi_ratio"]],
            epsilon=epsilon,
            source="frozen_best_single_teacher_oracle",
        )[0],
        audit_base,
    )
    multi = with_baseline_deltas(
        teacher_sweep(
            data,
            base,
            details,
            options,
            freeze["candidate_ids"],
            [freeze["multi_candidate_roi_ratio"]],
            epsilon=epsilon,
            source="frozen_multi_candidate_teacher_oracle",
        )[0],
        audit_base,
    )
    selected = ranked_candidate_actions(
        options_for_candidates(options, set(freeze["candidate_ids"])),
        count=round(
            float(freeze["multi_candidate_roi_ratio"])
            * len(data["samples"])
        ),
    )
    selection_rows, aggregates = selection_diagnostics(
        data, selected, epsilon=epsilon
    )
    output = Path(config["output_root"]) / "runs/evaluation/audit"
    write_csv(output / "frozen_multi_selection.csv", selection_rows)
    write_csv(output / "frozen_multi_decomposition.csv", aggregates)
    tolerance = float(config["evaluation"]["ap_tolerance"])
    diagnostic_ceiling = report["multi_candidate"][
        "lowest_kb_ap_safe_point"
    ]
    result = {
        "protocol": report["protocol"],
        "status": "completed",
        "frozen_best_single": single,
        "frozen_multi_candidate": multi,
        "diagnostic_audit_ceiling": diagnostic_ceiling,
        "diagnostic_ceiling_used_for_selection": False,
        "decomposition": aggregates,
        "success": (
            float(multi["map_delta_vs_base"]) >= -tolerance
            and float(multi["kb_delta_vs_base"]) < 0.0
        ),
        "calibration_freeze_sha256": file_sha256(
            Path(config["output_root"]) / "runs/CALIBRATION_FROZEN.json"
        ),
        "audit_used_for_selection": False,
        "registered_final_test_accessed": False,
    }
    write_json(Path(config["output_root"]) / "runs/final_report.json", result)
    (Path(config["output_root"]) / "COMPLETED").write_text(
        "ok\n", encoding="ascii"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs/protocol.json",
    )
    parser.add_argument(
        "--role", choices=("calibration", "audit"), required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    report = evaluate_role(config, args.role)
    if args.role == "calibration":
        result = freeze_calibration(config, report)
    else:
        freeze_path = (
            Path(config["output_root"]) / "runs/CALIBRATION_FROZEN.json"
        )
        if not freeze_path.is_file():
            raise RuntimeError("audit requires calibration freeze")
        result = finalize_audit(config, report, read_json(freeze_path))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
