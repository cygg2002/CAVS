"""Pure contracts for the same-selection ROI-fusion ablation."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence


TRANSPORT_FIELDS = (
    "selected_images",
    "original_count",
    "q90_count",
    "roi_count",
    "request_bytes",
)
METRIC_FIELDS = ("official_ap50_95", "official_ap50", "official_ap75")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if list(protocol.get("datasets", {})) != ["coco", "oregon"]:
        raise ValueError("dataset order or population changed")
    if protocol.get("method") != "CAVS-OQR" or protocol.get("policy") != "Joint-OQR":
        raise ValueError("source selection contract changed")
    if protocol.get("roi_action") != "current@640":
        raise ValueError("ROI action changed")
    if list(protocol.get("fusions", [])) != ["Reversible", "MAUR"]:
        raise ValueError("fusion pair changed")
    if list(protocol.get("roles", [])) != ["calibration", "final"]:
        raise ValueError("role contract changed")
    if list(protocol.get("modes", [])) != ["fixed_byte", "fixed_call"]:
        raise ValueError("mode contract changed")
    if list(map(float, protocol.get("budgets", []))) != [0.1, 0.25, 0.4]:
        raise ValueError("budget contract changed")
    expected_roles = {
        "coco": {"calibration": 571, "final": 5000},
        "oregon": {"calibration": 4961, "final": 4962},
    }
    for dataset, roles in expected_roles.items():
        if protocol["datasets"][dataset].get("roles") != roles:
            raise ValueError(f"role counts changed: {dataset}")
    bootstrap = protocol.get("bootstrap", {})
    if int(bootstrap.get("replicates", 0)) != 1000:
        raise ValueError("bootstrap replicate count changed")
    if int(bootstrap.get("workers", 0)) > 4:
        raise ValueError("bootstrap may use at most four workers")
    resource = protocol.get("resource", {})
    if int(resource.get("maximum_workers", 0)) > 4:
        raise ValueError("experiment may use at most four workers")
    if protocol.get("training_performed") is not False:
        raise ValueError("this ablation must not train")
    if protocol.get("detector_inference_required") is not False:
        raise ValueError("this ablation must not run detector inference")
    if protocol.get("source_experiments_modified") is not False:
        raise ValueError("source experiments must remain read-only")


def expected_point_rows(protocol: Mapping[str, Any]) -> int:
    validate_protocol(protocol)
    return (
        len(protocol["datasets"])
        * len(protocol["roles"])
        * len(protocol["fusions"])
        * len(protocol["modes"])
        * len(protocol["budgets"])
    )


def expected_bootstrap_jobs(protocol: Mapping[str, Any]) -> int:
    validate_protocol(protocol)
    return sum(
        len(protocol["budgets"]) * len(config["bootstrap_units"])
        for config in protocol["datasets"].values()
    )


def selection_key(role: str, mode: str, budget: float) -> str:
    return f"{role}|CAVS-OQR|Joint-OQR|{mode}|{float(budget):g}"


def group_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        row["image_id"] = int(row["image_id"])
        if "category_id" in row:
            row["category_id"] = int(row["category_id"])
        if "bbox" in row:
            row["bbox"] = list(map(float, row["bbox"]))
        if "score" in row:
            row["score"] = float(row["score"])
        grouped[row["image_id"]].append(row)
    return {image_id: grouped[image_id] for image_id in sorted(grouped)}


def validate_selection_actions(selection: Mapping[int, str]) -> None:
    allowed = {"original", "q90", "current@640"}
    invalid = sorted({str(value) for value in selection.values()} - allowed)
    if invalid:
        raise ValueError(f"unexpected selected action: {invalid}")


def validate_transport_pair(
    reversible: Mapping[str, Any], maur: Mapping[str, Any]
) -> None:
    for field in TRANSPORT_FIELDS:
        if field not in reversible or field not in maur:
            raise ValueError(f"missing transport field: {field}")
        left = float(reversible[field])
        right = float(maur[field])
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"fusion changed transport field {field}: {left} != {right}")
    if "kb_per_image" in reversible and "kb_per_image" in maur and not math.isclose(
        float(reversible["kb_per_image"]),
        float(maur["kb_per_image"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("fusion changed KB/image")


def validate_zero_roi_metrics(
    reversible: Mapping[str, Any], maur: Mapping[str, Any]
) -> None:
    if int(float(reversible.get("roi_count", -1))) != 0 or int(
        float(maur.get("roi_count", -1))
    ) != 0:
        return
    for field in METRIC_FIELDS:
        if not math.isclose(
            float(reversible[field]),
            float(maur[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"zero-ROI fusion changed {field}")


def validate_unique_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [
        (
            str(row["dataset"]),
            str(row["role"]),
            str(row["fusion"]),
            str(row["mode"]),
            float(row["budget"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate point rows")
