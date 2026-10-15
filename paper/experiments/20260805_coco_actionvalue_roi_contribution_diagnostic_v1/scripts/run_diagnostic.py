#!/usr/bin/env python3
"""Run the cache-only COCO ActionValue ROI contribution diagnostic."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE = REPO / PROTOCOL["source_experiment"]
sys.path[:0] = [str(ROOT / "scripts"), str(SOURCE / "scripts")]

from roi_core import (  # noqa: E402
    METHODS,
    apply_action_policy,
    build_action_score_map,
    evaluate_gain_gate,
    roi_risk_rates,
    validate_protocol,
)
import run_oof as source  # type: ignore  # noqa: E402


validate_protocol(PROTOCOL)

ROLE_SPECS = {
    "fit_oof": ("fit", SOURCE / "runs/oof/scores.pkl"),
    "calibration": ("calibration", SOURCE / "runs/calibration_scores/scores.pkl"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def source_files() -> list[Path]:
    paths = [
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/roi_core.py",
        SOURCE / "configs/protocol.json",
        SOURCE / "runs/preflight/report.json",
        SOURCE / "runs/evaluation/report.json",
        SOURCE / "runs/evaluation/selections.json",
        SOURCE / "runs/oof/scores.pkl",
        SOURCE / "runs/calibration_scores/scores.pkl",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen source files:\n" + "\n".join(missing))
    return [path.resolve() for path in paths]


def source_hashes() -> dict[str, str]:
    return {str(path): file_sha256(path) for path in source_files()}


def preflight() -> dict[str, Any]:
    source_report = json.loads(
        (SOURCE / "runs/evaluation/report.json").read_text(encoding="utf-8")
    )
    for field in ("audit_accessed", "final_accessed", "registered_final_accessed"):
        if source_report.get(field) is not False:
            raise RuntimeError(f"Source report violates {field}")
    source.verify_hashes()
    hashes = source_hashes()
    report = {
        "passed": True,
        "source_sha256": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "roles": PROTOCOL["roles"],
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(RUNS / "preflight/report.json", report)
    return report


def _full_source_row(
    report: Mapping[str, Any], role: str, budget: float
) -> dict[str, Any]:
    row = next(
        item
        for item in report["rows"]
        if item["role"] == role
        and item["method"] == "ActionValue-Joint"
        and item["mode"] == "fixed_byte"
        and float(item["budget"]) == float(budget)
    )
    output = dict(row)
    output["method"] = "ActionValue-OQR"
    return output


def _evaluate(
    *,
    role: str,
    method: str,
    budget: float,
    data: Mapping[str, Any],
    rewards: Mapping[int, Mapping[str, float]],
    schedule: Mapping[str, Any],
    capacity: float,
) -> dict[str, Any]:
    selection = schedule["selection"]
    predictions, _ = source.canonical.mixed_predictions(data, dict(selection), 0.001)
    official = data["evaluator"].evaluate(predictions)
    counts = Counter(map(str, selection.values()))
    risk = roi_risk_rates(selection, rewards)
    request_bytes = int(round(float(schedule["total_cost_bytes"])))
    return {
        "role": role,
        "method": method,
        "mode": "fixed_byte",
        "budget": float(budget),
        "official_ap50_95": float(official["map50_95"]),
        "official_ap50": float(official["map50"]),
        "official_ap75": float(official["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": request_bytes,
        "capacity_bytes": float(capacity),
        "capacity_ok": request_bytes <= capacity + float(PROTOCOL["capacity_tolerance_bytes"]),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "harmful_roi_rate": float(risk["harmful_roi_rate"]),
        "relative_suboptimal_roi_rate": float(risk["relative_suboptimal_roi_rate"]),
        "predicted_objective": float(schedule["total_value"]),
    }


def evaluate_role(
    role: str,
    source_role: str,
    score_path: Path,
    source_report: Mapping[str, Any],
    source_selections: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = source.load_role(source_role)
    records = source.records_for_role(source_role)
    rewards = source.reward_map(source_role)
    artifact = source.load_pickle(score_path)
    image_ids = [int(record["image_id"]) for record in records]
    if list(map(int, artifact["image_ids"])) != image_ids:
        raise RuntimeError(f"{role} score ordering changed")
    if len(image_ids) != int(PROTOCOL["roles"][role]) or len(set(image_ids)) != len(image_ids):
        raise RuntimeError(f"{role} population changed")
    numeric = source.numeric_score_arrays(artifact)
    full_scores = build_action_score_map(records, numeric["value"])
    policy_scores = {
        method: apply_action_policy(full_scores, rewards, method) for method in METHODS
    }
    samples = data["samples"]
    original_total = sum(float(sample["payload_bytes"]["original"]) for sample in samples)
    rows: list[dict[str, Any]] = []
    selection_report: dict[str, Any] = {"fixed_byte": {}, "fixed_call_invariant": {}}
    for raw_budget in PROTOCOL["budgets"]:
        budget = float(raw_budget)
        capacity = budget * original_total
        schedules = {
            method: source.schedule_fixed_byte(
                samples, values, capacity_bytes=capacity
            )
            for method, values in policy_scores.items()
        }
        full_key = f"ActionValue-Joint|fixed_byte|{budget:g}"
        expected_full = {
            int(image_id): action
            for image_id, action in source_selections[role][full_key].items()
        }
        if schedules["ActionValue-OQR"]["selection"] != expected_full:
            raise RuntimeError(f"{role}/{budget:g} full selection changed")
        full_row = _full_source_row(source_report, role, budget)
        full_row.update(roi_risk_rates(schedules["ActionValue-OQR"]["selection"], rewards))
        rows.append(full_row)
        for method in ("ActionValue-OQ", "ActionValue-OQR-OracleRelativeGate"):
            rows.append(
                _evaluate(
                    role=role,
                    method=method,
                    budget=budget,
                    data=data,
                    rewards=rewards,
                    schedule=schedules[method],
                    capacity=capacity,
                )
            )
        selection_report["fixed_byte"][f"{budget:g}"] = {
            method: {str(image_id): action for image_id, action in value["selection"].items()}
            for method, value in schedules.items()
        }
        call_schedules = {
            method: source.schedule_fixed_call(samples, values, budget=budget)["selection"]
            for method, values in policy_scores.items()
        }
        invariant = len({tuple(sorted(value.items())) for value in call_schedules.values()}) == 1
        selection_report["fixed_call_invariant"][f"{budget:g}"] = bool(invariant)
        if not invariant:
            raise RuntimeError(f"Fixed-call selection changed at {role}/{budget:g}")
    return rows, selection_report


def _role_gate(rows: list[dict[str, Any]], role: str, method: str) -> dict[str, Any]:
    deltas = []
    for budget in map(float, PROTOCOL["budgets"]):
        baseline = next(
            row for row in rows
            if row["role"] == role and row["method"] == "ActionValue-OQ" and row["budget"] == budget
        )
        candidate = next(
            row for row in rows
            if row["role"] == role and row["method"] == method and row["budget"] == budget
        )
        deltas.append(float(candidate["official_ap50_95"]) - float(baseline["official_ap50_95"]))
    output = evaluate_gain_gate(deltas, PROTOCOL)
    output["method"] = method
    return output


def write_notes(report: Mapping[str, Any]) -> None:
    lines = [
        "# COCO ActionValue ROI 贡献诊断",
        "",
        "> 缓存级诊断；Oracle-relative gate 使用 GT contextual reward，不可部署。未访问 audit/final。",
        "",
    ]
    for role, title in (("fit_oof", "5-fold OOF"), ("calibration", "Frozen calibration")):
        lines.extend([
            f"## {title}", "",
            "| 预算 | 方法 | AP | AP75 | KB/图 | O/Q90/ROI | 有害ROI | 相对次优ROI |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for budget in map(float, PROTOCOL["budgets"]):
            for method in METHODS:
                row = next(
                    item for item in report["rows"]
                    if item["role"] == role and item["method"] == method and item["budget"] == budget
                )
                lines.append(
                    f"| {budget:.0%} | {method} | {row['official_ap50_95']:.6f} | "
                    f"{row['official_ap75']:.6f} | {row['kb_per_image']:.2f} | "
                    f"{row['original_count']}/{row['q90_count']}/{row['roi_count']} | "
                    f"{row.get('harmful_roi_rate', 0.0):.2%} | "
                    f"{row.get('relative_suboptimal_roi_rate', 0.0):.2%} |"
                )
        lines.extend(["", "### Gates", ""])
        for name, gate in report["gates"][role].items():
            lines.append(f"- {name}: `passed={str(gate['passed']).lower()}`, deltas={gate['deltas']}")
        lines.append("")
    lines.extend([
        "## Decision", "",
        f"`decision={report['decision']}`", "",
        f"`retain_roi_research={str(report['retain_roi_research']).lower()}`", "",
    ])
    path = ROOT / "notes/results_zh.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    preflight_report = preflight()
    source_report = json.loads(
        (SOURCE / "runs/evaluation/report.json").read_text(encoding="utf-8")
    )
    source_selections = json.loads(
        (SOURCE / "runs/evaluation/selections.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    selections = {}
    for role, (source_role, score_path) in ROLE_SPECS.items():
        role_rows, role_selections = evaluate_role(
            role, source_role, score_path, source_report, source_selections
        )
        rows.extend(role_rows)
        selections[role] = role_selections
    gates = {
        role: {
            "learned_full_vs_oq": _role_gate(rows, role, "ActionValue-OQR"),
            "oracle_relative_vs_oq": _role_gate(
                rows, role, "ActionValue-OQR-OracleRelativeGate"
            ),
        }
        for role in ROLE_SPECS
    }
    oracle_potential = all(
        gates[role]["oracle_relative_vs_oq"]["passed"] for role in ROLE_SPECS
    )
    learned_effective = all(
        gates[role]["learned_full_vs_oq"]["passed"] for role in ROLE_SPECS
    )
    if not oracle_potential:
        decision = "drop_coco_roi_mainline"
    elif learned_effective:
        decision = "roi_adds_value_but_safety_requires_separate_gate"
    else:
        decision = "roi_potential_exists_but_eligibility_is_the_bottleneck"
    hashes = source_hashes()
    if hashes != preflight_report["source_sha256"]:
        raise RuntimeError("Frozen source files changed during evaluation")
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_type": "cache_only_gt_aware_roi_contribution_diagnostic",
        "rows": rows,
        "gates": gates,
        "fixed_call_invariant": all(
            all(value["fixed_call_invariant"].values()) for value in selections.values()
        ),
        "decision": decision,
        "retain_roi_research": bool(oracle_potential),
        "learned_roi_effective_by_ap_gate": bool(learned_effective),
        "source_digest": preflight_report["source_digest"],
        "oracle_uses_contextual_reward": True,
        "calibration_used_for_selection": False,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_csv(RUNS / "evaluation/results.csv", rows)
    atomic_json(RUNS / "evaluation/selections.json", selections)
    atomic_json(RUNS / "evaluation/report.json", report)
    write_notes(report)
    atomic_json(ROOT / "COMPLETED.json", {"complete": True, "decision": decision})
    print(json.dumps({"decision": decision, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
