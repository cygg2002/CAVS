#!/usr/bin/env python3
"""Compare batch-cache, one-image PyTorch, and RKNN FP16 CAVS-OQR plans."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
BUDGETS = (0.1, 0.25, 0.4)
ACTIONS = ("edge", "original", "q90", "current@640")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_plan(selection: Mapping[str | int, str], image_ids: Sequence[int]) -> dict[int, str]:
    """Expand sparse upload selections to the explicit Edge fallback."""
    plan = {int(image_id): "edge" for image_id in image_ids}
    for raw_image_id, raw_action in selection.items():
        image_id, action = int(raw_image_id), str(raw_action)
        if image_id not in plan:
            raise ValueError(f"selection image outside COCO val2017: {image_id}")
        if action not in ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        plan[image_id] = action
    return plan


def plan_counts(plan: Mapping[int, str]) -> dict[str, int]:
    counts = Counter(map(str, plan.values()))
    return {action: int(counts[action]) for action in ACTIONS}


def compare_plans(reference: Mapping[int, str], candidate: Mapping[int, str]) -> dict[str, Any]:
    """Keep upgrade membership and exact action agreement as separate metrics."""
    if set(reference) != set(candidate):
        raise ValueError("compared plans have different image IDs")
    image_ids = sorted(map(int, reference))
    exact = sum(reference[image_id] == candidate[image_id] for image_id in image_ids)
    reference_uploads = {image_id for image_id in image_ids if reference[image_id] != "edge"}
    candidate_uploads = {image_id for image_id in image_ids if candidate[image_id] != "edge"}
    shared_uploads = reference_uploads & candidate_uploads
    union_uploads = reference_uploads | candidate_uploads
    transitions = Counter(
        f"{reference[image_id]}->{candidate[image_id]}"
        for image_id in image_ids
        if reference[image_id] != candidate[image_id]
    )
    return {
        "images": len(image_ids),
        "exact_action_agreement": exact / max(1, len(image_ids)),
        "exact_action_matches": exact,
        "reference_uploads": len(reference_uploads),
        "candidate_uploads": len(candidate_uploads),
        "shared_uploads": len(shared_uploads),
        "upload_membership_jaccard": len(shared_uploads) / max(1, len(union_uploads)),
        "shared_upload_same_action_rate": (
            sum(reference[image_id] == candidate[image_id] for image_id in shared_uploads)
            / max(1, len(shared_uploads))
        ),
        "reference_action_counts": plan_counts(reference),
        "candidate_action_counts": plan_counts(candidate),
        "action_transition_counts": dict(sorted(transitions.items())),
    }


def result_metric(summary: Mapping[str, Any], budget: float) -> float:
    method = summary["methods"][f"cavs_oqr_{int(budget * 100)}"]
    metrics = method["metrics"]
    if "AP50_95" in metrics:
        return float(metrics["AP50_95"])
    return float(metrics["ap50_95"])


def result_capacity(summary: Mapping[str, Any], budget: float) -> dict[str, Any]:
    report = summary["plans"][str(int(budget * 100))]
    request = int(report["request_bytes"])
    capacity = float(report["capacity_bytes"])
    if request > capacity + 1e-6:
        raise RuntimeError(f"fixed-byte capacity exceeded at {budget:g}")
    return {
        "request_bytes": request,
        "capacity_bytes": capacity,
        "capacity_utilization": request / capacity,
    }


def canonical_results(path: Path) -> dict[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {
        float(row["budget"]): float(row["official_ap50_95"])
        for row in rows
        if row["role"] == "final"
        and row["method"] == "CAVS-Current-MAUR"
        and row["mode"] == "fixed_byte"
    }
    if set(output) != set(BUDGETS):
        raise RuntimeError(f"missing published Current+MAUR points: {sorted(output)}")
    return output


def published_plan_key(budget: float) -> str:
    return f"final|CAVS-Current-MAUR|fixed_byte|{budget:g}"


def render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# COCO CAVS-OQR Edge 来源定位诊断",
        "",
        "本诊断只读取已完成的 5,000 张结果。三种来源均使用同一冻结 CAVS-Current-MAUR 策略；"
        "没有重训、重新上传或再次运行 RK3588。",
        "",
        "| Edge 来源 | 10% AP50-95 | 25% | 40% | 相对批量缓存 (10% / 25% / 40%) |",
        "|---|---:|---:|---:|---:|",
    ]
    canonical = report["sources"]["batch_rect_cache"]["ap50_95"]
    for name, label in (
        ("batch_rect_cache", "论文批量 rect=True 缓存"),
        ("pytorch_single", "PyTorch 单图部署路径"),
        ("rknn_fp16", "RKNN FP16 真板路径"),
    ):
        ap = report["sources"][name]["ap50_95"]
        delta = " / ".join(f"{ap[key] - canonical[key]:+.6f}" for key in ("10", "25", "40"))
        lines.append(f"| {label} | {ap['10']:.6f} | {ap['25']:.6f} | {ap['40']:.6f} | {delta} |")

    lines.extend([
        "",
        "## 计划一致性（相对论文批量缓存）",
        "",
        "| 来源 | 预算 | 精确动作一致率 | 上传集合 Jaccard | 共享上传动作一致率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, label in (("pytorch_single", "PyTorch 单图"), ("rknn_fp16", "RKNN FP16")):
        for budget in ("10", "25", "40"):
            row = report["plan_comparisons"][name][budget]["vs_batch_rect_cache"]
            lines.append(
                f"| {label} | {budget}% | {row['exact_action_agreement']:.2%} | "
                f"{row['upload_membership_jaccard']:.2%} | {row['shared_upload_same_action_rate']:.2%} |"
            )

    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 批量 `rect=True` 缓存与单图路径的差值，包含 Edge 框、ROI 几何、动作计划和融合结果的联合变化。",
        "- RKNN FP16 与 PyTorch 单图的差值才是部署后端数值与 NMS 路径的剩余差异。",
        "- 因此不能把论文缓存与 RKNN 的全差距直接称为量化或上传损失。",
        "",
        "完整动作转换矩阵、容量核查与源文件 SHA-256 见 `report.json`。",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/formal")
    parser.add_argument(
        "--paper-results",
        type=Path,
        default=REPO / "paper/experiments/20260815_coco5000_cavs_oqr_paper_parity_e2e_v1/inputs/paper_results.csv",
    )
    parser.add_argument(
        "--paper-selections",
        type=Path,
        default=REPO / "paper/experiments/20260815_coco5000_cavs_oqr_paper_parity_e2e_v1/inputs/paper_selections.json",
    )
    parser.add_argument(
        "--pytorch-summary",
        type=Path,
        default=REPO / "paper/experiments/20260815_coco5000_cavs_oqr_paper_parity_e2e_v1/runs/formal/summary.json",
    )
    parser.add_argument(
        "--pytorch-plans",
        type=Path,
        default=REPO / "paper/experiments/20260815_coco5000_cavs_oqr_paper_parity_e2e_v1/runs/formal/plans.json",
    )
    parser.add_argument(
        "--rknn-summary",
        type=Path,
        default=Path("/Volumes/NO NAME/rk3588/cavs_oqr_edge_cloud/runs/formal_fp16_20260816_110906/fp16/summary.json"),
    )
    parser.add_argument(
        "--rknn-plans",
        type=Path,
        default=Path("/Volumes/NO NAME/rk3588/cavs_oqr_edge_cloud/runs/formal_fp16_20260816_110906/fp16/plans.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = {
        "paper_results": args.paper_results.resolve(),
        "paper_selections": args.paper_selections.resolve(),
        "pytorch_summary": args.pytorch_summary.resolve(),
        "pytorch_plans": args.pytorch_plans.resolve(),
        "rknn_summary": args.rknn_summary.resolve(),
        "rknn_plans": args.rknn_plans.resolve(),
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    published_results = canonical_results(source_paths["paper_results"])
    published_selections = read_json(source_paths["paper_selections"])
    pytorch_summary, rknn_summary = read_json(source_paths["pytorch_summary"]), read_json(source_paths["rknn_summary"])
    pytorch_plans, rknn_plans = read_json(source_paths["pytorch_plans"]), read_json(source_paths["rknn_plans"])

    pytorch_policy = str(pytorch_summary["policy_sha256"])
    rknn_policy = str(rknn_summary["policy_sha256"])
    if not pytorch_policy or pytorch_policy != rknn_policy:
        raise RuntimeError("PyTorch and RKNN runs do not use the same frozen policy")

    raw_plan_sets = [
        set(map(int, pytorch_plans["plans"][label]))
        for label in ("10", "25", "40")
    ] + [
        set(map(int, rknn_plans["plans"][label]))
        for label in ("10", "25", "40")
    ]
    if any(len(ids) != 5000 for ids in raw_plan_sets) or any(ids != raw_plan_sets[0] for ids in raw_plan_sets[1:]):
        raise RuntimeError("plan populations are not identical COCO val2017 sets")
    image_ids = sorted(raw_plan_sets[0])

    source_ap = {
        "batch_rect_cache": {str(int(budget * 100)): published_results[budget] for budget in BUDGETS},
        "pytorch_single": {str(int(budget * 100)): result_metric(pytorch_summary, budget) for budget in BUDGETS},
        "rknn_fp16": {str(int(budget * 100)): result_metric(rknn_summary, budget) for budget in BUDGETS},
    }
    comparisons: dict[str, dict[str, Any]] = {"pytorch_single": {}, "rknn_fp16": {}}
    capacities: dict[str, dict[str, Any]] = {"pytorch_single": {}, "rknn_fp16": {}}
    for budget in BUDGETS:
        label = str(int(budget * 100))
        batch_plan = normalize_plan(published_selections[published_plan_key(budget)], image_ids)
        pytorch_plan = normalize_plan(pytorch_plans["plans"][label], image_ids)
        rknn_plan = normalize_plan(rknn_plans["plans"][label], image_ids)
        comparisons["pytorch_single"][label] = {"vs_batch_rect_cache": compare_plans(batch_plan, pytorch_plan)}
        comparisons["rknn_fp16"][label] = {
            "vs_batch_rect_cache": compare_plans(batch_plan, rknn_plan),
            "vs_pytorch_single": compare_plans(pytorch_plan, rknn_plan),
        }
        capacities["pytorch_single"][label] = result_capacity(pytorch_summary, budget)
        capacities["rknn_fp16"][label] = result_capacity(rknn_summary, budget)

    report = {
        "complete": True,
        "protocol": "coco_cavs_oqr_edge_source_diagnostic_v1",
        "images": len(image_ids),
        "policy_sha256": pytorch_policy,
        "sources": {
            "batch_rect_cache": {"description": "published batch-shaped rect=True cache", "ap50_95": source_ap["batch_rect_cache"]},
            "pytorch_single": {"description": "one-image PyTorch deployment path", "ap50_95": source_ap["pytorch_single"]},
            "rknn_fp16": {"description": "RK3588 FP16 deployment path", "ap50_95": source_ap["rknn_fp16"]},
        },
        "ap50_95_deltas": {
            "pytorch_single_minus_batch_rect_cache": {
                label: source_ap["pytorch_single"][label] - source_ap["batch_rect_cache"][label]
                for label in ("10", "25", "40")
            },
            "rknn_fp16_minus_pytorch_single": {
                label: source_ap["rknn_fp16"][label] - source_ap["pytorch_single"][label]
                for label in ("10", "25", "40")
            },
            "rknn_fp16_minus_batch_rect_cache": {
                label: source_ap["rknn_fp16"][label] - source_ap["batch_rect_cache"][label]
                for label in ("10", "25", "40")
            },
        },
        "plan_comparisons": comparisons,
        "capacity_checks": capacities,
        "inputs": {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in source_paths.items()},
    }
    output = args.output.resolve()
    atomic_json(output / "report.json", report)
    (output / "RESULTS_ZH.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"complete": True, "output": str(output), "images": len(image_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
