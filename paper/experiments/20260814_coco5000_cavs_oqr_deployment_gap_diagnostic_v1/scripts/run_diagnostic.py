#!/usr/bin/env python3
"""Attribute the COCO CAVS-OQR cached-paper versus real-system AP gap."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import gc
import gzip
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
DEMO = REPO / "examples/YOLO26-Edge-Cloud"
sys.path.insert(0, str(ROOT / "inputs/deps"))
sys.path.insert(0, str(DEMO))

from paper_budget_scheduler import schedule_fixed_byte  # noqa: E402


ACTIONS = ("edge", "original", "q90", "current@640")
BUDGETS = (0.1, 0.25, 0.4)
PAPER_EXPECTED = {0.1: 0.48027713207508005, 0.25: 0.5329001188407024, 0.4: 0.5601780940076753}
REAL_EXPECTED = {0.1: 0.4590351050940802, 0.25: 0.5071096848878314, 0.4: 0.5387647914194968}


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def group_predictions(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["image_id"])].append(row)
    return dict(grouped)


def normalize_plan(selection: Mapping[int, str], image_ids: Sequence[int]) -> dict[int, str]:
    plan = {int(image_id): "edge" for image_id in image_ids}
    for image_id, action in selection.items():
        if int(image_id) not in plan:
            raise ValueError(f"selection image outside evaluation set: {image_id}")
        if str(action) not in ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        plan[int(image_id)] = str(action)
    return plan


def powerset(players: Sequence[str]) -> list[frozenset[str]]:
    return [
        frozenset(subset)
        for size in range(len(players) + 1)
        for subset in itertools.combinations(players, size)
    ]


def hybrid_predictions(
    plan: Mapping[int, str],
    real_mixed: Mapping[int, Sequence[Mapping[str, Any]]],
    cached: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    replaced_actions: frozenset[str],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for image_id in sorted(plan):
        action = str(plan[image_id])
        source = cached[action] if action in replaced_actions else real_mixed
        rows.extend(source.get(int(image_id), ()))
    return rows


def cached_predictions(
    plan: Mapping[int, str],
    cached: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
) -> list[Mapping[str, Any]]:
    return hybrid_predictions(plan, {}, cached, frozenset(ACTIONS))


def shapley_values(
    values: Mapping[frozenset[str], float], players: Sequence[str]
) -> dict[str, float]:
    n = len(players)
    factorial = math.factorial
    output: dict[str, float] = {}
    for player in players:
        contribution = 0.0
        others = [item for item in players if item != player]
        for subset in powerset(others):
            weight = factorial(len(subset)) * factorial(n - len(subset) - 1) / factorial(n)
            contribution += weight * (values[subset | {player}] - values[subset])
        output[player] = contribution
    return output


def apply_paper_costs(
    candidate: Mapping[str, Any], paper_sample: Mapping[str, Any]
) -> dict[str, Any]:
    paper_payloads = {
        str(action): int(value)
        for action, value in paper_sample.get("payload_bytes", {}).items()
        if value is not None
    }
    scores = {"edge": float(candidate["scores"]["edge"])}
    payloads = {"edge": 0}
    for action in ACTIONS[1:]:
        if action in candidate["scores"] and action in paper_payloads:
            scores[action] = float(candidate["scores"][action])
            payloads[action] = int(paper_payloads[action])
    return {"frame_id": int(candidate["frame_id"]), "scores": scores, "payload_bytes": payloads}


def validate_resume_report(
    report: Mapping[str, Any], *, paper_total_bytes: int, real_total_bytes: int
) -> set[str]:
    if int(report["paper_original_total_bytes"]) != int(paper_total_bytes):
        raise RuntimeError("partial report paper-byte anchor changed")
    if int(report["real_q95_original_total_bytes"]) != int(real_total_bytes):
        raise RuntimeError("partial report real-byte anchor changed")
    return set(map(str, report.get("budgets", {})))


def q90_path_decomposition(
    *,
    cached_paper_codec: float,
    opencv_codec_paper_inference: float,
    opencv_codec_real_inference: float,
) -> dict[str, float]:
    return {
        "total_q90_real_minus_cached": opencv_codec_real_inference - cached_paper_codec,
        "codec_opencv_minus_pillow_under_paper_inference": (
            opencv_codec_paper_inference - cached_paper_codec
        ),
        "realtime_inference_minus_paper_inference_under_opencv": (
            opencv_codec_real_inference - opencv_codec_paper_inference
        ),
    }


def evaluate_coco(ground_truth: Any, image_ids: Sequence[int], rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    from faster_coco_eval import COCOeval_faster

    detections = ground_truth.loadRes(list(rows), min_score=0.0)
    evaluator = COCOeval_faster(
        ground_truth,
        detections,
        "bbox",
        print_function=lambda *_args, **_kwargs: None,
    )
    evaluator.params.imgIds = list(map(int, image_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "ap50_95": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
    }


def plan_counts(plan: Mapping[int, str]) -> dict[str, int]:
    counts = Counter(map(str, plan.values()))
    return {action: int(counts[action]) for action in ACTIONS}


def load_candidates(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                output.append(json.loads(line)["candidate"])
    if len(output) != 5000:
        raise RuntimeError(f"real candidate count changed: {len(output)}")
    return output


def paper_point_rows(path: Path) -> dict[float, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        float(row["budget"]): row
        for row in rows
        if row["role"] == "final"
        and row["method"] == "CAVS-Current-MAUR"
        and row["mode"] == "fixed_byte"
    }


def paper_selection_key(budget: float) -> str:
    return f"final|CAVS-Current-MAUR|fixed_byte|{budget:g}"


def schedule_with_absolute_capacity(
    candidates: Sequence[Mapping[str, Any]], target_capacity: float
) -> tuple[dict[int, str], dict[str, Any]]:
    original_total = sum(int(row["payload_bytes"]["original"]) for row in candidates)
    result = schedule_fixed_byte(candidates, float(target_capacity) / float(original_total))
    return {int(key): str(value) for key, value in result["actions"].items()}, {
        key: value for key, value in result.items() if key != "actions"
    }


def iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0 else 0.0


def prediction_agreement(
    image_ids: Sequence[int],
    real: Mapping[int, Sequence[Mapping[str, Any]]],
    cached: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    top_k: int = 20,
) -> dict[str, float | int]:
    count_deltas = []
    category_matches = 0
    matched_ious = []
    score_deltas = []
    for image_id in image_ids:
        real_rows = sorted(real.get(int(image_id), ()), key=lambda row: float(row["score"]), reverse=True)[:top_k]
        cached_rows = sorted(cached.get(int(image_id), ()), key=lambda row: float(row["score"]), reverse=True)[:top_k]
        count_deltas.append(len(real.get(int(image_id), ())) - len(cached.get(int(image_id), ())))
        if sorted(int(row["category_id"]) for row in real_rows) == sorted(
            int(row["category_id"]) for row in cached_rows
        ):
            category_matches += 1
        remaining = set(range(len(cached_rows)))
        for real_row in real_rows:
            choices = [
                (iou_xywh(real_row["bbox"], cached_rows[index]["bbox"]), index)
                for index in remaining
                if int(cached_rows[index]["category_id"]) == int(real_row["category_id"])
            ]
            if not choices:
                continue
            best_iou, best_index = max(choices)
            remaining.remove(best_index)
            matched_ious.append(best_iou)
            score_deltas.append(abs(float(real_row["score"]) - float(cached_rows[best_index]["score"])))
    count = max(1, len(image_ids))
    return {
        "images": len(image_ids),
        "mean_detection_count_delta_real_minus_cached": sum(count_deltas) / count,
        "top20_category_multiset_match_rate": category_matches / count,
        "matched_top20_mean_iou": sum(matched_ious) / max(1, len(matched_ious)),
        "matched_top20_mean_abs_score_delta": sum(score_deltas) / max(1, len(score_deltas)),
        "matched_top20_pairs": len(matched_ious),
    }


def parse_args() -> argparse.Namespace:
    dataset_root = REPO.parents[1] / "datasets/coco"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/formal")
    parser.add_argument("--real-root", type=Path, default=ROOT / "inputs/real")
    parser.add_argument("--annotations", type=Path, default=dataset_root / "annotations/instances_val2017.json")
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=REPO / "paper/experiments/20260812_coco_profile_fusion_2x2_controlled_retrain_v2/runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    real_root = args.real_root.resolve()
    paper_root = args.paper_root.resolve()
    canonical_root = REPO / "paper/experiments/20260811_coco_currentoqr12_profile_scale_extension_v1/runs"

    paths = {
        "paper_selections": paper_root / "evaluation/selections.json",
        "paper_results": paper_root / "evaluation/results.csv",
        "paper_samples": canonical_root / "common/coco/final/samples.json",
        "cached_edge": REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/runs/canonical_v1/actions/edge/coco/final/validator/predictions.json",
        "cached_original": REPO / "paper/experiments/E24_new/runs/coco_val_full/native/cloud_yolo26x_640/predictions.json",
        "cached_q90": REPO / "paper/experiments/20260719_coco_compressed_full_q90/runs/q90_val2017/q90_cloud_predictions.json",
        "cached_roi": REPO / "paper/experiments/20260812_coco_profile_fusion_2x2_factorial_v1/runs/actions/final/current__640_maur.json",
    }
    required = [
        args.annotations,
        real_root / "plans.json",
        real_root / "candidates.jsonl",
        *[real_root / f"predictions_cavs_oqr_{int(budget * 100)}.json.gz" for budget in BUDGETS],
        *paths.values(),
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[:10])

    from faster_coco_eval import COCO

    ground_truth = COCO(args.annotations, print_function=lambda *_args, **_kwargs: None)
    image_ids = sorted(map(int, ground_truth.getImgIds()))
    if len(image_ids) != 5000:
        raise RuntimeError(f"COCO population changed: {len(image_ids)}")

    print("loading cached action predictions", flush=True)
    cached = {
        "edge": group_predictions(read_json(paths["cached_edge"])),
        "original": group_predictions(read_json(paths["cached_original"])),
        "q90": group_predictions(read_json(paths["cached_q90"])),
        "current@640": group_predictions(read_json(paths["cached_roi"])),
    }
    paper_samples = {int(row["image_id"]): row for row in read_json(paths["paper_samples"])}
    paper_total_bytes = sum(int(row["payload_bytes"]["original"]) for row in paper_samples.values())
    paper_rows = paper_point_rows(paths["paper_results"])
    paper_selections = read_json(paths["paper_selections"])
    real_plans_raw = read_json(real_root / "plans.json")["plans"]
    real_candidates = load_candidates(real_root / "candidates.jsonl")
    real_total_bytes = sum(int(row["payload_bytes"]["original"]) for row in real_candidates)
    paper_cost_candidates = [
        apply_paper_costs(candidate, paper_samples[int(candidate["frame_id"])])
        for candidate in real_candidates
    ]

    result: dict[str, Any] = {
        "complete": False,
        "paper_original_total_bytes": paper_total_bytes,
        "real_q95_original_total_bytes": real_total_bytes,
        "real_vs_paper_original_total_ratio": real_total_bytes / paper_total_bytes,
        "budgets": {},
    }
    partial_path = output / "partial_report.json"
    if partial_path.is_file():
        previous = read_json(partial_path)
        completed = validate_resume_report(
            previous,
            paper_total_bytes=paper_total_bytes,
            real_total_bytes=real_total_bytes,
        )
        result["budgets"] = dict(previous.get("budgets", {}))
        print(f"resuming completed budgets: {sorted(completed)}", flush=True)

    real_40_group = group_predictions(read_json(real_root / "predictions_cavs_oqr_40.json.gz"))
    real_edge_group = group_predictions(read_json(real_root / "predictions_edge.json.gz"))
    plan_40 = normalize_plan(
        {int(key): str(value) for key, value in real_plans_raw["40"].items()}, image_ids
    )
    agreement: dict[str, Any] = {
        "edge": prediction_agreement(image_ids, real_edge_group, cached["edge"]),
    }
    for action in ACTIONS[1:]:
        selected = [image_id for image_id, selected_action in plan_40.items() if selected_action == action]
        agreement[action] = prediction_agreement(selected, real_40_group, cached[action])
    result["prediction_agreement_40pct"] = agreement
    del real_edge_group, real_40_group
    gc.collect()

    for budget in BUDGETS:
        label = str(int(budget * 100))
        if label in result["budgets"]:
            print(f"budget {label}%: reusing completed checkpoint", flush=True)
            continue
        print(f"budget {label}%: loading real mixed predictions", flush=True)
        real_rows = read_json(real_root / f"predictions_cavs_oqr_{label}.json.gz")
        real_mixed = group_predictions(real_rows)
        actual_metrics = evaluate_coco(ground_truth, image_ids, real_rows)
        if abs(actual_metrics["ap50_95"] - REAL_EXPECTED[budget]) > 1e-6:
            raise RuntimeError(f"real anchor mismatch at {label}%: {actual_metrics}")

        real_plan = normalize_plan(
            {int(key): str(value) for key, value in real_plans_raw[label].items()}, image_ids
        )
        paper_plan = normalize_plan(
            {int(key): str(value) for key, value in paper_selections[paper_selection_key(budget)].items()},
            image_ids,
        )
        paper_metrics = evaluate_coco(ground_truth, image_ids, cached_predictions(paper_plan, cached))
        if abs(paper_metrics["ap50_95"] - PAPER_EXPECTED[budget]) > 1e-6:
            raise RuntimeError(f"paper anchor mismatch at {label}%: {paper_metrics}")

        target_capacity = paper_total_bytes * budget
        native_capacity_plan, native_capacity_report = schedule_with_absolute_capacity(
            real_candidates, target_capacity
        )
        paper_cost_plan, paper_cost_report = schedule_with_absolute_capacity(
            paper_cost_candidates, target_capacity
        )
        native_capacity_metrics = evaluate_coco(
            ground_truth, image_ids, cached_predictions(native_capacity_plan, cached)
        )
        paper_cost_metrics = evaluate_coco(
            ground_truth, image_ids, cached_predictions(paper_cost_plan, cached)
        )

        coalition_metrics: dict[frozenset[str], float] = {}
        coalition_rows = []
        for coalition in powerset(ACTIONS):
            if not coalition:
                metrics = actual_metrics
            elif coalition == frozenset(ACTIONS):
                metrics = evaluate_coco(
                    ground_truth, image_ids, cached_predictions(real_plan, cached)
                )
            else:
                metrics = evaluate_coco(
                    ground_truth,
                    image_ids,
                    hybrid_predictions(real_plan, real_mixed, cached, coalition),
                )
            coalition_metrics[coalition] = float(metrics["ap50_95"])
            coalition_rows.append(
                {
                    "cached_actions": sorted(coalition),
                    "ap50_95": float(metrics["ap50_95"]),
                    "ap50": float(metrics["ap50"]),
                    "ap75": float(metrics["ap75"]),
                }
            )
            print(
                f"budget {label}% coalition={','.join(sorted(coalition)) or 'none'} "
                f"AP={metrics['ap50_95']:.6f}",
                flush=True,
            )
            gc.collect()

        real_plan_cached = coalition_metrics[frozenset(ACTIONS)]
        actual = coalition_metrics[frozenset()]
        decomposition = {
            "total_paper_minus_real": paper_metrics["ap50_95"] - actual,
            "execution_cached_minus_real_same_selection": real_plan_cached - actual,
            "budget_denominator_effect_native_capacity_minus_real_plan": native_capacity_metrics["ap50_95"] - real_plan_cached,
            "paper_action_cost_effect_paper_cost_minus_native_capacity": paper_cost_metrics["ap50_95"] - native_capacity_metrics["ap50_95"],
            "score_availability_selection_effect_paper_plan_minus_paper_cost": paper_metrics["ap50_95"] - paper_cost_metrics["ap50_95"],
        }
        reconciled = sum(
            decomposition[key]
            for key in decomposition
            if key != "total_paper_minus_real"
        )
        if abs(reconciled - decomposition["total_paper_minus_real"]) > 1e-9:
            raise RuntimeError(f"decomposition failed at {label}%")

        shapley = shapley_values(coalition_metrics, ACTIONS)
        if abs(sum(shapley.values()) - (real_plan_cached - actual)) > 1e-9:
            raise RuntimeError(f"Shapley reconciliation failed at {label}%")

        result["budgets"][label] = {
            "actual_real": actual_metrics,
            "real_plan_cached_actions": {"ap50_95": real_plan_cached},
            "native_capacity_real_costs_cached_actions": native_capacity_metrics,
            "paper_costs_real_scores_cached_actions": paper_cost_metrics,
            "published_paper_plan_cached_actions": paper_metrics,
            "recorded_paper_point": paper_rows[budget],
            "plans": {
                "real": {"action_counts": plan_counts(real_plan)},
                "native_capacity_real_costs": {
                    "action_counts": plan_counts(native_capacity_plan),
                    "scheduler": native_capacity_report,
                },
                "paper_costs_real_scores": {
                    "action_counts": plan_counts(paper_cost_plan),
                    "scheduler": paper_cost_report,
                },
                "published_paper": {"action_counts": plan_counts(paper_plan)},
            },
            "decomposition": decomposition,
            "execution_shapley": shapley,
            "coalitions": coalition_rows,
        }
        atomic_json(output / "partial_report.json", result)
        del real_rows, real_mixed
        gc.collect()

    result["complete"] = True
    atomic_json(output / "report.json", result)
    (output / "RESULTS_ZH.md").write_text(render_report(result), encoding="utf-8")
    print(f"complete: {output / 'RESULTS_ZH.md'}", flush=True)


def render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# COCO5000 CAVS-OQR 部署差距诊断",
        "",
        "## 预算协议核查",
        "",
        f"- 论文原始文件总字节：{int(report['paper_original_total_bytes']):,}。",
        f"- 真实实验 Q95 重编码总字节：{int(report['real_q95_original_total_bytes']):,}。",
        f"- 真实/论文分母：{float(report['real_vs_paper_original_total_ratio']):.4%}。",
        "",
        "## AP50-95 分解",
        "",
        "| 预算 | 真实执行 | 同selection全缓存 | 修正原始分母 | 论文动作成本 | 论文selection | 总差距 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in report["budgets"].items():
        lines.append(
            f"| {label}% | {row['actual_real']['ap50_95']:.6f} | "
            f"{row['real_plan_cached_actions']['ap50_95']:.6f} | "
            f"{row['native_capacity_real_costs_cached_actions']['ap50_95']:.6f} | "
            f"{row['paper_costs_real_scores_cached_actions']['ap50_95']:.6f} | "
            f"{row['published_paper_plan_cached_actions']['ap50_95']:.6f} | "
            f"{row['decomposition']['total_paper_minus_real']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 路径分解",
            "",
            "| 预算 | 执行差异 | 预算分母 | 动作字节成本 | 分数/可用性/selection |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in report["budgets"].items():
        item = row["decomposition"]
        lines.append(
            f"| {label}% | {item['execution_cached_minus_real_same_selection']:+.6f} | "
            f"{item['budget_denominator_effect_native_capacity_minus_real_plan']:+.6f} | "
            f"{item['paper_action_cost_effect_paper_cost_minus_native_capacity']:+.6f} | "
            f"{item['score_availability_selection_effect_paper_plan_minus_paper_cost']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 同selection执行差异 Shapley 归因",
            "",
            "| 预算 | Edge | Original | Q90 | ROI+MAUR | 合计 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in report["budgets"].items():
        value = row["execution_shapley"]
        lines.append(
            f"| {label}% | {value['edge']:+.6f} | {value['original']:+.6f} | "
            f"{value['q90']:+.6f} | {value['current@640']:+.6f} | {sum(value.values()):+.6f} |"
        )
    lines.extend(["", "## 40% selection预测一致性", ""])
    for action, stats in report["prediction_agreement_40pct"].items():
        lines.append(
            f"- `{action}`：{stats['images']} 张；检测数差(real-cache) "
            f"{stats['mean_detection_count_delta_real_minus_cached']:+.3f}；top-20类别完全一致率 "
            f"{stats['top20_category_multiset_match_rate']:.2%}；匹配框平均IoU "
            f"{stats['matched_top20_mean_iou']:.4f}；置信度平均绝对差 "
            f"{stats['matched_top20_mean_abs_score_delta']:.4f}。"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 路径分解依赖顺序，用于诊断而不是声明独立因果效应。",
            "- 执行层四动作归因使用全部16种组合的精确Shapley值，可处理mAP非加性。",
            "- 所有计划与预测均冻结；本实验不训练、不运行检测器、不使用GT做selection。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
