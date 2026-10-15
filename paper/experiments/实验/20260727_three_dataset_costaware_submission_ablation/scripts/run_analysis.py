#!/usr/bin/env python3
"""Run isolated post-hoc submission ablations on frozen tabular outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from analysis_core import (
    choose_final_oracle,
    choose_no_cost_candidate,
    choose_reference_candidate,
    index_unique_rows,
    select_shared_pair,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
RISK = EXPERIMENTS / "20260725_three_dataset_costaware_selector_riskgate"
UPSTREAM = EXPERIMENTS / "20260723_three_dataset_nineranker_triaction_paper_exploration"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _budget(value: Any) -> float:
    return round(float(value), 8)


def normalize_selector_row(source: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "mode": str(source["mode"]),
        "budget": _budget(source["budget"]),
        "ranker": str(source["ranker"]),
        "policy": str(source["policy"]),
        "ap": float(source["ap"]),
        "kb": float(source["kb"]),
        "source_role": str(source.get("selection_source_role", "")),
    }


def normalize_final_row(source: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    ranker = source.get("ranker") or source.get("method")
    policy = source.get("policy") or source.get("action_policy")
    budget = source.get("nominal_budget") or source.get("budget")
    if ranker is None or policy is None or budget is None:
        raise ValueError("Final row is missing ranker, policy, or budget")
    return {
        "dataset": dataset,
        "mode": str(source["mode"]),
        "budget": _budget(budget),
        "ranker": str(ranker),
        "policy": str(policy),
        "ap": float(source["official_map50_95"]),
        "kb": float(source["request_kb_per_image"]),
        "e2e_ms_10mbps": float(source["e2e_mean_ms_10mbps"]),
        "roi_ratio": float(source.get("roi_ratio") or 0.0),
        "upload_ratio": float(source.get("upload_ratio") or 0.0),
        "estimated_e2e": str(source.get("estimated_e2e", "True")) == "True",
    }


def _registered(row: Mapping[str, Any], protocol: Mapping[str, Any]) -> bool:
    budgets = protocol["work_points"].get(str(row["mode"]), ())
    return any(abs(float(row["budget"]) - float(value)) <= 1e-8 for value in budgets)


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    kind: str,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalizer = normalize_selector_row if kind == "selector" else normalize_final_row
    normalized = [normalizer(row, dataset) for row in rows]
    rankers = set(map(str, protocol["ranker_order"]))
    return [
        row
        for row in normalized
        if row["ranker"] in rankers and _registered(row, protocol)
    ]


def load_analysis_inputs(root: Path = ROOT) -> dict[str, Any]:
    protocol = read_json(root / "configs" / "protocol.json")
    risk = root.parent / "20260725_three_dataset_costaware_selector_riskgate"
    upstream = root.parent / "20260723_three_dataset_nineranker_triaction_paper_exploration"
    paths = {
        "selector": {
            "coco": risk / "runs/coco/candidate_table.csv",
            "ena24": risk / "runs/ena24/strict_candidate_table.csv",
            "oregon": risk / "runs/oregon/candidate_table.csv",
        },
        "final_matrix": {
            "coco": upstream / "runs/coco/results.csv",
            "ena24": risk / "runs/ena24/final_candidate_results.csv",
            "oregon": upstream / "runs/oregon/results.csv",
        },
        "frozen": {
            dataset: risk / f"runs/{dataset}/final_results.csv"
            for dataset in protocol["datasets"]
        },
    }
    loaded: dict[str, Any] = {"protocol": protocol, "paths": paths}
    for kind in ("selector", "final_matrix", "frozen"):
        loaded[kind] = {}
        for dataset in protocol["datasets"]:
            normalized = _normalize_rows(
                read_csv(paths[kind][dataset]),
                dataset=dataset,
                kind="selector" if kind == "selector" else "final",
                protocol=protocol,
            )
            expected = 240 if kind != "frozen" else 10
            if len(normalized) != expected:
                raise ValueError(
                    f"Expected {expected} {kind} rows for {dataset}, got {len(normalized)}"
                )
            key_fields = (
                ("dataset", "mode", "budget", "ranker", "policy")
                if kind != "frozen"
                else ("dataset", "mode", "budget")
            )
            index_unique_rows(normalized, key_fields=key_fields)
            loaded[kind][dataset] = normalized
    return loaded


def registered_work_points(protocol: Mapping[str, Any]) -> list[tuple[str, float]]:
    return [
        (str(mode), _budget(budget))
        for mode, budgets in protocol["work_points"].items()
        for budget in budgets
    ]


def rows_at(
    rows: Iterable[Mapping[str, Any]], mode: str, budget: float
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row["mode"] == mode and abs(float(row["budget"]) - budget) <= 1e-8
    ]


def load_provisional_pairs(root: Path, protocol: Mapping[str, Any]) -> dict:
    risk = root.parent / "20260725_three_dataset_costaware_selector_riskgate"
    paths = {
        "coco": risk / "runs/coco/provisional_choices.json",
        "ena24": risk / "runs/ena24/strict_provisional_choices.json",
        "oregon": risk / "runs/oregon/provisional_choices.json",
    }
    pairs = {}
    for dataset, path in paths.items():
        for source in read_json(path):
            provisional = source.get("provisional")
            if provisional is None:
                ranker = source["provisional_ranker"]
                policy = source["provisional_policy"]
            else:
                ranker = provisional["ranker"]
                policy = provisional["policy"]
            key = (dataset, str(source["mode"]), _budget(source["budget"]))
            if key in pairs:
                raise ValueError(f"Duplicate provisional work point: {key}")
            pairs[key] = (str(ranker), str(policy))
    expected = len(protocol["datasets"]) * len(registered_work_points(protocol))
    if len(pairs) != expected:
        raise ValueError(f"Expected {expected} provisional pairs, got {len(pairs)}")
    return pairs


def diagnostic_row(
    *,
    variant: str,
    selection_role: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    oracle: Mapping[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    ap_delta = float(candidate["ap"]) - float(reference["ap"])
    kb_delta = float(candidate["kb"]) - float(reference["kb"])
    return {
        "dataset": candidate["dataset"],
        "mode": candidate["mode"],
        "budget": candidate["budget"],
        "variant": variant,
        "selection_role": selection_role,
        "ranker": candidate["ranker"],
        "policy": candidate["policy"],
        "final_ap": candidate["ap"],
        "final_kb": candidate["kb"],
        "estimated_e2e_ms_10mbps": candidate["e2e_ms_10mbps"],
        "roi_ratio": candidate["roi_ratio"],
        "upload_ratio": candidate["upload_ratio"],
        "reference_ranker": reference["ranker"],
        "reference_policy": reference["policy"],
        "reference_final_ap": reference["ap"],
        "reference_final_kb": reference["kb"],
        "final_ap_delta_vs_reference": ap_delta,
        "final_kb_delta_vs_reference": kb_delta,
        "final_relative_kb_delta": kb_delta / float(reference["kb"]),
        "epsilon_violation": ap_delta < -float(epsilon) - 1e-12,
        "final_oracle_ap": oracle["ap"],
        "final_oracle_regret": float(oracle["ap"]) - float(candidate["ap"]),
        "diagnostic_only": True,
    }


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source = list(map(dict, rows))
    variants = sorted({str(row["variant"]) for row in source})
    datasets = sorted({str(row["dataset"]) for row in source})
    modes = sorted({str(row["mode"]) for row in source})
    summaries = []
    for variant in variants:
        scopes = [
            ("all", "all"),
            *(("dataset", dataset) for dataset in datasets),
            *(("mode", mode) for mode in modes),
        ]
        for scope_type, scope in scopes:
            selected = [
                row
                for row in source
                if row["variant"] == variant
                and (
                    scope_type == "all"
                    or (scope_type == "dataset" and row["dataset"] == scope)
                    or (scope_type == "mode" and row["mode"] == scope)
                )
            ]
            if not selected:
                continue
            count = len(selected)
            summaries.append(
                {
                    "variant": variant,
                    "scope_type": scope_type,
                    "scope": scope,
                    "work_points": count,
                    "epsilon_violations": sum(bool(row["epsilon_violation"]) for row in selected),
                    "mean_final_ap_delta": sum(float(row["final_ap_delta_vs_reference"]) for row in selected) / count,
                    "worst_final_ap_delta": min(float(row["final_ap_delta_vs_reference"]) for row in selected),
                    "mean_final_kb_delta": sum(float(row["final_kb_delta_vs_reference"]) for row in selected) / count,
                    "mean_relative_kb_delta": sum(float(row["final_relative_kb_delta"]) for row in selected) / count,
                    "mean_oracle_regret": sum(float(row["final_oracle_regret"]) for row in selected) / count,
                    "max_oracle_regret": max(float(row["final_oracle_regret"]) for row in selected),
                }
            )
    return summaries


def build_analysis(loaded: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    protocol = loaded["protocol"]
    datasets = tuple(map(str, protocol["datasets"]))
    ranker_order = tuple(map(str, protocol["ranker_order"]))
    policy_order = tuple(map(str, protocol["policy_order"]))
    epsilon = float(protocol["ap_tolerance"])
    provisional_pairs = load_provisional_pairs(root, protocol)
    final_indexes = {
        dataset: index_unique_rows(
            loaded["final_matrix"][dataset],
            key_fields=("dataset", "mode", "budget", "ranker", "policy"),
        )
        for dataset in datasets
    }
    frozen_indexes = {
        dataset: index_unique_rows(
            loaded["frozen"][dataset],
            key_fields=("dataset", "mode", "budget"),
        )
        for dataset in datasets
    }

    shared_choices = []
    shared_pair_audit = []
    shared_final = []
    lesion_rows = []
    comparison_rows = []
    for mode, budget in registered_work_points(protocol):
        selector_groups = {
            dataset: rows_at(loaded["selector"][dataset], mode, budget)
            for dataset in datasets
        }
        shared = select_shared_pair(
            selector_groups,
            mode=mode,
            ap_tolerance=epsilon,
            ranker_order=ranker_order,
            policy_order=policy_order,
        )
        shared_choices.append(
            {
                "mode": mode,
                "budget": budget,
                "strict_feasible": shared["strict_feasible"],
                "strict_feasible_pairs": shared["strict_feasible_pairs"],
                "selection_status": shared["selection_status"],
                "selected_ranker": shared["selected_ranker"],
                "selected_policy": shared["selected_policy"],
                "violating_datasets": shared["violating_datasets"],
                "worst_selector_ap_delta": shared["worst_ap_delta"],
                "mean_selector_ap_delta": shared["mean_ap_delta"],
                "mean_normalized_selector_kb": shared["mean_normalized_kb"],
                "final_used_for_selection": False,
            }
        )
        for pair in shared["all_pair_summaries"]:
            audit = {
                "mode": mode,
                "budget": budget,
                "ranker": pair["ranker"],
                "policy": pair["policy"],
                "violating_datasets": pair["violating_datasets"],
                "worst_selector_ap_delta": pair["worst_ap_delta"],
                "mean_selector_ap_delta": pair["mean_ap_delta"],
                "mean_normalized_selector_kb": pair["mean_normalized_kb"],
            }
            for dataset in datasets:
                audit[f"{dataset}_selector_ap_delta"] = pair["per_dataset"][dataset]["ap_delta"]
                audit[f"{dataset}_normalized_selector_kb"] = pair["per_dataset"][dataset]["normalized_kb"]
            shared_pair_audit.append(audit)

        for dataset in datasets:
            selector_rows = selector_groups[dataset]
            final_rows = rows_at(loaded["final_matrix"][dataset], mode, budget)
            reference_selector = choose_reference_candidate(
                selector_rows, ranker_order=ranker_order
            )
            no_cost_selector = choose_no_cost_candidate(
                selector_rows,
                ranker_order=ranker_order,
                policy_order=policy_order,
            )
            oracle = choose_final_oracle(
                final_rows,
                ranker_order=ranker_order,
                policy_order=policy_order,
            )

            def final_for(ranker: str, policy: str) -> dict[str, Any]:
                key = (dataset, mode, budget, str(ranker), str(policy))
                try:
                    return final_indexes[dataset][key]
                except KeyError as error:
                    raise ValueError(f"Missing final candidate: {key}") from error

            reference = final_for(
                reference_selector["ranker"], reference_selector["policy"]
            )
            frozen = frozen_indexes[dataset][(dataset, mode, budget)]
            provisional_ranker, provisional_policy = provisional_pairs[
                (dataset, mode, budget)
            ]
            variants = {
                "full_protocol": (
                    "frozen_before_final",
                    final_for(frozen["ranker"], frozen["policy"]),
                ),
                "no_gate": (
                    "selector_only_provisional",
                    final_for(provisional_ranker, provisional_policy),
                ),
                "no_cost_objective": (
                    "selector_ap_only",
                    final_for(no_cost_selector["ranker"], no_cost_selector["policy"]),
                ),
                "fixed_confidence_shared": (
                    "predefined_baseline",
                    final_for("Confidence", "SharedAction"),
                ),
                "dataset_reference": ("selector_sharedaction_reference", reference),
                "final_oracle": ("final_only_oracle", oracle),
            }
            variant_rows = {
                name: diagnostic_row(
                    variant=name,
                    selection_role=role,
                    candidate=candidate,
                    reference=reference,
                    oracle=oracle,
                    epsilon=epsilon,
                )
                for name, (role, candidate) in variants.items()
            }
            lesion_rows.extend(variant_rows.values())

            shared_candidate = final_for(
                shared["selected_ranker"], shared["selected_policy"]
            )
            shared_row = diagnostic_row(
                variant="shared_least_violating",
                selection_role=shared["selection_status"],
                candidate=shared_candidate,
                reference=reference,
                oracle=oracle,
                epsilon=epsilon,
            )
            shared_row["strict_selector_feasible"] = shared["strict_feasible"]
            shared_row["selector_violating_datasets"] = shared["violating_datasets"]
            shared_final.append(shared_row)
            comparison_rows.extend(
                [
                    {**variant_rows["full_protocol"], "comparison_variant": "dataset_adaptive"},
                    {**shared_row, "comparison_variant": "shared_policy_family"},
                    {**variant_rows["dataset_reference"], "comparison_variant": "dataset_reference"},
                    {**variant_rows["final_oracle"], "comparison_variant": "final_oracle"},
                ]
            )

    return {
        "shared_choices": shared_choices,
        "shared_pair_audit": shared_pair_audit,
        "shared_final": shared_final,
        "lesion_rows": lesion_rows,
        "lesion_summary": summarize_rows(lesion_rows),
        "comparison_rows": comparison_rows,
        "comparison_summary": summarize_rows(
            {**row, "variant": row["comparison_variant"]}
            for row in comparison_rows
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_input_paths(root: Path = ROOT) -> list[Path]:
    risk = root.parent / "20260725_three_dataset_costaware_selector_riskgate"
    upstream = root.parent / "20260723_three_dataset_nineranker_triaction_paper_exploration"
    paths = [root / "configs/protocol.json"]
    paths.extend(
        risk / relative
        for relative in (
            "runs/coco/candidate_table.csv",
            "runs/ena24/strict_candidate_table.csv",
            "runs/oregon/candidate_table.csv",
            "runs/coco/final_results.csv",
            "runs/ena24/final_results.csv",
            "runs/oregon/final_results.csv",
            "runs/coco/provisional_choices.json",
            "runs/ena24/strict_provisional_choices.json",
            "runs/oregon/provisional_choices.json",
            "runs/ena24/final_candidate_results.csv",
            "runs/system/feasibility.json",
        )
    )
    paths.extend(
        upstream / relative
        for relative in ("runs/coco/results.csv", "runs/oregon/results.csv")
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs:\n" + "\n".join(missing))
    return sorted(paths)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(map(dict, rows))
    if not values:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def blocked_experiment_audit(root: Path = ROOT) -> list[dict[str, Any]]:
    feasibility = read_json(
        root.parent
        / "20260725_three_dataset_costaware_selector_riskgate"
        / "runs/system/feasibility.json"
    )
    return [
        {
            "experiment": "audit_bootstrap_1000_2000_bca",
            "status": "blocked_local_assets",
            "reason": "Local package has aggregate 200-replicate checkpoints but not per-image audit predictions/actions required for valid resampling.",
            "claim_allowed": False,
        },
        {
            "experiment": "ena24_independent_gate_expansion",
            "status": "blocked_new_independent_data",
            "reason": "Fit data cannot be reused as an independent gate; new data or a full retrain after a new split is required.",
            "claim_allowed": False,
        },
        {
            "experiment": "faithful_external_offloading_baseline",
            "status": "blocked_protocol_choice_and_predictions",
            "reason": "No faithful external implementation and frozen same-protocol prediction cache has been selected; MORIC-style/GEC-style remain internal approximations.",
            "claim_allowed": False,
        },
        {
            "experiment": "measured_end_to_end_testbed",
            "status": str(feasibility["claim_gate"]["status"]),
            "reason": "; ".join(feasibility["claim_gate"]["blocking_requirements"]),
            "claim_allowed": bool(feasibility["claim_gate"]["measured_e2e_claim_allowed"]),
        },
    ]


def _overall(summary: Iterable[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in summary
        if row["variant"] == variant and row["scope_type"] == "all"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one overall summary for {variant}")
    return matches[0]


def render_report(analysis: Mapping[str, Any], blocked: list[dict[str, Any]]) -> str:
    full = _overall(analysis["lesion_summary"], "full_protocol")
    no_gate = _overall(analysis["lesion_summary"], "no_gate")
    no_cost = _overall(analysis["lesion_summary"], "no_cost_objective")
    fixed = _overall(analysis["lesion_summary"], "fixed_confidence_shared")
    shared = _overall(analysis["comparison_summary"], "shared_policy_family")
    return f"""# 投稿补强实验结果（本地冻结表诊断）

## 口径

- 本实验是对既有冻结selector/final表的事后诊断，不是新的隐藏终评。
- 选择函数从不读取final；final只用于连接已存在候选并计算诊断差值。
- 所有E2E仍为估算值，没有冒充真机实测。

## 1. 跨域共享策略族

在10个注册工作点上，严格满足三域各自 `reference AP - 0.001` 的共享
Ranker家族与动作策略为 **0/10**。协议未放宽epsilon，而是另行报告最小违反
折中配置。该折中在30个数据集级final诊断点中有
**{shared['epsilon_violations']}/30** 个容忍违反，平均AP差
**{shared['mean_final_ap_delta']:+.6f}**，最差AP差
**{shared['worst_final_ap_delta']:+.6f}**，平均请求字节变化
**{100 * shared['mean_relative_kb_delta']:+.2f}%**。

这直接支持“同一Ranker家族+动作策略不能在当前三域和工作点下稳定满足统一
epsilon”的诊断性结论，但由于final已经在既有研究中被查看，不能写成确认性
跨域泛化证明。

## 2. 组件损伤汇总

| 变体 | epsilon违反 | 平均AP差 | 最差AP差 | 平均请求字节变化 | 平均oracle regret |
|---|---:|---:|---:|---:|---:|
| 完整协议 | {full['epsilon_violations']}/30 | {full['mean_final_ap_delta']:+.6f} | {full['worst_final_ap_delta']:+.6f} | {100 * full['mean_relative_kb_delta']:+.2f}% | {full['mean_oracle_regret']:.6f} |
| 无门禁 | {no_gate['epsilon_violations']}/30 | {no_gate['mean_final_ap_delta']:+.6f} | {no_gate['worst_final_ap_delta']:+.6f} | {100 * no_gate['mean_relative_kb_delta']:+.2f}% | {no_gate['mean_oracle_regret']:.6f} |
| 无成本目标（selector AP-only） | {no_cost['epsilon_violations']}/30 | {no_cost['mean_final_ap_delta']:+.6f} | {no_cost['worst_final_ap_delta']:+.6f} | {100 * no_cost['mean_relative_kb_delta']:+.2f}% | {no_cost['mean_oracle_regret']:.6f} |
| 固定Confidence+SharedAction | {fixed['epsilon_violations']}/30 | {fixed['mean_final_ap_delta']:+.6f} | {fixed['worst_final_ap_delta']:+.6f} | {100 * fixed['mean_relative_kb_delta']:+.2f}% | {fixed['mean_oracle_regret']:.6f} |

完整协议在本轮诊断中保持0次epsilon违反，但平均通信节省只有
**{-100 * full['mean_relative_kb_delta']:.2f}%**。无门禁减少更多字节并提高平均AP，
但出现{no_gate['epsilon_violations']}次违反，说明门禁的主要作用是约束尾部风险，
同时存在明显误拒成本。

“无成本目标”没有退化，反而在当前事后final诊断中获得更高平均AP和更低字节。
因此，这项消融**不能证明成本目标具有独立增益**；论文应将成本感知定位为问题
定义与选择约束，而不能声称成本目标模块已被消融验证优于AP-only选择。

固定Confidence在30点中有{fixed['epsilon_violations']}次违反，说明绑定单一简单
Ranker不适合作为跨域统一方案。

## 3. 尚未完成的实验

| 实验 | 状态 | 原因 |
|---|---|---|
""" + "\n".join(
        f"| {row['experiment']} | {row['status']} | {row['reason']} |"
        for row in blocked
    ) + """

这些项目未写完成标记，也不得在论文中作为已完成结果引用。

## 4. 论文含义

本轮最强的新证据是跨域共享策略族的严格不可行性和门禁的尾部风险约束作用；
最重要的负结果是当前数据不能证明成本目标本身优于AP-only选择。论文方向仍可
保持“可审计成本感知配置选择”，但创新重点应放在预算语义、实际字节口径、
配置数据隔离和经验门禁，而不是宣称成本目标学习器带来稳定性能提升。
"""


def write_analysis_outputs(
    analysis: Mapping[str, Any],
    *,
    root: Path = ROOT,
    input_hashes: Mapping[str, str],
) -> None:
    runs = root / "runs"
    write_csv(runs / "shared_policy_choices.csv", analysis["shared_choices"])
    write_csv(runs / "shared_policy_pair_audit.csv", analysis["shared_pair_audit"])
    write_csv(runs / "shared_policy_final_diagnostic.csv", analysis["shared_final"])
    write_csv(runs / "lesion_rows.csv", analysis["lesion_rows"])
    write_csv(runs / "lesion_summary.csv", analysis["lesion_summary"])
    write_csv(runs / "cross_dataset_comparison.csv", analysis["comparison_rows"])
    write_csv(runs / "cross_dataset_comparison_summary.csv", analysis["comparison_summary"])
    blocked = blocked_experiment_audit(root)
    write_json(runs / "blocked_experiments.json", blocked)
    write_json(
        runs / "provenance.json",
        {
            "protocol_id": "three_dataset_costaware_submission_ablation_v1",
            "output_role": "posthoc_diagnostic_not_confirmatory",
            "final_used_for_selection": False,
            "input_sha256": dict(sorted(input_hashes.items())),
        },
    )
    notes = root / "notes" / "results_zh.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(render_report(analysis, blocked), encoding="utf-8")


def main() -> None:
    sources = source_input_paths(ROOT)
    before = {str(path): file_sha256(path) for path in sources}
    loaded = load_analysis_inputs(ROOT)
    analysis = build_analysis(loaded, ROOT)
    write_analysis_outputs(analysis, root=ROOT, input_hashes=before)
    after = {str(path): file_sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("A frozen source input changed during analysis")
    (ROOT / "LOCAL_ANALYSIS_COMPLETED").write_text("ok\n", encoding="ascii")
    summary = {
        "status": "local_analysis_complete",
        "shared_work_points": len(analysis["shared_choices"]),
        "strict_feasible_work_points": sum(
            bool(row["strict_feasible"]) for row in analysis["shared_choices"]
        ),
        "shared_final_rows": len(analysis["shared_final"]),
        "lesion_rows": len(analysis["lesion_rows"]),
        "blocked_experiments": len(blocked_experiment_audit(ROOT)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
