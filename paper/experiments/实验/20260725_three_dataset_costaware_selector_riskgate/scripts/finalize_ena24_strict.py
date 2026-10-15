#!/usr/bin/env python3
"""Evaluate the strictly frozen ENA24 choices on final exactly once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import run_ena24_strict as strict


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
OUTPUT = EXP_ROOT / "runs/ena24"
SOURCE_20 = REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
SOURCE_23 = (
    REPO
    / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
)
CANONICAL = SOURCE_20 / "runs/canonical_v1"
COMMON = CANONICAL / "common/ena24"
SOURCE_RUN = SOURCE_23 / "runs/ena24"
EXPECTED_FINAL_IMAGES = 1758


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
    if not values:
        raise RuntimeError(f"Refusing to write an empty final table: {path}")
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
        raise FileNotFoundError("Missing frozen ENA24 final inputs:\n" + "\n".join(missing))
    return {str(path): file_sha256(path) for path in normalized}


def validate_gate_lock(output: Path) -> str:
    output = Path(output)
    marker_path = output / "STRICT_RISK_GATE_COMPLETED"
    choices_path = output / "strict_frozen_choices.json"
    if not marker_path.is_file() or not choices_path.is_file():
        raise RuntimeError("Strict risk gate must complete before final is opened")
    marker = read_json(marker_path)
    observed = file_sha256(choices_path)
    if marker.get("strict_frozen_choices_sha256") != observed:
        raise RuntimeError("Strict frozen choice hash differs from the gate marker")
    return observed


def choose_final_oracle(
    rows: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    budget: float,
    ranker_order: Sequence[str],
    policy_order: Sequence[str],
) -> dict[str, Any]:
    candidates = [
        dict(row)
        for row in rows
        if str(row["mode"]) == str(mode)
        and math.isclose(float(row["budget"]), float(budget), abs_tol=1e-12)
    ]
    if not candidates:
        raise RuntimeError(f"No ENA24 final oracle candidates for {mode} {budget}")
    chosen = min(
        candidates,
        key=lambda row: (
            -float(row["official_map50_95"]),
            float(row["request_kb_per_image"]),
            tuple(policy_order).index(str(row["policy"])),
            tuple(ranker_order).index(str(row["ranker"])),
        ),
    )
    chosen["diagnostic_only"] = True
    return chosen


def _manifest_paths(role: str) -> list[Path]:
    manifest_path = COMMON / role / "source_manifest.json"
    manifest = read_json(manifest_path)
    sources = manifest.get("source_paths", [])
    if isinstance(sources, Mapping):
        sources = list(sources)
    return [
        manifest_path,
        Path(manifest["samples"]),
        Path(manifest["gt"]),
        *(Path(path) for path in manifest["action_sources"].values()),
        *(Path(path) for path in sources),
    ]


def final_input_paths() -> list[Path]:
    return [
        SCRIPT,
        EXP_ROOT / "configs/protocol.json",
        OUTPUT / "STRICT_RISK_GATE_COMPLETED",
        OUTPUT / "strict_frozen_choices.json",
        OUTPUT / "strict_risk_gate_provenance.json",
        OUTPUT / "strict_candidate_table.csv",
        OUTPUT / "strict_threshold_choices.json",
        OUTPUT / "strict_split_manifest.json",
        CANONICAL / "benchmark/ena24/models.json",
        CANONICAL / "benchmark/ena24/action_gate.json",
        SOURCE_RUN / "models/current_controlled_32d.json",
        SOURCE_RUN / "models/saferoi_gbdt.pkl",
        *strict._calibration_manifest_inputs(),
        *_manifest_paths("final"),
    ]


def _scores(tri: Any, models: Mapping[str, Any], controlled: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values = tri.current.ranker_scores(models, data)
    values["Current-Controlled-32D"] = tri.current.predict_pairwise_ranker(
        controlled,
        np.asarray([sample["pca32"] for sample in data["samples"]], dtype=np.float64),
    )
    return values


def _runtime() -> dict[str, Any]:
    _, tri = strict._load_engine()
    selector, _ = strict._load_strict_roles(tri)
    final = tri.current.load_role("ena24", "final")
    if len(final["samples"]) != EXPECTED_FINAL_IMAGES:
        raise RuntimeError(
            f"ENA24 final size drift: {len(final['samples'])} != {EXPECTED_FINAL_IMAGES}"
        )
    models = tri.load_frozen_models(CANONICAL / "benchmark/ena24/models.json")
    controlled = read_json(SOURCE_RUN / "models/current_controlled_32d.json")
    action_gate = read_json(CANONICAL / "benchmark/ena24/action_gate.json")
    with (SOURCE_RUN / "models/saferoi_gbdt.pkl").open("rb") as handle:
        roi_model = pickle.load(handle)
    return {
        "tri": tri,
        "selector": selector,
        "final": final,
        "models": models,
        "selector_scores": _scores(tri, models, controlled, selector),
        "final_scores": _scores(tri, models, controlled, final),
        "base_actions": tri.current.chosen_actions(
            models["action_heads"], action_gate, final
        ),
        "roi_scores": tri.predict_gbdt(roi_model, tri.profile_features(final)),
    }


def _action_sha256(actions: Mapping[int, str]) -> str:
    payload = json.dumps(
        {str(key): value for key, value in actions.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pending_candidate_keys(
    all_keys: Iterable[tuple[str, float, str, str]],
    *,
    completed: set[tuple[str, float, str, str]],
    max_new_candidates: int,
) -> list[tuple[str, float, str, str]]:
    pending = [key for key in sorted(all_keys) if key not in completed]
    if int(max_new_candidates) > 0:
        pending = pending[: int(max_new_candidates)]
    return pending


def completed_final_is_valid(
    output: Path,
    *,
    frozen_sha: str,
    expected_candidates: int,
    expected_points: int,
) -> bool:
    output = Path(output)
    required = (
        "FINAL_COMPLETED",
        "final_integrity.json",
        "final_candidate_results.csv",
        "final_results.csv",
        "final_oracle_diagnostic.csv",
        "final_selections.json",
    )
    if any(not (output / name).is_file() for name in required):
        return False
    integrity = read_json(output / "final_integrity.json")
    return (
        integrity.get("strict_frozen_choices_sha256") == frozen_sha
        and int(integrity.get("candidate_rows", -1)) == int(expected_candidates)
        and int(integrity.get("final_rows", -1)) == int(expected_points)
        and int(integrity.get("oracle_rows", -1)) == int(expected_points)
    )


def _row_key(row: Mapping[str, Any]) -> tuple[str, float, str, str]:
    return (
        str(row["mode"]),
        float(row["budget"]),
        str(row["ranker"]),
        str(row["policy"]),
    )


def finalize(*, max_new_candidates: int = 0) -> bool:
    frozen_sha = validate_gate_lock(OUTPUT)
    gate_provenance = read_json(OUTPUT / "strict_risk_gate_provenance.json")
    if gate_provenance.get("strict_frozen_choices_sha256") != frozen_sha:
        raise RuntimeError("Strict gate provenance differs from frozen choices")
    protocol = read_json(EXP_ROOT / "configs/protocol.json")
    frozen_choices = read_json(OUTPUT / "strict_frozen_choices.json")
    expected_points = int(protocol["expected_frozen_rows_per_dataset"])
    if len(frozen_choices) != expected_points:
        raise RuntimeError(f"Strict frozen choice count drift: {len(frozen_choices)}")
    expected_candidates = expected_points * int(
        protocol["expected_candidates_per_work_point"]
    )
    if completed_final_is_valid(
        OUTPUT,
        frozen_sha=frozen_sha,
        expected_candidates=expected_candidates,
        expected_points=expected_points,
    ):
        print("[ena24-final] completed artifacts already match frozen choices", flush=True)
        return True

    before = snapshot_sha256(final_input_paths())
    candidate_lookup = strict._candidate_lookup()
    if len(candidate_lookup) != expected_candidates:
        raise RuntimeError(
            f"Strict candidate count drift: {len(candidate_lookup)} != {expected_candidates}"
        )
    frozen_by_key = {
        (
            str(choice["mode"]),
            float(choice["budget"]),
            str(choice["frozen"]["ranker"]),
            str(choice["frozen"]["policy"]),
        ): choice
        for choice in frozen_choices
    }
    if len(frozen_by_key) != expected_points:
        raise RuntimeError("Strict frozen work points are not unique")

    checkpoint_path = OUTPUT / "ena24_final_checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if checkpoint.get("strict_frozen_choices_sha256") != frozen_sha:
            raise RuntimeError("ENA24 final checkpoint frozen hash changed")
        if checkpoint.get("frozen_input_sha256") != before:
            raise RuntimeError("ENA24 final checkpoint input hashes changed")
        candidate_results = list(checkpoint.get("candidate_results", []))
        selections = dict(checkpoint.get("frozen_selections", {}))
    else:
        candidate_results = []
        selections = {}
    completed = {_row_key(row) for row in candidate_results}
    if len(completed) != len(candidate_results):
        raise RuntimeError("ENA24 final checkpoint contains duplicate candidates")
    pending = pending_candidate_keys(
        candidate_lookup,
        completed=completed,
        max_new_candidates=max_new_candidates,
    )

    runtime = _runtime() if pending else None
    tri = runtime["tri"] if runtime is not None else None
    for key in pending:
        if runtime is None or tri is None:
            raise RuntimeError("ENA24 final runtime unavailable for pending work")
        candidate = candidate_lookup[key]
        mode, budget, ranker, policy = key
        print(
            f"[ena24-final] {len(candidate_results) + 1}/{expected_candidates} "
            f"{mode} {budget:.2f} {ranker} {policy}",
            flush=True,
        )
        selected = strict._selected_actions(
            tri=tri,
            data=runtime["final"],
            selector_scores=runtime["selector_scores"][ranker],
            gate_scores=runtime["final_scores"][ranker],
            base_actions=runtime["base_actions"],
            roi_scores=runtime["roi_scores"],
            mode=mode,
            budget=budget,
            candidate=candidate,
            threshold=float(candidate["threshold"]),
        )
        evaluated = tri.current.evaluate(
            ranker,
            mode,
            budget,
            runtime["final"],
            selected,
            runtime["models"],
        )
        row = {
            **evaluated,
            "dataset": "ena24",
            "mode": mode,
            "budget": budget,
            "ranker": ranker,
            "policy": policy,
            "roi_threshold": float(candidate["threshold"]),
            "selection_source_role": "near_duplicate_grouped_selector",
            "gate_source_role": "near_duplicate_grouped_gate",
            "final_used_for_selection": False,
        }
        candidate_results.append(row)
        if key in frozen_by_key:
            selection_key = "|".join(
                (key[0], key[3], key[2], f"{key[1]:.2f}")
            )
            selections[selection_key] = {
                str(image_id): action for image_id, action in selected.items()
            }
        write_json(
            checkpoint_path,
            {
                "strict_frozen_choices_sha256": frozen_sha,
                "frozen_input_sha256": before,
                "candidate_results": candidate_results,
                "frozen_selections": selections,
            },
        )
        write_csv(OUTPUT / "final_candidate_results.csv", candidate_results)

    after_batch = snapshot_sha256(final_input_paths())
    if before != after_batch:
        raise RuntimeError("A frozen ENA24 input changed during final evaluation")
    if len(candidate_results) < expected_candidates:
        print(
            f"[ena24-final] checkpointed {len(candidate_results)}/"
            f"{expected_candidates} candidates",
            flush=True,
        )
        return False
    if len(candidate_results) != expected_candidates:
        raise RuntimeError(f"ENA24 final candidate row drift: {len(candidate_results)}")
    result_lookup = {
        (row["mode"], float(row["budget"]), row["ranker"], row["policy"]): row
        for row in candidate_results
    }
    if len(result_lookup) != expected_candidates:
        raise RuntimeError("ENA24 final candidate rows contain duplicate work points")

    final_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for choice in frozen_choices:
        frozen = choice["frozen"]
        key = (
            str(choice["mode"]),
            float(choice["budget"]),
            str(frozen["ranker"]),
            str(frozen["policy"]),
        )
        row = dict(result_lookup[key])
        selection_key = "|".join(
            (key[0], key[3], key[2], f"{key[1]:.2f}")
        )
        actions = selections[selection_key]
        row.update(
            {
                "gate_passed": bool(choice["gate_passed"]),
                "used_reference_fallback": bool(
                    choice["used_reference_fallback"]
                ),
                "frozen_action_map_sha256": _action_sha256(
                    {int(image_id): action for image_id, action in actions.items()}
                ),
            }
        )
        final_rows.append(row)
        oracle = choose_final_oracle(
            candidate_results,
            mode=key[0],
            budget=key[1],
            ranker_order=protocol["selectable_rankers"],
            policy_order=protocol["policy_tie_break_order"],
        )
        oracle_rows.append(
            {
                **oracle,
                "frozen_ranker": key[2],
                "frozen_policy": key[3],
                "frozen_official_map50_95": float(row["official_map50_95"]),
                "final_ap_regret": float(oracle["official_map50_95"])
                - float(row["official_map50_95"]),
                "diagnostic_only": True,
            }
        )

    write_csv(OUTPUT / "final_candidate_results.csv", candidate_results)
    write_csv(OUTPUT / "final_results.csv", final_rows)
    write_csv(OUTPUT / "final_oracle_diagnostic.csv", oracle_rows)
    write_json(OUTPUT / "final_selections.json", selections)
    after = snapshot_sha256(final_input_paths())
    if before != after:
        raise RuntimeError("A frozen ENA24 final input changed during evaluation")
    write_json(
        OUTPUT / "final_integrity.json",
        {
            "dataset": "ena24",
            "strict_frozen_choices_sha256": frozen_sha,
            "frozen_input_sha256_before": before,
            "frozen_input_sha256_after": after,
            "candidate_rows": len(candidate_results),
            "final_rows": len(final_rows),
            "oracle_rows": len(oracle_rows),
            "final_used_for_selection": False,
        },
    )
    (OUTPUT / "FINAL_COMPLETED").write_text("ok\n", encoding="ascii")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-candidates", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finalize(max_new_candidates=args.max_new_candidates)
