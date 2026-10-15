#!/usr/bin/env python3
"""Finalize frozen COCO/Oregon choices from immutable 20260723 final artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_ROOT = (
    REPO
    / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
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


def action_map_sha256(actions: dict[str, str]) -> str:
    payload = json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def action_selection_key(
    *, mode: str, policy: str, ranker: str, budget: float
) -> str:
    return f"{mode}|{policy}|{ranker}|{float(budget):.2f}"


def _same_point(row: dict[str, Any], *, mode: str, budget: float) -> bool:
    return str(row["mode"]) == str(mode) and math.isclose(
        float(row["nominal_budget"]), float(budget), abs_tol=1e-12
    )


def find_frozen_final_row(
    rows: Iterable[dict[str, Any]], choice: dict[str, Any]
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in rows
        if _same_point(row, mode=choice["mode"], budget=float(choice["budget"]))
        and str(row["method"]) == str(choice["frozen_ranker"])
        and str(row["action_policy"]) == str(choice["frozen_policy"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one frozen final row, got {len(matches)}")
    return matches[0]


def choose_final_oracle(
    rows: Iterable[dict[str, Any]],
    *,
    mode: str,
    budget: float,
    selectable_rankers: Sequence[str],
    ranker_order: Sequence[str],
    policy_order: Sequence[str],
) -> dict[str, Any]:
    rankers = tuple(map(str, selectable_rankers))
    candidates = [
        dict(row)
        for row in rows
        if _same_point(row, mode=mode, budget=budget)
        and str(row["method"]) in set(rankers)
    ]
    if not candidates:
        raise RuntimeError(f"No final oracle candidates for {mode} {budget}")

    def order_index(value: str, order: Sequence[str]) -> int:
        return tuple(map(str, order)).index(str(value))

    chosen = min(
        candidates,
        key=lambda row: (
            -float(row["official_map50_95"]),
            float(row["request_kb_per_image"]),
            order_index(str(row["action_policy"]), policy_order),
            order_index(str(row["method"]), ranker_order),
        ),
    )
    chosen["diagnostic_only"] = True
    return chosen


def finalize(dataset: str) -> None:
    if dataset not in {"coco", "oregon"}:
        raise ValueError("Source-row finalization is valid only for COCO/Oregon")
    output = EXP_ROOT / "runs" / dataset
    frozen_path = output / "frozen_choices.json"
    marker = output / "RISK_GATE_COMPLETED"
    gate_provenance_path = output / "audit_gate_provenance.json"
    if (
        not marker.is_file()
        or not frozen_path.is_file()
        or not gate_provenance_path.is_file()
    ):
        raise RuntimeError("Frozen choices must be locked before final can be read")
    gate_provenance = read_json(gate_provenance_path)
    if gate_provenance.get("frozen_choices_sha256") != file_sha256(frozen_path):
        raise RuntimeError("Frozen choice hash differs from the completed risk gate")
    protocol = read_json(EXP_ROOT / "configs/protocol.json")
    choices = read_json(frozen_path)
    if len(choices) != int(protocol["expected_frozen_rows_per_dataset"]):
        raise RuntimeError(f"Unexpected frozen choice count: {len(choices)}")

    source = SOURCE_ROOT / "runs" / dataset
    result_path = source / "results.csv"
    selection_path = source / "selections.json"
    before = {
        str(result_path.resolve()): file_sha256(result_path),
        str(selection_path.resolve()): file_sha256(selection_path),
    }
    rows = read_csv(result_path)
    selections = read_json(selection_path)
    final_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for choice in choices:
        frozen = find_frozen_final_row(rows, choice)
        key = action_selection_key(
            mode=str(choice["mode"]),
            policy=str(choice["frozen_policy"]),
            ranker=str(choice["frozen_ranker"]),
            budget=float(choice["budget"]),
        )
        if key not in selections:
            raise RuntimeError(f"Missing frozen source action map: {key}")
        final_row = {
            **frozen,
            **{f"selector_{key}": value for key, value in choice.items()},
            "frozen_action_map_sha256": action_map_sha256(selections[key]),
            "final_used_for_selection": False,
        }
        oracle = choose_final_oracle(
            rows,
            mode=str(choice["mode"]),
            budget=float(choice["budget"]),
            selectable_rankers=protocol["selectable_rankers"],
            ranker_order=protocol["selectable_rankers"],
            policy_order=protocol["policy_tie_break_order"],
        )
        oracle_rows.append(
            {
                **oracle,
                "dataset": dataset,
                "mode": choice["mode"],
                "budget": choice["budget"],
                "frozen_ranker": choice["frozen_ranker"],
                "frozen_policy": choice["frozen_policy"],
                "frozen_official_map50_95": float(frozen["official_map50_95"]),
                "final_ap_regret": float(oracle["official_map50_95"])
                - float(frozen["official_map50_95"]),
                "diagnostic_only": True,
            }
        )
        final_rows.append(final_row)
    write_csv(output / "final_results.csv", final_rows)
    write_csv(output / "final_oracle_diagnostic.csv", oracle_rows)
    after = {
        str(result_path.resolve()): file_sha256(result_path),
        str(selection_path.resolve()): file_sha256(selection_path),
    }
    if before != after:
        raise RuntimeError("A frozen source final artifact changed")
    write_json(
        output / "final_integrity.json",
        {
            "dataset": dataset,
            "frozen_choices_sha256": file_sha256(frozen_path),
            "source_final_sha256_before": before,
            "source_final_sha256_after": after,
            "final_used_for_selection": False,
            "rows": len(final_rows),
            "oracle_rows": len(oracle_rows),
        },
    )
    (output / "FINAL_COMPLETED").write_text("ok\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "oregon"), required=True)
    args = parser.parse_args()
    finalize(args.dataset)


if __name__ == "__main__":
    main()
