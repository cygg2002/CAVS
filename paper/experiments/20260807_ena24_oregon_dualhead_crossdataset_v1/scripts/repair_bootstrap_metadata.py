#!/usr/bin/env python3
"""Repair stale labels in completed Track B bootstrap artifacts.

The paired bootstrap statistics are preserved byte-for-byte at the semantic
level.  Only labels are rebuilt from the frozen Track B ``results.csv`` rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parents[1]
DATASETS = ("ena24", "oregon")
NUMERIC_FIELDS = (
    "replicates",
    "official_map50_95_delta_mean",
    "official_map50_95_delta_ci_low",
    "official_map50_95_delta_ci_high",
    "official_map50_95_probability_gt_zero",
    "request_kb_per_image_delta_mean",
    "request_kb_per_image_delta_ci_low",
    "request_kb_per_image_delta_ci_high",
)

if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
from crossdataset_core import track_b_bootstrap_metadata  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{path.stat().st_ino if path.exists() else 'new'}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.repair.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def numeric_snapshot(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    snapshot = []
    for row in rows:
        item = {"key": str(row.get("key", ""))}
        item.update({field: row.get(field) for field in NUMERIC_FIELDS})
        snapshot.append(item)
    return sorted(snapshot, key=lambda row: row["key"])


def core_index(track_b: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(track_b / "results.csv")
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        key = f"{row['action_policy']}|{float(row['nominal_budget']):g}"
        if key in index:
            raise RuntimeError(f"Duplicate Track B core row: {key}")
        index[key] = row
    return index


def repair_dataset(dataset_root: Path, *, dry_run: bool) -> dict[str, Any]:
    track_b = dataset_root / "track_b"
    bootstrap = track_b / "bootstrap"
    if not (track_b / "results.csv").is_file():
        raise FileNotFoundError(track_b / "results.csv")
    if not bootstrap.is_dir():
        raise FileNotFoundError(bootstrap)

    index = core_index(track_b)
    json_paths = sorted(
        path
        for path in bootstrap.glob("*.json")
        if not path.name.endswith(".completion.json")
    )
    if not json_paths:
        raise RuntimeError(f"No bootstrap JSON artifacts found: {bootstrap}")

    before_rows = read_csv(track_b / "paired_bootstrap.csv")
    expected_keys = set(index)
    observed_keys = {str(row.get("key", "")) for row in before_rows}
    if observed_keys != expected_keys or len(before_rows) != len(expected_keys):
        raise RuntimeError(
            f"Bootstrap CSV matrix mismatch for {dataset_root.name}: "
            f"missing={sorted(expected_keys - observed_keys)} "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    before_numeric_hash = sha256_json(numeric_snapshot(before_rows))
    before_pair_hashes = {
        str(row["key"]): str(row.get("action_pair_sha256", "")) for row in before_rows
    }
    before_json_numeric: dict[str, dict[str, Any]] = {}

    repaired_json = 0
    repaired_markers = 0
    for path in json_paths:
        payload = read_json(path)
        key = str(payload.get("key", ""))
        if key not in index:
            raise RuntimeError(f"Unknown bootstrap JSON key: {dataset_root.name}/{key}")
        before_json_numeric[key] = {
            field: payload.get(field) for field in NUMERIC_FIELDS
        }
        metadata = track_b_bootstrap_metadata(key, index[key])
        payload.update(metadata)
        if not dry_run:
            atomic_json(path, payload)
        repaired_json += 1

        marker = path.with_suffix(".completion.json")
        if marker.is_file():
            marker_payload = read_json(marker)
            marker_payload["complete"] = True
            marker_payload["sha256"] = hashlib.sha256(
                path.read_bytes() if not dry_run else json.dumps(
                    payload, indent=2, ensure_ascii=False, allow_nan=False
                ).encode("utf-8")
            ).hexdigest()
            if not dry_run:
                atomic_json(marker, marker_payload)
            repaired_markers += 1

    after_rows = []
    for row in before_rows:
        key = str(row["key"])
        updated = dict(row)
        updated.update(track_b_bootstrap_metadata(key, index[key]))
        after_rows.append(updated)
    after_numeric_hash = sha256_json(numeric_snapshot(after_rows))
    after_pair_hashes = {
        str(row["key"]): str(row.get("action_pair_sha256", "")) for row in after_rows
    }
    if before_numeric_hash != after_numeric_hash:
        raise RuntimeError(f"Numeric bootstrap fields changed for {dataset_root.name}")
    if before_pair_hashes != after_pair_hashes:
        raise RuntimeError(f"Action-pair hashes changed for {dataset_root.name}")
    after_json_numeric = {
        key: {
            field: read_json(
                bootstrap / f"{key.replace('|', '_')}.json"
            ).get(field)
            for field in NUMERIC_FIELDS
        }
        for key in before_json_numeric
    }
    if before_json_numeric != after_json_numeric:
        raise RuntimeError(f"Numeric bootstrap JSON fields changed for {dataset_root.name}")
    if not dry_run:
        write_csv(track_b / "paired_bootstrap.csv", after_rows)

    return {
        "dataset": dataset_root.name,
        "rows": len(after_rows),
        "json_artifacts": repaired_json,
        "completion_markers": repaired_markers,
        "numeric_fields_unchanged": before_numeric_hash == after_numeric_hash,
        "numeric_json_fields_unchanged": before_json_numeric == after_json_numeric,
        "action_pair_hashes_unchanged": before_pair_hashes == after_pair_hashes,
        "numeric_snapshot_sha256": after_numeric_hash,
        "dry_run": dry_run,
    }


def repair_root(root: Path, *, dry_run: bool) -> dict[str, Any]:
    datasets = [repair_dataset(root / dataset, dry_run=dry_run) for dataset in DATASETS]
    report = {
        "repair": "track_b_bootstrap_metadata_v1",
        "experiment_root": str(root.parent),
        "datasets": datasets,
        "dry_run": dry_run,
    }
    if not dry_run:
        report_path = root / "bootstrap_metadata_repair_report.json"
        atomic_json(report_path, report)
        report["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        atomic_json(report_path, report)
        for dataset in DATASETS:
            track_b = root / dataset / "track_b"
            dataset_report_path = track_b / "report.json"
            if dataset_report_path.is_file():
                dataset_report = read_json(dataset_report_path)
                dataset_report.update(
                    {
                        "bootstrap_metadata_repaired": True,
                        "bootstrap_metadata_repair_report": str(report_path),
                    }
                )
                atomic_json(dataset_report_path, dataset_report)
            completion_path = track_b / "completion.json"
            if completion_path.is_file():
                completion = read_json(completion_path)
                completion.update(
                    {
                        "complete": True,
                        "bootstrap_metadata_repaired": True,
                    }
                )
                atomic_json(completion_path, completion)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=EXPERIMENT_ROOT / "runs",
        help="Experiment runs directory containing ena24/ and oregon/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without writing artifacts.",
    )
    args = parser.parse_args()
    print(json.dumps(repair_root(args.root, dry_run=args.dry_run), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
