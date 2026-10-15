#!/usr/bin/env python3
"""Run the canonical COCO fit=9,678 DualHead paper comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

import canonical_core
from canonical_core import (
    validate_protocol,
    validate_reward_population,
    validate_role_populations,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
LEGACY_ROOT = (
    REPO
    / "paper/experiments/20260804_coco_dualhead_fixedbyte_registered_v1"
)
REWARD_CORE_PATH = (
    REPO
    / "paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/paper_core.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for module_root in (
    LEGACY_ROOT / "scripts",
    REPO / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts",
    REPO / "paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/scripts",
    REPO / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines/scripts",
    REPO / "paper/experiments/CA_AORIC",
    REPO,
):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

LEGACY = _load_module("canonical9678_legacy_runner", LEGACY_ROOT / "scripts/run_paper.py")
REWARD_CORE = _load_module("canonical9678_reward_core", REWARD_CORE_PATH)


ROLE_DIRECTORY = {
    "fit": "fit",
    "development": "fit",
    "calibration": "calibration",
    "audit": "calibration",
    "registered_final": "final",
    "final": "final",
}
CONTRACT_ROLE = {
    "fit": "fit",
    "development": "fit",
    "calibration": "calibration",
    "audit": "calibration",
    "registered_final": "registered_final",
    "final": "registered_final",
}
_ROLE_CACHE: dict[str, list[dict]] = {}
_REWARD_STATE: dict[str, Any] = {}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    return protocol


def _canonical_root(protocol: Mapping[str, Any]) -> Path:
    return Path(protocol["paths"]["canonical_common"])


def _role_root(role: str, protocol: Mapping[str, Any]) -> Path:
    try:
        directory = ROLE_DIRECTORY[str(role)]
    except KeyError as error:
        raise ValueError(f"Unknown canonical role: {role}") from error
    return _canonical_root(protocol) / directory


def _role_samples(role: str, protocol: Mapping[str, Any]) -> list[dict]:
    return [dict(row) for row in read_json(_role_root(role, protocol) / "samples.json")]


def _source_fingerprint(protocol: Mapping[str, Any]) -> dict[str, str]:
    observed = {
        "protocol": file_sha256(PROTOCOL_PATH),
        "canonical_runner": file_sha256(Path(__file__)),
        "canonical_core": file_sha256(ROOT / "scripts/canonical_core.py"),
        "legacy_runner": file_sha256(LEGACY_ROOT / "scripts/run_paper.py"),
        "reward_core": file_sha256(REWARD_CORE_PATH),
        "edge_model": file_sha256(Path(protocol["paths"]["edge_model"])),
        "cloud_model": file_sha256(Path(protocol["paths"]["cloud_model"])),
    }
    for role in ("fit", "calibration", "final"):
        root = _canonical_root(protocol) / role
        for name in ("samples.json", "source_manifest.json", "gt.json"):
            observed[f"canonical_{role}_{name}"] = file_sha256(root / name)
    if observed["edge_model"] != protocol["hashes"]["edge_model"]:
        raise RuntimeError("Edge model hash mismatch")
    if observed["cloud_model"] != protocol["hashes"]["cloud_model"]:
        raise RuntimeError("Cloud model hash mismatch")
    return observed


def _prediction_rows(path: Path, image_ids: set[int]) -> list[dict]:
    return [
        dict(row)
        for row in read_json(path)
        if int(row["image_id"]) in image_ids
    ]


def _records_for_role(role: str, protocol: Mapping[str, Any]) -> list[dict]:
    canonical_role = CONTRACT_ROLE[str(role)]
    if canonical_role in _ROLE_CACHE:
        return _ROLE_CACHE[canonical_role]
    root = _role_root(role, protocol)
    samples = _role_samples(role, protocol)
    image_ids = {int(row["image_id"]) for row in samples}
    manifest = read_json(root / "source_manifest.json")
    action_sources = manifest.get("action_sources", {})
    if not {"edge", "original"}.issubset(action_sources):
        raise RuntimeError(f"Missing Edge/Original source for {canonical_role}")
    edge_rows = _prediction_rows(Path(action_sources["edge"]), image_ids)
    cloud_rows = _prediction_rows(Path(action_sources["original"]), image_ids)
    sources = []
    for sample in samples:
        original_bytes = sample.get("payload_bytes", {}).get("original")
        if original_bytes is None:
            raise RuntimeError(f"Missing Original payload: {sample['image_id']}")
        sources.append(
            {
                "image_id": int(sample["image_id"]),
                "file_name": Path(str(sample["identity"])).name,
                "path": str(sample["identity"]),
                "identity": str(sample["identity"]),
                "width": int(sample["width"]),
                "height": int(sample["height"]),
                "payload_bytes": int(original_bytes),
            }
        )
    records = LEGACY._records_from_sources(
        sources, edge_rows, cloud_rows, protocol
    )
    expected = int(protocol["roles"][canonical_role])
    if len(records) != expected:
        raise RuntimeError(
            f"Canonical {canonical_role} population changed: {len(records)} != {expected}"
        )
    _ROLE_CACHE[canonical_role] = records
    return records


def _evaluate_role(
    role: str,
    protocol: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> tuple[list[dict], dict[str, list[int]]]:
    role_protocol = copy.deepcopy(dict(protocol))
    role_protocol["paths"]["annotations"] = str(_role_root(role, protocol) / "gt.json")
    return _LEGACY_EVALUATE_ROLE(role, role_protocol, artifact)


def _registered_ground_truth(
    protocol: Mapping[str, Any], image_ids: Sequence[int]
) -> dict:
    source = read_json(_role_root("registered_final", protocol) / "gt.json")
    allowed = set(map(int, image_ids))
    output = {
        key: value
        for key, value in source.items()
        if key not in {"images", "annotations"}
    }
    output["images"] = [
        row for row in source["images"] if int(row["id"]) in allowed
    ]
    output["annotations"] = [
        row for row in source["annotations"] if int(row["image_id"]) in allowed
    ]
    if {int(row["id"]) for row in output["images"]} != allowed:
        raise RuntimeError("Canonical val2017 ground-truth population mismatch")
    return output


_LEGACY_EVALUATE_ROLE = LEGACY._evaluate_role
LEGACY.ROOT = ROOT
LEGACY.RUNS = RUNS
LEGACY.PROTOCOL_PATH = PROTOCOL_PATH
LEGACY.validate_protocol = validate_protocol
LEGACY.source_fingerprint = _source_fingerprint
LEGACY._role_records = _records_for_role
LEGACY._development_records = lambda protocol: _records_for_role("fit", protocol)
LEGACY._evaluate_role = _evaluate_role
LEGACY._registered_ground_truth = _registered_ground_truth


def _paired_bootstrap(
    *,
    ground_truth: dict,
    image_ids: list[int],
    candidate_predictions: list[dict],
    comparator_predictions: list[dict],
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0,
        len(image_ids),
        size=(int(replicates), len(image_ids)),
        dtype=np.int32,
    )
    indices_sha256 = hashlib.sha256(indices.tobytes()).hexdigest()
    canonical_core.configure_bootstrap(
        ground_truth,
        image_ids,
        candidate_predictions,
        comparator_predictions,
    )
    values = np.empty(int(replicates), dtype=np.float64)
    workers = min(24, max(1, (os.cpu_count() or 1) - 2))
    context = mp.get_context("fork")
    try:
        with context.Pool(processes=workers) as pool:
            tasks = ((index, indices[index]) for index in range(int(replicates)))
            for completed, (replicate, delta) in enumerate(
                pool.imap_unordered(
                    canonical_core.bootstrap_worker,
                    tasks,
                    chunksize=1,
                ),
                start=1,
            ):
                values[replicate] = delta
                if completed % 25 == 0 or completed == int(replicates):
                    print(f"[bootstrap] {completed}/{replicates}", flush=True)
    finally:
        canonical_core.clear_bootstrap()
    return values, indices_sha256


LEGACY._paired_bootstrap = _paired_bootstrap


def run_prepare() -> dict:
    protocol = _protocol()
    completion_path = RUNS / "prepare/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "prepare/report.json")
    samples = {
        role: _role_samples(role, protocol)
        for role in ("fit", "calibration", "registered_final")
    }
    validate_role_populations(samples)
    for role in samples:
        root = _role_root(role, protocol)
        manifest = read_json(root / "source_manifest.json")
        expected = int(protocol["roles"][role])
        if int(manifest.get("images", -1)) != expected:
            raise RuntimeError(f"Canonical source manifest changed for {role}")
        for action in protocol["actions"]:
            source = manifest.get("action_sources", {}).get(action)
            if source is None or not Path(source).is_file():
                raise RuntimeError(f"Missing canonical {role}/{action} cache")
            expected_hash = manifest.get("source_paths", {}).get(str(source))
            if expected_hash is None or file_sha256(Path(source)) != expected_hash:
                raise RuntimeError(f"Canonical {role}/{action} cache hash changed")
    fingerprint = _source_fingerprint(protocol)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "roles": {
            role: [
                {"image_id": int(row["image_id"]), "identity": str(row["identity"])}
                for row in rows
            ]
            for role, rows in samples.items()
        },
        "role_counts": {role: len(rows) for role, rows in samples.items()},
        "role_ids_sha256": {
            role: LEGACY.canonical_hash([int(row["image_id"]) for row in rows])
            for role, rows in samples.items()
        },
        "source_fingerprint": fingerprint,
        "temporary_5000_checkpoint_reused": False,
        "registered_final_used_for_selection": False,
    }
    LEGACY.atomic_json(RUNS / "prepare/split_manifest.json", manifest)
    report = {
        "protocol_id": protocol["protocol_id"],
        "role_counts": manifest["role_counts"],
        "role_ids_sha256": manifest["role_ids_sha256"],
        "fit_source_split": "train2017",
        "final_source_split": "val2017",
        "temporary_5000_checkpoint_reused": False,
        "source_fingerprint": fingerprint,
    }
    LEGACY.atomic_json(RUNS / "prepare/report.json", report)
    LEGACY.atomic_json(
        completion_path,
        {
            "complete": True,
            "manifest_sha256": file_sha256(RUNS / "prepare/split_manifest.json"),
            "source_fingerprint": fingerprint,
        },
    )
    return report


def _reward_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid reward checkpoint line {line_number}") from error
    return rows


def _init_reward_worker(annotation_path: str) -> None:
    from ca_aoric import CocoMetricBackend

    _REWARD_STATE["metric"] = CocoMetricBackend(Path(annotation_path))


def _reward_worker(target_id: int) -> dict:
    ordered_ids = _REWARD_STATE["ordered_ids"]
    context_ids = REWARD_CORE.context_indices(
        ordered_ids,
        int(target_id),
        context_size=int(_REWARD_STATE["context_size"]),
        seed=int(_REWARD_STATE["seed"]),
    )
    by_id = _REWARD_STATE["by_id"]
    return REWARD_CORE.dual_metric_context_reward(
        _REWARD_STATE["metric"],
        [by_id[image_id] for image_id in context_ids],
        by_id[int(target_id)],
    )


def run_rewards() -> dict:
    protocol = _protocol()
    completion_path = RUNS / "rewards/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "rewards/report.json")
    prepared = read_json(RUNS / "prepare/completion.json")
    if prepared.get("complete") is not True:
        raise RuntimeError("Prepare must complete before rewards")
    records = _records_for_role("fit", protocol)
    fit_ids = [int(row["image_id"]) for row in records]
    reward_path = RUNS / "rewards/contextual_rewards.jsonl"
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _reward_rows(reward_path)
    completed = [int(row["target_image_id"]) for row in existing]
    if len(completed) != len(set(completed)):
        raise RuntimeError("Duplicate target in reward checkpoint")
    if not set(completed).issubset(set(fit_ids)):
        raise RuntimeError("Non-fit target in reward checkpoint")
    pending = [image_id for image_id in fit_ids if image_id not in set(completed)]
    _REWARD_STATE.clear()
    _REWARD_STATE.update(
        {
            "ordered_ids": fit_ids,
            "by_id": {int(row["image_id"]): row for row in records},
            "context_size": int(protocol["reward"]["context_size"]),
            "seed": int(protocol["reward"]["seed"]),
        }
    )
    workers = min(
        int(protocol["reward"]["workers"]),
        max(1, (os.cpu_count() or 1) - 2),
    )
    context = mp.get_context("fork")
    with reward_path.open("a", encoding="utf-8") as handle:
        with context.Pool(
            processes=workers,
            initializer=_init_reward_worker,
            initargs=(str(_role_root("fit", protocol) / "gt.json"),),
        ) as pool:
            for completed_count, row in enumerate(
                pool.imap_unordered(_reward_worker, pending, chunksize=1),
                start=len(completed) + 1,
            ):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                if completed_count % int(protocol["reward"]["progress_every"]) == 0:
                    os.fsync(handle.fileno())
                    print(f"[reward] {completed_count}/{len(fit_ids)}", flush=True)
        os.fsync(handle.fileno())
    rows = _reward_rows(reward_path)
    reward_ids = {int(row["target_image_id"]) for row in rows}
    role_manifest = read_json(RUNS / "prepare/split_manifest.json")["roles"]
    validate_reward_population(
        reward_ids,
        {int(row["image_id"]) for row in role_manifest["fit"]},
        {int(row["image_id"]) for row in role_manifest["calibration"]},
        {int(row["image_id"]) for row in role_manifest["registered_final"]},
    )
    report = {
        "protocol_id": protocol["protocol_id"],
        "role": "fit",
        "images": len(rows),
        "context_size": int(protocol["reward"]["context_size"]),
        "workers": workers,
        "reward_ap50_mean": sum(float(row["reward_ap50"]) for row in rows) / len(rows),
        "reward_ap50_95_mean": sum(float(row["reward_ap50_95"]) for row in rows) / len(rows),
        "calibration_ids_used": False,
        "registered_final_ids_used": False,
    }
    LEGACY.atomic_json(RUNS / "rewards/report.json", report)
    LEGACY.atomic_json(
        completion_path,
        {
            "complete": True,
            "rewards_sha256": file_sha256(reward_path),
            "report_sha256": file_sha256(RUNS / "rewards/report.json"),
            "calibration_ids_used": False,
            "registered_final_ids_used": False,
        },
    )
    return report


def run_train() -> dict:
    protocol = _protocol()
    rewards = read_json(RUNS / "rewards/completion.json")
    reward_path = RUNS / "rewards/contextual_rewards.jsonl"
    if rewards.get("complete") is not True or rewards.get("rewards_sha256") != file_sha256(reward_path):
        raise RuntimeError("Frozen fit=9,678 rewards are incomplete or changed")
    report = LEGACY.run_train()
    if int(report.get("development_images", -1)) != 9678:
        raise RuntimeError("Training did not use canonical fit=9,678")
    return report


def run_calibration() -> dict:
    protocol = _protocol()
    output = RUNS / "audit"
    completion_path = output / "completion.json"
    if completion_path.is_file():
        return read_json(output / "report.json")
    artifact = LEGACY._load_checkpoints()
    rows, selections = LEGACY._evaluate_role("calibration", protocol, artifact)
    frozen = LEGACY.freeze_primary_comparators(rows, mode="fixed_byte")
    LEGACY.atomic_csv(output / "results.csv", rows)
    LEGACY.atomic_json(output / "selections.json", selections)
    LEGACY.atomic_json(output / "frozen_comparators.json", frozen)
    report = {
        "protocol_id": protocol["protocol_id"],
        "role": "calibration",
        "images": 571,
        "purpose": "freeze external fixed-byte comparator per budget only",
        "frozen_comparators": frozen,
        "model_training": False,
        "registered_final_accessed": False,
        "rows": rows,
    }
    LEGACY.atomic_json(output / "report.json", report)
    LEGACY.atomic_json(
        completion_path,
        {
            "complete": True,
            "passed": True,
            "selection_only": True,
            "report_sha256": file_sha256(output / "report.json"),
            "results_sha256": file_sha256(output / "results.csv"),
            "selections_sha256": file_sha256(output / "selections.json"),
            "comparators_sha256": file_sha256(output / "frozen_comparators.json"),
            "registered_final_accessed": False,
        },
    )
    return report


def run_final() -> dict:
    protocol = _protocol()
    completion_path = RUNS / "registered_final/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "registered_final/report.json")
    calibration = read_json(RUNS / "audit/completion.json")
    if calibration.get("complete") is not True or calibration.get("selection_only") is not True:
        raise RuntimeError("Calibration comparator freeze must complete before val2017")
    access = RUNS / "registered_final/accessed.json"
    if not access.is_file():
        manifest = read_json(RUNS / "prepare/split_manifest.json")
        LEGACY.atomic_json(
            access,
            {
                "accessed": True,
                "role": "canonical val2017 final",
                "images": 5000,
                "protocol_sha256": file_sha256(PROTOCOL_PATH),
                "fit_images": 9678,
                "calibration_images": 571,
                "image_ids_sha256": manifest["role_ids_sha256"]["registered_final"],
                "comparators_sha256": file_sha256(RUNS / "audit/frozen_comparators.json"),
                "model_or_threshold_selection_on_final": False,
            },
        )
    return LEGACY.run_final()


def run_all() -> dict:
    reports = {}
    for stage in ("prepare", "rewards", "train", "calibration", "final"):
        print(f"[stage] {stage}", flush=True)
        reports[stage] = globals()[f"run_{stage}"]()
    LEGACY.atomic_json(
        RUNS / "COMPLETED.json",
        {
            "complete": True,
            "protocol_id": _protocol()["protocol_id"],
            "fit_images": 9678,
            "calibration_images": 571,
            "final_images": 5000,
            "temporary_5000_checkpoint_reused": False,
        },
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "rewards", "train", "calibration", "final", "all"),
        required=True,
    )
    args = parser.parse_args()
    report = globals()[f"run_{args.stage}"]()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
