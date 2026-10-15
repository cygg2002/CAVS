#!/usr/bin/env python3
"""Run the unified COCO/Oregon CAVS-OQR paper benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
TRAINING_ROLES = ("fit",)
POINT_ROLES = ("fit_oof", "calibration", "final")
SCALAR_ROLES = ("calibration", "final")
sys.dont_write_bytecode = True


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_import_path(path: Path) -> None:
    value = str(Path(path))
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


CORE = load_module("unified_cavs_contract", ROOT / "scripts/core.py")
CORE.validate_protocol(PROTOCOL)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def is_complete(path: Path) -> bool:
    return Path(path).is_file() and read_json(path).get("complete") is True


class DatasetAdapter:
    """Read-only adapter around one detector-specific completed experiment."""

    def __init__(self, dataset: str, *, load_modules: bool = True) -> None:
        if dataset not in PROTOCOL["datasets"]:
            raise ValueError(dataset)
        self.dataset = str(dataset)
        self.config = PROTOCOL["datasets"][dataset]
        self.cavs_root = REPO / self.config["cavs_source"]
        self.score_root = (
            REPO / self.config["score_source"] if dataset == "coco" else None
        )
        self.cloud_root = (
            REPO / self.config["cloud_source"] if dataset == "oregon" else None
        )
        self.dcsb_root = REPO / self.config["dcsb_source"]
        self.cavs_protocol = read_json(self.cavs_root / "configs/protocol.json")
        self.external_root = (
            REPO
            / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines"
        )
        self.cavs = None
        self.cloud = None
        self.external = None
        self.bootstrap_core = None
        self.selector = None
        if load_modules:
            self._load_modules()

    def _load_modules(self) -> None:
        self.cavs = load_module(
            f"unified_{self.dataset}_cavs_source",
            self.cavs_root / "scripts/run_experiment.py",
        )
        self.cloud = self.cavs if self.dataset == "coco" else self.cavs.SOURCE_EVAL
        ensure_import_path(self.external_root / "scripts")
        self.external = load_module(
            f"unified_{self.dataset}_external_source",
            self.external_root / "scripts/run_dataset.py",
        )
        self.bootstrap_core = load_module(
            f"unified_{self.dataset}_bootstrap_core",
            REPO / "paper/experiments/20260804_coco_dualhead_canonical9678_val2017_paper_v1/scripts/canonical_core.py",
        )
        self.selector = load_module(
            f"unified_{self.dataset}_capacity_selector",
            REPO / "paper/experiments/GAIR_v5/gair/selector.py",
        )

    @property
    def roles(self) -> Mapping[str, int]:
        return self.config["roles"]

    @property
    def seeds(self) -> list[int]:
        return list(map(int, self.config["model_seeds"]))

    def load_role(self, role: str) -> dict[str, Any]:
        if role not in self.roles:
            raise ValueError(role)
        if self.cavs is None:
            raise RuntimeError("heavy evaluator modules were not loaded")
        return self.cavs.load_role(role) if self.dataset == "coco" else self.cloud.load_role(role)

    def common_root(self, role: str) -> Path:
        if self.dataset == "coco":
            return self.cavs_root / f"runs/common/coco/{role}"
        return self.cloud_root / f"runs/common/oregon/{role}"

    def load_samples(self, role: str) -> list[dict[str, Any]]:
        if role not in self.roles:
            raise ValueError(role)
        values = read_json(self.common_root(role) / "samples.json")
        return list(values["samples"] if isinstance(values, dict) else values)

    def lightweight_actions(self, role: str) -> set[str]:
        samples = self.load_samples(role)
        return {
            action
            for sample in samples
            for action in PROTOCOL["action_spaces"]["OQR"]
            if action == "edge" or bool(sample.get("available", {}).get(action, False))
        }

    def records(self, role: str) -> list[dict[str, Any]]:
        if self.dataset == "coco":
            return self.cavs._filtered(
                self.cavs.load_role(role)["records"], PROTOCOL["action_spaces"]["OQR"]
            )
        return self.cavs._filtered(self.cavs.SOURCE_TRAIN.load_role(role)["records"])

    def source_score_path(self, seed: int, fold: int | str) -> Path:
        token = "full" if fold == "full" else f"fold_{int(fold)}"
        if self.dataset == "coco":
            method = self.cavs_protocol["methods"]["single"].replace("/", "_")
            return self.cavs_root / f"runs/training/jobs/{method}/seed_{seed}/{token}/scores.npz"
        return self.cavs_root / f"runs/training/jobs/seed_{seed}/{token}/scores.npz"

    def source_checkpoint_path(self, seed: int, fold: int | str) -> Path:
        return self.source_score_path(seed, fold).with_name("artifact.pkl")

    def common_manifest_path(self, role: str) -> Path:
        if self.dataset == "coco":
            return self.cavs_root / f"runs/common/coco/{role}/source_manifest.json"
        return self.cloud_root / f"runs/common/oregon/{role}/source_manifest.json"

    def gt_path(self, role: str) -> Path:
        return Path(read_json(self.common_manifest_path(role))["gt"])

    def evaluator(self, role: str) -> Any:
        return self.load_role(role)["evaluator"]

    def mixed_predictions(
        self, role: str, selection: Mapping[int, str]
    ) -> dict[int, list[dict[str, Any]]]:
        predictions, _ = self.cloud.CANONICAL.mixed_predictions(
            self.load_role(role), selection, 0.001
        )
        return predictions


def _required_sources(adapter: DatasetAdapter) -> list[Path]:
    required = [
        ROOT / "configs/protocol.json",
        ROOT / "scripts/core.py",
        SCRIPT,
        adapter.cavs_root / "configs/protocol.json",
        adapter.cavs_root / "runs/training/completion.json",
        adapter.cavs_root / "runs/training/report.json",
    ]
    for role in adapter.roles:
        required.append(adapter.common_manifest_path(role))
    for seed in adapter.seeds:
        for fold in (*range(int(PROTOCOL["fold_count"])), "full"):
            required.extend(
                (
                    adapter.source_score_path(seed, fold),
                    adapter.source_checkpoint_path(seed, fold),
                    adapter.source_score_path(seed, fold).with_name("report.json"),
                )
            )
    if adapter.dataset == "coco":
        required.extend(
            (
                adapter.score_root / "runs/scores/frozen_scores.npz",
                adapter.score_root / "runs/scores/report.json",
                adapter.dcsb_root / "runs/coco/rewards/dcsb_model.json",
            )
        )
    else:
        required.extend(
            (
                adapter.cloud_root / "runs/COMPLETED.json",
                adapter.cloud_root / "runs/learners/completion.json",
                adapter.cloud_root / "runs/historical/artifact.pkl",
                adapter.cloud_root / "runs/evaluation/point/completion.json",
                adapter.dcsb_root / "runs/dcsb_oregon/model.json",
                adapter.dcsb_root / "runs/dcsb_oregon/fit_manifest.json",
            )
        )
    common_learner_root = REPO / "paper/experiments/20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1"
    required.extend(
        (
            common_learner_root / "scripts/training_adapter.py",
            common_learner_root / "scripts/train_core.py",
            common_learner_root / "scripts/train_learners.py",
            REPO / "paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1/scripts/training.py",
            REPO / "paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1/scripts/core.py",
        )
    )
    return required


def _score_ids(path: Path, key: str) -> tuple[list[int], np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return (
            list(map(int, artifact[f"image_ids_{key}"])),
            np.asarray(artifact[f"scores_{key}"], dtype=np.float64),
        )


def run_preflight(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "preflight"
    completion = output / "completion.json"
    adapter = DatasetAdapter(dataset, load_modules=False)
    if is_complete(completion):
        previous = read_json(output / "report.json")
        current_external = {
            str(path.resolve()): file_sha256(path)
            for path in _required_sources(adapter)
            if ROOT not in path.parents
        }
        previous_external = {
            path: digest
            for path, digest in previous["source_sha256"].items()
            if str(ROOT.resolve()) not in path
        }
        if current_external != previous_external:
            raise RuntimeError(f"frozen external sources changed after preflight: {dataset}")
        current_own = {
            str(path.resolve()): file_sha256(path)
            for path in _required_sources(adapter)
            if ROOT in path.parents
        }
        previous_own = {
            path: digest
            for path, digest in previous["source_sha256"].items()
            if str(ROOT.resolve()) in path
        }
        if current_own == previous_own:
            return previous
    required = _required_sources(adapter)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen sources:\n" + "\n".join(missing[:20]))
    training = read_json(adapter.cavs_root / "runs/training/report.json")
    if training.get("old_learner_checkpoints_reused") is True:
        raise RuntimeError("old learner checkpoints were reused")
    if training.get("final_used_for_training_or_selection") is not False:
        raise RuntimeError("training provenance does not protect final")
    roles = {role: adapter.load_samples(role) for role in adapter.roles}
    counts = {role: len(samples) for role, samples in roles.items()}
    if counts != adapter.roles:
        raise RuntimeError(f"role population drift: {counts}")
    identities = {
        role: {str(sample["identity"]) for sample in samples}
        for role, samples in roles.items()
    }
    overlap = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(identities)
        for right in list(identities)[index + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError(f"identity leakage: {overlap}")
    current_available = {}
    for role, samples in roles.items():
        manifest = read_json(adapter.common_manifest_path(role))
        if manifest.get("cloud_model_sha256") != adapter.config["cloud_model_sha256"]:
            raise RuntimeError(f"wrong detector cache: {dataset}/{role}")
        if manifest.get("primary_roi_fusion") != "Reversible":
            raise RuntimeError(f"wrong ROI fusion: {dataset}/{role}")
        if manifest.get("reversible_parameters") != PROTOCOL["reversible_fusion"]:
            raise RuntimeError(f"Reversible parameters drift: {dataset}/{role}")
        current_available[role] = sum(
            bool(sample.get("available", {}).get("current@640", False))
            for sample in samples
        )
        if not current_available[role]:
            raise RuntimeError(f"current@640 is absent: {dataset}/{role}")
        expected_actions = set(PROTOCOL["action_spaces"]["OQR"])
        observed = adapter.lightweight_actions(role)
        if observed != expected_actions:
            raise RuntimeError(f"OQR record drift: {dataset}/{role}/{sorted(observed)}")
    fit_ids = [int(sample["image_id"]) for sample in roles["fit"]]
    fold_sizes: dict[str, list[int]] = {}
    checkpoint_hashes: dict[str, str] = {}
    for seed in adapter.seeds:
        observed: list[int] = []
        sizes = []
        seed_folds: list[list[int]] = []
        for fold in range(int(PROTOCOL["fold_count"])):
            ids, values = _score_ids(adapter.source_score_path(seed, fold), "fit_oof")
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite OOF scores: {dataset}/{seed}/{fold}")
            observed.extend(ids)
            sizes.append(len(ids))
            seed_folds.append(ids)
            checkpoint_hashes[f"seed={seed}/fold={fold}"] = file_sha256(
                adapter.source_checkpoint_path(seed, fold)
            )
        if len(observed) != len(set(observed)) or set(observed) != set(fit_ids):
            raise RuntimeError(f"OOF population changed: {dataset}/{seed}")
        for role in ("calibration", "final"):
            ids, values = _score_ids(adapter.source_score_path(seed, "full"), role)
            expected = [int(sample["image_id"]) for sample in roles[role]]
            if ids != expected or not np.isfinite(values).all():
                raise RuntimeError(f"full-fit score alignment changed: {dataset}/{seed}/{role}")
        checkpoint_hashes[f"seed={seed}/full"] = file_sha256(
            adapter.source_checkpoint_path(seed, "full")
        )
        fold_sizes[str(seed)] = sizes
        if dataset == "oregon":
            group_by_id = {
                int(sample["image_id"]): str(sample["event_id"])
                for sample in roles["fit"]
            }
            CORE.validate_grouped_folds(fit_ids, seed_folds, group_by_id)
    source_hashes = {str(path.resolve()): file_sha256(path) for path in required}
    source_digest = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "complete": True,
        "dataset": dataset,
        "role_counts": counts,
        "identity_overlap": overlap,
        "current640_available": current_available,
        "model_seeds": adapter.seeds,
        "fold_sizes": fold_sizes,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": source_hashes,
        "source_digest": source_digest,
        "primary_roi_fusion": "Reversible",
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
        "source_experiments_modified": False,
        "common_learner_implementation": "CurrentActionValue via the same frozen training_adapter/AFPD implementation",
        "common_reward_definition": "fold-contained contextual AP50-95, context_size=32",
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "source_digest": source_digest})
    return report


def verify_preflight(dataset: str, adapter: DatasetAdapter) -> dict[str, Any]:
    report = read_json(RUNS / dataset / "preflight/report.json")
    current = {
        str(path.resolve()): file_sha256(path) for path in _required_sources(adapter)
    }
    if current != report["source_sha256"]:
        changed = sorted(
            set(current) ^ set(report["source_sha256"])
            | {
                path
                for path in set(current) & set(report["source_sha256"])
                if current[path] != report["source_sha256"][path]
            }
        )
        raise RuntimeError(f"frozen sources changed after preflight: {changed[:10]}")
    return report


def _load_seed_oof(adapter: DatasetAdapter, seed: int) -> np.ndarray:
    expected = [int(sample["image_id"]) for sample in adapter.load_role("fit")["samples"]]
    folds = []
    for fold in range(int(PROTOCOL["fold_count"])):
        ids, values = _score_ids(adapter.source_score_path(seed, fold), "fit_oof")
        folds.append((ids, values))
    return CORE.assemble_oof_scores(expected, folds)


def _load_seed_full(adapter: DatasetAdapter, seed: int, role: str) -> np.ndarray:
    expected = [int(sample["image_id"]) for sample in adapter.load_role(role)["samples"]]
    ids, values = _score_ids(adapter.source_score_path(seed, "full"), role)
    if ids != expected or values.ndim != 3 or values.shape[1] != len(expected):
        raise RuntimeError(f"full-fit score alignment changed: {adapter.dataset}/{seed}/{role}")
    return values


def cavs_scores(adapter: DatasetAdapter) -> dict[str, np.ndarray]:
    return {
        "fit_oof": np.mean([_load_seed_oof(adapter, seed) for seed in adapter.seeds], axis=0),
        "calibration": np.mean([_load_seed_full(adapter, seed, "calibration") for seed in adapter.seeds], axis=0),
        "final": np.mean([_load_seed_full(adapter, seed, "final") for seed in adapter.seeds], axis=0),
    }


def _role_for_point(role: str) -> str:
    return "fit" if role == "fit_oof" else role


def joint_selection(
    adapter: DatasetAdapter,
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    if mode == "fixed_call":
        return CORE.fixed_call_joint(samples, scores, budget=float(budget))
    if mode != "fixed_byte":
        raise ValueError(mode)
    groups = []
    for sample in samples:
        image_id = int(sample["image_id"])
        actions = [{"image_id": image_id, "action_id": "edge", "value": 0.0, "cost": 0.0}]
        for action, value in scores[image_id].items():
            if action not in sample["payload_bytes"] or not bool(sample.get("available", {}).get(action, True)):
                continue
            actions.append(
                {
                    "image_id": image_id,
                    "action_id": str(action),
                    "value": float(value),
                    "cost": float(sample["payload_bytes"][action]),
                }
            )
        if len(actions) == 1:
            raise RuntimeError(f"no available cloud action: {image_id}")
        groups.append(actions)
    capacity = float(budget) * sum(float(sample["payload_bytes"]["original"]) for sample in samples)
    result = adapter.selector.select_actions_for_budget(groups, capacity)
    selection = {
        int(row["image_id"]): str(row["action_id"])
        for row in result["selected"]
        if str(row["action_id"]) != "edge"
    }
    CORE.validate_selection(samples, selection, mode=mode, budget=float(budget))
    return selection


def _shared_features(samples: Sequence[Mapping[str, Any]], *, roi: bool) -> np.ndarray:
    rows = []
    for sample in samples:
        values = list(map(float, sample["action_features"]))
        if roi:
            original = max(1.0, float(sample["payload_bytes"]["original"]))
            values.extend(
                (
                    float(sample.get("roi_area_ratio", 0.0)),
                    float(sample["payload_bytes"].get("current@640", original)) / original,
                )
            )
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError("shared-action feature matrix changed")
    return matrix


def _evaluate_map(adapter: DatasetAdapter, role: str, action_map: Mapping[int, str]) -> tuple[float, int]:
    predictions = adapter.mixed_predictions(role, action_map)
    metric = adapter.evaluator(role).evaluate(predictions)
    samples = adapter.load_role(role)["samples"]
    by_id = {int(sample["image_id"]): sample for sample in samples}
    requested = sum(float(by_id[image_id]["payload_bytes"][action]) for image_id, action in action_map.items())
    return float(metric["map50_95"]), int(requested)


def train_shared_action(adapter: DatasetAdapter) -> dict[str, Any]:
    output = RUNS / adapter.dataset / "shared_action"
    completion = output / "completion.json"
    artifact_path = output / "artifact.pkl"
    if is_complete(completion):
        artifact = load_pickle(artifact_path)
        if read_json(completion)["artifact_sha256"] != file_sha256(artifact_path):
            raise RuntimeError(f"shared-action artifact drift: {adapter.dataset}")
        return artifact
    if artifact_path.is_file():
        artifact = load_pickle(artifact_path)
        artifact_hash = file_sha256(artifact_path)
        q90_choice = dict(artifact["q90_choice"])
        roi_choice = dict(artifact["roi_choice"])
        q90_choice["threshold"] = CORE.json_threshold(q90_choice["threshold"])
        roi_choice["threshold"] = CORE.json_threshold(roi_choice["threshold"])
        atomic_json(
            output / "report.json",
            {
                "complete": True,
                "q90_choice": q90_choice,
                "roi_choice": roi_choice,
                "artifact_sha256": artifact_hash,
                "fit_only_training": True,
                "final_used_for_training_or_selection": False,
                "recovered_after_report_serialization_interrupt": True,
            },
        )
        atomic_json(completion, {"complete": True, "artifact_sha256": artifact_hash})
        return artifact
    benchmark_core = load_module(
        f"unified_{adapter.dataset}_benchmark_core",
        REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/benchmark_core.py",
    )
    fit = adapter.load_role("fit")
    samples = fit["samples"]
    epsilon = float(PROTOCOL["shared_action"]["epsilon"])
    q90_labels = np.asarray(
        [
            bool(sample["available"].get("q90", False))
            and float(sample["qualities"]["q90"]) >= float(sample["qualities"]["original"]) - epsilon
            for sample in samples
        ],
        dtype=np.float64,
    )
    roi_indices = [
        index for index, sample in enumerate(samples)
        if bool(sample["available"].get("current@640", False))
    ]
    roi_labels = np.asarray(
        [
            float(samples[index]["qualities"]["current@640"])
            >= max(float(samples[index]["qualities"]["original"]), float(samples[index]["qualities"]["q90"])) - epsilon
            for index in roi_indices
        ],
        dtype=np.float64,
    )
    q90_model = benchmark_core.fit_weighted_logistic(_shared_features(samples, roi=False), q90_labels)
    roi_features = _shared_features(samples, roi=True)[roi_indices]
    roi_model = benchmark_core.fit_weighted_logistic(roi_features, roi_labels)
    calibration = adapter.load_role("calibration")
    cal_samples = calibration["samples"]
    q90_probabilities = benchmark_core.predict_weighted_logistic(
        q90_model, _shared_features(cal_samples, roi=False)
    )
    roi_all = benchmark_core.predict_weighted_logistic(
        roi_model, _shared_features(cal_samples, roi=True)
    )
    roi_probabilities = {
        int(sample["image_id"]): float(value)
        for sample, value in zip(cal_samples, roi_all)
        if bool(sample["available"].get("current@640", False))
    }
    fractions = list(map(float, PROTOCOL["shared_action"]["action_fractions"]))
    q90_eligible = np.asarray([bool(sample["available"].get("q90", False)) for sample in cal_samples])
    q90_thresholds = sorted(
        {
            benchmark_core.threshold_for_fraction(q90_probabilities, q90_eligible, fraction)
            for fraction in fractions
        },
        reverse=True,
    )
    original_map = {int(sample["image_id"]): "original" for sample in cal_samples}
    original_ap, _ = _evaluate_map(adapter, "calibration", original_map)
    q90_candidates = []
    for threshold in q90_thresholds:
        action_map = CORE.shared_action_map(
            cal_samples,
            q90_probabilities=q90_probabilities,
            roi_probabilities={},
            q90_threshold=float(threshold),
            roi_threshold=float("inf"),
        )
        ap, requested = _evaluate_map(adapter, "calibration", action_map)
        q90_candidates.append({"threshold": float(threshold), "ap": ap, "request_bytes": requested})
    maximum_loss = float(PROTOCOL["shared_action"]["maximum_calibration_ap_loss"])
    q90_safe = [row for row in q90_candidates if row["ap"] >= original_ap - maximum_loss]
    q90_choice = min(q90_safe, key=lambda row: (row["request_bytes"], -row["ap"], -row["threshold"]))
    q90_threshold = float(q90_choice["threshold"])
    base_map = CORE.shared_action_map(
        cal_samples,
        q90_probabilities=q90_probabilities,
        roi_probabilities={},
        q90_threshold=q90_threshold,
        roi_threshold=float("inf"),
    )
    base_ap, _ = _evaluate_map(adapter, "calibration", base_map)
    available_roi_values = np.asarray(list(roi_probabilities.values()), dtype=np.float64)
    roi_thresholds = sorted(
        {
            benchmark_core.threshold_for_fraction(
                available_roi_values, np.ones(len(available_roi_values), dtype=bool), fraction
            )
            for fraction in fractions
        },
        reverse=True,
    )
    roi_candidates = []
    for threshold in roi_thresholds:
        action_map = CORE.shared_action_map(
            cal_samples,
            q90_probabilities=q90_probabilities,
            roi_probabilities=roi_probabilities,
            q90_threshold=q90_threshold,
            roi_threshold=float(threshold),
        )
        ap, requested = _evaluate_map(adapter, "calibration", action_map)
        roi_candidates.append(
            {
                "threshold": float(threshold),
                "ap": ap,
                "request_bytes": requested,
                "roi_count": sum(action == "current@640" for action in action_map.values()),
            }
        )
    roi_safe = [row for row in roi_candidates if row["ap"] >= base_ap - maximum_loss]
    roi_choice = min(roi_safe, key=lambda row: (row["request_bytes"], -row["ap"], -row["threshold"]))
    artifact = {
        "q90_model": q90_model,
        "roi_model": roi_model,
        "q90_threshold": float(q90_choice["threshold"]),
        "roi_threshold": float(roi_choice["threshold"]),
        "q90_choice": q90_choice,
        "roi_choice": roi_choice,
        "q90_candidates": q90_candidates,
        "roi_candidates": roi_candidates,
        "fit_q90_positive_rate": float(q90_labels.mean()),
        "fit_roi_positive_rate": float(roi_labels.mean()),
        "fit_only_training": True,
        "calibration_only_thresholds": True,
        "final_used_for_training_or_selection": False,
    }
    atomic_pickle(artifact_path, artifact)
    artifact_hash = file_sha256(artifact_path)
    report_q90_choice = dict(q90_choice)
    report_roi_choice = dict(roi_choice)
    report_q90_choice["threshold"] = CORE.json_threshold(report_q90_choice["threshold"])
    report_roi_choice["threshold"] = CORE.json_threshold(report_roi_choice["threshold"])
    atomic_json(
        output / "report.json",
        {
            "complete": True,
            "q90_choice": report_q90_choice,
            "roi_choice": report_roi_choice,
            "artifact_sha256": artifact_hash,
            "fit_only_training": True,
            "final_used_for_training_or_selection": False,
        },
    )
    atomic_json(completion, {"complete": True, "artifact_sha256": artifact_hash})
    return artifact


def shared_action_maps(adapter: DatasetAdapter, artifact: Mapping[str, Any]) -> dict[str, dict[int, str]]:
    benchmark_core = load_module(
        f"unified_{adapter.dataset}_benchmark_predict",
        REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/benchmark_core.py",
    )
    output = {}
    for role in SCALAR_ROLES:
        samples = adapter.load_role(role)["samples"]
        q90 = benchmark_core.predict_weighted_logistic(artifact["q90_model"], _shared_features(samples, roi=False))
        roi_values = benchmark_core.predict_weighted_logistic(artifact["roi_model"], _shared_features(samples, roi=True))
        roi = {
            int(sample["image_id"]): float(value)
            for sample, value in zip(samples, roi_values)
            if bool(sample["available"].get("current@640", False))
        }
        output[role] = CORE.shared_action_map(
            samples,
            q90_probabilities=q90,
            roi_probabilities=roi,
            q90_threshold=float(artifact["q90_threshold"]),
            roi_threshold=float(artifact["roi_threshold"]),
        )
    return output


def scalar_scores(adapter: DatasetAdapter) -> dict[str, dict[str, np.ndarray]]:
    methods = list(map(str, PROTOCOL["scalar_methods"]))
    output = {role: {} for role in SCALAR_ROLES}
    if adapter.dataset == "coco":
        path = adapter.score_root / "runs/scores/frozen_scores.npz"
        with np.load(path, allow_pickle=False) as source:
            source_methods = list(map(str, source["methods"]))
            for role in SCALAR_ROLES:
                expected = [int(sample["image_id"]) for sample in adapter.load_role(role)["samples"]]
                ids = list(map(int, source[f"{role}_ids"]))
                if ids != expected:
                    raise RuntimeError(f"COCO scalar score alignment changed: {role}")
                for method in methods:
                    if method == "DCSB-Score-Adapted":
                        continue
                    matrix = np.asarray(source[f"{role}_scores"][source_methods.index(method)], dtype=np.float64)
                    output[role][method] = matrix
        dcsb_model = read_json(adapter.dcsb_root / "runs/coco/rewards/dcsb_model.json")
    else:
        for role in SCALAR_ROLES:
            all_scores = adapter.cloud.scalar_scores(role)
            for method in methods:
                if method in {"Random", "DCSB-Score-Adapted"}:
                    continue
                output[role][method] = np.asarray(all_scores[method], dtype=np.float64)
        dcsb_model = read_json(adapter.dcsb_root / "runs/dcsb_oregon/model.json")
    for role in SCALAR_ROLES:
        data = adapter.load_role(role)
        dcsb = []
        for sample in data["samples"]:
            image_id = int(sample["image_id"])
            _, score = adapter.external.dcsb_decision_and_score(
                data["actions"]["edge"].get(image_id, []),
                int(sample["width"]),
                int(sample["height"]),
                dcsb_model,
            )
            dcsb.append(float(score))
        output[role]["DCSB-Score-Adapted"] = np.repeat(
            np.asarray(dcsb, dtype=np.float64)[None, :], len(PROTOCOL["budgets"]), axis=0
        )
        output[role].pop("Random", None)
        expected = set(methods) - {"Random"}
        if set(output[role]) != expected:
            raise RuntimeError(f"scalar method matrix changed: {adapter.dataset}/{role}/{sorted(output[role])}")
        for method, matrix in output[role].items():
            array = np.asarray(matrix, dtype=np.float64)
            if array.ndim == 1:
                array = np.repeat(array[None, :], len(PROTOCOL["budgets"]), axis=0)
            if array.shape != (len(PROTOCOL["budgets"]), len(adapter.load_role(role)["samples"])):
                raise RuntimeError(f"scalar score shape changed: {adapter.dataset}/{role}/{method}/{array.shape}")
            output[role][method] = array
    return output


def scalar_fixed_call(
    samples: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    action_map: Mapping[int, str],
    *,
    budget: float,
) -> dict[int, str]:
    if len(samples) != len(scores):
        raise ValueError("scalar fixed-call score alignment changed")
    count = int(math.floor(len(samples) * float(budget) + 0.5))
    order = sorted(
        range(len(samples)),
        key=lambda index: (-float(scores[index]), int(samples[index]["image_id"])),
    )[:count]
    selected = {
        int(samples[index]["image_id"]): str(action_map[int(samples[index]["image_id"])])
        for index in order
    }
    CORE.validate_selection(samples, selected, mode="fixed_call", budget=float(budget))
    return selected


def scalar_fixed_byte(
    adapter: DatasetAdapter,
    samples: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    action_map: Mapping[int, str],
    *,
    budget: float,
    reference_scores: Sequence[float],
) -> dict[int, str]:
    percentiles = CORE.empirical_percentile(reference_scores, scores)
    score_map = {
        int(sample["image_id"]): {
            str(action_map[int(sample["image_id"])]): float(percentiles[index])
        }
        for index, sample in enumerate(samples)
    }
    return joint_selection(
        adapter, samples, score_map, mode="fixed_byte", budget=float(budget)
    )


def evaluate_selection(
    adapter: DatasetAdapter,
    role: str,
    *,
    method: str,
    policy: str,
    mode: str,
    budget: float,
    selection: Mapping[int, str],
    seed: int | str = "ensemble",
) -> dict[str, Any]:
    data = adapter.load_role(role)
    predictions = adapter.mixed_predictions(role, selection)
    metric = data["evaluator"].evaluate(predictions)
    validation = CORE.validate_selection(data["samples"], selection, mode=mode, budget=float(budget))
    counts = Counter(selection.values())
    return {
        "dataset": adapter.dataset,
        "role": role,
        "method": str(method),
        "policy": str(policy),
        "mode": str(mode),
        "budget": float(budget),
        "seed": seed,
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": validation["request_bytes"] / 1024.0 / len(data["samples"]),
        "request_bytes": int(validation["request_bytes"]),
        "capacity_utilization": float(validation["utilization"]),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["current@640"]),
        "primary_roi_fusion": "Reversible",
        "cloud_model_sha256": adapter.config["cloud_model_sha256"],
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }


_POINT_STATE: dict[str, Any] = {}


def _evaluate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    adapter = _POINT_STATE["adapter"]
    return evaluate_selection(
        adapter,
        str(task["role"]),
        method=str(task["method"]),
        policy=str(task["policy"]),
        mode=str(task["mode"]),
        budget=float(task["budget"]),
        selection={int(key): str(value) for key, value in task["selection"].items()},
        seed=task.get("seed", "ensemble"),
    )


def aggregate_random(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["role"]),
            str(row["policy"]),
            str(row["mode"]),
            float(row["budget"]),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for (_, _, _, _), values in sorted(grouped.items()):
        item = dict(values[0])
        for key in (
            "official_ap50_95", "official_ap50", "official_ap75", "kb_per_image",
            "request_bytes", "capacity_utilization", "selected_images", "original_count",
            "q90_count", "roi_count",
        ):
            item[key] = float(np.mean([float(value[key]) for value in values]))
        item["seed"] = "mean50"
        item["random_seed_count"] = len(values)
        output.append(item)
    return output


def audit_point_rows(
    dataset: str,
    aggregate_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    CORE.validate_unique_rows(aggregate_rows)
    expected_aggregate = CORE.expected_cavs_rows(PROTOCOL, roles=3) + CORE.expected_scalar_rows(
        PROTOCOL, roles=2
    )
    expected_random = (
        len(SCALAR_ROLES)
        * len(PROTOCOL["modes"])
        * len(PROTOCOL["budgets"])
        * int(PROTOCOL["random"]["seeds"])
    )
    if len(aggregate_rows) != expected_aggregate:
        raise RuntimeError(
            f"point matrix incomplete: {dataset}/{len(aggregate_rows)} != {expected_aggregate}"
        )
    if len(random_rows) != expected_random:
        raise RuntimeError(
            f"random seed matrix incomplete: {dataset}/{len(random_rows)} != {expected_random}"
        )
    expected_keys = {
        (role, method, f"Joint-{method.split('-')[-1]}", mode, float(budget))
        for role in POINT_ROLES
        for method in PROTOCOL["cavs_methods"]
        for mode in PROTOCOL["modes"]
        for budget in PROTOCOL["budgets"]
    } | {
        (role, method, "SharedAction-OQR", mode, float(budget))
        for role in SCALAR_ROLES
        for method in PROTOCOL["scalar_methods"]
        for mode in PROTOCOL["modes"]
        for budget in PROTOCOL["budgets"]
    }
    observed_keys = {
        (
            str(row["role"]),
            str(row["method"]),
            str(row["policy"]),
            str(row["mode"]),
            float(row["budget"]),
        )
        for row in aggregate_rows
    }
    if observed_keys != expected_keys:
        raise RuntimeError(
            f"point key set changed: {dataset}/missing={len(expected_keys - observed_keys)}"
            f"/extra={len(observed_keys - expected_keys)}"
        )
    invalid = [
        row
        for row in aggregate_rows
        if bool(row.get("final_used_for_training_or_selection"))
        or str(row.get("primary_roi_fusion")) != "Reversible"
        or not (0.0 <= float(row.get("capacity_utilization", 0.0)) <= 1.0 + 1e-9)
    ]
    if invalid:
        raise RuntimeError(f"invalid point provenance or capacity: {dataset}/{len(invalid)}")
    return {
        "aggregate_rows": len(aggregate_rows),
        "random_seed_rows": len(random_rows),
        "expected_aggregate_rows": expected_aggregate,
        "expected_random_seed_rows": expected_random,
        "complete": True,
    }


def evaluate_references(adapter: DatasetAdapter) -> dict[str, Any]:
    output = RUNS / adapter.dataset / "references"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    rows = []
    for role in ("calibration", "final"):
        samples = adapter.load_role(role)["samples"]
        for method, selection in CORE.reference_action_maps(samples).items():
            predictions = adapter.mixed_predictions(role, selection)
            metric = adapter.evaluator(role).evaluate(predictions)
            counts = Counter(selection.values())
            request_bytes = sum(
                float(sample["payload_bytes"][selection[int(sample["image_id"])]])
                for sample in samples
                if int(sample["image_id"]) in selection
            )
            rows.append(
                {
                    "dataset": adapter.dataset,
                    "role": role,
                    "method": method,
                    "official_ap50_95": float(metric["map50_95"]),
                    "official_ap50": float(metric["map50"]),
                    "official_ap75": float(metric["map75"]),
                    "kb_per_image": request_bytes / 1024.0 / len(samples),
                    "request_bytes": int(request_bytes),
                    "selected_images": len(selection),
                    "original_count": int(counts["original"]),
                    "q90_count": int(counts["q90"]),
                    "roi_count": 0,
                    "cloud_model_sha256": adapter.config["cloud_model_sha256"],
                    "final_is_hidden": False,
                    "final_used_for_training_or_selection": False,
                }
            )
    if len(rows) != 6:
        raise RuntimeError(f"reference matrix incomplete: {adapter.dataset}/{len(rows)}")
    atomic_csv(output / "results.csv", rows)
    report = {
        "complete": True,
        "dataset": adapter.dataset,
        "rows": len(rows),
        "results_sha256": file_sha256(output / "results.csv"),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"]})
    return report


def run_point(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "point"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_preflight(dataset)
    adapter = DatasetAdapter(dataset)
    verify_preflight(dataset, adapter)
    cavs = cavs_scores(adapter)
    shared = train_shared_action(adapter)
    evaluate_references(adapter)
    action_maps = shared_action_maps(adapter, shared)
    scalar = scalar_scores(adapter)
    tasks: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    for point_role in POINT_ROLES:
        role = _role_for_point(point_role)
        data = adapter.load_role(role)
        records = adapter.records(role)
        matrix = cavs[point_role]
        if matrix.shape[1] != len(records):
            raise RuntimeError(f"CAVS score population changed: {dataset}/{point_role}")
        for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
            base_scores = CORE.action_score_map(records, matrix[budget_index])
            for mask_name, method in (("O", "CAVS-O"), ("OQ", "CAVS-OQ"), ("OQR", "CAVS-OQR")):
                masked = CORE.mask_scores(base_scores, PROTOCOL["action_spaces"][mask_name])
                for mode in PROTOCOL["modes"]:
                    selection = joint_selection(adapter, data["samples"], masked, mode=mode, budget=budget)
                    key = f"{point_role}|{method}|Joint-{mask_name}|{mode}|{budget:g}"
                    selections[key] = {str(image_id): action for image_id, action in selection.items()}
                    tasks.append({"role": role, "reported_role": point_role, "method": method, "policy": f"Joint-{mask_name}", "mode": mode, "budget": budget, "selection": selection})
    random_tasks = []
    for role in SCALAR_ROLES:
        data = adapter.load_role(role)
        samples = data["samples"]
        for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
            for method, matrices in scalar[role].items():
                scores = matrices[budget_index]
                for mode in PROTOCOL["modes"]:
                    if mode == "fixed_call":
                        selection = scalar_fixed_call(samples, scores, action_maps[role], budget=budget)
                    else:
                        selection = scalar_fixed_byte(
                            adapter,
                            samples,
                            scores=scores,
                            action_map=action_maps[role],
                            budget=budget,
                            reference_scores=scalar["calibration"][method][budget_index],
                        )
                    key = f"{role}|{method}|SharedAction-OQR|{mode}|{budget:g}"
                    selections[key] = {str(image_id): action for image_id, action in selection.items()}
                    tasks.append({"role": role, "reported_role": role, "method": method, "policy": "SharedAction-OQR", "mode": mode, "budget": budget, "selection": selection})
            for offset in range(int(PROTOCOL["random"]["seeds"])):
                seed = int(PROTOCOL["random"]["base_seed"]) + offset
                rng = np.random.default_rng(seed)
                scores = rng.random(len(samples))
                reference = np.random.default_rng(seed).random(len(adapter.load_role("calibration")["samples"]))
                for mode in PROTOCOL["modes"]:
                    if mode == "fixed_call":
                        selection = scalar_fixed_call(samples, scores, action_maps[role], budget=budget)
                    else:
                        selection = scalar_fixed_byte(
                            adapter,
                            samples,
                            scores=scores,
                            action_map=action_maps[role],
                            budget=budget,
                            reference_scores=reference,
                        )
                    task = {"role": role, "reported_role": role, "method": "Random", "policy": "SharedAction-OQR", "mode": mode, "budget": budget, "selection": selection, "seed": seed}
                    random_tasks.append(task)
                    if offset == 0:
                        key = f"{role}|Random|SharedAction-OQR|{mode}|{budget:g}"
                        selections[key] = {str(image_id): action for image_id, action in selection.items()}
    _POINT_STATE.clear()
    _POINT_STATE["adapter"] = adapter
    workers = min(int(PROTOCOL["cpu"]["point_workers"]), len(tasks), max(1, (os.cpu_count() or 1) - 4))
    with mp.get_context("fork").Pool(workers) as pool:
        rows = list(pool.imap_unordered(_evaluate_task, tasks, chunksize=1))
    random_workers = min(workers, len(random_tasks))
    with mp.get_context("fork").Pool(random_workers) as pool:
        random_rows = list(pool.imap_unordered(_evaluate_task, random_tasks, chunksize=1))
    _POINT_STATE.clear()
    for row, task in zip([], []):
        pass
    # Workers evaluate the real role; restore the OOF reporting label from task keys.
    for row in rows:
        if row["method"].startswith("CAVS") and row["role"] == "fit":
            row["role"] = "fit_oof"
    random_aggregate = aggregate_random(random_rows)
    aggregate_rows = rows + random_aggregate
    point_audit = audit_point_rows(dataset, aggregate_rows, random_rows)
    calibration_rows = [row for row in aggregate_rows if row["role"] == "calibration"]
    strongest = {
        f"{budget:g}": CORE.strongest_external(calibration_rows, budget, PROTOCOL)
        for budget in map(float, PROTOCOL["budgets"])
    }
    atomic_csv(output / "results.csv", aggregate_rows)
    atomic_csv(output / "random_seed_results.csv", random_rows)
    atomic_json(output / "selections.json", selections)
    atomic_json(output / "strongest_external.json", strongest)
    report = {
        "complete": True,
        "dataset": dataset,
        "aggregate_rows": len(aggregate_rows),
        "random_seed_rows": len(random_rows),
        "point_audit": point_audit,
        "shared_action_sha256": file_sha256(RUNS / dataset / "shared_action/artifact.pkl"),
        "strongest_external": strongest,
        "results_sha256": file_sha256(output / "results.csv"),
        "selections_sha256": file_sha256(output / "selections.json"),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"], "selections_sha256": report["selections_sha256"]})
    return report


_BOOT_STATE: dict[str, Any] = {}


def _sample_bootstrap_ids(rng: np.random.Generator, unit: str) -> list[int]:
    if unit == "image":
        values = _BOOT_STATE["image_ids"]
        return [int(values[int(index)]) for index in rng.integers(0, len(values), size=len(values))]
    if unit == "event":
        groups = _BOOT_STATE["event_groups"]
        return [
            int(image_id)
            for index in rng.integers(0, len(groups), size=len(groups))
            for image_id in groups[int(index)]
        ]
    raise ValueError(unit)


def _bootstrap_job(job: Mapping[str, Any]) -> dict[str, Any]:
    safe = hashlib.sha256(str(job["id"]).encode()).hexdigest()[:16]
    path = RUNS / str(job["dataset"]) / f"bootstrap/jobs/{safe}.json"
    partial = path.with_suffix(".partial.json")
    if path.is_file():
        cached = read_json(path)
        if int(cached["replicates"]) != int(PROTOCOL["bootstrap"]["replicates"]):
            raise RuntimeError(f"incomplete bootstrap cache: {path}")
        return cached
    values = list(map(float, read_json(partial)["values"])) if partial.is_file() else []
    token = hashlib.sha256(f"{PROTOCOL['bootstrap']['seed']}:{job['id']}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
    for _ in values:
        _sample_bootstrap_ids(rng, str(job["unit"]))
    candidate, comparator = _BOOT_STATE["prediction_pairs"][int(job["prediction_index"])]
    for replicate in range(len(values), int(PROTOCOL["bootstrap"]["replicates"])):
        sampled = _sample_bootstrap_ids(rng, str(job["unit"]))
        gt, left, right = _BOOT_STATE["core"]._bootstrap_remap(
            _BOOT_STATE["ground_truth"], candidate, comparator, sampled
        )
        values.append(float(_BOOT_STATE["core"]._in_memory_ap(gt, left) - _BOOT_STATE["core"]._in_memory_ap(gt, right)))
        if (replicate + 1) % 100 == 0:
            atomic_json(partial, {"values": values})
    array = np.asarray(values, dtype=np.float64)
    result = {
        **{key: value for key, value in job.items() if key != "prediction_index"},
        "replicates": len(values),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0)),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(path, result)
    partial.unlink(missing_ok=True)
    return result


def run_bootstrap(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "bootstrap"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_point(dataset)
    adapter = DatasetAdapter(dataset)
    rows = read_csv(RUNS / dataset / "point/results.csv")
    stored = read_json(RUNS / dataset / "point/selections.json")
    strongest = read_json(RUNS / dataset / "point/strongest_external.json")
    data = adapter.load_role("final")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    events: dict[str, list[int]] = {}
    for sample in data["samples"]:
        events.setdefault(str(sample.get("event_id", sample["image_id"])), []).append(int(sample["image_id"]))
    units = list(map(str, adapter.config["bootstrap_units"]))

    def selection(key: str) -> dict[int, str]:
        if key not in stored:
            raise RuntimeError(f"missing frozen selection: {key}")
        return {int(image_id): str(action) for image_id, action in stored[key].items()}

    jobs = []
    prediction_pairs = []
    cache: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for budget in map(float, PROTOCOL["budgets"]):
        candidate_key = f"final|CAVS-OQR|Joint-OQR|fixed_byte|{budget:g}"
        comparator_keys = {
            "CAVS-OQ": f"final|CAVS-OQ|Joint-OQ|fixed_byte|{budget:g}",
            "Confidence": f"final|Confidence|SharedAction-OQR|fixed_byte|{budget:g}",
            "DCSB-Score-Adapted": f"final|DCSB-Score-Adapted|SharedAction-OQR|fixed_byte|{budget:g}",
            "Random": f"final|Random|SharedAction-OQR|fixed_byte|{budget:g}",
            "ORIC-AP5095": f"final|ORIC-AP5095|SharedAction-OQR|fixed_byte|{budget:g}",
            "MORIC-AP5095": f"final|MORIC-AP5095|SharedAction-OQR|fixed_byte|{budget:g}",
            "Strongest-External": f"final|{strongest[f'{budget:g}']}|SharedAction-OQR|fixed_byte|{budget:g}",
        }
        pairs = [("CAVS-OQR", candidate_key, name, key) for name, key in comparator_keys.items()]
        pairs.append(("CAVS-OQ", comparator_keys["CAVS-OQ"], "Strongest-External", comparator_keys["Strongest-External"]))
        for candidate_name, left_key, comparator_name, right_key in pairs:
            for key in (left_key, right_key):
                if key not in cache:
                    cache[key] = adapter.mixed_predictions("final", selection(key))
            for unit in units:
                job = {
                    "id": f"{candidate_name}|vs|{comparator_name}|budget={budget:g}|{unit}",
                    "dataset": dataset,
                    "candidate": candidate_name,
                    "comparator": comparator_name,
                    "budget": budget,
                    "unit": unit,
                    "candidate_key": left_key,
                    "comparator_key": right_key,
                    "prediction_index": len(prediction_pairs),
                }
                prediction_pairs.append((cache[left_key], cache[right_key]))
                jobs.append(job)
    if len(jobs) != CORE.expected_bootstrap_jobs(PROTOCOL, dataset):
        raise RuntimeError(f"bootstrap matrix incomplete: {dataset}/{len(jobs)}")
    observed_job_ids = {str(job["id"]) for job in jobs}
    expected_job_ids = CORE.expected_bootstrap_job_ids(PROTOCOL, dataset)
    if observed_job_ids != expected_job_ids:
        raise RuntimeError(
            f"bootstrap job set changed: {dataset}/missing={len(expected_job_ids - observed_job_ids)}"
            f"/extra={len(observed_job_ids - expected_job_ids)}"
        )
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        ground_truth=read_json(adapter.gt_path("final")),
        image_ids=image_ids,
        event_groups=[events[key] for key in sorted(events)],
        prediction_pairs=prediction_pairs,
        core=adapter.bootstrap_core,
    )
    workers = min(int(PROTOCOL["bootstrap"]["workers"]), len(jobs), max(1, (os.cpu_count() or 1) - 4))
    with mp.get_context("fork").Pool(workers) as pool:
        results = list(pool.imap_unordered(_bootstrap_job, jobs, chunksize=1))
    _BOOT_STATE.clear()
    public = sorted(results, key=lambda row: (row["unit"], row["budget"], row["candidate"], row["comparator"]))
    completed_ids = {str(row["id"]) for row in public}
    if completed_ids != expected_job_ids or any(
        int(row["replicates"]) != int(PROTOCOL["bootstrap"]["replicates"])
        for row in public
    ):
        raise RuntimeError(f"bootstrap output matrix incomplete: {dataset}")
    report = {
        "complete": True,
        "dataset": dataset,
        "jobs": len(public),
        "replicates_per_job": int(PROTOCOL["bootstrap"]["replicates"]),
        "comparisons": public,
        "event_bootstrap_primary": dataset == "oregon",
        "image_bootstrap_sensitivity_only": dataset == "oregon",
        "random_comparator_seed": int(PROTOCOL["random"]["base_seed"]),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(public)})
    return report


def run_report() -> dict[str, Any]:
    reports = {dataset: run_bootstrap(dataset) for dataset in PROTOCOL["datasets"]}
    lines = [
        "# COCO / Oregon 统一 CAVS-OQR 论文结果",
        "",
        "本结果为 post-hoc paper benchmark；evaluation role 未参与训练、阈值或外部方法选择。",
    ]
    for dataset in PROTOCOL["datasets"]:
        rows = read_csv(RUNS / dataset / "point/results.csv")
        references = read_csv(RUNS / dataset / "references/results.csv")
        lines.extend(["", f"## {dataset.upper()} Fixed-byte final", "", "| 方法 | 策略 | 10% | 25% | 40% |", "|---|---|---:|---:|---:|"])
        methods = [*PROTOCOL["cavs_methods"], *PROTOCOL["scalar_methods"]]
        for method in methods:
            policy = f"Joint-{method.split('-')[-1]}" if method.startswith("CAVS-") else "SharedAction-OQR"
            cells = []
            for budget in map(float, PROTOCOL["budgets"]):
                found = [row for row in rows if row["role"] == "final" and row["method"] == method and row["policy"] == policy and row["mode"] == "fixed_byte" and math.isclose(float(row["budget"]), budget)]
                if len(found) != 1:
                    raise RuntimeError(f"missing report row: {dataset}/{method}/{budget}")
                cells.append(f"{float(found[0]['official_ap50_95']):.6f} / {float(found[0]['kb_per_image']):.2f}")
            lines.append(f"| {method} | {policy} | " + " | ".join(cells) + " |")
        lines.extend(["", f"## {dataset.upper()} Fixed-call final", "", "| 方法 | 策略 | 10% | 25% | 40% |", "|---|---|---:|---:|---:|"])
        for method in methods:
            policy = f"Joint-{method.split('-')[-1]}" if method.startswith("CAVS-") else "SharedAction-OQR"
            cells = []
            for budget in map(float, PROTOCOL["budgets"]):
                found = [row for row in rows if row["role"] == "final" and row["method"] == method and row["policy"] == policy and row["mode"] == "fixed_call" and math.isclose(float(row["budget"]), budget)]
                if len(found) != 1:
                    raise RuntimeError(f"missing fixed-call report row: {dataset}/{method}/{budget}")
                cells.append(f"{float(found[0]['official_ap50_95']):.6f} / {float(found[0]['kb_per_image']):.2f}")
            lines.append(f"| {method} | {policy} | " + " | ".join(cells) + " |")
        lines.extend(["", f"## {dataset.upper()} detector references", "", "| 参考 | AP50-95 | AP50 | AP75 | KB/图 |", "|---|---:|---:|---:|---:|"])
        for method in ("Edge-only", "Cloud-only Original", "Cloud-only Q90"):
            found = [row for row in references if row["role"] == "final" and row["method"] == method]
            if len(found) != 1:
                raise RuntimeError(f"missing reference row: {dataset}/{method}")
            row = found[0]
            lines.append(f"| {method} | {float(row['official_ap50_95']):.6f} | {float(row['official_ap50']):.6f} | {float(row['official_ap75']):.6f} | {float(row['kb_per_image']):.2f} |")
        lines.extend(["", f"## {dataset.upper()} CAVS action ablation", "", "| 方法 | 预算 | AP50-95 | KB/图 | Original | Q90 | ROI |", "|---|---:|---:|---:|---:|---:|---:|"])
        for method in PROTOCOL["cavs_methods"]:
            policy = f"Joint-{method.split('-')[-1]}"
            for budget in map(float, PROTOCOL["budgets"]):
                found = [row for row in rows if row["role"] == "final" and row["method"] == method and row["policy"] == policy and row["mode"] == "fixed_byte" and math.isclose(float(row["budget"]), budget)]
                row = found[0]
                lines.append(f"| {method} | {budget:.0%} | {float(row['official_ap50_95']):.6f} | {float(row['kb_per_image']):.2f} | {int(float(row['original_count']))} | {int(float(row['q90_count']))} | {int(float(row['roi_count']))} |")
        bootstrap = read_json(RUNS / dataset / "bootstrap/report.json")
        lines.extend(["", f"## {dataset.upper()} paired bootstrap", "", "| 单位 | 预算 | 对比 | ΔAP均值 | 95% CI | P(Δ>0) |", "|---|---:|---|---:|---:|---:|"])
        for comparison in bootstrap["comparisons"]:
            lines.append(
                f"| {comparison['unit']} | {float(comparison['budget']):.0%} | {comparison['candidate']} vs {comparison['comparator']} | "
                f"{float(comparison['mean_delta']):+.6f} | [{float(comparison['ci95_low']):+.6f}, {float(comparison['ci95_high']):+.6f}] | {float(comparison['probability_gt_zero']):.3f} |"
            )
    path = ROOT / "RESULTS_ZH.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"complete": True, "datasets": reports, "results_sha256": file_sha256(path), "final_used_for_training_or_selection": False}
    atomic_json(RUNS / "report.json", report)
    atomic_json(RUNS / "COMPLETED.json", {"complete": True, "results_sha256": report["results_sha256"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "oregon"), default=None)
    parser.add_argument("--stage", choices=("preflight", "shared", "point", "bootstrap", "report", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "report":
        print(json.dumps(run_report(), indent=2, ensure_ascii=False))
        return
    if args.dataset is None:
        raise SystemExit("--dataset is required except for --stage report")
    if args.stage == "preflight":
        result = run_preflight(args.dataset)
    elif args.stage == "shared":
        run_preflight(args.dataset)
        result = train_shared_action(DatasetAdapter(args.dataset))
    elif args.stage == "point":
        result = run_point(args.dataset)
    elif args.stage == "bootstrap":
        result = run_bootstrap(args.dataset)
    else:
        result = {
            "preflight": run_preflight(args.dataset),
            "point": run_point(args.dataset),
            "bootstrap": run_bootstrap(args.dataset),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
