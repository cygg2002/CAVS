#!/usr/bin/env python3
"""Train, evaluate, and export three-seed RKNN-domain ORIC/MORIC policies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import external_baseline_core as EXTERNAL_CORE


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
RUNS = ROOT / "runs/external"
METHODS = ("oric", "moric")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest(hashes: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_pickle(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)


def atomic_npz(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def atomic_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    fields = sorted({str(key) for row in rows for key in row})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def role_paths(role: str) -> dict[str, Path]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(f"unknown role: {role}")
    root = Path(PROTOCOL["source_backend_root"]) / "common/coco" / role
    manifest_path = root / "source_manifest.json"
    manifest = read_json(manifest_path)
    return {
        "root": root,
        "manifest": manifest_path,
        "samples": Path(manifest["samples"]),
        "gt": Path(manifest["gt"]),
        "edge": Path(manifest["action_sources"]["edge"]),
        "original": Path(manifest["action_sources"]["original"]),
    }


def group_predictions(
    rows: Sequence[Mapping[str, Any]], allowed_image_ids: set[int]
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in allowed_image_ids:
            grouped[image_id].append(dict(row))
    return dict(grouped)


def load_role(role: str, *, include_original: bool) -> dict[str, Any]:
    paths = role_paths(role)
    manifest = read_json(paths["manifest"])
    if str(manifest.get("backend")) != "fp16":
        raise RuntimeError(f"wrong source backend: {role}")
    if str(manifest.get("rknn_model_sha256")) != str(
        PROTOCOL["rknn_models"]["fp16"]["sha256"]
    ):
        raise RuntimeError(f"wrong RKNN model: {role}")
    if str(manifest.get("cloud_model_sha256")) != str(PROTOCOL["cloud_model_sha256"]):
        raise RuntimeError(f"wrong cloud model: {role}")
    samples = read_json(paths["samples"])
    expected = int(PROTOCOL["roles"][role])
    image_ids = [int(sample["image_id"]) for sample in samples]
    if len(image_ids) != expected or len(image_ids) != len(set(image_ids)):
        raise RuntimeError(f"role population changed: {role}")
    allowed = set(image_ids)
    edge = group_predictions(read_json(paths["edge"]), allowed)
    output = {
        "paths": paths,
        "samples": samples,
        "image_ids": image_ids,
        "edge": edge,
        "gt": read_json(paths["gt"]),
    }
    if include_original:
        output["original"] = group_predictions(read_json(paths["original"]), allowed)
    return output


def feature_matrix(
    samples: Sequence[Mapping[str, Any]],
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    category_ids: Sequence[int],
    *,
    top_k: int,
) -> np.ndarray:
    return np.asarray(
        [
            EXTERNAL_CORE.build_output_feature(
                predictions.get(int(sample["image_id"]), ()),
                width=int(sample["width"]),
                height=int(sample["height"]),
                category_ids=category_ids,
                top_k=int(top_k),
            )
            for sample in samples
        ],
        dtype=np.float32,
    )


def build_targets(rewards: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw = np.asarray(rewards, dtype=np.float64)
    ranks = EXTERNAL_CORE.empirical_cdf(raw, raw)
    return {
        "oric": (raw, np.ones_like(raw)),
        "moric": (ranks, ranks),
    }


def _source_hashes_for_roles(roles: Sequence[str]) -> dict[str, str]:
    files = {PROTOCOL_PATH, SCRIPT, SCRIPT.parent / "external_baseline_core.py"}
    for role in roles:
        paths = role_paths(role)
        files.update(paths[key] for key in ("manifest", "samples", "gt", "edge", "original"))
    return {str(path.resolve()): file_sha256(path) for path in sorted(files)}


def run_preflight() -> dict[str, Any]:
    report_path = RUNS / "preflight/report.json"
    completion = RUNS / "preflight/completion.json"
    if completion.is_file():
        return read_json(report_path)
    populations: dict[str, int] = {}
    identities: dict[str, set[int]] = {}
    for role in PROTOCOL["roles"]:
        data = load_role(role, include_original=False)
        populations[role] = len(data["image_ids"])
        identities[role] = set(data["image_ids"])
    names = list(identities)
    overlaps = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"role leakage: {overlaps}")
    hashes = _source_hashes_for_roles(tuple(PROTOCOL["roles"]))
    report = {
        "complete": True,
        "protocol_id": PROTOCOL["protocol_id"],
        "populations": populations,
        "identity_overlap": overlaps,
        "source_hashes": hashes,
        "source_digest": source_digest(hashes),
        "fit_only_training": True,
        "calibration_labels_used_for_training": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


def _reward_at(
    index: int,
    edge_stats: Sequence[Any],
    original_stats: Sequence[Any],
    context_size: int,
    context_seed: int,
) -> dict[str, float]:
    context = EXTERNAL_CORE.context_indices(
        len(edge_stats), index, context_size=int(context_size), seed=int(context_seed)
    )
    return EXTERNAL_CORE.oric_reward(
        [edge_stats[other] for other in context], edge_stats[index], original_stats[index]
    )


def _load_reward_checkpoint(
    path: Path, expected_ids: Sequence[int], expected_digest: str
) -> tuple[list[float], list[float], list[float]]:
    if not path.is_file():
        return [], [], []
    with np.load(path, allow_pickle=False) as archive:
        digest = str(archive["source_digest"].item())
        image_ids = list(map(int, archive["image_ids"]))
        rewards = list(map(float, archive["rewards"]))
        edge_maps = list(map(float, archive["edge_maps"]))
        original_maps = list(map(float, archive["original_maps"]))
    if (
        digest != expected_digest
        or image_ids != list(map(int, expected_ids[: len(image_ids)]))
        or len({len(image_ids), len(rewards), len(edge_maps), len(original_maps)}) != 1
    ):
        raise RuntimeError("reward checkpoint does not match frozen sources")
    return rewards, edge_maps, original_maps


def run_rewards() -> dict[str, Any]:
    report_path = RUNS / "rewards/report.json"
    completion = RUNS / "rewards/completion.json"
    artifact_path = RUNS / "rewards/fit_artifacts.npz"
    if completion.is_file():
        report = read_json(report_path)
        if str(report.get("artifact_sha256")) != file_sha256(artifact_path):
            raise RuntimeError("external reward artifact drift")
        return report
    run_preflight()
    started = time.time()
    data = load_role("fit", include_original=True)
    category_ids = [int(row["id"]) for row in data["gt"]["categories"]]
    if len(category_ids) != 80 or len(category_ids) != len(set(category_ids)):
        raise RuntimeError("COCO category schema changed")
    gt_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["gt"]["annotations"]:
        gt_map[int(annotation["image_id"])].append(dict(annotation))
    thresholds = list(map(float, PROTOCOL["external"]["context_metric_iou_thresholds"]))
    edge_stats = []
    original_stats = []
    for image_id in data["image_ids"]:
        ground_truth = gt_map.get(int(image_id), [])
        edge_stats.append(
            EXTERNAL_CORE.detection_stats(data["edge"].get(int(image_id), []), ground_truth, thresholds)
        )
        original_stats.append(
            EXTERNAL_CORE.detection_stats(
                data["original"].get(int(image_id), []), ground_truth, thresholds
            )
        )
    features = feature_matrix(
        data["samples"],
        data["edge"],
        category_ids,
        top_k=int(PROTOCOL["external"]["top_k"]),
    )
    hashes = _source_hashes_for_roles(("fit",))
    digest = source_digest(hashes)
    checkpoint = RUNS / "rewards/reward_checkpoint.npz"
    rewards, edge_maps, original_maps = _load_reward_checkpoint(
        checkpoint, data["image_ids"], digest
    )
    interval = int(PROTOCOL["external"]["checkpoint_interval"])
    workers = int(PROTOCOL["external"]["reward_workers"])
    for start in range(len(rewards), len(data["image_ids"]), interval):
        indices = list(range(start, min(start + interval, len(data["image_ids"]))))
        with ThreadPoolExecutor(max_workers=min(workers, len(indices))) as executor:
            rows = list(
                executor.map(
                    lambda index: _reward_at(
                        index,
                        edge_stats,
                        original_stats,
                        int(PROTOCOL["external"]["context_size"]),
                        int(PROTOCOL["external"]["context_seed"]),
                    ),
                    indices,
                )
            )
        rewards.extend(float(row["reward"]) for row in rows)
        edge_maps.extend(float(row["edge_map"]) for row in rows)
        original_maps.extend(float(row["cloud_map"]) for row in rows)
        atomic_npz(
            checkpoint,
            source_digest=np.asarray(digest),
            image_ids=np.asarray(data["image_ids"][: len(rewards)], dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float64),
            edge_maps=np.asarray(edge_maps, dtype=np.float64),
            original_maps=np.asarray(original_maps, dtype=np.float64),
        )
        if len(rewards) % 250 == 0 or len(rewards) == len(data["image_ids"]):
            print(f"[external-reward] {len(rewards)}/{len(data['image_ids'])}", flush=True)
    atomic_npz(
        artifact_path,
        source_digest=np.asarray(digest),
        image_ids=np.asarray(data["image_ids"], dtype=np.int64),
        features=features,
        rewards=np.asarray(rewards, dtype=np.float64),
        edge_maps=np.asarray(edge_maps, dtype=np.float64),
        original_maps=np.asarray(original_maps, dtype=np.float64),
        category_ids=np.asarray(category_ids, dtype=np.int64),
    )
    values = np.asarray(rewards, dtype=np.float64)
    report = {
        "complete": True,
        "images": len(values),
        "feature_dim": int(features.shape[1]),
        "context_size": int(PROTOCOL["external"]["context_size"]),
        "context_seed": int(PROTOCOL["external"]["context_seed"]),
        "metric_iou_thresholds": thresholds,
        "reward_min": float(values.min()),
        "reward_mean": float(values.mean()),
        "reward_max": float(values.max()),
        "elapsed_seconds": time.time() - started,
        "source_hashes": hashes,
        "source_digest": digest,
        "artifact_sha256": file_sha256(artifact_path),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {"complete": True, "images": len(values), "artifact_sha256": report["artifact_sha256"]},
    )
    return report


def _fit_job(payload: tuple[str, int, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]) -> tuple[str, int, dict[str, Any]]:
    method, seed, features, targets, weights, config = payload
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    selected = dict(config)
    selected["seed"] = int(seed)
    checkpoint = EXTERNAL_CORE.fit_reward_mlp(features, targets, weights, selected)
    return method, int(seed), checkpoint


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_train() -> dict[str, Any]:
    report_path = RUNS / "learners/report.json"
    completion = RUNS / "learners/completion.json"
    if completion.is_file():
        return read_json(report_path)
    run_rewards()
    started = time.time()
    with np.load(RUNS / "rewards/fit_artifacts.npz", allow_pickle=False) as artifact:
        fit_features = np.asarray(artifact["features"], dtype=np.float32)
        fit_rewards = np.asarray(artifact["rewards"], dtype=np.float64)
        category_ids = list(map(int, artifact["category_ids"]))
        fit_digest = str(artifact["source_digest"].item())
    role_data = {
        role: load_role(role, include_original=False) for role in ("calibration", "final")
    }
    role_features = {
        role: feature_matrix(
            data["samples"],
            data["edge"],
            category_ids,
            top_k=int(PROTOCOL["external"]["top_k"]),
        )
        for role, data in role_data.items()
    }
    targets = build_targets(fit_rewards)
    payloads = [
        (
            method,
            int(seed),
            fit_features,
            np.asarray(targets[method][0], dtype=np.float32),
            np.asarray(targets[method][1], dtype=np.float32),
            dict(PROTOCOL["external"]["mlp"]),
        )
        for method in METHODS
        for seed in map(int, PROTOCOL["seeds"])
    ]
    with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
        trained = list(executor.map(_fit_job, payloads))
    reports = []
    for method, seed, checkpoint in trained:
        root = RUNS / f"learners/{method}/seed_{seed}"
        checkpoint_path = root / "checkpoint.pkl"
        atomic_pickle(checkpoint_path, checkpoint)
        arrays: dict[str, np.ndarray] = {}
        for role, features in role_features.items():
            arrays[f"image_ids_{role}"] = np.asarray(
                role_data[role]["image_ids"], dtype=np.int64
            )
            arrays[f"scores_{role}"] = EXTERNAL_CORE.predict_reward_mlp(
                checkpoint, features
            )
        scores_path = root / "scores.npz"
        atomic_npz(scores_path, **arrays)
        row = {
            "complete": True,
            "method": method,
            "seed": int(seed),
            "selected_epoch": int(checkpoint["selected_epoch"]),
            "fit_validation_loss": float(checkpoint["fit_validation_loss"]),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "scores_sha256": file_sha256(scores_path),
            "fit_reward_source_digest": fit_digest,
            "fit_images": len(fit_features),
            "calibration_labels_used_for_training": False,
            "final_used_for_training_or_selection": False,
        }
        atomic_json(root / "report.json", row)
        atomic_json(root / "completion.json", {"complete": True, "scores_sha256": row["scores_sha256"]})
        reports.append(row)
    expected = {(method, int(seed)) for method in METHODS for seed in PROTOCOL["seeds"]}
    observed = {(str(row["method"]), int(row["seed"])) for row in reports}
    if observed != expected:
        raise RuntimeError("external learner matrix is incomplete")
    report = {
        "complete": True,
        "jobs": len(reports),
        "methods": list(METHODS),
        "seeds": list(map(int, PROTOCOL["seeds"])),
        "elapsed_seconds": time.time() - started,
        "calibration_labels_used_for_training": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "jobs": len(reports)})
    return report


def checkpoint_arrays(checkpoint: Mapping[str, Any]) -> dict[str, np.ndarray]:
    state = checkpoint["state_dict"]
    hidden_dims = list(map(int, checkpoint["hidden_dims"]))

    def tensor(key: str) -> np.ndarray:
        value = state[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite external policy tensor: {key}")
        return np.array(array, dtype=np.float32, copy=True)

    arrays: dict[str, np.ndarray] = {
        "layer_count": np.asarray(len(hidden_dims), dtype=np.int64)
    }
    for index in range(len(hidden_dims)):
        linear = index * 4
        batch_norm = linear + 1
        prefix = f"layer_{index}_"
        arrays.update(
            {
                prefix + "weight": tensor(f"{linear}.weight"),
                prefix + "bias": tensor(f"{linear}.bias"),
                prefix + "bn_weight": tensor(f"{batch_norm}.weight"),
                prefix + "bn_bias": tensor(f"{batch_norm}.bias"),
                prefix + "bn_mean": tensor(f"{batch_norm}.running_mean"),
                prefix + "bn_var": tensor(f"{batch_norm}.running_var"),
            }
        )
    output = len(hidden_dims) * 4
    arrays["output_weight"] = tensor(f"{output}.weight")
    arrays["output_bias"] = tensor(f"{output}.bias")
    return arrays


def predict_exported_mlp(arrays: Mapping[str, np.ndarray], feature: np.ndarray) -> float:
    value = np.asarray(feature, dtype=np.float32).reshape(1, -1)
    layer_count = int(np.asarray(arrays["layer_count"]).item())
    for index in range(layer_count):
        prefix = f"layer_{index}_"
        value = value @ np.asarray(arrays[prefix + "weight"], dtype=np.float32).T
        value += np.asarray(arrays[prefix + "bias"], dtype=np.float32)
        value = (value - np.asarray(arrays[prefix + "bn_mean"], dtype=np.float32)) / np.sqrt(
            np.asarray(arrays[prefix + "bn_var"], dtype=np.float32) + np.float32(1e-5)
        )
        value *= np.asarray(arrays[prefix + "bn_weight"], dtype=np.float32)
        value += np.asarray(arrays[prefix + "bn_bias"], dtype=np.float32)
        value = np.maximum(value, np.float32(0.0))
    value = value @ np.asarray(arrays["output_weight"], dtype=np.float32).T
    value += np.asarray(arrays["output_bias"], dtype=np.float32)
    result = float(value.reshape(-1)[0])
    if not math.isfinite(result):
        raise RuntimeError("external NumPy policy output is not finite")
    return result


def build_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    category_ids: Sequence[int],
    source_hashes: Mapping[str, str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    seeds = tuple(map(int, protocol["seeds"]))
    expected = {(method, seed) for method in METHODS for seed in seeds}
    observed = {(str(row["name"]), int(row["seed"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("external export matrix is incomplete")
    categories = list(map(int, category_ids))
    if len(categories) != 80 or len(categories) != len(set(categories)):
        raise RuntimeError("external export requires 80 unique categories")
    external = protocol["external"]
    return {
        "schema_version": 2,
        "policy_type": "oric_moric_original_only_rknn",
        "policy_id": "oric_moric_rknn_fp16_three_seed_v1",
        "display_name": "ORIC/MORIC RKNN-FP16 original-action (3-seed)",
        "actions": ["edge", "original"],
        "aggregation": "mean_score",
        "ensemble_seeds": list(seeds),
        "rknn_model_sha256": str(protocol["rknn_models"]["fp16"]["sha256"]),
        "cloud_model_sha256": str(protocol["cloud_model_sha256"]),
        "top_k": int(external["top_k"]),
        "category_ids": categories,
        "reward_contract": {
            "metric": "AP50",
            "context_size": int(external["context_size"]),
            "context_seed": int(external["context_seed"]),
            "moric_target": "right_continuous_fit_empirical_cdf",
            "fit_only": True,
        },
        "source_protocol": str(protocol["protocol_id"]),
        "source_hashes": dict(source_hashes),
        "models": [dict(row) for row in sorted(rows, key=lambda row: (str(row["name"]), int(row["seed"])))],
    }


def run_export() -> dict[str, Any]:
    completion = RUNS / "export/completion.json"
    report_path = RUNS / "export/report.json"
    if completion.is_file():
        return read_json(report_path)
    run_train()
    with np.load(RUNS / "rewards/fit_artifacts.npz", allow_pickle=False) as artifact:
        category_ids = list(map(int, artifact["category_ids"]))
    output = RUNS / "export/policy"
    rows = []
    maximum_error = 0.0
    rng = np.random.default_rng(20260828)
    checkpoint_hashes: dict[str, str] = {}
    for method in METHODS:
        for seed in map(int, PROTOCOL["seeds"]):
            checkpoint_path = RUNS / f"learners/{method}/seed_{seed}/checkpoint.pkl"
            checkpoint = _load_checkpoint(checkpoint_path)
            arrays = checkpoint_arrays(checkpoint)
            destination = output / f"{method}_seed_{seed}.npz"
            atomic_npz(destination, **arrays)
            values = rng.normal(size=(17, int(checkpoint["input_dim"]))).astype(np.float32)
            expected = EXTERNAL_CORE.predict_reward_mlp(checkpoint, values)
            actual = np.asarray([predict_exported_mlp(arrays, row) for row in values])
            maximum_error = max(maximum_error, float(np.max(np.abs(expected - actual))))
            checkpoint_hashes[str(checkpoint_path.resolve())] = file_sha256(checkpoint_path)
            rows.append(
                {
                    "name": method,
                    "seed": seed,
                    "file": destination.name,
                    "sha256": file_sha256(destination),
                    "input_dim": int(checkpoint["input_dim"]),
                    "hidden_dims": list(map(int, checkpoint["hidden_dims"])),
                    "selected_epoch": int(checkpoint["selected_epoch"]),
                    "source_checkpoint_sha256": checkpoint_hashes[str(checkpoint_path.resolve())],
                }
            )
    if maximum_error > 1e-5:
        raise RuntimeError(f"PyTorch/NumPy external parity failed: {maximum_error:.8g}")
    hashes = {
        str(PROTOCOL_PATH.resolve()): file_sha256(PROTOCOL_PATH),
        str(SCRIPT.resolve()): file_sha256(SCRIPT),
        str((SCRIPT.parent / "external_baseline_core.py").resolve()): file_sha256(
            SCRIPT.parent / "external_baseline_core.py"
        ),
        **checkpoint_hashes,
    }
    manifest = build_manifest(
        rows, category_ids=category_ids, source_hashes=hashes, protocol=PROTOCOL
    )
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)
    report = {
        "complete": True,
        "models": len(rows),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "maximum_pytorch_numpy_error": maximum_error,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "manifest_sha256": report["manifest_sha256"]})
    return report


def _score_variants(method: str, role: str) -> tuple[list[int], dict[str, np.ndarray]]:
    expected_ids: list[int] | None = None
    variants: dict[str, np.ndarray] = {}
    for seed in map(int, PROTOCOL["seeds"]):
        with np.load(
            RUNS / f"learners/{method}/seed_{seed}/scores.npz", allow_pickle=False
        ) as archive:
            image_ids = list(map(int, archive[f"image_ids_{role}"]))
            values = np.asarray(archive[f"scores_{role}"], dtype=np.float64)
        if expected_ids is None:
            expected_ids = image_ids
        elif image_ids != expected_ids:
            raise RuntimeError(f"external score population changed: {method}/{role}/{seed}")
        variants[f"seed_{seed}"] = values
    variants["ensemble"] = np.mean(np.stack(list(variants.values())), axis=0)
    assert expected_ids is not None
    return expected_ids, variants


def run_evaluate() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    report_path = RUNS / "evaluation/report.json"
    if completion.is_file():
        return read_json(report_path)
    run_train()
    import train_backend

    sources = train_backend.load_sources("fp16")
    rows: list[dict[str, Any]] = []
    for role in ("calibration", "final"):
        data = train_backend.load_role("fp16", role, sources)
        expected = [int(sample["image_id"]) for sample in data["samples"]]
        for method in METHODS:
            image_ids, variants = _score_variants(method, role)
            if image_ids != expected:
                raise RuntimeError(f"external score/sample alignment changed: {method}/{role}")
            for variant, values in variants.items():
                scores = {
                    image_id: {"original": float(score)}
                    for image_id, score in zip(image_ids, values)
                }
                for budget in map(float, PROTOCOL["budgets"]):
                    selection, used, capacity = train_backend.schedule_fixed_byte(
                        data["samples"], scores, budget, sources
                    )
                    if set(selection.values()) - {"original"}:
                        raise RuntimeError("external Original-only action contract changed")
                    predictions, _ = sources["canonical"].mixed_predictions(
                        data, selection, 0.001
                    )
                    metric = data["evaluator"].evaluate(predictions)
                    rows.append(
                        {
                            "role": role,
                            "method": method,
                            "model_variant": variant,
                            "mode": "fixed_byte",
                            "budget": budget,
                            "official_ap50_95": float(metric["map50_95"]),
                            "official_ap50": float(metric["map50"]),
                            "official_ap75": float(metric["map75"]),
                            "request_bytes": int(used),
                            "capacity_bytes": float(capacity),
                            "capacity_utilization": used / max(1.0, capacity),
                            "selected_images": len(selection),
                            "calibration_labels_used_for_training": False,
                            "final_used_for_training_or_selection": False,
                        }
                    )
    expected_rows = 2 * len(METHODS) * (len(PROTOCOL["seeds"]) + 1) * len(
        PROTOCOL["budgets"]
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"external point matrix incomplete: {len(rows)}/{expected_rows}")
    atomic_csv(RUNS / "evaluation/results.csv", rows)
    summary: dict[str, Any] = {}
    for method in METHODS:
        summary[method] = {}
        for budget in map(float, PROTOCOL["budgets"]):
            single = np.asarray(
                [
                    float(row["official_ap50_95"])
                    for row in rows
                    if row["role"] == "final"
                    and row["method"] == method
                    and float(row["budget"]) == budget
                    and str(row["model_variant"]).startswith("seed_")
                ]
            )
            ensemble = next(
                float(row["official_ap50_95"])
                for row in rows
                if row["role"] == "final"
                and row["method"] == method
                and float(row["budget"]) == budget
                and row["model_variant"] == "ensemble"
            )
            summary[method][f"{budget:g}"] = {
                "single_model_mean": float(single.mean()),
                "single_model_sd": float(single.std(ddof=1)),
                "ensemble": ensemble,
            }
    report = {
        "complete": True,
        "rows": len(rows),
        "final_ap50_95": summary,
        "calibration_labels_used_for_training": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "rows": len(rows)})
    return report


def run_stage(stage: str) -> dict[str, Any]:
    stages = {
        "preflight": run_preflight,
        "rewards": run_rewards,
        "train": run_train,
        "evaluate": run_evaluate,
        "export": run_export,
    }
    if stage != "all":
        return stages[stage]()
    return {
        "preflight": run_preflight(),
        "rewards": run_rewards(),
        "train": run_train(),
        "evaluate": run_evaluate(),
        "export": run_export(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "rewards", "train", "evaluate", "export", "all"), default="all"
    )
    args = parser.parse_args()
    print(json.dumps(run_stage(args.stage), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
