#!/usr/bin/env python3
"""Resumable deployment-consistent COCO CAVS-OQR retraining pipeline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
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
import torch


os.environ.setdefault("YOLO_AUTOINSTALL", "False")
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
sys.path[:0] = [str(SCRIPT.parent), str(REPO), str(REPO / "examples/YOLO26-Edge-Cloud")]

import contracts
from inference_contract import (
    ScaleDownDetectionPredictor,
    current_roi_xyxy,
    decode_jpeg,
    edge_boxes,
    encoded_jpeg,
    native_jpeg,
    official_predictions,
)
from ultralytics import YOLO


SOURCE = REPO / "paper/experiments/20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1/scripts"
CANONICAL_SCRIPTS = REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN_CORE = load_module("perframe_train_core", SOURCE / "train_core.py")
ADAPTER = load_module("perframe_training_adapter", SOURCE / "training_adapter.py")
DATA_CORE = load_module("perframe_data_core", SOURCE / "data_core.py")
CANONICAL_PROTOCOL = load_module("perframe_canonical_protocol", CANONICAL_SCRIPTS / "protocol_core.py")
CANONICAL = load_module("perframe_canonical_eval", CANONICAL_SCRIPTS / "run_multiranker_benchmark.py")
PAPER = load_module(
    "perframe_reward_paper_core",
    REPO / "paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/paper_core.py",
)
CA_AORIC = load_module("perframe_reward_metric", REPO / "paper/experiments/CA_AORIC/ca_aoric.py")
sys.path.insert(0, str(REPO / "paper/experiments/GAIR_v5"))
from gair.selector import select_actions_for_budget


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def role_source(role: str) -> dict[str, Any]:
    manifest_path = contracts.role_manifest(PROTOCOL["canonical_root"], role)
    manifest = read_json(manifest_path)
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "gt": Path(manifest["gt"]),
        "samples": Path(manifest["samples"]),
    }


def run_preflight() -> dict[str, Any]:
    completion = RUNS / "preflight/completion.json"
    report_path = RUNS / "preflight/report.json"
    if completion.is_file():
        return read_json(report_path)
    contracts.validate_protocol(PROTOCOL)
    sources = [ROOT / "configs/protocol.json", SCRIPT, SCRIPT.parent / "contracts.py", SCRIPT.parent / "inference_contract.py"]
    role_report = {}
    identities = {}
    for role, expected in PROTOCOL["roles"].items():
        source = role_source(role)
        samples = read_json(source["samples"])
        ids = [int(sample["image_id"]) for sample in samples]
        contracts.validate_population(ids, role=role, expected=int(expected))
        role_report[role] = {"images": len(ids), "gt": str(source["gt"]), "samples": str(source["samples"])}
        identities[role] = {str(sample["identity"]) for sample in samples}
        sources.extend((source["manifest_path"], source["gt"], source["samples"]))
    overlaps = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(PROTOCOL["roles"])
        for right in list(PROTOCOL["roles"])[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"role identity leakage: {overlaps}")
    for model in (PROTOCOL["edge_model"], PROTOCOL["cloud_model"]):
        path = Path(model["path"])
        if file_sha256(path) != str(model["sha256"]):
            raise RuntimeError(f"model hash changed: {path}")
        sources.append(path)
    folds = read_json(Path(PROTOCOL["frozen_folds"]))["folds"]
    flat = [int(image_id) for fold in folds for image_id in fold]
    fit_ids = {int(sample["image_id"]) for sample in read_json(role_source("fit")["samples"])}
    if len(folds) != 5 or set(flat) != fit_ids or len(flat) != len(set(flat)):
        raise RuntimeError("frozen folds changed")
    sources.append(Path(PROTOCOL["frozen_folds"]))
    hashes = {str(path.resolve()): file_sha256(path) for path in sources}
    report = {
        "complete": True,
        "roles": role_report,
        "identity_overlap": overlaps,
        "training_jobs": len(contracts.training_jobs(PROTOCOL)),
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "single_image_inference": True,
        "batch_shape_replay": False,
        "source_experiments_read_only": True,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


class InferenceModels:
    def __init__(self, gt: Mapping[str, Any]) -> None:
        self.edge = YOLO(PROTOCOL["edge_model"]["path"])
        self.full = YOLO(PROTOCOL["cloud_model"]["path"])
        self.roi = YOLO(PROTOCOL["cloud_model"]["path"])
        self.edge_category_map = CANONICAL_PROTOCOL.category_map_from_names(self.edge.names, gt["categories"])
        self.cloud_category_map = CANONICAL_PROTOCOL.category_map_from_names(self.full.names, gt["categories"])

    @staticmethod
    def _kwargs(action: str) -> dict[str, Any]:
        values = contracts.action_predict_kwargs(PROTOCOL, action)
        values.pop("scaleup")
        return values

    def predict_edge(self, image: np.ndarray, image_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        contracts.require_single_image(image)
        result = self.edge.predict(source=image, predictor=ScaleDownDetectionPredictor, **self._kwargs("edge"))[0]
        return edge_boxes(result), official_predictions(result, image_id=image_id, category_map=self.edge_category_map)

    def predict_full(self, image: np.ndarray, image_id: int, action: str) -> list[dict[str, Any]]:
        contracts.require_single_image(image)
        result = self.full.predict(source=image, predictor=ScaleDownDetectionPredictor, **self._kwargs(action))[0]
        return official_predictions(result, image_id=image_id, category_map=self.cloud_category_map)

    def predict_roi(self, image: np.ndarray, image_id: int, xyxy: Sequence[int]) -> list[dict[str, Any]]:
        contracts.require_single_image(image)
        result = self.roi.predict(source=image, **self._kwargs("current@640"))[0]
        return official_predictions(
            result,
            image_id=image_id,
            category_map=self.cloud_category_map,
            offset_xy=(int(xyxy[0]), int(xyxy[1])),
        )


def cache_signature() -> str:
    payload = {
        "protocol": PROTOCOL["protocol_id"],
        "edge": PROTOCOL["edge_model"]["sha256"],
        "cloud": PROTOCOL["cloud_model"]["sha256"],
        "inference": PROTOCOL["inference"],
        "codec": PROTOCOL["codec"],
        "roi": PROTOCOL["roi"],
        "maur": PROTOCOL["maur"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_one(models: InferenceModels, role: str, sample: Mapping[str, Any]) -> dict[str, Any]:
    image_id = int(sample["image_id"])
    path = Path(str(sample["identity"]))
    original_payload = native_jpeg(path)
    image = decode_jpeg(original_payload)
    height, width = image.shape[:2]
    if (width, height) != (int(sample["width"]), int(sample["height"])):
        raise RuntimeError(f"image dimensions changed: {role}/{image_id}")
    edge_local, edge_official = models.predict_edge(image, image_id)
    original_predictions = models.predict_full(image, image_id, "original")
    codec = PROTOCOL["codec"]
    q90_payload = encoded_jpeg(path, long_side=codec["long_side"], quality=codec["quality"], optimize=codec["optimize"], subsampling=codec["subsampling"])
    q90_predictions = models.predict_full(decode_jpeg(q90_payload), image_id, "q90")
    roi_cfg = PROTOCOL["roi"]
    roi_xyxy = current_roi_xyxy(
        width,
        height,
        edge_local,
        confidence=float(roi_cfg["confidence"]),
        max_area_ratio=float(roi_cfg["max_area_ratio"]),
    )
    roi_payload = b""
    roi_raw: list[dict[str, Any]] = []
    roi_maur: list[dict[str, Any]] = []
    if roi_xyxy is not None:
        roi_payload = encoded_jpeg(
            path,
            crop_xyxy=roi_xyxy,
            long_side=codec["long_side"],
            quality=codec["quality"],
            optimize=codec["optimize"],
            subsampling=codec["subsampling"],
        )
        roi_raw = models.predict_roi(decode_jpeg(roi_payload), image_id, roi_xyxy)
        roi_maur = CANONICAL_PROTOCOL.class_aware_nms(
            CANONICAL_PROTOCOL.maur_candidates(
                edge_official,
                roi_raw,
                tuple(map(int, roi_xyxy)),
                duplicate_iou=float(PROTOCOL["maur"]["duplicate_iou"]),
            ),
            iou_threshold=float(PROTOCOL["maur"]["nms_iou"]),
        )
    return {
        "schema": "cavs_perframe_action_shard_v1",
        "signature": cache_signature(),
        "role": role,
        "image_id": image_id,
        "identity": str(path),
        "file_name": str(sample.get("file_name") or path.name),
        "width": width,
        "height": height,
        "payload_bytes": {
            "original": len(original_payload),
            "q90": len(q90_payload),
            "current@640": len(roi_payload) if roi_xyxy is not None else 0,
        },
        "available": {
            "edge": True,
            "original": True,
            "q90": 0 < len(q90_payload) < len(original_payload),
            "current@640": roi_xyxy is not None,
        },
        "roi_xyxy": list(map(int, roi_xyxy)) if roi_xyxy is not None else None,
        "predictions": {
            "edge": edge_official,
            "original": original_predictions,
            "q90": q90_predictions,
            "current@640": roi_maur,
            "current@640_raw": roi_raw,
        },
    }


def run_cache_role(role: str, *, limit: int = 0, smoke: bool = False) -> dict[str, Any]:
    source = role_source(role)
    samples = sorted(read_json(source["samples"]), key=lambda row: int(row["image_id"]))
    if limit:
        samples = samples[: int(limit)]
    output = RUNS / (f"smoke/cache/{role}" if smoke else f"cache/{role}")
    completion = output / "completion.json"
    report_path = output / "report.json"
    if completion.is_file() and not smoke:
        return read_json(report_path)
    models = InferenceModels(read_json(source["gt"]))
    started = time.time()
    completed = 0
    for index, sample in enumerate(samples, 1):
        image_id = int(sample["image_id"])
        shard = output / f"shards/{image_id:012d}.json"
        if shard.is_file():
            current = read_json(shard)
            if current.get("signature") != cache_signature() or int(current.get("image_id", -1)) != image_id:
                raise RuntimeError(f"cache shard drift: {shard}")
        else:
            atomic_json(shard, cache_one(models, role, sample))
        completed += 1
        if index % 25 == 0 or index == len(samples):
            print(f"[cache {role}] {index}/{len(samples)}", flush=True)
    report = {
        "complete": completed == len(samples),
        "role": role,
        "images": completed,
        "expected": len(samples),
        "formal_expected": int(PROTOCOL["roles"][role]),
        "signature": cache_signature(),
        "single_image_inference": True,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(report_path, report)
    if not smoke:
        contracts.validate_population(
            [int(path.stem) for path in (output / "shards").glob("*.json")], role=role
        )
        atomic_json(completion, {"complete": True, "images": completed, "signature": cache_signature()})
    return report


def run_cache(*, limit: int = 0, smoke: bool = False) -> dict[str, Any]:
    run_preflight()
    roles = ("final",) if smoke else tuple(PROTOCOL["roles"])
    reports = {role: run_cache_role(role, limit=limit, smoke=smoke) for role in roles}
    if not smoke:
        atomic_json(RUNS / "cache/completion.json", {"complete": True, "roles": {key: value["images"] for key, value in reports.items()}})
    return reports


def flat_predictions(shards: Sequence[Mapping[str, Any]], action: str) -> list[dict[str, Any]]:
    return [row for shard in shards for row in shard["predictions"][action]]


def build_common_role(role: str) -> dict[str, Any]:
    output = RUNS / f"common/coco/{role}"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(completion)
    source = role_source(role)
    gt = read_json(source["gt"])
    old = {int(sample["image_id"]): sample for sample in read_json(source["samples"])}
    paths = sorted((RUNS / f"cache/{role}/shards").glob("*.json"))
    contracts.validate_population([int(path.stem) for path in paths], role=role)
    shards = [read_json(path) for path in paths]
    samples = []
    for shard in shards:
        image_id = int(shard["image_id"])
        source_sample = old[image_id]
        payloads = {key: int(value) for key, value in shard["payload_bytes"].items()}
        available = {key: bool(value) for key, value in shard["available"].items()}
        actions = [
            {"action_id": "edge", "payload_bytes": 0, "available": True},
            {"action_id": "original", "payload_bytes": payloads["original"], "available": True},
        ]
        if available["q90"]:
            actions.append({"action_id": "q90", "payload_bytes": payloads["q90"], "available": True})
        if available["current@640"]:
            x1, y1, x2, y2 = map(int, shard["roi_xyxy"])
            actions.append(
                {
                    "action_id": "current@640",
                    "payload_bytes": payloads["current@640"],
                    "available": True,
                    "profile": "current",
                    "imgsz": 640,
                    "xyxy": [x1, y1, x2, y2],
                    "crop_width": x2 - x1,
                    "crop_height": y2 - y1,
                    "area_ratio": (x2 - x1) * (y2 - y1) / max(1, int(shard["width"]) * int(shard["height"])),
                }
            )
        samples.append(
            {
                "image_id": image_id,
                "identity": str(source_sample["identity"]),
                "file_name": str(source_sample.get("file_name") or Path(str(source_sample["identity"])).name),
                "width": int(shard["width"]),
                "height": int(shard["height"]),
                "event_id": f"coco:{image_id}",
                "camera": "coco",
                "payload_bytes": payloads,
                "available": available,
                "actions": actions,
                "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
            }
        )
    actions_dir = output / "actions"
    action_sources = {}
    for action in contracts.ACTIONS:
        path = actions_dir / f"{action.replace('@', '__')}.json"
        atomic_json(path, flat_predictions(shards, action))
        action_sources[action] = str(path.resolve())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "gt.json", gt)
    atomic_json(output / "samples.json", samples)
    manifest = {
        "dataset": "coco",
        "role": role,
        "images": len(samples),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "edge_model_sha256": PROTOCOL["edge_model"]["sha256"],
        "action_sources": action_sources,
        "gt": str((output / "gt.json").resolve()),
        "samples": str((output / "samples.json").resolve()),
        "single_image_inference": True,
        "primary_roi_fusion": "MAUR",
        "batch_shape_replay": False,
    }
    atomic_json(output / "source_manifest.json", manifest)
    report = {"complete": True, "role": role, "images": len(samples), "samples_sha256": file_sha256(output / "samples.json")}
    atomic_json(completion, report)
    return report


def run_common() -> dict[str, Any]:
    if not (RUNS / "cache/completion.json").is_file():
        raise RuntimeError("per-frame cache must complete first")
    reports = {role: build_common_role(role) for role in PROTOCOL["roles"]}
    atomic_json(RUNS / "common/completion.json", {"complete": True, "roles": {key: value["images"] for key, value in reports.items()}})
    return reports


def group_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json(path):
        output[int(row["image_id"])].append(dict(row))
    return dict(output)


def load_role(role: str) -> dict[str, Any]:
    root = RUNS / f"common/coco/{role}"
    manifest = read_json(root / "source_manifest.json")
    samples = read_json(root / "samples.json")
    return {
        "root": root,
        "gt": root / "gt.json",
        "samples": samples,
        "actions": {name: group_predictions(Path(path)) for name, path in manifest["action_sources"].items()},
        "category_ids": [int(row["id"]) for row in read_json(root / "gt.json")["categories"]],
    }


_REWARD_STATE: dict[str, Any] = {}


def reward_folds(role: str, samples: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    if role == "calibration":
        return [sorted(int(sample["image_id"]) for sample in samples)]
    folds = [list(map(int, fold)) for fold in read_json(Path(PROTOCOL["frozen_folds"]))["folds"]]
    expected = {int(sample["image_id"]) for sample in samples}
    observed = [image_id for fold in folds for image_id in fold]
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise RuntimeError("reward folds changed")
    return folds


def _reward_init(gt_path: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _REWARD_STATE["metric"] = CA_AORIC.CocoMetricBackend(Path(gt_path))


def _reward_one(task: tuple[int, str]) -> dict[str, Any]:
    image_id, action = int(task[0]), str(task[1])
    edge = _REWARD_STATE["actions"]["edge"]
    context = [
        {"image_id": value, "edge_predictions": edge.get(value, [])}
        for value in _REWARD_STATE["contexts"][image_id]
    ]
    target = {
        "image_id": image_id,
        "edge_predictions": edge.get(image_id, []),
        "cloud_predictions": _REWARD_STATE["actions"][action].get(image_id, []),
    }
    row = PAPER.dual_metric_context_reward(_REWARD_STATE["metric"], context, target)
    row.update(action_id=action, reward=float(row["reward_ap50_95"]), fold_index=int(_REWARD_STATE["membership"][image_id]))
    return row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reward_action(role: str, action: str, data: Mapping[str, Any], contexts: Mapping[int, list[int]], membership: Mapping[int, int]) -> dict[str, Any]:
    output = RUNS / f"rewards/{role}/{action.replace('@', '__')}"
    rows_path = output / "rows.jsonl"
    completion = output / "completion.json"
    if completion.is_file():
        report = read_json(completion)
        if report["rows_sha256"] != file_sha256(rows_path):
            raise RuntimeError(f"reward shard drift: {role}/{action}")
        return report
    existing_rows = read_jsonl(rows_path)
    existing = {int(row["target_image_id"]) for row in existing_rows}
    expected = {int(sample["image_id"]) for sample in data["samples"]}
    tasks = [(image_id, action) for image_id in sorted(expected - existing)]
    _REWARD_STATE.clear()
    _REWARD_STATE.update(actions=data["actions"], contexts=dict(contexts), membership=dict(membership))
    output.mkdir(parents=True, exist_ok=True)
    workers = min(int(PROTOCOL["reward"]["workers"]), max(1, os.cpu_count() or 1))
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        if tasks:
            with mp.get_context("fork").Pool(workers, initializer=_reward_init, initargs=(str(data["gt"]),)) as pool:
                for count, row in enumerate(pool.imap_unordered(_reward_one, tasks, chunksize=2), 1):
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    if count % 100 == 0:
                        handle.flush()
                    if count % 500 == 0 or count == len(tasks):
                        print(f"[reward {role}/{action}] {len(existing) + count}/{len(expected)}", flush=True)
    rows = read_jsonl(rows_path)
    if len(rows) != len(expected) or {int(row["target_image_id"]) for row in rows} != expected:
        raise RuntimeError(f"incomplete reward shard: {role}/{action}")
    report = {"complete": True, "role": role, "action": action, "rows": len(rows), "rows_sha256": file_sha256(rows_path), "elapsed_seconds": time.time() - started}
    atomic_json(completion, report)
    return report


def run_rewards_role(role: str) -> dict[str, Any]:
    output = RUNS / f"rewards/{role}"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(completion)
    data = load_role(role)
    folds = reward_folds(role, data["samples"])
    contexts = DATA_CORE.fold_contained_contexts(folds, context_size=int(PROTOCOL["reward"]["context_size"]), seed=int(PROTOCOL["fold_seed"]))
    membership = {int(image_id): index for index, fold in enumerate(folds) for image_id in fold}
    atomic_json(output / "folds.json", {"folds": folds})
    atomic_json(output / "contexts.json", {str(key): value for key, value in contexts.items()})
    reward_map = {int(sample["image_id"]): {} for sample in data["samples"]}
    for action in ("original", "q90", "current@640"):
        reward_action(role, action, data, contexts, membership)
        for row in read_jsonl(output / action.replace("@", "__") / "rows.jsonl"):
            reward_map[int(row["target_image_id"])][action] = float(row["reward"])
    atomic_pickle(output / "reward_map.pkl", reward_map)
    report = {"complete": True, "role": role, "images": len(reward_map), "tasks": len(reward_map) * 3, "reward_map_sha256": file_sha256(output / "reward_map.pkl")}
    atomic_json(output / "report.json", report)
    atomic_json(completion, report)
    return report


def run_rewards() -> dict[str, Any]:
    if not (RUNS / "common/completion.json").is_file():
        raise RuntimeError("common cache must complete first")
    reports = {role: run_rewards_role(role) for role in ("fit", "calibration")}
    atomic_json(RUNS / "rewards/completion.json", {"complete": True, "roles": {key: value["tasks"] for key, value in reports.items()}})
    return reports


_TRAIN_STATE: dict[str, Any] = {}


def records_for(role: str) -> list[dict[str, Any]]:
    data = load_role(role)
    return TRAIN_CORE.build_action_records(data["samples"], data["actions"]["edge"], category_ids=data["category_ids"])


def initialize_training() -> None:
    fit_records = records_for("fit")
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        fit_records=fit_records,
        fit_rewards=TRAIN_CORE.reward_matrix(fit_records, load_pickle(RUNS / "rewards/fit/reward_map.pkl")),
        folds=reward_folds("fit", load_role("fit")["samples"]),
        category_ids=load_role("fit")["category_ids"],
        calibration_records=records_for("calibration"),
        final_records=records_for("final"),
    )


def training_root(job: Mapping[str, Any]) -> Path:
    fold = "full" if job["fold"] == "full" else f"fold_{int(job['fold'])}"
    return RUNS / f"training/jobs/CAVS-Current-MAUR/seed_{int(job['seed'])}/{fold}"


def train_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    root = training_root(job)
    completion = root / "completion.json"
    report_path = root / "report.json"
    if completion.is_file():
        return read_json(report_path)
    records = _TRAIN_STATE["fit_records"]
    rewards = _TRAIN_STATE["fit_rewards"]
    ids = [int(record["image_id"]) for record in records]
    if job["fold"] == "full":
        train_indices = list(range(len(records)))
        predictions = {"calibration": _TRAIN_STATE["calibration_records"], "final": _TRAIN_STATE["final_records"]}
    else:
        heldout_ids = set(_TRAIN_STATE["folds"][int(job["fold"])])
        heldout = [index for index, image_id in enumerate(ids) if image_id in heldout_ids]
        train_indices = [index for index, image_id in enumerate(ids) if image_id not in heldout_ids]
        predictions = {"fit_oof": [records[index] for index in heldout]}
    train_records = [records[index] for index in train_indices]
    started = time.time()
    artifact = ADAPTER.fit_model(
        "current_action",
        train_records,
        rewards[train_indices],
        PROTOCOL,
        category_count=len(_TRAIN_STATE["category_ids"]),
        seed=int(job["seed"]) + (0 if job["fold"] == "full" else int(job["fold"]) * 10_007),
    )
    arrays = {}
    for role, role_records in predictions.items():
        arrays[f"scores_{role}"] = np.asarray(ADAPTER.predict_model(artifact, role_records, PROTOCOL, category_count=len(_TRAIN_STATE["category_ids"])), dtype=np.float64)
        arrays[f"image_ids_{role}"] = np.asarray([int(record["image_id"]) for record in role_records], dtype=np.int64)
    outer = {
        "method": "CAVS-Current-MAUR",
        "profile": "current",
        "fusion": "MAUR",
        "action_space": list(contracts.ACTIONS),
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "category_ids": _TRAIN_STATE["category_ids"],
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "artifact": artifact,
    }
    artifact_path = root / "artifact.pkl"
    score_path = root / "scores.npz"
    atomic_pickle(artifact_path, outer)
    atomic_npz(score_path, **arrays)
    metadata = artifact.get("metadata", {})
    report = {
        "complete": True,
        "id": job["id"],
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "fit_images": len(train_indices),
        "selected_epoch": int(metadata.get("selected_epoch", -1)),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": file_sha256(artifact_path),
        "scores_sha256": file_sha256(score_path),
        "old_learner_checkpoint_reused": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "artifact_sha256": report["artifact_sha256"], "scores_sha256": report["scores_sha256"]})
    print(f"[train] {job['id']} {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def run_train() -> dict[str, Any]:
    completion = RUNS / "training/completion.json"
    if completion.is_file():
        return read_json(RUNS / "training/report.json")
    initialize_training()
    jobs = contracts.training_jobs(PROTOCOL)
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(jobs))
    started = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        reports = list(pool.imap_unordered(train_job, jobs, chunksize=1))
    report = {"complete": True, "jobs": len(reports), "workers": workers, "elapsed_seconds": time.time() - started, "old_learner_checkpoints_reused": False}
    atomic_json(RUNS / "training/report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(reports)})
    _TRAIN_STATE.clear()
    return report


def ensemble_scores(role: str) -> np.ndarray:
    expected = [int(record["image_id"]) for record in records_for(role)]
    values = []
    for seed in PROTOCOL["model_seeds"]:
        path = RUNS / f"training/jobs/CAVS-Current-MAUR/seed_{int(seed)}/full/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            ids = list(map(int, artifact[f"image_ids_{role}"]))
            scores = np.asarray(artifact[f"scores_{role}"], dtype=np.float64)
        if ids != expected:
            raise RuntimeError(f"score alignment changed: {role}/{seed}")
        if scores.ndim == 3:
            if scores.shape[0] != len(PROTOCOL["budgets"]):
                raise RuntimeError(f"unexpected budget score axis: {role}/{seed} {scores.shape}")
            if not np.array_equal(scores, np.broadcast_to(scores[0], scores.shape)):
                raise RuntimeError(f"static ActionValue scores changed across budgets: {role}/{seed}")
            scores = scores[0]
        if scores.ndim != 2:
            raise RuntimeError(f"unexpected score shape: {role}/{seed} {scores.shape}")
        values.append(scores)
    return np.mean(values, axis=0)


def score_map(records: Sequence[Mapping[str, Any]], scores: np.ndarray) -> dict[int, dict[str, float]]:
    return {
        int(record["image_id"]): {
            str(action["action_id"]): float(scores[row, column])
            for column, action in enumerate(record["actions"])
            if str(action["action_id"]) != "edge"
        }
        for row, record in enumerate(records)
    }


def schedule(samples: Sequence[Mapping[str, Any]], scores: Mapping[int, Mapping[str, float]], *, mode: str, budget: float) -> dict[int, str]:
    groups = []
    best = []
    for sample in samples:
        image_id = int(sample["image_id"])
        actions = [{"action_id": "edge", "image_id": image_id, "value": 0.0, "cost": 0.0}]
        for action, value in scores[image_id].items():
            if not sample["available"].get(action, False):
                continue
            actions.append({"action_id": action, "image_id": image_id, "value": float(value), "cost": float(sample["payload_bytes"][action])})
        groups.append(actions)
        best.append(max(actions[1:], key=lambda row: (row["value"], -row["cost"], row["action_id"])))
    if mode == "fixed_byte":
        capacity = float(budget) * sum(int(sample["payload_bytes"]["original"]) for sample in samples)
        result = select_actions_for_budget(groups, capacity)
        selection = {int(row["image_id"]): str(row["action_id"]) for row in result["selected"] if str(row["action_id"]) != "edge"}
        used = sum(int(sample["payload_bytes"][selection[int(sample["image_id"])]]) for sample in samples if int(sample["image_id"]) in selection)
        if used > capacity + 1e-6:
            raise RuntimeError("fixed-byte capacity exceeded")
        return selection
    count = int(math.floor(len(samples) * float(budget) + 0.5))
    selected = sorted(best, key=lambda row: (-row["value"], row["cost"], row["image_id"]))[:count]
    return {int(row["image_id"]): str(row["action_id"]) for row in selected}


def evaluate_selection(role: str, mode: str, budget: float, selection: Mapping[int, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    CANONICAL.COMMON = RUNS / "common"
    data = CANONICAL.load_role("coco", role)
    predictions, _ = CANONICAL.mixed_predictions(data, dict(selection), 0.001)
    metric = data["evaluator"].evaluate(predictions)
    by_id = {int(sample["image_id"]): sample for sample in data["samples"]}
    used = sum(int(by_id[image_id]["payload_bytes"][action]) for image_id, action in selection.items())
    total = sum(int(sample["payload_bytes"]["original"]) for sample in data["samples"])
    counts = Counter(selection.values())
    return {
        "role": role,
        "method": "CAVS-Current-MAUR-PerFrame",
        "mode": mode,
        "budget": budget,
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "request_bytes": used,
        "capacity_bytes": float(budget) * total if mode == "fixed_byte" else None,
        "capacity_utilization": used / max(1.0, float(budget) * total) if mode == "fixed_byte" else 1.0,
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["current@640"]),
        "kb_per_image": used / 1024.0 / len(data["samples"]),
    }, predictions


def run_evaluate() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    if completion.is_file():
        return read_json(RUNS / "evaluation/report.json")
    run_train()
    rows = []
    selections = {}
    for role in ("calibration", "final"):
        role_records = records_for(role)
        scores = score_map(role_records, ensemble_scores(role))
        samples = load_role(role)["samples"]
        for mode in PROTOCOL["modes"]:
            for budget in map(float, PROTOCOL["budgets"]):
                selection = schedule(samples, scores, mode=mode, budget=budget)
                key = f"{role}|CAVS-Current-MAUR-PerFrame|{mode}|{budget:g}"
                selections[key] = {str(image_id): action for image_id, action in selection.items()}
                row, _ = evaluate_selection(role, mode, budget, selection)
                rows.append(row)
    atomic_csv(RUNS / "evaluation/results.csv", rows)
    atomic_json(RUNS / "evaluation/selections.json", selections)
    report = {"complete": True, "rows": len(rows), "single_image_inference": True, "final_used_for_training_or_selection": False}
    atomic_json(RUNS / "evaluation/report.json", report)
    atomic_json(completion, report)
    return report


def export_policy() -> dict[str, Any]:
    output = RUNS / "export/policy"
    completion = RUNS / "export/completion.json"
    if completion.is_file():
        return read_json(completion)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    category_ids = None
    for seed in map(int, PROTOCOL["model_seeds"]):
        root = RUNS / f"training/jobs/CAVS-Current-MAUR/seed_{seed}/full"
        with (root / "artifact.pkl").open("rb") as handle:
            outer = pickle.load(handle)
        inner = outer["artifact"]
        current_categories = list(map(int, outer["category_ids"]))
        category_ids = current_categories if category_ids is None else category_ids
        if current_categories != category_ids:
            raise RuntimeError("category schema differs across seeds")
        arrays = {name: tensor.detach().cpu().numpy() for name, tensor in inner["state_dict"].items()}
        path = output / f"seed_{seed}.npz"
        atomic_npz(path, **arrays)
        metadata = inner["metadata"]
        entries.append(
            {
                "seed": seed,
                "file": path.name,
                "sha256": file_sha256(path),
                "reward_mean": float(metadata["reward_mean"]),
                "reward_std": float(metadata["reward_std"]),
                "selected_epoch": int(metadata["selected_epoch"]),
                "source_artifact_sha256": file_sha256(root / "artifact.pkl"),
            }
        )
    model = PROTOCOL["model"]
    manifest = {
        "schema_version": 1,
        "policy_type": "cavs_current_maur",
        "policy_id": "cavs_current_maur_perframe_deployment_v1",
        "display_name": "CAVS-Current-MAUR PerFrame (3-seed)",
        "method": "CAVS-Current-MAUR",
        "profile": "current",
        "fusion": "MAUR",
        "action_ids": list(contracts.ACTIONS),
        "category_count": 80,
        "category_ids": category_ids,
        "edge_model": {"name": "yolo26n.pt", "sha256": PROTOCOL["edge_model"]["sha256"]},
        "deploy_confidence": float(model["deploy_confidence"]),
        "architecture": {
            "class_embedding_dim": int(model["class_embedding_dim"]),
            "box_hidden_dim": int(model["box_hidden_dim"]),
            "action_hidden_dim": int(model["action_hidden_dim"]),
            "joint_hidden_dim": int(model["joint_hidden_dim"]),
            "dropout": float(model["dropout"]),
            "max_boxes": int(model["max_boxes"]),
        },
        "roi": {"action": "current@640", "confidence": 0.25, "max_area_ratio": 0.65, "imgsz": 640},
        "maur": PROTOCOL["maur"],
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "source_protocol": PROTOCOL["protocol_id"],
        "training_jobs_complete": 18,
        "seeds": entries,
    }
    atomic_json(output / "manifest.json", manifest)
    report = {"complete": True, "manifest": str((output / "manifest.json").resolve()), "manifest_sha256": file_sha256(output / "manifest.json")}
    atomic_json(completion, report)
    return report


def run_report() -> dict[str, Any]:
    run_evaluate()
    export = export_policy()
    with (RUNS / "evaluation/results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# COCO CAVS-OQR 逐帧部署一致重训结果",
        "",
        "本实验不复刻 `model.val(rect=True)` 的批次形状；所有检测缓存均为单图调用。COCO val2017 仍属于 post-hoc 公开验证集。",
        "",
        "| 角色/模式 | 10% | 25% | 40% |",
        "|---|---:|---:|---:|",
    ]
    for role in ("calibration", "final"):
        for mode in PROTOCOL["modes"]:
            cells = []
            for budget in PROTOCOL["budgets"]:
                found = [row for row in rows if row["role"] == role and row["mode"] == mode and math.isclose(float(row["budget"]), float(budget))]
                cells.append(f"{float(found[0]['official_ap50_95']):.6f} / {float(found[0]['kb_per_image']):.2f}")
            lines.append(f"| {role}/{mode} | " + " | ".join(cells) + " |")
    lines.extend(["", f"策略：`{export['manifest']}`"])
    path = ROOT / "RESULTS_ZH.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"complete": True, "result": str(path), "policy": export["manifest"]}


def run_all() -> dict[str, Any]:
    return {
        "preflight": run_preflight(),
        "cache": run_cache(),
        "common": run_common(),
        "rewards": run_rewards(),
        "train": run_train(),
        "evaluate": run_evaluate(),
        "export": export_policy(),
        "report": run_report(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "cache", "common", "rewards", "train", "evaluate", "export", "report", "all"), default="all")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    actions = {
        "preflight": run_preflight,
        "smoke": lambda: run_cache(limit=args.limit, smoke=True),
        "cache": run_cache,
        "common": run_common,
        "rewards": run_rewards,
        "train": run_train,
        "evaluate": run_evaluate,
        "export": export_policy,
        "report": run_report,
        "all": run_all,
    }
    print(json.dumps(actions[args.stage](), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
