"""Fail-closed contracts for RK3588 deployment-aligned CAVS retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


BACKENDS = ("fp16", "int8-heads")
ROLES = {"fit": 9678, "calibration": 571, "final": 5000}
SEEDS = (20260809, 20260810, 20260811)
FOLD_COUNT = 5


def role_manifest(canonical_root: str | Path, role: str) -> Path:
    if role not in ROLES:
        raise ValueError(role)
    return Path(canonical_root) / "common" / "coco" / role / "source_manifest.json"


def validate_backend(backend: str) -> str:
    value = str(backend)
    if value not in BACKENDS:
        raise RuntimeError(f"unsupported RKNN backend: {value}")
    return value


def validate_model_sha256(value: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
        raise RuntimeError("model SHA-256 is invalid")
    return digest


def validate_population(values: Any, *, role: str) -> None:
    if role not in ROLES:
        raise RuntimeError(f"unknown COCO role: {role}")
    image_ids = [int(value) for value in values]
    if len(image_ids) != ROLES[role] or len(image_ids) != len(set(image_ids)):
        raise RuntimeError(f"role population changed: {role}")


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    backends = tuple(map(validate_backend, protocol.get("backends", ())))
    if not backends or len(backends) != len(set(backends)):
        raise RuntimeError(f"backend matrix changed: {backends}")
    jobs = []
    for backend in backends:
        for seed in SEEDS:
            for fold in (*range(FOLD_COUNT), "full"):
                fold_label = "full" if fold == "full" else f"fold_{fold}"
                jobs.append(
                    {
                        "id": f"backend={backend}|seed={seed}|{fold_label}",
                        "backend": backend,
                        "seed": seed,
                        "fold": fold,
                    }
                )
    return jobs


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    backend: str,
    model_sha256: str,
    role: str,
) -> None:
    expected_backend = validate_backend(backend)
    expected_digest = validate_model_sha256(model_sha256)
    if role not in ROLES:
        raise RuntimeError(f"unknown COCO role: {role}")
    if str(manifest.get("backend")) != expected_backend:
        raise RuntimeError("candidate backend does not match the training run")
    if validate_model_sha256(str(manifest.get("model_sha256", ""))) != expected_digest:
        raise RuntimeError("candidate model hash does not match the training run")
    if str(manifest.get("role")) != role:
        raise RuntimeError("candidate role does not match the training run")
    if int(manifest.get("images", -1)) != ROLES[role]:
        raise RuntimeError(f"candidate population is incomplete for {role}")
