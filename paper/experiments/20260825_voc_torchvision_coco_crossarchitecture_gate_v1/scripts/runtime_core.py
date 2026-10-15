"""Small filesystem helpers for the Torchvision VOC gate experiment."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def gpu_gate_open(
    free_mib: int, utilization_percent: int, contract: Mapping[str, Any]
) -> bool:
    return int(free_mib) >= int(contract["minimum_free_gpu_mib"]) and int(
        utilization_percent
    ) <= int(contract["maximum_utilization_percent"])


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def source_role_paths(
    experiment_root: str | Path, protocol: Mapping[str, Any], role: str
) -> dict[str, Path]:
    value = str(role)
    if value not in set(map(str, protocol.get("allowed_roles", ()))):
        raise ValueError(f"forbidden data role: {value}")
    source = (Path(experiment_root) / str(protocol["source_experiment"])).resolve()
    role_root = source / f"runs/data/roles/{value}"
    return {
        "source_experiment": source,
        "role_root": role_root,
        "gt": role_root / "gt.json",
        "images": role_root / "images.txt",
        "completion": role_root / "completion.json",
        "metric": source / "scripts/voc_metric.py",
    }


def load_source_role(
    experiment_root: str | Path,
    protocol: Mapping[str, Any],
    role: str,
    *,
    require_images: bool = True,
) -> dict[str, Any]:
    paths = source_role_paths(experiment_root, protocol, role)
    contract = protocol.get("roles", {}).get(str(role))
    if not isinstance(contract, Mapping):
        raise ValueError(f"missing frozen role contract: {role}")
    if file_sha256(paths["gt"]) != str(contract["gt_sha256"]):
        raise RuntimeError(f"ground-truth hash changed: {role}")
    if file_sha256(paths["images"]) != str(contract["images_sha256"]):
        raise RuntimeError(f"image-list hash changed: {role}")
    payload = json.loads(paths["gt"].read_text(encoding="utf-8"))
    images = [dict(row) for row in payload.get("images", ())]
    if len(images) != int(contract["images"]):
        raise RuntimeError(f"frozen image count changed: {role}")
    identities = [str(row["identity"]) for row in images]
    observed_identity_hash = text_sha256("\n".join(sorted(identities)))
    if observed_identity_hash != str(contract["identity_sha256"]):
        raise RuntimeError(f"identity hash changed: {role}")
    listed = [line.strip() for line in paths["images"].read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_paths = [str(row["absolute_path"]) for row in images]
    if listed != expected_paths:
        raise RuntimeError(f"image-list order changed: {role}")
    if require_images:
        missing = [path for path in expected_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} source images for {role}: {missing[0]}")
    by_image: dict[int, set[int]] = defaultdict(set)
    for annotation in payload.get("annotations", ()):
        by_image[int(annotation["image_id"])].add(int(annotation["category_id"]))
    for image in images:
        image["category_ids"] = sorted(by_image[int(image["id"])])
    return {
        "role": str(role),
        "paths": paths,
        "gt": payload,
        "images": images,
        "categories": [dict(row) for row in payload.get("categories", ())],
    }


def read_checkpoint(
    path: str | Path,
    *,
    action: str,
    expected_ids: set[int],
    protocol_sha256: str | None = None,
) -> dict[int, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return {}
    output = {}
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid checkpoint JSON at line {number}: {source}") from error
            image_id = int(row["image_id"])
            if image_id in output:
                raise RuntimeError(f"duplicate checkpoint image: {image_id}")
            if image_id not in expected_ids:
                raise RuntimeError(f"checkpoint contains unexpected image: {image_id}")
            if str(row.get("action")) != str(action):
                raise RuntimeError(f"checkpoint action changed: {row.get('action')}/{action}")
            if protocol_sha256 is not None and str(row.get("protocol_sha256")) != str(
                protocol_sha256
            ):
                raise RuntimeError(f"checkpoint protocol changed: {source}")
            output[image_id] = row
    return output


def read_protocol_bound_json(
    path: str | Path, protocol_sha256: str
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if str(payload.get("protocol_sha256")) != str(protocol_sha256):
        raise RuntimeError(f"artifact protocol changed: {source}")
    return payload


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
