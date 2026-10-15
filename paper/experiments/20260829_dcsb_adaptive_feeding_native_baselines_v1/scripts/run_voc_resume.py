#!/usr/bin/env python3
"""VOC-only recovery runner with the correct detector-hash binding chain.

VOC role manifests bind prediction hashes, while the frozen VOC protocol binds
those predictions to the detector weights.  COCO/Oregon manifests carry the
cloud hash directly.  This runner keeps the already completed COCO/Oregon
provenance untouched and records the extra VOC protocol in its own digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]


def _load_base() -> Any:
    import importlib.util
    import sys

    path = ROOT / "scripts/run_experiment.py"
    spec = importlib.util.spec_from_file_location("dcsb_af_voc_recovery_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
VOC_PROTOCOL_PATH = BASE.DATA.VOC_ROOT / "configs/protocol.json"


def voc_cloud_hash(protocol: Mapping[str, Any]) -> str:
    """Return the cloud-weight hash bound by the frozen VOC protocol."""
    return str(protocol["detectors"]["cloud"]["weights_sha256"])


def _source_digest(dataset: Any) -> tuple[str, dict[str, str]]:
    paths = [
        BASE.PROTOCOL_PATH,
        ROOT / "scripts/baseline_core.py",
        ROOT / "scripts/data_adapter.py",
        ROOT / "scripts/run_experiment.py",
        SCRIPT,
        VOC_PROTOCOL_PATH,
        *dataset.source_paths(),
    ]
    hashes = {str(path.resolve()): BASE.file_sha256(path) for path in paths}
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, hashes


def run_voc_preflight() -> dict[str, Any]:
    BASE.validate_protocol()
    dataset_name = "voc"
    output = BASE.RUNS / dataset_name / "preflight"
    dataset = BASE.DATA.FrozenDataset(dataset_name)
    expected = BASE.PROTOCOL["datasets"][dataset_name]
    source_protocol = BASE.read_json(VOC_PROTOCOL_PATH)
    observed_cloud_hash = voc_cloud_hash(source_protocol)
    expected_cloud_hash = str(expected["cloud_model_sha256"])
    if observed_cloud_hash != expected_cloud_hash:
        raise RuntimeError(
            f"VOC cloud detector hash changed: {observed_cloud_hash} != {expected_cloud_hash}"
        )

    populations, identities, manifests = {}, {}, {}
    required_prediction_keys = {"edge", "original"}
    for role_name in BASE.PROTOCOL["roles"]:
        role = dataset.role(role_name)
        samples = role["samples"]
        if len(samples) != int(expected["roles"][role_name]):
            raise RuntimeError(f"role population changed: voc/{role_name}")
        ids = [int(row["image_id"]) for row in samples]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate image ID: voc/{role_name}")
        if any(int(row["payload_bytes"]["original"]) <= 0 for row in samples):
            raise RuntimeError(f"invalid Original payload: voc/{role_name}")
        if set(role["actions"]) != required_prediction_keys:
            raise RuntimeError(f"external action projection changed: voc/{role_name}")
        manifest_sources = set(role["manifest"].get("source_sha256", {}))
        if not required_prediction_keys <= manifest_sources:
            raise RuntimeError(
                f"VOC role manifest does not bind Edge/Original predictions: {role_name}"
            )
        populations[role_name] = len(samples)
        identities[role_name] = {
            str(row.get("identity", row["image_id"])) for row in samples
        }
        manifests[role_name] = BASE.file_sha256(role["manifest_path"])

    overlaps = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(BASE.PROTOCOL["roles"])
        for right in BASE.PROTOCOL["roles"][index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"role identity leakage: voc/{overlaps}")

    import sklearn

    digest, hashes = _source_digest(dataset)
    report = {
        "complete": True,
        "dataset": dataset_name,
        "role_counts": populations,
        "identity_overlap": overlaps,
        "manifest_sha256": manifests,
        "source_digest": digest,
        "source_hashes": hashes,
        "cloud_model_sha256": expected_cloud_hash,
        "cloud_hash_binding": {
            "source": str(VOC_PROTOCOL_PATH.resolve()),
            "source_sha256": BASE.file_sha256(VOC_PROTOCOL_PATH),
            "field": "detectors.cloud.weights_sha256",
            "role_manifests_bind": "source_sha256.edge_and_original",
        },
        "payload_actions": ["edge", "original"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source_experiments_modified": False,
        "final_used_for_training_or_selection": False,
        "recovered_from_incorrect_direct_manifest_hash_assumption": True,
    }
    BASE.atomic_json(output / "report.json", report)
    BASE.atomic_json(
        output / "completion.json", {"complete": True, "source_digest": digest}
    )
    return report


def main() -> None:
    # Training and point stages call verify_preflight, so bind them to the
    # expanded VOC digest that includes this runner and the source protocol.
    BASE._source_digest = _source_digest
    run_voc_preflight()
    BASE.run_train("voc")
    report = BASE.run_point("voc")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

