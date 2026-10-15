from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rknn_cavs_contracts", ROOT / "scripts" / "contracts.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_protocol_has_two_isolated_backends_and_36_jobs() -> None:
    protocol = {"backends": ["fp16", "int8-heads"]}

    jobs = MODULE.training_jobs(protocol)

    assert len(jobs) == 36
    assert len({job["id"] for job in jobs}) == 36
    assert {job["backend"] for job in jobs} == {"fp16", "int8-heads"}


def test_role_manifest_is_canonical_and_role_scoped() -> None:
    assert MODULE.role_manifest("/tmp/canonical", "fit") == Path("/tmp/canonical/common/coco/fit/source_manifest.json")


def test_population_rejects_missing_or_duplicate_ids() -> None:
    MODULE.validate_population(range(9678), role="fit")
    for values in (range(9677), [*range(9677), 0]):
        try:
            MODULE.validate_population(values, role="fit")
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid frozen role population was accepted")


def test_candidate_manifest_rejects_cross_backend_model_hash() -> None:
    manifest = {"backend": "fp16", "model_sha256": "a" * 64, "role": "fit", "images": 9678}

    MODULE.validate_manifest(manifest, backend="fp16", model_sha256="a" * 64, role="fit")

    try:
        MODULE.validate_manifest(manifest, backend="int8-heads", model_sha256="a" * 64, role="fit")
    except RuntimeError as error:
        assert "backend" in str(error)
    else:
        raise AssertionError("cross-backend candidate manifest was accepted")


def test_candidate_manifest_requires_the_frozen_role_population() -> None:
    manifest = {"backend": "int8-heads", "model_sha256": "b" * 64, "role": "final", "images": 4999}

    try:
        MODULE.validate_manifest(manifest, backend="int8-heads", model_sha256="b" * 64, role="final")
    except RuntimeError as error:
        assert "population" in str(error)
    else:
        raise AssertionError("incomplete final candidate manifest was accepted")
