#!/usr/bin/env python3
"""Run resumable detector inference while enforcing the registered-final seal."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
SOURCE = (
    REPO
    / "paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/infer_original.py"
)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(REPO))

# Pin the imported helper names to this experiment before loading the reviewed,
# resumable detector implementation from the preceding registered protocol.
import paper_core as local_paper_core  # noqa: E402
import run_paper as local_run_paper  # noqa: E402

sys.modules["paper_core"] = local_paper_core
sys.modules["run_paper"] = local_run_paper


def _load_inference_implementation():
    spec = importlib.util.spec_from_file_location("_registered_infer_original", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference implementation: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.REPO = REPO
    module.RUNS = ROOT / "runs"
    module.PROTOCOL_PATH = ROOT / "configs/protocol.json"
    module.__file__ = str(Path(__file__).resolve())
    return module


def _requested_role(argv: list[str]) -> str:
    if "--role" not in argv:
        return "all"
    index = argv.index("--role")
    if index + 1 >= len(argv):
        raise ValueError("--role requires a value")
    return str(argv[index + 1])


def _enforce_role_seal(role: str) -> None:
    if role == "all":
        raise RuntimeError("Role 'all' is forbidden because registered final is audit-gated")
    if role != "registered_final":
        return
    runs = ROOT / "runs"
    audit_path = runs / "audit/completion.json"
    access_path = runs / "registered_final/accessed.json"
    if not audit_path.is_file() or not access_path.is_file():
        raise RuntimeError("Registered-final inference is sealed until open_final succeeds")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError("Registered-final inference is forbidden after a failed audit gate")


def main() -> None:
    _enforce_role_seal(_requested_role(sys.argv[1:]))
    _load_inference_implementation().main()


if __name__ == "__main__":
    main()
