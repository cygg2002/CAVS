#!/usr/bin/env python3
"""Verify immutable policy and sibling RKNN model identities before a formal run."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from board_client import ROOT, resolved, sha256, validate_policy_binding
from cavs_numpy import load_policy


def main() -> None:
    config = json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))
    policy = resolved(config["models"]["policy"])
    if sha256(policy) != config["models"]["policy_sha256"]:
        raise RuntimeError("policy manifest SHA-256 mismatch")
    loaded = load_policy(policy)
    label = "fp16"
    model = resolved(config["models"][label]["path"])
    if not model.is_file() or sha256(model) != config["models"][label]["sha256"]:
        raise RuntimeError(f"{label} RKNN SHA-256 mismatch")
    validate_policy_binding(loaded, label, config["models"][label]["sha256"])
    print(json.dumps({"status": "ok", "policy": loaded["policy_id"], "seeds": [item["seed"] for item in loaded["seeds_loaded"]], "models": [label]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
