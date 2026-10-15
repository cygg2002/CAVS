"""Pure contracts shared by the isolated Oregon Cloud-V2 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


CANONICAL_INFERENCE = {
    "imgsz": 640,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": True,
    "end2end": False,
}


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    roles = {str(name): int(count) for name, count in protocol["roles"].items()}
    dev_roles = tuple(map(str, protocol["dev_roles"]))
    final_role = str(protocol["final_role"])
    if set(roles) != {"fit", "calibration", "audit", "final"}:
        raise ValueError("role contract changed")
    if dev_roles != ("fit", "calibration", "audit"):
        raise ValueError("dev roles changed or final leaked into dev")
    if final_role != "final" or final_role in dev_roles:
        raise ValueError("final must remain sealed from dev")
    observed = {
        key: protocol["inference"][key] for key in CANONICAL_INFERENCE
    }
    if observed != CANONICAL_INFERENCE:
        raise ValueError(f"canonical inference contract changed: {observed}")
    if protocol.get("final_used_for_selection") is not False:
        raise ValueError("final cannot be used for selection")
    if protocol.get("final_claim") != "posthoc_detector_update_final_reuse":
        raise ValueError("Oregon Cloud-V2 final claim is not auditable")
    return {
        "roles": roles,
        "dev_roles": dev_roles,
        "final_role": final_role,
        "end2end": observed["end2end"],
    }


def stage_roles(protocol: Mapping[str, Any], stage: str) -> tuple[str, ...]:
    validate_protocol(protocol)
    if stage == "dev":
        return tuple(map(str, protocol["dev_roles"]))
    if stage == "final":
        return (str(protocol["final_role"]),)
    raise ValueError(stage)


def gpu_has_capacity(
    *, total_mib: int, used_mib: int, minimum_free_mib: int
) -> bool:
    return int(total_mib) - int(used_mib) >= int(minimum_free_mib)


def ensure_symlink(source: Path, target: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(
                f"symlink target drift: {target} -> {target.resolve()} != {source}"
            )
        return
    if target.exists():
        raise RuntimeError(f"refusing to replace existing path: {target}")
    target.symlink_to(source, target_is_directory=source.is_dir())

