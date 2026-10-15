"""Pure contracts for the COCO Profile × Fusion factorial experiment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


CELL_ORDER = (
    "CAVS-Current-Reversible",
    "CAVS-Current-MAUR",
    "CAVS-Historical-Reversible",
    "CAVS-Historical-MAUR",
)


def normalize_oof_scores(
    matrix: np.ndarray, *, expected_images: int, expected_budgets: int
) -> np.ndarray:
    """Return image-major OOF scores after removing a redundant budget axis."""
    values = np.asarray(matrix)
    if values.ndim == 3:
        if values.shape[0] != int(expected_budgets):
            raise ValueError(f"unexpected OOF score shape: {values.shape}")
        if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
            raise ValueError("budget-conditioned OOF scores cannot be collapsed")
        values = values[0]
    if values.ndim != 2 or values.shape[0] != int(expected_images):
        raise ValueError(f"unexpected OOF score shape: {values.shape}")
    return values


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("dataset must be COCO")
    if protocol.get("roles") != {"fit": 9678, "calibration": 571, "final": 5000}:
        raise ValueError("canonical role counts changed")
    if list(protocol.get("profiles", [])) != ["current", "historical_cost"]:
        raise ValueError("Profile factor changed")
    if list(protocol.get("fusions", [])) != ["Reversible", "MAUR"]:
        raise ValueError("Fusion factor changed")
    if int(protocol.get("scale", 0)) != 640:
        raise ValueError("ROI inference scale changed")
    if tuple(protocol.get("cells", {})) != CELL_ORDER:
        raise ValueError("factorial cell order or population changed")
    expected = {
        "CAVS-Current-Reversible": ("current", "Reversible", True),
        "CAVS-Current-MAUR": ("current", "MAUR", False),
        "CAVS-Historical-Reversible": ("historical_cost", "Reversible", False),
        "CAVS-Historical-MAUR": ("historical_cost", "MAUR", False),
    }
    for method, (profile, fusion, reused) in expected.items():
        cell = protocol["cells"][method]
        observed = (
            str(cell["profile"]),
            str(cell["fusion"]),
            bool(cell["reused_anchor"]),
        )
        if observed != (profile, fusion, reused):
            raise ValueError(f"factorial cell changed: {method}")
    if list(protocol.get("base_actions", [])) != ["edge", "original", "q90"]:
        raise ValueError("base action space changed")
    if list(map(float, protocol.get("budgets", []))) != [0.1, 0.25, 0.4]:
        raise ValueError("budget grid changed")
    if list(protocol.get("modes", [])) != ["fixed_byte", "fixed_call"]:
        raise ValueError("evaluation modes changed")
    if int(protocol.get("fold_count", 0)) != 5:
        raise ValueError("fold count changed")
    if list(map(int, protocol.get("model_seeds", []))) != [20260809, 20260810, 20260811]:
        raise ValueError("model seeds changed")
    if int(protocol["cpu"]["training_workers"]) > 12:
        raise ValueError("training workers exceed isolation limit")
    if int(protocol["cpu"]["reward_workers"]) > 12:
        raise ValueError("reward workers exceed isolation limit")
    if int(protocol["cpu"]["evaluation_workers"]) > 12:
        raise ValueError("evaluation workers exceed isolation limit")
    if int(protocol["cpu"]["bootstrap_workers"]) > 12:
        raise ValueError("bootstrap workers exceed isolation limit")
    if protocol.get("detector_inference_required") is not False:
        raise ValueError("detector inference is forbidden")
    if protocol.get("final_used_for_training_gate_or_selection") is not False:
        raise ValueError("final cannot influence the experiment")
    if protocol.get("source_experiments_modified") is not False:
        raise ValueError("source experiments must remain read-only")


def action_space(protocol: Mapping[str, Any], method: str) -> list[str]:
    validate_protocol(protocol)
    cell = protocol["cells"][str(method)]
    return [
        *map(str, protocol["base_actions"]),
        f"{cell['profile']}@{int(protocol['scale'])}",
    ]


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    folds: list[int | str] = [*range(int(protocol["fold_count"])), "full"]
    output = []
    for method, cell in protocol["cells"].items():
        if bool(cell["reused_anchor"]):
            continue
        for seed in map(int, protocol["model_seeds"]):
            for fold in folds:
                token = "full" if fold == "full" else f"fold_{fold}"
                output.append(
                    {
                        "id": f"{method}|seed={seed}|{token}",
                        "method": method,
                        "profile": str(cell["profile"]),
                        "fusion": str(cell["fusion"]),
                        "action_space": action_space(protocol, method),
                        "seed": seed,
                        "fold": fold,
                    }
                )
    return output


def expected_new_training_jobs(protocol: Mapping[str, Any]) -> int:
    return len(training_jobs(protocol))


def expected_point_rows(protocol: Mapping[str, Any]) -> int:
    validate_protocol(protocol)
    return (
        3
        * len(protocol["cells"])
        * len(protocol["modes"])
        * len(protocol["budgets"])
    )


def evaluation_jobs(protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    validate_protocol(protocol)
    return [
        {
            "id": f"{point_role}|{method}",
            "point_role": point_role,
            "method": str(method),
        }
        for point_role in ("fit_oof", "calibration", "final")
        for method in protocol["cells"]
    ]


def bootstrap_contrasts() -> list[dict[str, Any]]:
    cr = "CAVS-Current-Reversible"
    cm = "CAVS-Current-MAUR"
    hr = "CAVS-Historical-Reversible"
    hm = "CAVS-Historical-MAUR"
    return [
        {"name": "fusion_at_current", "coefficients": {cm: 1.0, cr: -1.0}},
        {"name": "fusion_at_historical", "coefficients": {hm: 1.0, hr: -1.0}},
        {"name": "profile_at_reversible", "coefficients": {hr: 1.0, cr: -1.0}},
        {"name": "profile_at_maur", "coefficients": {hm: 1.0, cm: -1.0}},
        {
            "name": "interaction",
            "coefficients": {hm: 1.0, hr: -1.0, cm: -1.0, cr: 1.0},
        },
    ]


def expected_bootstrap_jobs(protocol: Mapping[str, Any]) -> int:
    validate_protocol(protocol)
    return len(protocol["budgets"]) * len(bootstrap_contrasts())


def cell_passes_gate(
    deltas: Sequence[float], config: Mapping[str, Any]
) -> bool:
    values = list(map(float, deltas))
    minimum = float(config["minimum_improvement"])
    floor = float(config["noninferiority"])
    required = int(config["minimum_improved_budgets"])
    return sum(value >= minimum for value in values) >= required and all(
        value >= floor for value in values
    )


def factorial_effects(values: Mapping[str, float]) -> dict[str, float]:
    missing = set(CELL_ORDER) - set(values)
    if missing:
        raise ValueError(f"missing factorial cells: {sorted(missing)}")
    cr, cm, hr, hm = (float(values[name]) for name in CELL_ORDER)
    fusion_current = cm - cr
    fusion_historical = hm - hr
    profile_reversible = hr - cr
    profile_maur = hm - cm
    return {
        "fusion_at_current": fusion_current,
        "fusion_at_historical": fusion_historical,
        "profile_at_reversible": profile_reversible,
        "profile_at_maur": profile_maur,
        "fusion_main_effect": 0.5 * (fusion_current + fusion_historical),
        "profile_main_effect": 0.5 * (profile_reversible + profile_maur),
        "interaction": fusion_historical - fusion_current,
    }
