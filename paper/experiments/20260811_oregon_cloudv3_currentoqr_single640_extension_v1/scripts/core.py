"""Pure contracts for the Oregon single-ROI completion experiment."""

from __future__ import annotations

from typing import Any, Mapping


EXPECTED_ACTIONS = ("edge", "original", "q90", "current@640")
EXPECTED_COMPARATORS = ("CurrentActionValue-OQ", "CurrentActionValue-OQR12")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "oregon":
        raise ValueError("dataset must be Oregon")
    if tuple(protocol.get("action_space", ())) != EXPECTED_ACTIONS:
        raise ValueError("single ROI must be the COCO-matched current@640 action")
    if tuple(protocol.get("comparators", ())) != EXPECTED_COMPARATORS:
        raise ValueError("comparator contract changed")
    if protocol.get("roles") != {"fit": 9923, "calibration": 4961, "final": 4962}:
        raise ValueError("canonical Oregon population changed")
    if bool(protocol.get("confirmatory")) or bool(protocol.get("final_is_hidden")):
        raise ValueError("this extension is post-hoc, not confirmatory")
    if bool(protocol.get("final_used_for_training_or_selection")):
        raise ValueError("final may not select or train the method")


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    jobs = []
    folds: list[int | str] = [*range(int(protocol["fold_count"])), "full"]
    for seed in map(int, protocol["model_seeds"]):
        for fold in folds:
            token = "full" if fold == "full" else f"fold_{fold}"
            jobs.append(
                {
                    "id": f"{protocol['method']}|seed={seed}|{token}",
                    "method": str(protocol["method"]),
                    "seed": seed,
                    "fold": fold,
                }
            )
    return jobs


def expected_point_rows(protocol: Mapping[str, Any]) -> int:
    return 2 * len(protocol["budgets"]) * len(protocol["modes"])


def bootstrap_jobs(protocol: Mapping[str, Any]) -> int:
    return (
        len(protocol["comparators"])
        * len(protocol["budgets"])
        * len(protocol["bootstrap"]["units"])
    )


def validate_action_selection(
    selection: Mapping[int, str], protocol: Mapping[str, Any]
) -> None:
    allowed = set(map(str, protocol["action_space"])) - {"edge"}
    invalid = sorted({str(action) for action in selection.values()} - allowed)
    if invalid:
        raise ValueError(f"selection contains forbidden actions: {invalid}")
