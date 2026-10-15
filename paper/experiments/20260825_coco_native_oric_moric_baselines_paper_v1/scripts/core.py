"""Pure contracts for the native COCO external-baseline completion."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def original_action_map(samples: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    return {int(sample["image_id"]): "original" for sample in samples}


def validate_native_selection(selection: Mapping[int, str]) -> None:
    invalid = sorted({str(action) for action in selection.values()} - {"original"})
    if invalid:
        raise ValueError(f"native selection contains non-Original actions: {invalid}")


def expected_point_rows(protocol: Mapping[str, Any]) -> int:
    roles = 2
    operating_points = len(protocol["modes"]) * len(protocol["budgets"])
    seeded = 2 * (len(protocol["model_seeds"]) + 1)
    unseeded = 2
    return roles * operating_points * (seeded + unseeded)


def bootstrap_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for budget in map(float, protocol["budgets"]):
        for method in map(str, protocol["native_methods"]):
            jobs.append(
                {
                    "id": f"CAVS-OQR|vs|{method}|budget={budget:g}|image",
                    "left": "CAVS-OQR",
                    "right": method,
                    "budget": budget,
                    "unit": "image",
                    "kind": "nested_random" if method == "Random-Original" else "paired",
                }
            )
    return jobs


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol["runtime_domain"] != "pytorch_paper_cache":
        raise ValueError("native baseline completion must use the paper cache")
    if list(protocol["action_space"]) != ["edge", "original"]:
        raise ValueError("native baseline action space changed")
    if list(map(float, protocol["budgets"])) != [0.1, 0.25, 0.4]:
        raise ValueError("registered budgets changed")
    if int(protocol["bootstrap"]["random_seeds"]) != 50:
        raise ValueError("Random must use 50 policies")
