"""Hard protocol contracts for the canonical COCO 9,678 comparison."""

from __future__ import annotations

from collections import defaultdict
import copy
from typing import Any, Mapping, Sequence


EXPECTED_ROLES = {"fit": 9678, "calibration": 571, "registered_final": 5000}
EXPECTED_METHODS = (
    "ORIC-AP50",
    "MORIC-AP50",
    "ORIC-AP5095",
    "MORIC-AP5095",
    "Current-BudgetSet",
    "SetValue",
    "DualHead",
)
_BOOTSTRAP_STATE: dict[str, Any] = {}


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco_canonical_train9678_val2017_5000":
        raise ValueError("Unexpected canonical COCO dataset contract")
    roles = dict(protocol.get("roles", {}))
    if roles.get("fit") != 9678:
        raise ValueError("Canonical training must contain exactly 9,678 fit images")
    if roles != EXPECTED_ROLES:
        raise ValueError("Canonical role counts changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("Canonical budgets changed")
    if tuple(protocol.get("modes", ())) != ("fixed_call", "fixed_byte"):
        raise ValueError("Canonical evaluation modes changed")
    if tuple(protocol.get("methods", ())) != EXPECTED_METHODS:
        raise ValueError("Canonical method matrix changed")
    if tuple(protocol.get("actions", ())) != ("edge", "original"):
        raise ValueError("This comparison must remain Edge/Original-only")
    if protocol.get("primary_mode") != "fixed_byte":
        raise ValueError("Fixed-byte must remain primary")
    if protocol.get("primary_candidate") != "DualHead":
        raise ValueError("DualHead must remain the primary candidate")
    if protocol.get("fit_only_training") is not True:
        raise ValueError("Training must remain fit-only")
    if protocol.get("calibration_used_for_training") is not False:
        raise ValueError("Calibration cannot train a model")
    if protocol.get("registered_final_used_for_selection") is not False:
        raise ValueError("val2017 cannot select models or comparators")
    reward = protocol.get("reward", {})
    if int(reward.get("context_size", -1)) != 32:
        raise ValueError("Contextual reward size changed")


def validate_role_populations(roles: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    observed_counts = {name: len(rows) for name, rows in roles.items()}
    if observed_counts != EXPECTED_ROLES:
        raise ValueError(f"Canonical role populations changed: {observed_counts}")
    expected_split = {
        "fit": "train2017",
        "calibration": "train2017",
        "registered_final": "val2017",
    }
    populations: dict[str, set[int]] = {}
    for role, rows in roles.items():
        image_ids = [int(row["image_id"]) for row in rows]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"Duplicate image ID in {role}")
        split = expected_split[role]
        if any(split not in str(row.get("identity", "")) for row in rows):
            raise ValueError(f"{role} must contain only {split} images")
        populations[role] = set(image_ids)
    if populations["fit"] & populations["calibration"]:
        raise ValueError("fit and calibration overlap")


def validate_reward_population(
    reward_ids: set[int],
    fit_ids: set[int],
    calibration_ids: set[int],
    final_ids: set[int],
) -> None:
    if set(reward_ids) != set(fit_ids):
        raise ValueError("Reward IDs must exactly equal canonical fit IDs")
    if set(reward_ids) & set(calibration_ids):
        raise ValueError("Calibration IDs leaked into rewards")
    # COCO train2017 and val2017 may reuse numeric IDs. Role isolation is also
    # enforced by the source split path, so only disjoint IDs can be checked here.
    disjoint_final = set(final_ids) - set(fit_ids)
    if set(reward_ids) & disjoint_final:
        raise ValueError("val2017 IDs leaked into rewards")


def configure_bootstrap(
    ground_truth: Mapping[str, Any],
    image_ids: Sequence[int],
    candidate_predictions: Sequence[Mapping[str, Any]],
    comparator_predictions: Sequence[Mapping[str, Any]],
) -> None:
    """Install read-only bootstrap inputs before forking worker processes."""
    _BOOTSTRAP_STATE.clear()
    _BOOTSTRAP_STATE.update(
        {
            "ground_truth": ground_truth,
            "image_ids": list(map(int, image_ids)),
            "candidate_predictions": list(candidate_predictions),
            "comparator_predictions": list(comparator_predictions),
        }
    )


def clear_bootstrap() -> None:
    _BOOTSTRAP_STATE.clear()


def _bootstrap_remap(
    ground_truth: Mapping[str, Any],
    candidate_predictions: Sequence[Mapping[str, Any]],
    comparator_predictions: Sequence[Mapping[str, Any]],
    sampled_image_ids: Sequence[int],
) -> tuple[dict, list[dict], list[dict]]:
    images = {int(row["id"]): row for row in ground_truth["images"]}
    annotations: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ground_truth["annotations"]:
        annotations[int(row["image_id"])].append(row)
    prediction_groups = []
    for predictions in (candidate_predictions, comparator_predictions):
        groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in predictions:
            groups[int(row["image_id"])].append(row)
        prediction_groups.append(groups)
    remapped_gt = {
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(ground_truth["categories"]),
    }
    remapped_predictions: list[list[dict]] = [[], []]
    annotation_id = 1
    for new_id, source_id in enumerate(map(int, sampled_image_ids), start=1):
        image = copy.deepcopy(images[source_id])
        image["id"] = new_id
        remapped_gt["images"].append(image)
        for source in annotations[source_id]:
            row = copy.deepcopy(source)
            row["id"] = annotation_id
            row["image_id"] = new_id
            annotation_id += 1
            remapped_gt["annotations"].append(row)
        for target, groups in zip(remapped_predictions, prediction_groups):
            for source in groups[source_id]:
                row = copy.deepcopy(source)
                row["image_id"] = new_id
                target.append(row)
    return remapped_gt, remapped_predictions[0], remapped_predictions[1]


def _in_memory_ap(ground_truth: dict, predictions: list[dict]) -> float:
    from faster_coco_eval import COCO, COCOeval_faster

    annotation = COCO()
    annotation.dataset = ground_truth
    annotation.createIndex()
    result = annotation.loadRes(predictions)
    evaluator = COCOeval_faster(
        annotation,
        result,
        iouType="bbox",
        print_function=lambda *args, **kwargs: None,
    )
    evaluator.params.imgIds = list(range(1, len(ground_truth["images"]) + 1))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats_as_dict["AP_all"])


def bootstrap_worker(item: tuple[int, Any]) -> tuple[int, float]:
    """Picklable paired-bootstrap worker used by the canonical runner."""
    if not _BOOTSTRAP_STATE:
        raise RuntimeError("Bootstrap state is not configured")
    replicate, indices = item
    source_ids = [
        _BOOTSTRAP_STATE["image_ids"][int(index)] for index in indices
    ]
    ground_truth, candidate, comparator = _bootstrap_remap(
        _BOOTSTRAP_STATE["ground_truth"],
        _BOOTSTRAP_STATE["candidate_predictions"],
        _BOOTSTRAP_STATE["comparator_predictions"],
        source_ids,
    )
    delta = _in_memory_ap(ground_truth, candidate) - _in_memory_ap(
        ground_truth, comparator
    )
    return int(replicate), float(delta)
