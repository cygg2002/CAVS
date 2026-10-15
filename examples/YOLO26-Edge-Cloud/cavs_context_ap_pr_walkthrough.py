#!/usr/bin/env python3
"""Detection-level AP and CAVS contextual-reward walkthrough.

This is a deliberately small, one-class teaching example.  It keeps the paper's
formal context size (32 images) and COCO-style 101-point interpolation over IoU
thresholds 0.50:0.05:0.95.  The detections are synthetic and are not reported
experimental measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


CONTEXT_IMAGE_IDS = tuple(f"c{index:02d}" for index in range(1, 33))
TARGET_IMAGE_ID = "target"
CONTEXT_SIZE = len(CONTEXT_IMAGE_IDS)
GROUND_TRUTH_COUNT = 10  # g1..g8 in context, g9..g10 in the target image.
IOU_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(10))


@dataclass(frozen=True)
class Prediction:
    image_id: str
    score: float
    best_gt_id: str | None
    best_iou: float


@dataclass(frozen=True)
class PrRow:
    rank: int
    prediction: Prediction
    outcome: str
    cumulative_tp: int
    cumulative_fp: int
    recall: float
    precision: float


@dataclass(frozen=True)
class Evaluation:
    rows: tuple[PrRow, ...]
    interpolated_precision: tuple[float, ...]
    ap: float


# Eight context images contain one GT traffic light each.  The remaining 24
# context images have no GT of this class.  c09/c10/c11 contribute false alarms.
CONTEXT_EDGE_PREDICTIONS = (
    Prediction("c01", 0.99, "g1", 0.92),
    Prediction("c09", 0.97, None, 0.00),
    Prediction("c02", 0.95, "g2", 0.87),
    Prediction("c03", 0.90, "g3", 0.81),
    Prediction("c10", 0.88, None, 0.00),
    Prediction("c04", 0.82, "g4", 0.78),
    Prediction("c05", 0.74, "g5", 0.72),
    Prediction("c06", 0.68, "g6", 0.65),
    Prediction("c11", 0.62, None, 0.00),
    Prediction("c07", 0.55, "g7", 0.59),
    Prediction("c08", 0.45, "g8", 0.53),
)


TARGET_PREDICTIONS = {
    # g10 is missed.  The second g9 prediction is a duplicate and therefore FP.
    "edge": (
        Prediction(TARGET_IMAGE_ID, 0.86, "g9", 0.68),
        Prediction(TARGET_IMAGE_ID, 0.60, "g9", 0.62),
    ),
    "original": (
        Prediction(TARGET_IMAGE_ID, 0.93, "g9", 0.90),
        Prediction(TARGET_IMAGE_ID, 0.79, "g10", 0.83),
        Prediction(TARGET_IMAGE_ID, 0.40, None, 0.00),
    ),
    "q90": (
        Prediction(TARGET_IMAGE_ID, 0.91, "g9", 0.88),
        Prediction(TARGET_IMAGE_ID, 0.76, None, 0.00),
        Prediction(TARGET_IMAGE_ID, 0.70, "g10", 0.74),
    ),
    "roi": (
        Prediction(TARGET_IMAGE_ID, 0.89, "g9", 0.86),
        Prediction(TARGET_IMAGE_ID, 0.67, "g10", 0.70),
    ),
}


def evaluate_at_iou(
    target_predictions: Sequence[Prediction], *, iou_threshold: float
) -> Evaluation:
    """Evaluate one class at one IoU threshold with COCO's 101 recall samples."""
    predictions = sorted(
        (*CONTEXT_EDGE_PREDICTIONS, *target_predictions),
        key=lambda prediction: -prediction.score,
    )
    matched_gt: set[str] = set()
    cumulative_tp = 0
    cumulative_fp = 0
    rows: list[PrRow] = []

    for rank, prediction in enumerate(predictions, start=1):
        is_tp = (
            prediction.best_gt_id is not None
            and prediction.best_iou >= iou_threshold
            and prediction.best_gt_id not in matched_gt
        )
        if is_tp:
            matched_gt.add(str(prediction.best_gt_id))
            cumulative_tp += 1
            outcome = "TP"
        else:
            cumulative_fp += 1
            outcome = "FP"
        rows.append(
            PrRow(
                rank=rank,
                prediction=prediction,
                outcome=outcome,
                cumulative_tp=cumulative_tp,
                cumulative_fp=cumulative_fp,
                recall=cumulative_tp / GROUND_TRUTH_COUNT,
                precision=cumulative_tp / (cumulative_tp + cumulative_fp),
            )
        )

    # p_interp(r) = max precision observed at any recall r' >= r.
    interpolated = tuple(
        max(
            (row.precision for row in rows if row.recall >= recall_index / 100),
            default=0.0,
        )
        for recall_index in range(101)
    )
    return Evaluation(tuple(rows), interpolated, sum(interpolated) / 101)


def ap50_95(action: str) -> tuple[tuple[float, ...], float]:
    values = tuple(
        evaluate_at_iou(TARGET_PREDICTIONS[action], iou_threshold=threshold).ap
        for threshold in IOU_THRESHOLDS
    )
    return values, sum(values) / len(values)


def contextual_reward(action: str) -> float:
    """Paper Eq. (8), using AP50-95 as AP and Edge as the baseline action."""
    _, edge_ap = ap50_95("edge")
    _, action_ap = ap50_95(action)
    return (CONTEXT_SIZE + 1) * (action_ap - edge_ap)


def print_pr_table(action: str) -> None:
    evaluation = evaluate_at_iou(TARGET_PREDICTIONS[action], iou_threshold=0.50)
    print(f"\n{action.upper()} @ IoU=0.50")
    print("rank score source outcome TP FP recall precision")
    for row in evaluation.rows:
        source = "target" if row.prediction.image_id == TARGET_IMAGE_ID else "context"
        print(
            f"{row.rank:>4} {row.prediction.score:>5.2f} {source:>7} "
            f"{row.outcome:>7} {row.cumulative_tp:>2} {row.cumulative_fp:>2} "
            f"{row.recall:>6.2f} {row.precision:>9.6f}"
        )
    print(f"AP50 = {evaluation.ap:.9f}")


def main() -> None:
    assert CONTEXT_SIZE == 32
    print_pr_table("edge")
    print_pr_table("original")

    print("\nAP at each IoU threshold")
    print("action    " + " ".join(f"{threshold:.2f}" for threshold in IOU_THRESHOLDS) + "  mean")
    expected = {
        "edge": 0.37233706887172235,
        "original": 0.4776910383346027,
        "q90": 0.44342390282984345,
        "roi": 0.4551427120734052,
    }
    for action in TARGET_PREDICTIONS:
        values, mean = ap50_95(action)
        print(
            f"{action:<9} "
            + " ".join(f"{value:.4f}" for value in values)
            + f"  {mean:.6f}"
        )
        assert math.isclose(mean, expected[action], rel_tol=0.0, abs_tol=1e-12)

    print("\nCAVS rewards from Eq. (8)")
    for action in ("original", "q90", "roi"):
        print(f"r[{action}] = {contextual_reward(action):.9f}")
    assert math.isclose(contextual_reward("original"), 3.476680992275051, abs_tol=1e-12)
    print("\nAll AP and contextual-reward assertions passed.")


if __name__ == "__main__":
    main()
