#!/usr/bin/env python3
"""Five-image numerical walkthrough for the CAVS paper protocol.

The payload sizes and predicted action values below are copied from the formal
RKNN-FP16 online run.  The context-AP values are deliberately small teaching
values: COCO val2017 is never reused as fit data in the paper experiment.

This script demonstrates, with executable arithmetic:

1. deterministic fold-contained context sampling;
2. contextual rewards (paper Eq. 8);
3. reward standardization and the regression-ranking loss (paper Eq. 9);
4. fixed-byte full-information scheduling (paper Eq. 10); and
5. zero-lookahead primal-dual scheduling with prefix safety (paper Eq. 11).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


CONTEXT_SIZE = 32
ALPHA = 0.5
ACTION_ORDER = ("edge", "original", "q90", "current@640")
REPO = Path(__file__).resolve().parents[2]
IMAGE_ROOT = (
    REPO
    / "paper/experiments/20260816_coco_rknn_cavs_oqr_deployment_retrain_v1"
    / "runs/usb_snapshot_20260816/rk3588/yolo26n_coco_val"
    / "coco_val2017/images"
)


@dataclass(frozen=True)
class Frame:
    image_id: int
    file_name: str
    description: str
    payload_bytes: Mapping[str, int]
    predicted_value: Mapping[str, float]
    # Pedagogical targets only; these are not claimed as formal val2017 rewards.
    contextual_reward: Mapping[str, float]
    context_edge_ap: float

    @property
    def image_path(self) -> Path:
        return IMAGE_ROOT / self.file_name


# The arrival order is intentional: it makes Edge, Original, Q90 and ROI all
# appear in one short 25%-budget online trace.  The formal paper trace itself is
# sorted by image ID and contains all 5,000 COCO val2017 images.
FRAMES = (
    Frame(
        image_id=285,
        file_name="000000000285.jpg",
        description="close-up brown bear",
        payload_bytes={"edge": 0, "original": 335_861, "q90": 169_904},
        predicted_value={"edge": 0.0, "original": 0.0389698992, "q90": 0.0371610111},
        contextual_reward={"original": 0.040, "q90": 0.035},
        context_edge_ap=0.405,
    ),
    Frame(
        image_id=8762,
        file_name="000000008762.jpg",
        description="night intersection and traffic lights",
        payload_bytes={
            "edge": 0,
            "original": 48_914,
            "q90": 36_653,
            "current@640": 18_962,
        },
        predicted_value={
            "edge": 0.0,
            "original": 0.2356141022,
            "q90": 0.2101788347,
            "current@640": 0.1434427593,
        },
        contextual_reward={"original": 0.240, "q90": 0.210, "current@640": 0.140},
        context_edge_ap=0.410,
    ),
    Frame(
        image_id=785,
        file_name="000000000785.jpg",
        description="skier on a snowy slope",
        payload_bytes={"edge": 0, "original": 133_674, "q90": 60_462},
        predicted_value={"edge": 0.0, "original": 0.1373895983, "q90": 0.1281077586},
        contextual_reward={"original": 0.140, "q90": 0.130},
        context_edge_ap=0.395,
    ),
    Frame(
        image_id=5586,
        file_name="000000005586.jpg",
        description="tennis match",
        payload_bytes={
            "edge": 0,
            "original": 36_367,
            "q90": 21_529,
            "current@640": 11_681,
        },
        predicted_value={
            "edge": 0.0,
            "original": 0.2631093718,
            "q90": 0.2375691543,
            "current@640": 0.2307741218,
        },
        contextual_reward={"original": 0.260, "q90": 0.240, "current@640": 0.230},
        context_edge_ap=0.420,
    ),
    Frame(
        image_id=632,
        file_name="000000000632.jpg",
        description="bedroom with bed, plants and bookcase",
        payload_bytes={
            "edge": 0,
            "original": 155_667,
            "q90": 121_067,
            "current@640": 75_179,
        },
        predicted_value={
            "edge": 0.0,
            "original": 0.3208358216,
            "q90": 0.2987714489,
            "current@640": 0.2435817714,
        },
        contextual_reward={"original": 0.320, "q90": 0.300, "current@640": 0.245},
        context_edge_ap=0.400,
    ),
)


def load_module(name: str, path: Path) -> ModuleType:
    """Load one repository module without requiring package installation."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves forward annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_context(
    fold_image_ids: Sequence[int], *, target_id: int, fold_index: int, context_seed: int
) -> list[int]:
    """Reproduce the paper's SHA-256 fold-contained sampling protocol."""
    candidates = sorted(int(value) for value in fold_image_ids if int(value) != int(target_id))
    count = min(CONTEXT_SIZE, len(candidates))
    token = hashlib.sha256(
        f"{int(context_seed)}:{int(fold_index)}:{int(target_id)}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(token[:8], "big"))
    return sorted(rng.sample(candidates, count))


def contextual_ap(action_edge_ap: float, reward: float) -> float:
    """Invert Eq. (8) to make its numerical example easy to inspect."""
    return action_edge_ap + reward / (CONTEXT_SIZE + 1)


def reward_from_ap(action_ap: float, edge_ap: float) -> float:
    """Paper Eq. (8): (|C_i|+1) times the contextual AP difference."""
    return (CONTEXT_SIZE + 1) * (action_ap - edge_ap)


def training_mean_std(frames: Sequence[Frame]) -> tuple[float, float]:
    """Compute population statistics over raw cloud-action rewards only."""
    rewards = [
        float(value)
        for frame in frames
        for value in frame.contextual_reward.values()
    ]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    return mean, std


def smooth_l1(value: float) -> float:
    """PyTorch SmoothL1 with beta=1, written explicitly."""
    absolute = abs(value)
    return 0.5 * absolute * absolute if absolute < 1.0 else absolute - 0.5


def joint_loss(frames: Sequence[Frame], *, alpha: float = ALPHA) -> dict[str, float | int]:
    """Evaluate the exact structure of paper Eq. (9) on the five images."""
    mean, std = training_mean_std(frames)
    regression_sum = 0.0
    observation_count = 0
    ranking_numerator = 0.0
    ranking_weight = 0.0
    pair_count = 0

    for frame in frames:
        normalized: list[tuple[str, float, float]] = []
        for action in ACTION_ORDER:
            if action == "edge":
                raw_target = raw_prediction = 0.0
            elif action in frame.contextual_reward:
                raw_target = float(frame.contextual_reward[action])
                raw_prediction = float(frame.predicted_value[action])
            else:
                continue
            target = (raw_target - mean) / std
            prediction = (raw_prediction - mean) / std
            regression_sum += smooth_l1(prediction - target)
            observation_count += 1
            normalized.append((action, target, prediction))

        # Upper-triangle pairs in the fixed action order.
        for left in range(len(normalized)):
            for right in range(left + 1, len(normalized)):
                _, target_a, prediction_a = normalized[left]
                _, target_b, prediction_b = normalized[right]
                difference = target_a - target_b
                if abs(difference) <= 1e-8:
                    continue
                sign = 1.0 if difference > 0.0 else -1.0
                weight = abs(difference)
                ranking_numerator += weight * math.log1p(
                    math.exp(-sign * (prediction_a - prediction_b))
                )
                ranking_weight += weight
                pair_count += 1

    regression = regression_sum / observation_count
    ranking = ranking_numerator / ranking_weight if ranking_weight else 0.0
    return {
        "mu_train": mean,
        "sigma_train": std,
        "observations": observation_count,
        "ranking_pairs": pair_count,
        "regression_loss": regression,
        "ranking_loss": ranking,
        "joint_loss": alpha * regression + (1.0 - alpha) * ranking,
    }


def fixed_byte_candidates(frames: Sequence[Frame]) -> list[dict[str, Any]]:
    return [
        {
            "frame_id": frame.image_id,
            "payload_bytes": dict(frame.payload_bytes),
            "scores": dict(frame.predicted_value),
        }
        for frame in frames
    ]


def exact_tiny_mckp(frames: Sequence[Frame], budget_bytes: float) -> dict[str, Any]:
    """Brute force is only practical here because the example has five images."""
    action_lists = [
        [action for action in ACTION_ORDER if action in frame.payload_bytes]
        for frame in frames
    ]
    best: tuple[float, int, tuple[str, ...]] | None = None
    for actions in product(*action_lists):
        cost = sum(frame.payload_bytes[action] for frame, action in zip(frames, actions))
        value = sum(frame.predicted_value[action] for frame, action in zip(frames, actions))
        if cost <= budget_bytes and (
            best is None or (value, -cost, actions) > (best[0], -best[1], best[2])
        ):
            best = (value, cost, actions)
    if best is None:
        raise RuntimeError("Edge makes the tiny MCKP feasible, so this cannot happen")
    return {"total_value": best[0], "total_cost": best[1], "actions": best[2]}


def online_trace(frames: Sequence[Frame]) -> list[dict[str, Any]]:
    """Call the repository's exact Eq. (11) implementation."""
    online = load_module(
        "cavs_walkthrough_online_control",
        REPO
        / "paper/experiments/20260828_coco_rknn_fp16_online_hil_final_v2"
        / "board_bundle/scripts/online_control.py",
    )
    controller = online.PrimalDualByteController(
        budget=Fraction(1, 4),
        initial_price=0.30,
        step_size=0.0015,
        minimum_price=0.0,
        maximum_price=5.0,
        median_original_bytes=150_008,
        action_ids=ACTION_ORDER,
    )
    rows: list[dict[str, Any]] = []
    for position, frame in enumerate(frames, start=1):
        cumulative = controller.cumulative_original_bytes + frame.payload_bytes["original"]
        capacity = cumulative // 4
        remaining_before = capacity - controller.spent_bytes
        feasible = [
            action
            for action in ACTION_ORDER
            if action in frame.payload_bytes
            and controller.spent_bytes + frame.payload_bytes[action] <= capacity
        ]
        net_values = {
            action: frame.predicted_value[action]
            - controller.current_price * frame.payload_bytes[action] / 150_008.0
            for action in feasible
        }
        decision = controller.decide(
            frame.image_id, frame.payload_bytes, frame.predicted_value
        )
        rows.append(
            {
                "position": position,
                "image_id": frame.image_id,
                "capacity": capacity,
                "remaining_before": remaining_before,
                "feasible": feasible,
                "net_values": net_values,
                **asdict(decision),
            }
        )
    return rows


def print_input_table(frames: Sequence[Frame]) -> None:
    print("\n=== 1. Five real inputs and measured deployment candidates ===")
    for frame in frames:
        actions = ", ".join(
            f"{action}={frame.payload_bytes[action]} B / v={frame.predicted_value[action]:.6f}"
            for action in ACTION_ORDER
            if action in frame.payload_bytes
        )
        print(f"{frame.image_id:>5} | {frame.description:<42} | {actions}")
        print(f"      image: {frame.image_path}")


def print_reward_table(frames: Sequence[Frame]) -> None:
    print("\n=== 2. Pedagogical context-AP rewards (Eq. 8) ===")
    for frame in frames:
        for action, target in frame.contextual_reward.items():
            action_ap = contextual_ap(frame.context_edge_ap, target)
            reconstructed = reward_from_ap(action_ap, frame.context_edge_ap)
            print(
                f"{frame.image_id:>5} {action:<11} | edge AP={frame.context_edge_ap:.6f} "
                f"action AP={action_ap:.6f} | r=33*delta={reconstructed:.6f}"
            )


def main() -> None:
    print(__doc__.strip())
    print_input_table(FRAMES)

    # A miniature fold only demonstrates deterministic sampling.  The formal
    # experiment uses the full fit split and a 32-image context per target.
    miniature_fold = [frame.image_id for frame in FRAMES] + list(range(10_000, 10_035))
    context = frozen_context(
        miniature_fold, target_id=FRAMES[0].image_id, fold_index=0, context_seed=20_260_806
    )
    repeated = frozen_context(
        miniature_fold, target_id=FRAMES[0].image_id, fold_index=0, context_seed=20_260_806
    )
    assert context == repeated and FRAMES[0].image_id not in context and len(context) == 32
    print("\nFrozen context for image 285 (first eight IDs):", context[:8])

    print_reward_table(FRAMES)
    losses = joint_loss(FRAMES)
    print("\n=== 3. Standardization and joint loss (Eq. 9) ===")
    print(json.dumps(losses, indent=2, ensure_ascii=False))

    scheduler = load_module(
        "cavs_walkthrough_fixed_scheduler", Path(__file__).with_name("paper_budget_scheduler.py")
    )
    total_original = sum(frame.payload_bytes["original"] for frame in FRAMES)
    capacity = 0.25 * total_original
    plan = scheduler.schedule_fixed_byte(fixed_byte_candidates(FRAMES), 0.25)
    exact = exact_tiny_mckp(FRAMES, capacity)
    print("\n=== 4. Full-information fixed-byte reference (Eq. 10) ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print("Five-image brute-force optimum (comparison only):")
    print(json.dumps(exact, indent=2, ensure_ascii=False))

    trace = online_trace(FRAMES)
    print("\n=== 5. Zero-lookahead 25% online trace (Eq. 11) ===")
    for row in trace:
        compact = {
            "t": row["position"],
            "image_id": row["image_id"],
            "prefix_capacity": row["capacity"],
            "remaining_before": row["remaining_before"],
            "feasible": row["feasible"],
            "net_values": {key: round(value, 6) for key, value in row["net_values"].items()},
            "selected": row["action"],
            "selected_bytes": row["payload_bytes"],
            "spent_after": row["spent_bytes"],
            "remaining_after": row["credit_floor_bytes"],
            "price_before": round(row["price_before"], 9),
            "price_after": round(row["price_after"], 9),
        }
        print(json.dumps(compact, ensure_ascii=False))

    selected = tuple(row["action"] for row in trace)
    assert selected == ("edge", "original", "q90", "current@640", "edge")
    assert trace[-1]["spent_bytes"] <= trace[-1]["capacity"]
    print("\nAll arithmetic and prefix-safety assertions passed.")


if __name__ == "__main__":
    main()
