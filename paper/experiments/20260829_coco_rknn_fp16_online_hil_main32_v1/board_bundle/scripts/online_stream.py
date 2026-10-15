"""Small sequential stream engine used by the board runner and contract tests."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

from online_control import CausalByteController, decision_dict


def execute_stream(
    sequence: Sequence[Mapping[str, Any]],
    *,
    controller: CausalByteController,
    observe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    execute: Callable[[Mapping[str, Any], Mapping[str, Any], Any], Mapping[str, Any]],
    commit: Callable[[dict[str, Any]], None],
    start_position: int = 0,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> None:
    """Execute and commit one complete frame before observing the next frame."""
    if start_position < 0 or start_position > len(sequence):
        raise ValueError("invalid start position")
    previous_done = 0
    for position in range(start_position, len(sequence)):
        item = sequence[position]
        frame_start = int(monotonic_ns())
        if frame_start < previous_done:
            raise RuntimeError("frame clock moved backwards")
        observation = dict(observe(item))
        image_id = int(item["image_id"])
        if int(observation.get("frame_id", -1)) != image_id:
            raise RuntimeError("frame observation identity mismatch")
        decision = controller.decide(
            image_id,
            observation["payload_bytes"],
            observation["scores"],
        )
        decision_ns = int(monotonic_ns())
        outcome = dict(execute(item, observation, decision))
        frame_done = int(monotonic_ns())
        if not frame_start <= decision_ns <= frame_done:
            raise RuntimeError("invalid per-frame event order")
        record = {
            "position": position,
            "image_id": image_id,
            "file_name": str(item.get("file_name", "")),
            "frame_start_ns": frame_start,
            "decision_ns": decision_ns,
            "frame_done_ns": frame_done,
            "decision": decision_dict(decision),
            "controller_state": controller.snapshot(),
            **outcome,
        }
        commit(record)
        previous_done = frame_done

