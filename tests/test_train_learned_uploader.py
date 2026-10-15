# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the learned uploader training protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO_DIR))

from train_learned_uploader import (  # noqa: E402
    batched_predict,
    prepare_cloud_inputs,
    resolve_end2end_modes,
    training_protocol,
)


class FakeModel:
    """Capture prediction arguments without loading detector weights."""

    def __init__(self):
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return []


def test_batched_predict_forwards_frozen_end2many_protocol():
    """Training labels must use the same head and NMS settings as E24."""
    model = FakeModel()
    assert (
        batched_predict(
            model,
            [],
            imgsz=512,
            conf=0.25,
            device="0",
            batch_size=16,
            half=False,
            model_nms_iou=0.7,
            max_det=300,
            end2end=False,
        )
        == []
    )

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    model.predict = lambda **kwargs: setattr(model, "kwargs", kwargs) or []
    batched_predict(
        model,
        [image],
        imgsz=512,
        conf=0.25,
        device="0",
        batch_size=16,
        half=False,
        model_nms_iou=0.7,
        max_det=300,
        end2end=False,
    )
    assert model.kwargs["imgsz"] == 512
    assert model.kwargs["iou"] == 0.7
    assert model.kwargs["max_det"] == 300
    assert model.kwargs["end2end"] is False


def test_cloud_training_inputs_use_decoded_jpeg_pixels():
    """Codec-aware labels must be generated from transmitted JPEG pixels."""
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, 12:] = 255

    raw = prepare_cloud_inputs([image], simulate_codec=False, jpeg_quality=40)
    decoded = prepare_cloud_inputs([image], simulate_codec=True, jpeg_quality=40)

    assert raw[0] is image
    assert decoded[0].shape == image.shape
    assert decoded[0] is not image


def test_separate_edge_and_cloud_head_modes_override_shared_default():
    """Mixed M2E training must control the edge and cloud heads independently."""
    args = argparse.Namespace(end2end=None, edge_end2end=False, cloud_end2end=True)
    assert resolve_end2end_modes(args) == (False, True)

    shared_args = argparse.Namespace(end2end=False, edge_end2end=None, cloud_end2end=None)
    assert resolve_end2end_modes(shared_args) == (False, False)


def test_training_protocol_is_saved_with_separate_head_metadata():
    """The policy artifact must identify both detector heads and the codec protocol."""
    args = argparse.Namespace(
        imgsz=640,
        cloud_imgsz=512,
        conf=0.25,
        cloud_conf=0.25,
        model_nms_iou=0.7,
        max_det=300,
        end2end=None,
        edge_end2end=False,
        cloud_end2end=True,
        simulate_cloud_codec=True,
        jpeg_quality=80,
        half=False,
    )

    assert training_protocol(args) == {
        "edge_imgsz": 640,
        "cloud_imgsz": 512,
        "edge_conf": 0.25,
        "cloud_conf": 0.25,
        "model_nms_iou": 0.7,
        "max_det": 300,
        "end2end": None,
        "head_mode": "mixed",
        "edge_end2end": False,
        "cloud_end2end": True,
        "edge_head_mode": "end2many",
        "cloud_head_mode": "end2end",
        "simulate_cloud_codec": True,
        "jpeg_quality": 80,
        "half": False,
    }
