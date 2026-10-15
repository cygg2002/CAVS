#!/usr/bin/env python3
"""Inference and JPEG contracts shared by the paper-parity client and Cloud."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops


class ScaleDownDetectionPredictor(DetectionPredictor):
    """YOLO predictor using validation-style letterboxing and multi-label NMS."""

    def pre_transform(self, images: list[np.ndarray]) -> list[np.ndarray]:
        same_shapes = len({image.shape for image in images}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (
                self.model.format == "pt"
                or (getattr(self.model, "dynamic", False) and self.model.format != "imx")
            ),
            stride=self.model.stride,
            scaleup=False,
        )
        return [letterbox(image=image) for image in images]

    def postprocess(self, preds, image, original_images, **kwargs):
        """Match the detection validator's multi-label NMS semantics."""
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            multi_label=True,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )
        if not isinstance(original_images, list):
            original_images = ops.convert_torch2numpy_batch(original_images)[..., ::-1]
        return self.construct_results(outputs, image, original_images)


def native_jpeg(path: str | Path) -> bytes:
    """Return the source JPEG bytes without decode/re-encode drift."""
    return Path(path).read_bytes()


def paper_q90_jpeg(
    path: str | Path,
    *,
    crop_xyxy: Sequence[int] | None = None,
    long_side: int = 640,
) -> bytes:
    """Encode a full image or ROI exactly like the frozen paper action cache."""
    with Image.open(path) as opened:
        image = opened.crop(tuple(map(int, crop_xyxy))) if crop_xyxy is not None else opened.copy()
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(long_side) / max(1, width, height))
        size = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True, subsampling=2)
        return output.getvalue()


def roi_record(sample: dict, action: str = "current@640") -> dict | None:
    """Return the frozen action record containing ROI geometry, when available."""
    for record in sample.get("actions", ()):
        if str(record.get("action_id")) == action and bool(record.get("available", False)):
            return dict(record)
    return None


def validate_cloud_health(
    health: dict,
    *,
    cloud_sha256: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> None:
    """Reject a Cloud service that does not implement the frozen paper contract."""
    expected = {
        "model_sha256": str(cloud_sha256),
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
    }
    for key, value in expected.items():
        if health.get(key) != value:
            raise RuntimeError(f"cloud health mismatch for {key}: {health.get(key)!r} != {value!r}")
    protocols = health.get("action_protocols")
    required = {
        "original": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
        "q90": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
        "current@640": {"end2end": False, "rect": True, "scaleup": True, "half": True, "multi_label": False},
    }
    if protocols != required:
        raise RuntimeError(f"cloud action protocol mismatch: {protocols!r}")
