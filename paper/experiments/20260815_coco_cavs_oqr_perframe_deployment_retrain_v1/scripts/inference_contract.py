"""Shared one-image inference, JPEG and prediction conversion helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops


class ScaleDownDetectionPredictor(DetectionPredictor):
    """Single-image predictor matching validation NMS without batch-shaped rect."""

    def pre_transform(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if len(images) != 1:
            raise RuntimeError("deployment-consistent predictor requires one image")
        letterbox = LetterBox(
            self.imgsz,
            auto=self.args.rect
            and (
                self.model.format == "pt"
                or (getattr(self.model, "dynamic", False) and self.model.format != "imx")
            ),
            stride=self.model.stride,
            scaleup=False,
        )
        return [letterbox(image=images[0])]

    def postprocess(self, preds, image, original_images, **kwargs):
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


def _current_expansion(area_ratio: float) -> float:
    if area_ratio < 0.01:
        return 1.8
    if area_ratio < 0.10:
        return 1.6
    return 1.4


def current_roi_xyxy(
    width: int,
    height: int,
    boxes: Sequence[Mapping[str, Any]],
    *,
    confidence: float = 0.25,
    max_area_ratio: float = 0.65,
) -> list[int] | None:
    image_width = max(1, int(width))
    image_height = max(1, int(height))
    expanded = []
    for box in boxes:
        if float(box.get("conf", box.get("score", 0.0))) < float(confidence):
            continue
        x1, y1, x2, y2 = map(float, box["xyxy"])
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        area_ratio = box_width * box_height / float(image_width * image_height)
        expansion = _current_expansion(area_ratio)
        half_width = box_width * expansion / 2.0
        half_height = box_height * expansion / 2.0
        expanded.append(
            (
                max(0.0, center_x - half_width),
                max(0.0, center_y - half_height),
                min(float(image_width), center_x + half_width),
                min(float(image_height), center_y + half_height),
            )
        )
    if not expanded:
        return None
    x1 = max(0, int(min(value[0] for value in expanded)))
    y1 = max(0, int(min(value[1] for value in expanded)))
    x2 = min(image_width, int(max(value[2] for value in expanded) + 0.999999))
    y2 = min(image_height, int(max(value[3] for value in expanded) + 0.999999))
    if x2 <= x1 or y2 <= y1:
        return None
    merged_area = (x2 - x1) * (y2 - y1) / float(image_width * image_height)
    return None if merged_area > float(max_area_ratio) else [x1, y1, x2, y2]


def native_jpeg(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def encoded_jpeg(
    path: str | Path,
    *,
    crop_xyxy: Sequence[int] | None = None,
    long_side: int = 640,
    quality: int = 90,
    optimize: bool = True,
    subsampling: int = 2,
) -> bytes:
    with Image.open(path) as opened:
        image = opened.crop(tuple(map(int, crop_xyxy))) if crop_xyxy is not None else opened.copy()
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(long_side) / max(1, width, height))
        size = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=int(quality),
            optimize=bool(optimize),
            subsampling=int(subsampling),
        )
        return output.getvalue()


def decode_jpeg(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decode failed")
    return image


def edge_boxes(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    output = []
    for box, score, class_index in zip(
        result.boxes.xyxy.detach().cpu().numpy(),
        result.boxes.conf.detach().cpu().numpy(),
        result.boxes.cls.detach().cpu().numpy(),
    ):
        output.append(
            {
                "xyxy": list(map(float, box)),
                "conf": float(score),
                "cls": int(class_index),
            }
        )
    return output


def official_predictions(
    result: Any,
    *,
    image_id: int,
    category_map: Mapping[int, int],
    offset_xy: tuple[int, int] = (0, 0),
) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    offset_x, offset_y = map(float, offset_xy)
    output = []
    for box, score, class_index in zip(
        result.boxes.xyxy.detach().cpu().numpy(),
        result.boxes.conf.detach().cpu().numpy(),
        result.boxes.cls.detach().cpu().numpy(),
    ):
        x1, y1, x2, y2 = map(float, box)
        output.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_map[int(class_index)]),
                "bbox": [x1 + offset_x, y1 + offset_y, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return output
