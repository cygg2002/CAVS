# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Deployable contracts for the CAVS Current@640 + MAUR uploader.

The experiment artifact is exported to hash-pinned NPZ files before use. This
module deliberately never deserializes pickle files from the UI-provided path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

ACTION_IDS = ("edge", "original", "q90", "current@640")
EXPECTED_METHOD = "CAVS-Current-MAUR"
EXPECTED_SEEDS = (20260809, 20260810, 20260811)
EXPECTED_EDGE_MODEL = "yolo26n.pt"
EXPECTED_EDGE_MODEL_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
EXPECTED_CLOUD_MODEL_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
ACTION_TYPES = {"edge": 0, "full": 1, "roi": 2, "pair": 3, "full_region": 4}
SCENE_DIM = 9
ACTION_DIM = 18
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_SEED_BYTES = 32 * 1024 * 1024
MAX_SEED_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the bounded, non-executable deployment manifest schema."""
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported Current+MAUR manifest schema")
    if manifest.get("policy_type") != "cavs_current_maur":
        raise ValueError("not a Current+MAUR manifest")
    if manifest.get("method") != EXPECTED_METHOD:
        raise ValueError("unexpected Current+MAUR method")
    if manifest.get("profile") != "current" or manifest.get("fusion") != "MAUR":
        raise ValueError("manifest must use Current profile with MAUR fusion")
    if tuple(map(str, manifest.get("action_ids", ()))) != ACTION_IDS:
        raise ValueError("Current+MAUR action space changed")
    if int(manifest.get("category_count", 0)) != 80:
        raise ValueError("Current+MAUR requires the 80-class COCO schema")

    edge_model = manifest.get("edge_model")
    if not isinstance(edge_model, Mapping):
        raise TypeError("missing Current+MAUR edge model identity")
    if edge_model.get("name") != EXPECTED_EDGE_MODEL:
        raise ValueError(f"Current+MAUR requires {EXPECTED_EDGE_MODEL}")
    if str(edge_model.get("sha256", "")).lower() != EXPECTED_EDGE_MODEL_SHA256:
        raise ValueError("Current+MAUR edge model checksum changed")
    if str(manifest.get("cloud_model_sha256", "")).lower() != EXPECTED_CLOUD_MODEL_SHA256:
        raise ValueError("Current+MAUR cloud model checksum changed")

    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping):
        raise TypeError("missing Current+MAUR architecture")
    expected_architecture = {
        "class_embedding_dim": 8,
        "box_hidden_dim": 32,
        "action_hidden_dim": 32,
        "joint_hidden_dim": 64,
        "max_boxes": 64,
    }
    for key, expected in expected_architecture.items():
        if int(architecture.get(key, -1)) != expected:
            raise ValueError(f"unexpected architecture value: {key}")
    dropout = float(architecture.get("dropout", -1.0))
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("invalid architecture dropout")

    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3:
        raise ValueError("Current+MAUR requires exactly three seed weights")
    observed = []
    for item in seeds:
        if not isinstance(item, Mapping):
            raise TypeError("invalid seed entry")
        observed.append(int(item.get("seed", -1)))
        filename = str(item.get("file", ""))
        if Path(filename).suffix.lower() != ".npz":
            raise ValueError("Current+MAUR weights must be NPZ files")
        if Path(filename).is_absolute() or Path(filename).name != filename:
            raise ValueError("seed file must be one relative filename")
        checksum = str(item.get("sha256", "")).lower()
        if HEX_SHA256.fullmatch(checksum) is None:
            raise ValueError("seed file requires a SHA-256 checksum")
    if tuple(observed) != EXPECTED_SEEDS:
        raise ValueError("Current+MAUR seed ensemble changed")


class StaticActionValueRouter(nn.Module):
    """Inference-only replica of the frozen CurrentActionValue architecture."""

    def __init__(self, architecture: Mapping[str, Any], category_count: int) -> None:
        super().__init__()
        class_dim = int(architecture["class_embedding_dim"])
        box_dim = int(architecture["box_hidden_dim"])
        action_dim = int(architecture["action_hidden_dim"])
        joint_dim = int(architecture["joint_hidden_dim"])
        self.category_count = int(category_count)
        self.class_embedding = nn.Embedding(self.category_count, class_dim)
        self.box_encoder = nn.Sequential(
            nn.Linear(6 + class_dim, box_dim),
            nn.LayerNorm(box_dim),
            nn.SiLU(),
            nn.Linear(box_dim, box_dim),
            nn.SiLU(),
        )
        self.scene_encoder = nn.Sequential(
            nn.Linear(SCENE_DIM, box_dim),
            nn.LayerNorm(box_dim),
            nn.SiLU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(box_dim * 3, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.SiLU(),
        )
        self.action_type_embedding = nn.Embedding(len(ACTION_TYPES), 8)
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM + 8 + 1, action_dim),
            nn.LayerNorm(action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(joint_dim + action_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.SiLU(),
            nn.Dropout(float(architecture["dropout"])),
            nn.Linear(joint_dim, 1),
        )

    def forward(
        self,
        *,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        box_mask: torch.Tensor,
        scene: torch.Tensor,
        actions: torch.Tensor,
        action_type: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.class_embedding(classes.clamp(0, self.category_count - 1))
        encoded = self.box_encoder(torch.cat((boxes, embedded), dim=-1))
        valid = box_mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid, -torch.inf).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        scene_encoded = self.scene_encoder(scene)
        image = self.image_encoder(torch.cat((mean, maximum, scene_encoded), dim=-1))

        action_count = actions.shape[1]
        budget = torch.zeros(
            (len(actions), action_count, 1), dtype=actions.dtype, device=actions.device
        )
        encoded_action = self.action_encoder(
            torch.cat(
                (actions, self.action_type_embedding(action_type.clamp(0, len(ACTION_TYPES) - 1)), budget),
                dim=-1,
            )
        )
        image = image[:, None, :].expand(-1, action_count, -1)
        scores = self.head(torch.cat((image, encoded_action), dim=-1)).squeeze(-1)
        return scores.masked_fill(~action_mask, -1e9)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_npz_archive(path: Path, expected_keys: set[str]) -> None:
    """Bound NPZ expansion and require exactly the model tensor members."""
    expected_members = {f"{key}.npy" for key in expected_keys}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(expected_members) or set(names) != expected_members:
                raise ValueError(f"Current+MAUR archive members changed: {path.name}")
            if any(member.flag_bits & 0x1 for member in members):
                raise ValueError(f"Current+MAUR archive must not be encrypted: {path.name}")
            if sum(member.file_size for member in members) > MAX_SEED_UNCOMPRESSED_BYTES:
                raise ValueError(f"Current+MAUR archive expands beyond the safe limit: {path.name}")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Current+MAUR seed is not a valid NPZ: {path.name}") from exc


def _load_seed_model(
    path: Path,
    architecture: Mapping[str, Any],
    category_count: int,
) -> StaticActionValueRouter:
    model = StaticActionValueRouter(architecture, category_count)
    expected = model.state_dict()
    _validate_npz_archive(path, set(expected))
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(expected):
            raise ValueError(f"Current+MAUR state dictionary changed: {path.name}")
        state = {}
        for name, reference in expected.items():
            array = np.asarray(payload[name])
            if tuple(array.shape) != tuple(reference.shape):
                raise ValueError(f"Current+MAUR tensor shape changed: {name}")
            if array.dtype.kind not in {"f", "i", "u"}:
                raise ValueError(f"Current+MAUR tensor dtype is unsafe: {name}")
            state[name] = torch.from_numpy(array.copy()).to(dtype=reference.dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_current_maur_policy(manifest_path: str | Path) -> dict[str, Any]:
    """Load a hash-pinned Current+MAUR ensemble without executable serialization."""
    path = Path(manifest_path).expanduser().resolve()
    if path.suffix.lower() != ".json" or not path.is_file():
        raise FileNotFoundError(f"Current+MAUR manifest not found: {path}")
    if not 0 < path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("Current+MAUR manifest size is invalid")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Current+MAUR manifest must be a JSON object")
    validate_manifest(manifest)

    models = []
    root = path.parent.resolve()
    for item in manifest["seeds"]:
        seed_path = (root / str(item["file"])).resolve()
        if seed_path.parent != root or not seed_path.is_file():
            raise FileNotFoundError(f"Current+MAUR seed file not found: {item['file']}")
        if not 0 < seed_path.stat().st_size <= MAX_SEED_BYTES:
            raise ValueError(f"Current+MAUR seed file size is invalid: {seed_path.name}")
        if _file_sha256(seed_path) != str(item["sha256"]).lower():
            raise ValueError(f"Current+MAUR seed checksum mismatch: {seed_path.name}")
        reward_mean = float(item.get("reward_mean", math.nan))
        reward_std = float(item.get("reward_std", math.nan))
        if not math.isfinite(reward_mean) or not math.isfinite(reward_std) or reward_std <= 0.0:
            raise ValueError(f"Current+MAUR reward normalization is invalid: {seed_path.name}")
        models.append(
            {
                "seed": int(item["seed"]),
                "model": _load_seed_model(
                    seed_path, manifest["architecture"], int(manifest["category_count"])
                ),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
            }
        )
    return {
        **manifest,
        "kind": "current_maur",
        "manifest_path": str(path),
        "ensemble_seeds": [int(item["seed"]) for item in models],
        "models": models,
    }


def _current_expansion(area_ratio: float) -> float:
    if area_ratio < 0.01:
        return 1.8
    if area_ratio < 0.10:
        return 1.6
    return 1.4


def current_roi_xyxy(
    width: int,
    height: int,
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    confidence: float = 0.25,
    max_area_ratio: float = 0.65,
) -> list[int] | None:
    """Build the frozen merged ``current@640`` ROI from Edge detections."""
    image_width = max(1, int(width))
    image_height = max(1, int(height))
    expanded: list[tuple[float, float, float, float]] = []
    for box in edge_boxes:
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


def _box_inputs(
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    max_boxes: int,
    confidence: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_width = max(1.0, float(width))
    image_height = max(1.0, float(height))
    ordered = [
        dict(box)
        for box in edge_boxes
        if float(box.get("conf", box.get("score", 0.0))) >= float(confidence)
    ]
    ordered.sort(
        key=lambda box: (
            -float(box.get("conf", box.get("score", 0.0))),
            int(box.get("cls", 0)),
            tuple(map(float, box["xyxy"])),
        )
    )
    numeric = torch.zeros((1, int(max_boxes), 6), dtype=torch.float32)
    classes = torch.zeros((1, int(max_boxes)), dtype=torch.long)
    mask = torch.zeros((1, int(max_boxes)), dtype=torch.bool)
    for index, box in enumerate(ordered[: int(max_boxes)]):
        x1, y1, x2, y2 = map(float, box["xyxy"])
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        numeric[0, index] = torch.tensor(
            (
                (x1 + x2) / 2.0 / image_width,
                (y1 + y2) / 2.0 / image_height,
                box_width / image_width,
                box_height / image_height,
                box_width * box_height / (image_width * image_height),
                float(box.get("conf", box.get("score", 0.0))),
            ),
            dtype=torch.float32,
        )
        classes[0, index] = int(box.get("cls", 0))
        mask[0, index] = True
    return numeric, classes, mask


def _scene_inputs(
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    confidence: float,
) -> torch.Tensor:
    image_width = max(1.0, float(width))
    image_height = max(1.0, float(height))
    boxes = [
        box
        for box in edge_boxes
        if float(box.get("conf", box.get("score", 0.0))) >= float(confidence)
    ]
    scores = [float(box.get("conf", box.get("score", 0.0))) for box in boxes]
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = map(float, box["xyxy"])
        areas.append(
            max(0.0, x2 - x1)
            * max(0.0, y2 - y1)
            / (image_width * image_height)
        )
    # Index 1 is intentionally zero: training removes the direct payload input.
    return torch.tensor(
        [[
            math.log(image_width / image_height),
            0.0,
            min(1.0, len(boxes) / 64.0),
            max(scores, default=0.0),
            statistics.fmean(scores) if scores else 0.0,
            min(1.0, sum(areas)),
            0.0,
            0.0,
            0.0,
        ]],
        dtype=torch.float32,
    )


def _action_inputs(
    roi_xyxy: Sequence[int] | None,
    *,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = torch.zeros((1, len(ACTION_IDS), ACTION_DIM), dtype=torch.float32)
    action_types = torch.tensor(
        [[ACTION_TYPES["edge"], ACTION_TYPES["full"], ACTION_TYPES["pair"], ACTION_TYPES["roi"]]],
        dtype=torch.long,
    )
    action_mask = torch.tensor([[True, True, True, roi_xyxy is not None]], dtype=torch.bool)
    for index in (1, 2, 3):
        actions[0, index, 0:3] = 1.0
        actions[0, index, 4] = 0.5
    if roi_xyxy is not None:
        x1, y1, x2, y2 = map(float, roi_xyxy)
        area_ratio = (
            max(0.0, x2 - x1)
            * max(0.0, y2 - y1)
            / max(1.0, float(width) * float(height))
        )
        actions[0, 3, 5] = 0.5
        actions[0, 3, 13] = area_ratio
        actions[0, 3, 14] = area_ratio
        actions[0, 3, 17] = 1.0
    # Cost-bearing indices 6:13 remain zero, exactly as in CurrentActionValue training.
    return actions, action_types, action_mask


def predict_current_maur_scores(
    policy: Mapping[str, Any],
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    roi_xyxy: Sequence[int] | None,
) -> dict[str, float]:
    """Return the three-seed mean contextual value for every available OQR action."""
    if policy.get("kind") != "current_maur" or len(policy.get("models", ())) != 3:
        raise ValueError("a loaded three-seed Current+MAUR policy is required")
    architecture = policy["architecture"]
    deploy_confidence = float(policy.get("deploy_confidence", 0.001))
    boxes, classes, box_mask = _box_inputs(
        edge_boxes,
        width=int(width),
        height=int(height),
        max_boxes=int(architecture["max_boxes"]),
        confidence=deploy_confidence,
    )
    scene = _scene_inputs(
        edge_boxes,
        width=int(width),
        height=int(height),
        confidence=deploy_confidence,
    )
    actions, action_type, action_mask = _action_inputs(
        roi_xyxy, width=int(width), height=int(height)
    )
    rows = []
    with torch.inference_mode():
        for seed in policy["models"]:
            raw = seed["model"](
                boxes=boxes,
                classes=classes,
                box_mask=box_mask,
                scene=scene,
                actions=actions,
                action_type=action_type,
                action_mask=action_mask,
            )[0]
            values = raw.cpu().numpy().astype(np.float64)
            values = values * float(seed["reward_std"]) + float(seed["reward_mean"])
            values[0] = 0.0
            rows.append(values)
    ensemble = np.mean(np.stack(rows), axis=0)
    return {
        action: float(ensemble[index])
        for index, action in enumerate(ACTION_IDS)
        if bool(action_mask[0, index]) and math.isfinite(float(ensemble[index]))
    }


def _iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def maur_fuse_boxes(
    edge_boxes: Sequence[Mapping[str, Any]],
    cloud_boxes: Sequence[Mapping[str, Any]],
    roi_xyxy: Sequence[int],
    *,
    duplicate_iou: float = 0.5,
    nms_iou: float = 0.7,
) -> list[dict[str, Any]]:
    """Apply the frozen MAUR ROI fusion to UI-format ``xyxy`` boxes."""
    roi_x1, roi_y1, roi_x2, roi_y2 = map(float, roi_xyxy)
    candidates: list[dict[str, Any]] = []
    for source in edge_boxes:
        edge = dict(source)
        x1, y1, x2, y2 = map(float, edge["xyxy"])
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
            continue
        duplicate = any(
            int(edge["cls"]) == int(cloud["cls"])
            and _iou_xyxy(edge["xyxy"], cloud["xyxy"]) >= float(duplicate_iou)
            for cloud in cloud_boxes
        )
        if not duplicate:
            candidates.append(edge)
    candidates.extend(dict(box) for box in cloud_boxes)

    kept: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda box: (
            -float(box.get("conf", box.get("score", 0.0))),
            int(box["cls"]),
            tuple(map(float, box["xyxy"])),
        ),
    ):
        if any(
            int(candidate["cls"]) == int(existing["cls"])
            and _iou_xyxy(candidate["xyxy"], existing["xyxy"]) >= float(nms_iou)
            for existing in kept
        ):
            continue
        kept.append(dict(candidate))
    return kept


def select_current_maur_action(
    scores: Mapping[str, float],
    payload_bytes: Mapping[str, int],
    *,
    full_payload_bytes: int,
    cost_weight: float,
) -> dict[str, Any]:
    """Choose one online action by CAVS value minus normalized upload cost."""
    full_bytes = int(full_payload_bytes)
    weight = float(cost_weight)
    if full_bytes <= 0:
        raise ValueError("full payload must be positive")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("cost weight must be finite and non-negative")
    utilities: dict[str, float] = {}
    normalized_scores: dict[str, float] = {}
    for action in ACTION_IDS:
        if action not in scores:
            continue
        score = float(scores[action])
        if not math.isfinite(score):
            continue
        if action not in payload_bytes:
            raise ValueError(f"missing payload estimate for {action}")
        ratio = max(0.0, float(payload_bytes[action])) / float(full_bytes)
        normalized_scores[action] = score
        utilities[action] = score - weight * ratio
    if "edge" not in utilities:
        raise ValueError("Current+MAUR decision requires the Edge fallback")
    chosen = max(utilities, key=lambda action: (utilities[action], -ACTION_IDS.index(action)))
    return {
        "action": chosen,
        "score": normalized_scores[chosen],
        "utility": utilities[chosen],
        "scores": normalized_scores,
        "utilities": utilities,
    }
