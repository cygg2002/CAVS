#!/usr/bin/env python3
"""Resumable per-image GluonCV inference with frozen preprocessing and output audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import numpy as np
from pathlib import Path
import sys
import time
from typing import Any, Mapping


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_gluoncv_infer_data", ROOT / "scripts/data_core.py")
NORMALIZE = load_module("voc_gluoncv_infer_normalize", ROOT / "scripts/inference_core.py")


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _construct_network(model_zoo: Any, model_name: str, *, params: Path, context: Any) -> Any:
    """Construct the frozen architecture and load only the already verified parameters."""
    network = model_zoo.get_model(
        str(model_name), pretrained=False, pretrained_base=False
    )
    # These legacy GluonCV constructors initialize some parameters on CPU.
    # Load into that existing context first, then move the complete network as
    # one unit; direct GPU loading is rejected by MXNet 1.9 for such blocks.
    network.load_parameters(str(params), ctx=None)
    network.collect_params().reset_ctx(context)
    return network


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid inference checkpoint line {number}: {path}") from error
    return rows


def _image_map(path: Path | None) -> tuple[dict[int, str], str | None]:
    if path is None:
        return {}, None
    payload = DATA.read_json(path)
    rows = payload.get("records", payload)
    if not isinstance(rows, list):
        raise ValueError("image map must be a record list")
    mapping = {int(row["image_id"]): str(Path(row["path"]).resolve()) for row in rows}
    if len(mapping) != len(rows):
        raise RuntimeError("image map contains duplicate image IDs")
    return mapping, DATA.file_sha256(path)


def _load_network(role: str, model_root: Path, device: int):
    import mxnet as mx
    from gluoncv import model_zoo
    from gluoncv.model_zoo.model_store import get_model_file

    contract = PROTOCOL["detectors"][role]
    if int(device) >= 0:
        if mx.context.num_gpus() <= int(device):
            raise RuntimeError(f"MXNet GPU {device} is unavailable")
        context = mx.gpu(int(device))
    else:
        context = mx.cpu()
    params = Path(get_model_file(str(contract["model"]), root=str(model_root))).resolve()
    observed_sha1 = file_sha1(params)
    if observed_sha1 != str(contract["official_sha1"]):
        raise RuntimeError(f"official parameter SHA-1 changed: {role}/{observed_sha1}")
    network = _construct_network(
        model_zoo,
        str(contract["model"]),
        params=params,
        context=context,
    )
    return network, context, params


def _prepare_center_net_input(raw: Any, *, input_size: int, transform_factory: Any) -> Any:
    transform = transform_factory(int(input_size), int(input_size))
    tensor, _ = transform(raw, np.empty((0, 6), dtype=np.float32))
    return tensor.expand_dims(0)


def _predict(network: Any, context: Any, role: str, path: Path) -> tuple[Any, Any, Any, int, int]:
    import mxnet as mx
    from gluoncv import data

    raw = mx.image.imread(str(path), 1)
    contract = PROTOCOL["detectors"][role]
    if role == "edge":
        preset = data.transforms.presets.center_net
        input_size = int(contract["input_size"])
        tensor = _prepare_center_net_input(
            raw,
            input_size=input_size,
            transform_factory=preset.CenterNetDefaultValTransform,
        )
    else:
        tensor, resized = data.transforms.presets.rcnn.transform_test(
            raw, short=int(contract["short"]), max_size=int(contract["max_size"])
        )
    class_ids, scores, boxes = network(tensor.as_in_context(context))
    mx.nd.waitall()
    class_array = class_ids[0].asnumpy()
    score_array = scores[0].asnumpy()
    box_array = boxes[0].asnumpy()
    if role == "edge":
        original_width, original_height = int(raw.shape[1]), int(raw.shape[0])
        inverse = preset.get_post_transform(
            original_width, original_height, input_size, input_size
        )
        box_array = np.asarray(
            [
                NORMALIZE.map_affine_xyxy(
                    box,
                    transform=inverse,
                    original_width=original_width,
                    original_height=original_height,
                )
                for box in box_array
            ],
            dtype=np.float32,
        ).reshape((-1, 4))
        return class_array, score_array, box_array, original_width, original_height
    return class_array, score_array, box_array, int(resized.shape[1]), int(resized.shape[0])


def run(
    *,
    gt_json: Path,
    role: str,
    action: str,
    output: Path,
    model_root: Path,
    image_map_path: Path | None,
    device: int,
    limit: int | None,
) -> dict[str, Any]:
    if role not in {"edge", "cloud"}:
        raise ValueError(role)
    if action not in {"edge", "original", "q90"}:
        raise ValueError(action)
    if (role == "edge") != (action == "edge"):
        raise ValueError("edge role/action must be paired; cloud serves original/q90")
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "completion.json"
    predictions_path = output / "predictions.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("predictions_sha256") != DATA.file_sha256(predictions_path):
            raise RuntimeError(f"completed predictions changed: {output}")
        return report
    gt = DATA.read_json(gt_json)
    images = sorted(gt["images"], key=lambda row: int(row["id"]))
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be positive")
        images = images[: int(limit)]
    override, image_map_sha256 = _image_map(image_map_path)
    image_ids = {int(image["id"]) for image in images}
    if override and not image_ids <= set(override):
        raise RuntimeError("image-map is missing a requested inference image")
    if override:
        override = {image_id: override[image_id] for image_id in image_ids}
    network, context, params = _load_network(role, model_root, device)
    detector_contract = PROTOCOL["detectors"][role]
    contract = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": role,
        "action": action,
        "model": PROTOCOL["detectors"][role]["model"],
        "params_sha1": file_sha1(params),
        "params_sha256": DATA.file_sha256(params),
        "gt_sha256": DATA.file_sha256(gt_json),
        "image_map_sha256": image_map_sha256,
        "score_floor": float(PROTOCOL["inference"]["score_floor"]),
        "device": int(device),
        "limit": int(limit) if limit is not None else None,
        "expected_images": len(images),
        "per_image_inference": True,
        "flip_tta": False,
    }
    if role == "edge":
        contract.update(
            {
                "preprocess": str(detector_contract["preprocess"]),
                "input_size": int(detector_contract["input_size"]),
                "boxes_mapped_to_original_with": "get_post_transform",
            }
        )
    else:
        contract.update(
            {
                "short": int(detector_contract["short"]),
                "max_size": int(detector_contract["max_size"]),
            }
        )
    contract_path = output / "run_contract.json"
    if contract_path.is_file() and DATA.read_json(contract_path) != contract:
        raise RuntimeError(f"inference run contract changed: {output}")
    DATA.atomic_json(contract_path, contract)
    rows_path = output / "records.jsonl"
    existing_rows = _read_jsonl(rows_path)
    existing = {int(row["image_id"]): row for row in existing_rows}
    if len(existing) != len(existing_rows) or not set(existing) <= image_ids:
        raise RuntimeError(f"invalid inference checkpoint: {output}")
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        for count, image in enumerate(images, 1):
            image_id = int(image["id"])
            if image_id in existing:
                continue
            path = Path(override.get(image_id, str(image["absolute_path"]))).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            before = time.perf_counter()
            class_ids, scores, boxes, resized_width, resized_height = _predict(
                network, context, role, path
            )
            predictions, audit = NORMALIZE.normalize_detections(
                image_id=image_id,
                class_ids=class_ids,
                scores=scores,
                boxes=boxes,
                original_width=int(image["width"]),
                original_height=int(image["height"]),
                resized_width=resized_width,
                resized_height=resized_height,
                category_count=len(PROTOCOL["data"]["category_names"]),
                score_floor=float(PROTOCOL["inference"]["score_floor"]),
            )
            row = {
                "image_id": image_id,
                "source": str(path),
                "source_bytes": path.stat().st_size,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "elapsed_ms": (time.perf_counter() - before) * 1000.0,
                "audit": audit,
                "predictions": predictions,
            }
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            existing[image_id] = row
            if count % 100 == 0 or count == len(images):
                print(f"[{role}/{action}] {len(existing)}/{len(images)}", flush=True)
    rows = _read_jsonl(rows_path)
    if len(rows) != len(images) or {int(row["image_id"]) for row in rows} != image_ids:
        raise RuntimeError(f"inference output is incomplete: {output}")
    rows.sort(key=lambda row: int(row["image_id"]))
    predictions = [prediction for row in rows for prediction in row["predictions"]]
    DATA.atomic_json(predictions_path, predictions)
    audit = Counter()
    for row in rows:
        audit.update({str(key): int(value) for key, value in row["audit"].items()})
    elapsed_values = [float(row["elapsed_ms"]) for row in rows]
    report = {
        "complete": True,
        "role": role,
        "action": action,
        "images": len(rows),
        "predictions": len(predictions),
        "audit": dict(audit),
        "mean_inference_ms": sum(elapsed_values) / len(elapsed_values),
        "elapsed_seconds_this_invocation": time.time() - started,
        "params_sha1": contract["params_sha1"],
        "params_sha256": contract["params_sha256"],
        "records_sha256": DATA.file_sha256(rows_path),
        "predictions_sha256": DATA.file_sha256(predictions_path),
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    parser.add_argument("--action", choices=("edge", "original", "q90"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", default=str(ROOT / "assets/models"))
    parser.add_argument("--image-map")
    parser.add_argument("--device", type=int, default=int(PROTOCOL["inference"]["device"]))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run(
        gt_json=Path(args.gt_json),
        role=str(args.role),
        action=str(args.action),
        output=Path(args.output),
        model_root=Path(args.model_root),
        image_map_path=Path(args.image_map) if args.image_map else None,
        device=int(args.device),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
