#!/usr/bin/env python3
"""Run OpenCV-Q90 images through the cached paper validator protocol."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import yaml


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
Q90_SCRIPTS = REPO / "paper/experiments/20260719_coco_compressed_full_q90/scripts"
sys.path.insert(0, str(Q90_SCRIPTS))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DIAG = load_module("deployment_gap_core", ROOT / "scripts/run_diagnostic.py")
Q90_BUILD = load_module(
    "q90_cache_builder",
    REPO / "paper/experiments/20260719_coco_compressed_full_q90/scripts/build_q90_cloud_cache.py",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_opencv_q90(
    output: Path,
    real_candidates: Path,
    source_images: Path,
    source_q90_root: Path,
) -> Path:
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    expected: dict[int, int] = {}
    with real_candidates.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            expected[int(row["image_id"])] = int(row["candidate"]["payload_bytes"]["q90"])
    if len(expected) != 5000:
        raise RuntimeError(f"candidate population changed: {len(expected)}")

    for offset, image_id in enumerate(sorted(expected), start=1):
        target = images / f"{image_id:012d}.jpg"
        if not target.is_file() or target.stat().st_size != expected[image_id]:
            image = cv2.imread(str(source_images / f"{image_id:012d}.jpg"))
            if image is None:
                raise FileNotFoundError(image_id)
            ok, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if not ok:
                raise RuntimeError(f"Q90 encoding failed: {image_id}")
            if len(encoded) != expected[image_id]:
                raise RuntimeError(
                    f"OpenCV Q90 bytes changed: {image_id} {len(encoded)} != {expected[image_id]}"
                )
            target.write_bytes(encoded.tobytes())
        if offset % 500 == 0:
            print(f"materialized {offset}/5000", flush=True)

    labels = output / "labels"
    if not labels.exists():
        labels.symlink_to((source_q90_root / "labels").resolve(), target_is_directory=True)
    annotations = output / "annotations"
    annotations.mkdir(exist_ok=True)
    annotation_link = annotations / "instances_val2017.json"
    if not annotation_link.exists():
        annotation_link.symlink_to(
            (source_q90_root / "annotations/instances_val2017.json").resolve()
        )
    val = output / "val.txt"
    val.write_text(
        "\n".join(str((images / f"{image_id:012d}.jpg").resolve()) for image_id in sorted(expected))
        + "\n",
        encoding="utf-8",
    )
    source_yaml = yaml.safe_load((source_q90_root / "dataset.yaml").read_text(encoding="utf-8"))
    data_yaml = output / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": str(val.resolve()),
                "val": str(val.resolve()),
                "names": source_yaml["names"],
            },
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def run_paper_validator(
    output: Path,
    data_yaml: Path,
    model_path: Path,
    source_manifest: Path,
    category_ids: list[int],
    *,
    batch: int,
    workers: int,
) -> Path:
    mapped_path = output / "opencv_q90_paper_validator_predictions.json"
    if mapped_path.is_file():
        return mapped_path
    from ultralytics import YOLO

    validator = output / "validator"
    result = YOLO(str(model_path)).val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        batch=int(batch),
        workers=int(workers),
        device="0",
        rect=True,
        end2end=False,
        plots=False,
        save_json=True,
        project=str(output),
        name=validator.name,
        exist_ok=True,
        verbose=False,
    )
    del result
    raw = json.loads((validator / "predictions.json").read_text(encoding="utf-8"))
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    mapped = Q90_BUILD.scaled_rows(raw, manifest, category_ids, remap=True)
    atomic_json(mapped_path, mapped)
    del raw, mapped
    gc.collect()
    return mapped_path


def coalition_ap(report: Mapping[str, Any], label: str, cached_actions: set[str]) -> float:
    for row in report["budgets"][label]["coalitions"]:
        if set(row["cached_actions"]) == cached_actions:
            return float(row["ap50_95"])
    raise KeyError((label, cached_actions))


def parse_args() -> argparse.Namespace:
    dataset = REPO.parents[1] / "datasets/coco"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/q90_cross")
    parser.add_argument("--real-root", type=Path, default=ROOT / "inputs/real")
    parser.add_argument("--images", type=Path, default=dataset / "images/val2017")
    parser.add_argument("--annotations", type=Path, default=dataset / "annotations/instances_val2017.json")
    parser.add_argument("--model", type=Path, default=REPO / "yolo26x.pt")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    real_root = args.real_root.resolve()
    source_q90 = REPO / "paper/experiments/20260719_coco_compressed_full_q90/runs/q90_val2017"
    main_report = DIAG.read_json(ROOT / "runs/formal/report.json")
    manifest = DIAG.read_json(
        REPO / "paper/experiments/20260811_coco_currentoqr12_profile_scale_extension_v1/runs/common/coco/final/source_manifest.json"
    )
    source_q90_manifest = DIAG.read_json(source_q90 / "q90_manifest.json")
    cache_manifest = DIAG.read_json(
        Path(source_q90_manifest["source_cache"]) / "manifest.json"
    )
    category_ids = list(map(int, cache_manifest["metadata"]["category_ids"]))

    data_yaml = materialize_opencv_q90(
        output,
        real_root / "candidates.jsonl",
        args.images.resolve(),
        source_q90,
    )
    crossover_path = run_paper_validator(
        output,
        data_yaml,
        args.model.resolve(),
        source_q90 / "q90_manifest.json",
        category_ids,
        batch=args.batch,
        workers=args.workers,
    )

    from faster_coco_eval import COCO

    gt = COCO(args.annotations, print_function=lambda *_args, **_kwargs: None)
    image_ids = sorted(map(int, gt.getImgIds()))
    cached = {
        "edge": DIAG.group_predictions(DIAG.read_json(Path(manifest["action_sources"]["edge"]))),
        "original": DIAG.group_predictions(DIAG.read_json(Path(manifest["action_sources"]["original"]))),
        "q90": DIAG.group_predictions(DIAG.read_json(Path(manifest["action_sources"]["q90"]))),
        "current@640": DIAG.group_predictions(
            DIAG.read_json(
                REPO / "paper/experiments/20260812_coco_profile_fusion_2x2_factorial_v1/runs/actions/final/current__640_maur.json"
            )
        ),
    }
    crossover = dict(cached)
    crossover["q90"] = DIAG.group_predictions(DIAG.read_json(crossover_path))
    plans = DIAG.read_json(real_root / "plans.json")["plans"]
    report: dict[str, Any] = {"complete": False, "batch": args.batch, "budgets": {}}
    for budget in DIAG.BUDGETS:
        label = str(int(budget * 100))
        plan = DIAG.normalize_plan(
            {int(key): str(value) for key, value in plans[label].items()}, image_ids
        )
        real_mixed = DIAG.group_predictions(
            DIAG.read_json(real_root / f"predictions_cavs_oqr_{label}.json.gz")
        )
        cached_metrics = DIAG.evaluate_coco(
            gt, image_ids, DIAG.cached_predictions(plan, cached)
        )
        cross_metrics = DIAG.evaluate_coco(
            gt, image_ids, DIAG.cached_predictions(plan, crossover)
        )
        actual_q90_metrics = DIAG.evaluate_coco(
            gt,
            image_ids,
            DIAG.hybrid_predictions(
                plan,
                real_mixed,
                cached,
                frozenset({"edge", "original", "current@640"}),
            ),
        )
        expected_cached = float(main_report["budgets"][label]["real_plan_cached_actions"]["ap50_95"])
        expected_actual_q90 = coalition_ap(
            main_report, label, {"edge", "original", "current@640"}
        )
        if abs(cached_metrics["ap50_95"] - expected_cached) > 1e-6:
            raise RuntimeError(f"cached anchor mismatch: {label}")
        if abs(actual_q90_metrics["ap50_95"] - expected_actual_q90) > 1e-6:
            raise RuntimeError(f"actual Q90 anchor mismatch: {label}")
        decomposition = DIAG.q90_path_decomposition(
            cached_paper_codec=cached_metrics["ap50_95"],
            opencv_codec_paper_inference=cross_metrics["ap50_95"],
            opencv_codec_real_inference=actual_q90_metrics["ap50_95"],
        )
        report["budgets"][label] = {
            "cached_pillow_paper_inference": cached_metrics,
            "opencv_paper_inference": cross_metrics,
            "opencv_real_inference": actual_q90_metrics,
            "decomposition": decomposition,
        }
        print(
            f"budget {label}% cached={cached_metrics['ap50_95']:.6f} "
            f"opencv+paper={cross_metrics['ap50_95']:.6f} "
            f"opencv+real={actual_q90_metrics['ap50_95']:.6f}",
            flush=True,
        )
        del real_mixed
        gc.collect()

    report["complete"] = True
    atomic_json(output / "report.json", report)
    lines = [
        "# COCO CAVS-OQR Q90 编码/推理交叉诊断",
        "",
        "| 预算 | Pillow+论文推理 | OpenCV+论文推理 | OpenCV+实时推理 | Q90总差 | 编码器 | 实时推理 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in report["budgets"].items():
        value = row["decomposition"]
        lines.append(
            f"| {label}% | {row['cached_pillow_paper_inference']['ap50_95']:.6f} | "
            f"{row['opencv_paper_inference']['ap50_95']:.6f} | "
            f"{row['opencv_real_inference']['ap50_95']:.6f} | "
            f"{value['total_q90_real_minus_cached']:+.6f} | "
            f"{value['codec_opencv_minus_pillow_under_paper_inference']:+.6f} | "
            f"{value['realtime_inference_minus_paper_inference_under_opencv']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "- 三列使用相同真实 selection，Edge/Original/ROI 始终使用论文缓存，只替换 Q90。",
            "- 编码器与实时推理分解是按 Pillow论文推理 -> OpenCV论文推理 -> OpenCV实时推理 的路径效应。",
        ]
    )
    (output / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
