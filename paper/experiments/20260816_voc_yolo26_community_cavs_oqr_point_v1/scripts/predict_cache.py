#!/usr/bin/env python3
"""Run exact VOC detector inference and write a COCO-format cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    gt = json.loads(Path(args.gt_json).read_text(encoding="utf-8"))
    path_to_id = {
        str(Path(image["absolute_path"]).resolve()): int(image["id"])
        for image in gt["images"]
    }
    image_paths = list(path_to_id)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as source:
        source.write("".join(f"{path}\n" for path in image_paths))
        source.flush()
        rows = []
        for result in YOLO(args.model).predict(
            source.name,
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            device=str(args.device),
            batch=int(args.batch),
            half=bool(args.half),
            rect=True,
            end2end=False,
            verbose=False,
            stream=True,
        ):
            image_id = path_to_id[str(Path(result.path).resolve())]
            if result.boxes is None:
                continue
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(float, box)
                rows.append(
                    {
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
    target = Path(args.out_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    print(json.dumps({"images": len(image_paths), "predictions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
