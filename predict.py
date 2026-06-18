"""
Auto-label images with a trained YOLO segmentation model and export X-AnyLabeling JSON.

Defaults target the current YOLOv8 iteration:
  model:  D:\\Aiparking\\Aiparking For YOLO\\runs\\parking_yolov8_seg\\weights\\best.pt
  target: D:\\Aiparking\\image backcup\\images（6）
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "runs" / "parking_yolov8_seg" / "weights" / "best.pt"
DEFAULT_TARGET = Path(r"D:\Aiparking\image backcup\images（6）")
DEFAULT_CONFIDENCE = 0.4
CLASS_NAMES = {0: "Parking", 1: "barrier"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def masks_to_shapes(result) -> list[dict]:
    detections: list[dict] = []
    if result.masks is not None and result.masks.xy is not None:
        for index, polygon in enumerate(result.masks.xy):
            points = polygon.tolist()
            if len(points) < 3:
                continue
            detections.append(
                {
                    "points": points,
                    "confidence": float(result.boxes.conf[index]),
                    "class_id": int(result.boxes.cls[index]),
                }
            )
    elif result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append(
                {
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                }
            )
    return detections


def shape_for_detection(det: dict) -> dict:
    label = CLASS_NAMES.get(det["class_id"], f"class_{det['class_id']}")
    points = det["points"]
    shape_type = "polygon"

    if label == "barrier":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        points = [[min(xs), min(ys)], [max(xs), max(ys)]]
        shape_type = "rectangle"

    return {
        "label": label,
        "score": det["confidence"],
        "points": points,
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": shape_type,
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def create_xanylabeling_json(image_path: Path, detections: list[dict], width: int, height: int) -> dict:
    return {
        "version": "4.0.0-beta.7",
        "flags": {},
        "checked": False,
        "shapes": [shape_for_detection(det) for det in detections],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
    }


def collect_images(target: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"):
        images.extend(Path(path) for path in glob.glob(str(target / pattern)))
    return sorted(set(images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-seg auto-label to X-AnyLabeling JSON")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained .pt model")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="Image directory to annotate")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON annotations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    target = Path(args.target)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not target.exists():
        raise SystemExit(f"Target directory not found: {target}")

    print("=" * 60)
    print("AiParking auto-label")
    print("=" * 60)
    print(f"model: {model_path}")
    print(f"target: {target}")
    print(f"confidence: {args.conf}")

    model = YOLO(str(model_path))
    images = collect_images(target)
    to_process = [
        image for image in images
        if args.overwrite or not image.with_suffix(".json").exists()
    ]

    print(f"images found: {len(images)}")
    print(f"already annotated: {len(images) - len(to_process)}")
    print(f"to process: {len(to_process)}")
    if not to_process:
        return

    device = "0" if torch.cuda.is_available() else "cpu"
    total_detections = 0
    empty_images = 0
    processed = 0

    for start in range(0, len(to_process), args.batch):
        batch = to_process[start:start + args.batch]
        results = model.predict(
            source=[str(path) for path in batch],
            conf=args.conf,
            device=device,
            verbose=False,
        )

        for index, result in enumerate(results):
            image_path = batch[index]
            height, width = result.orig_shape
            detections = masks_to_shapes(result)
            if not detections:
                empty_images += 1
            else:
                out = create_xanylabeling_json(image_path, detections, width, height)
                image_path.with_suffix(".json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                total_detections += len(detections)
            processed += 1

        print(
            f"progress: {processed}/{len(to_process)} "
            f"({total_detections} detections, {empty_images} empty)"
        )

    print("\nFinished")
    print(f"processed: {processed}")
    print(f"detections: {total_detections}")
    print(f"empty images without JSON: {empty_images}")
    print(f"annotations saved to: {target}")


if __name__ == "__main__":
    main()
