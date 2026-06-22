"""
Select premium images from a new unlabeled batch using dedupe + teacher predictions.

Default source:
    D:\\Aiparking\\image backcup\\Photo\\images（7）\\images

Default output:
    D:\\Aiparking\\Premium photo\\images7_teacher_selected

The script does not change original images. It writes:
    high_conf/         images + teacher JSON for lightweight student training
    uncertain/         images + teacher JSON for manual review
    missed_or_empty/   images with no teacher detections
    reports/           summary and manifest files
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from dedupe_similar_images import (
    ImageItem,
    select_unique_images,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(r"D:\Aiparking\image backcup\Photo\images（7）\images")
DEFAULT_OUTPUT = Path(r"D:\Aiparking\Premium photo\images7_teacher_selected")
DEFAULT_MODEL = PROJECT_DIR / "models" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "Parking", 1: "barrier"}


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    points: list[list[float]]


@dataclass(frozen=True)
class SelectedImage:
    path: Path
    group_id: int


@dataclass(frozen=True)
class DuplicateImage:
    path: Path
    kept_path: Path
    group_id: int
    distance: int


@dataclass(frozen=True)
class FailedImage:
    path: Path
    error: str


def collect_images_recursive(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def dedupe_images(paths: list[Path], threshold: int) -> tuple[list[SelectedImage], list[DuplicateImage], list[FailedImage]]:
    items = [ImageItem(path=path, source_name=path.parent.name) for path in paths]

    def progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % 200 == 0:
            print(f"dedupe progress: {done}/{total}", flush=True)

    result = select_unique_images(items, threshold, progress=progress)
    selected = [SelectedImage(item.item.path, item.group_id) for item in result.selected]
    duplicates = [
        DuplicateImage(item.item.path, item.kept_path, item.group_id, item.distance)
        for item in result.duplicates
    ]
    failed = [FailedImage(item.item.path, item.error) for item in result.failed]
    return selected, duplicates, failed


def detections_from_result(result) -> list[Detection]:
    detections: list[Detection] = []
    if result.masks is not None and result.masks.xy is not None:
        for index, polygon in enumerate(result.masks.xy):
            points = polygon.tolist()
            if len(points) < 3:
                continue
            detections.append(
                Detection(
                    class_id=int(result.boxes.cls[index]),
                    confidence=float(result.boxes.conf[index]),
                    points=points,
                )
            )
    elif result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append(
                Detection(
                    class_id=int(box.cls[0]),
                    confidence=float(box.conf[0]),
                    points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                )
            )
    return detections


def bucket_for_detections(detections: list[Detection], high_confidence: float) -> str:
    if not detections:
        return "missed_or_empty"
    if min(det.confidence for det in detections) >= high_confidence:
        return "high_conf"
    return "uncertain"


def shape_for_detection(det: Detection) -> dict:
    label = CLASS_NAMES.get(det.class_id, f"class_{det.class_id}")
    points = det.points
    shape_type = "polygon"

    if label == "barrier":
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        points = [[min(xs), min(ys)], [max(xs), max(ys)]]
        shape_type = "rectangle"

    return {
        "label": label,
        "score": det.confidence,
        "points": points,
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": shape_type,
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def create_label_json(image_path: Path, detections: list[Detection], width: int, height: int) -> dict:
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


def reset_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for name in ["high_conf", "uncertain", "missed_or_empty", "reports"]:
        (output / name).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def unique_output_name(path: Path, used_names: set[str]) -> str:
    candidate = path.name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    index = 1
    while True:
        candidate = f"{path.stem}_{index}{path.suffix.lower()}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def write_reports(
    output: Path,
    source: Path,
    total_images: int,
    selected: list[SelectedImage],
    duplicates: list[DuplicateImage],
    failed: list[FailedImage],
    bucket_counts: dict[str, int],
    manifest_rows: list[dict],
    threshold: int,
    high_confidence: float,
) -> None:
    reports = output / "reports"
    summary = {
        "source": str(source),
        "output": str(output),
        "dedupe_threshold": threshold,
        "high_confidence": high_confidence,
        "total_images": total_images,
        "selected_after_dedupe": len(selected),
        "filtered_duplicates": len(duplicates),
        "failed_images": len(failed),
        "bucket_counts": bucket_counts,
    }
    (reports / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (reports / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["bucket", "source_path", "output_image", "output_json", "detections", "min_confidence"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (reports / "duplicates.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group_id", "path", "kept_path", "distance"])
        for item in duplicates:
            writer.writerow([item.group_id, item.path, item.kept_path, item.distance])

    with (reports / "failed_images.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "error"])
        for item in failed:
            writer.writerow([item.path, item.error])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select premium images with dedupe + teacher pseudo-labeling")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dedupe-threshold", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.4, help="Teacher inference minimum confidence")
    parser.add_argument("--high-conf", type=float, default=0.9, help="All detections must be >= this for high_conf")
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    model_path = Path(args.model)
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    if not model_path.exists():
        raise SystemExit(f"Teacher model not found: {model_path}")

    print("=" * 60, flush=True)
    print("AiParking premium image selection", flush=True)
    print("=" * 60, flush=True)
    print(f"source: {source}", flush=True)
    print(f"output: {output}", flush=True)
    print(f"teacher: {model_path}", flush=True)
    print(f"dedupe threshold: {args.dedupe_threshold}", flush=True)
    print(f"teacher conf: {args.conf}", flush=True)
    print(f"high conf: {args.high_conf}", flush=True)

    reset_output(output)
    paths = collect_images_recursive(source)
    selected, duplicates, failed = dedupe_images(paths, args.dedupe_threshold)
    print(f"images found: {len(paths)}", flush=True)
    print(f"selected after dedupe: {len(selected)}", flush=True)
    print(f"duplicates filtered: {len(duplicates)}", flush=True)
    print(f"failed images: {len(failed)}", flush=True)

    model = YOLO(str(model_path))
    device = "0" if torch.cuda.is_available() else "cpu"
    bucket_counts = {"high_conf": 0, "uncertain": 0, "missed_or_empty": 0}
    manifest_rows: list[dict] = []
    used_names: dict[str, set[str]] = {name: set() for name in bucket_counts}

    for start in range(0, len(selected), args.batch):
        batch = selected[start:start + args.batch]
        results = model.predict(
            source=[str(item.path) for item in batch],
            conf=args.conf,
            device=device,
            verbose=False,
        )
        for index, result in enumerate(results):
            item = batch[index]
            detections = detections_from_result(result)
            bucket = bucket_for_detections(detections, args.high_conf)
            bucket_counts[bucket] += 1

            out_name = unique_output_name(item.path, used_names[bucket])
            out_image = output / bucket / out_name
            link_or_copy(item.path, out_image)

            out_json = ""
            min_confidence = ""
            if detections:
                height, width = result.orig_shape
                label_json = create_label_json(out_image, detections, width, height)
                json_path = out_image.with_suffix(".json")
                json_path.write_text(json.dumps(label_json, ensure_ascii=False, indent=2), encoding="utf-8")
                out_json = str(json_path)
                min_confidence = f"{min(det.confidence for det in detections):.6f}"

            manifest_rows.append(
                {
                    "bucket": bucket,
                    "source_path": str(item.path),
                    "output_image": str(out_image),
                    "output_json": out_json,
                    "detections": len(detections),
                    "min_confidence": min_confidence,
                }
            )

        processed = min(start + args.batch, len(selected))
        print(
            f"progress: {processed}/{len(selected)} "
            f"(high={bucket_counts['high_conf']}, uncertain={bucket_counts['uncertain']}, "
            f"empty={bucket_counts['missed_or_empty']})",
            flush=True,
        )

    write_reports(
        output=output,
        source=source,
        total_images=len(paths),
        selected=selected,
        duplicates=duplicates,
        failed=failed,
        bucket_counts=bucket_counts,
        manifest_rows=manifest_rows,
        threshold=args.dedupe_threshold,
        high_confidence=args.high_conf,
    )

    print("\nFinished", flush=True)
    print(f"high_conf: {bucket_counts['high_conf']}", flush=True)
    print(f"uncertain: {bucket_counts['uncertain']}", flush=True)
    print(f"missed_or_empty: {bucket_counts['missed_or_empty']}", flush=True)
    print(f"reports: {output / 'reports'}", flush=True)


if __name__ == "__main__":
    main()
