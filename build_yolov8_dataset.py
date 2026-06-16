"""
Build a weighted YOLOv8 segmentation dataset from X-AnyLabeling/LabelMe JSON files.

The source images stay outside the Git repository under D:\\Aiparking\\image backcup.
The generated training dataset also stays outside the repository by default.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"D:\Aiparking\image backcup")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "dataset_yolov8_weighted"
DEFAULT_CLASSES = ["Parking", "barrier"]
DEFAULT_MIN_CONFIDENCE = 0.4
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SEED = 20260616
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]


@dataclass(frozen=True)
class SourceConfig:
    path: Path
    weight: int


@dataclass(frozen=True)
class WeightedItem:
    source: SourceConfig
    image_path: Path
    json_path: Path
    repeat_index: int


@dataclass(frozen=True)
class BaseItem:
    source: SourceConfig
    image_path: Path
    json_path: Path
    label_lines: list[str]
    class_counts: Counter
    stats: Counter


def source_weight(name: str) -> int:
    """Return a conservative recency weight based on the source folder name."""
    if name == "images":
        return 1
    match = re.search(r"（(\d+)）", name)
    if not match:
        return 1
    index = int(match.group(1))
    if index <= 1:
        return 1
    return min(index, 4)


def source_weight(name: str) -> int:
    """Return a conservative recency weight based on the source folder name."""
    if name == "images":
        return 1
    match = re.search(r"（(\d+)）", name)
    if not match:
        return 1
    index = int(match.group(1))
    if index <= 1:
        return 1
    return min(index, 4)


def discover_sources(data_root: Path, exclude: set[str]) -> list[SourceConfig]:
    sources: list[SourceConfig] = []
    for path in sorted(data_root.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or not path.name.startswith("images"):
            continue
        if path.name in exclude:
            continue
        if not any(path.glob("*.json")):
            continue
        sources.append(SourceConfig(path=path, weight=source_weight(path.name)))
    return sources


def parse_sources(data_root: Path, raw_sources: list[str] | None, exclude: set[str]) -> list[SourceConfig]:
    if not raw_sources:
        return discover_sources(data_root, exclude)

    sources: list[SourceConfig] = []
    for item in raw_sources:
        if ":" in item:
            name, raw_weight = item.rsplit(":", 1)
            weight = int(raw_weight)
        else:
            name = item
            weight = source_weight(name)
        path = Path(name)
        if not path.is_absolute():
            path = data_root / name
        if path.name in exclude:
            continue
        sources.append(SourceConfig(path=path, weight=max(1, weight)))
    return sources


def find_image_for_json(json_path: Path, data: dict) -> Path | None:
    image_name = data.get("imagePath")
    candidates: list[Path] = []
    if image_name:
        candidates.append(json_path.parent / image_name)
    for ext in IMAGE_EXTENSIONS:
        candidates.append(json_path.with_suffix(ext))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def convert_points_to_polygon(points: list[list[float]], shape_type: str) -> list[list[float]]:
    if shape_type in {"rectangle", "cuboid"} and len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        left, right = sorted([float(x1), float(x2)])
        top, bottom = sorted([float(y1), float(y2)])
        return [[left, top], [right, top], [right, bottom], [left, bottom]]
    return [[float(x), float(y)] for x, y in points]


def normalize(value: float, size: int) -> float:
    if size <= 0:
        return 0.0
    return min(1.0, max(0.0, value / size))


def extract_yolo_segments(
    data: dict,
    label_to_id: dict[str, int],
    min_confidence: float,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[str], Counter]:
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if image_size and (width <= 0 or height <= 0):
        width, height = image_size

    stats: Counter = Counter()
    lines: list[str] = []

    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in label_to_id:
            stats["unknown_label"] += 1
            continue

        score = shape.get("score")
        if score is not None and float(score) < min_confidence:
            stats["low_confidence"] += 1
            continue

        points = shape.get("points") or []
        polygon = convert_points_to_polygon(points, shape.get("shape_type", "polygon"))
        if len(polygon) < 3:
            stats["invalid_polygon"] += 1
            continue

        coords: list[str] = []
        for x, y in polygon:
            coords.append(f"{normalize(x, width):.6f}")
            coords.append(f"{normalize(y, height):.6f}")

        lines.append(f"{label_to_id[label]} " + " ".join(coords))
        stats[f"class:{label}"] += 1

    return lines, stats


def iter_weighted_items(sources: Iterable[SourceConfig]) -> Iterable[WeightedItem]:
    for source in sources:
        for json_path in sorted(source.path.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            image_path = find_image_for_json(json_path, data)
            if image_path is None:
                continue
            for repeat_index in range(source.weight):
                yield WeightedItem(source, image_path, json_path, repeat_index)


def load_base_items(
    sources: Iterable[SourceConfig],
    label_to_id: dict[str, int],
    min_confidence: float,
) -> tuple[list[BaseItem], Counter]:
    items: list[BaseItem] = []
    totals: Counter = Counter()

    for source in sources:
        for json_path in sorted(source.path.glob("*.json")):
            totals["json_files"] += 1
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                totals["bad_json"] += 1
                continue

            image_path = find_image_for_json(json_path, data)
            if image_path is None:
                totals["missing_image"] += 1
                continue

            image_size = None
            if not data.get("imageWidth") or not data.get("imageHeight"):
                with Image.open(image_path) as image:
                    image_size = image.size

            lines, stats = extract_yolo_segments(data, label_to_id, min_confidence, image_size)
            totals.update(stats)
            if not lines:
                totals["empty_after_filter"] += 1
                continue

            class_counts = Counter()
            for line in lines:
                class_id = int(line.split(" ", 1)[0])
                class_counts[class_id] += 1

            items.append(
                BaseItem(
                    source=source,
                    image_path=image_path,
                    json_path=json_path,
                    label_lines=lines,
                    class_counts=class_counts,
                    stats=stats,
                )
            )
            totals["usable_images"] += 1

    return items, totals


def safe_reset_output_dir(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    root = DEFAULT_DATA_ROOT.resolve()
    if root not in output_dir.parents:
        raise ValueError(f"Refusing to delete output outside data root: {output_dir}")
    if not output_dir.name.startswith("dataset_yolov8"):
        raise ValueError(f"Refusing to delete unexpected output directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def safe_name(value: str) -> str:
    value = value.replace("（", "_").replace("）", "")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "source"


def safe_name(value: str) -> str:
    value = value.replace("（", "_").replace("）", "")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "source"


def link_or_copy(src: Path, dst: Path) -> str:
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_dataset(
    items: list[BaseItem],
    output_dir: Path,
    classes: list[str],
    val_ratio: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = {
        "train_images": 0,
        "val_images": 0,
        "train_instances": Counter(),
        "val_instances": Counter(),
        "link_modes": Counter(),
        "source_images": defaultdict(int),
        "source_weighted_train_images": defaultdict(int),
    }

    def write_one(item: BaseItem, split: str, repeat_index: int) -> None:
        source_label = safe_name(item.source.path.name)
        stem = f"{source_label}__{item.image_path.stem}"
        if split == "train":
            stem = f"{stem}__r{repeat_index}"
        image_dst = output_dir / "images" / split / f"{stem}{item.image_path.suffix.lower()}"
        label_dst = output_dir / "labels" / split / f"{stem}.txt"

        mode = link_or_copy(item.image_path, image_dst)
        label_dst.write_text("\n".join(item.label_lines), encoding="utf-8")

        summary["link_modes"][mode] += 1
        summary[f"{split}_images"] += 1
        for class_id, count in item.class_counts.items():
            summary[f"{split}_instances"][classes[class_id]] += count

    for item in val_items:
        summary["source_images"][item.source.path.name] += 1
        write_one(item, "val", 0)

    for item in train_items:
        summary["source_images"][item.source.path.name] += 1
        for repeat_index in range(item.source.weight):
            summary["source_weighted_train_images"][item.source.path.name] += 1
            write_one(item, "train", repeat_index)

    yaml_path = output_dir / "parking_yolov8.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    yaml_path.write_text(
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )

    serializable = {}
    for key, value in summary.items():
        if isinstance(value, Counter) or isinstance(value, defaultdict):
            serializable[key] = dict(value)
        else:
            serializable[key] = value
    (output_dir / "build_summary.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weighted YOLOv8-seg dataset")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="External image/JSON root")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Generated YOLO dataset directory")
    parser.add_argument("--source", action="append", help="Source folder or folder:weight. Repeatable.")
    parser.add_argument("--exclude", action="append", default=["images（5）"], help="Source folder name to exclude")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    sources = parse_sources(data_root, args.source, set(args.exclude or []))
    if not sources:
        raise SystemExit("No source directories with JSON annotations found.")

    print("Training sources:")
    for source in sources:
        print(f"  {source.path}  weight={source.weight}")

    label_to_id = {name: idx for idx, name in enumerate(args.classes)}
    items, totals = load_base_items(sources, label_to_id, args.min_conf)
    if not items:
        raise SystemExit("No usable annotated images after filtering.")

    safe_reset_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = write_dataset(items, output_dir, args.classes, args.val_ratio, args.seed)

    print("\nRaw annotation stats:")
    for key, value in sorted(totals.items()):
        print(f"  {key}: {value}")
    print("\nGenerated dataset:")
    print(f"  output: {output_dir}")
    print(f"  train images: {summary['train_images']}")
    print(f"  val images: {summary['val_images']}")
    print(f"  train instances: {summary['train_instances']}")
    print(f"  val instances: {summary['val_instances']}")
    print(f"  link modes: {summary['link_modes']}")
    print(f"  yaml: {output_dir / 'parking_yolov8.yaml'}")


if __name__ == "__main__":
    main()
