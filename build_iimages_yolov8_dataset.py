"""Build a fresh YOLOv8-seg dataset from the recursive IImages tree.

This builder is intentionally separate from the older weighted-source builder.
It keeps every image with a same-directory X-AnyLabeling JSON, including true
empty JSON files as negative samples, while excluding metadata and derived
annotation folders that do not contain the corresponding images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_NAME = "Parking"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    json_path: Path
    relative_dir: str
    group: str
    label_lines: tuple[str, ...]
    instances: int
    is_negative: bool


def distance(first: list[float], second: list[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def cross(first: list[float], middle: list[float], last: list[float]) -> float:
    return abs(
        (middle[0] - first[0]) * (last[1] - middle[1])
        - (middle[1] - first[1]) * (last[0] - middle[0])
    )


def normalize_points(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    clean: list[list[float]] = []
    for point in points:
        if len(point) < 2:
            continue
        x = min(float(width), max(0.0, float(point[0])))
        y = min(float(height), max(0.0, float(point[1])))
        if not clean or distance(clean[-1], [x, y]) > 1e-4:
            clean.append([x, y])

    if len(clean) > 1 and distance(clean[0], clean[-1]) <= 1e-4:
        clean.pop()

    # Reduce extra vertices by removing the point with the smallest local
    # triangle contribution. This preserves the original point order.
    while len(clean) > 4:
        candidates = []
        for index in range(len(clean)):
            previous = clean[index - 1]
            current = clean[index]
            following = clean[(index + 1) % len(clean)]
            base = distance(previous, following)
            contribution = cross(previous, current, following) / max(base, 1e-6)
            candidates.append((contribution, index))
        _, remove_index = min(candidates)
        clean.pop(remove_index)

    return clean


def canonical_label(raw_label: object) -> str | None:
    value = str(raw_label or "").strip().lower()
    if value == "parking":
        return LABEL_NAME
    return None


def find_sibling_image(json_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def group_name(relative_dir: Path) -> str:
    parts = relative_dir.parts
    if len(parts) >= 2 and parts[1].lower().startswith("ground"):
        return "\\".join(parts[:2])
    return parts[0] if parts else "root"


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return token.strip("_") or "source"


def parse_json_sample(
    json_path: Path,
    image_path: Path,
    root: Path,
    min_confidence: float,
    include_empty: bool,
) -> tuple[Sample | None, Counter]:
    stats: Counter = Counter(json_files=1)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        stats["bad_json"] += 1
        return None, stats

    try:
        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            with Image.open(image_path) as image:
                width, height = image.size
    except (OSError, ValueError):
        stats["bad_image"] += 1
        return None, stats

    shapes = data.get("shapes") or []
    if not shapes:
        stats["empty_json"] += 1
        if not include_empty:
            stats["empty_excluded"] += 1
            return None, stats
        relative_dir = image_path.parent.relative_to(root)
        return Sample(
            image_path=image_path,
            json_path=json_path,
            relative_dir=str(relative_dir),
            group=group_name(relative_dir),
            label_lines=(),
            instances=0,
            is_negative=True,
        ), stats

    label_lines: list[str] = []
    for shape in shapes:
        label = canonical_label(shape.get("label"))
        if label is None:
            stats["unknown_label"] += 1
            continue

        score = shape.get("score")
        if score is not None:
            try:
                if float(score) < min_confidence:
                    stats["low_confidence_shapes"] += 1
                    continue
            except (TypeError, ValueError):
                stats["invalid_score"] += 1

        raw_points = shape.get("points") or []
        if shape.get("shape_type") in {"rectangle", "cuboid"} and len(raw_points) == 2:
            x1, y1 = raw_points[0]
            x2, y2 = raw_points[1]
            raw_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        points = normalize_points(raw_points, width, height)
        if len(points) != 4:
            stats["non_quadrilateral_shapes"] += 1
            continue

        coordinates = []
        for x, y in points:
            coordinates.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
        label_lines.append("0 " + " ".join(coordinates))
        stats["instances"] += 1

    if not label_lines:
        # A JSON with shapes that all fail confidence/geometry checks is not a
        # confirmed negative image, so exclude it instead of teaching a false
        # background example.
        stats["excluded_after_filter"] += 1
        return None, stats

    relative_dir = image_path.parent.relative_to(root)
    return Sample(
        image_path=image_path,
        json_path=json_path,
        relative_dir=str(relative_dir),
        group=group_name(relative_dir),
        label_lines=tuple(label_lines),
        instances=len(label_lines),
        is_negative=False,
    ), stats


def discover_samples(
    root: Path,
    min_confidence: float,
    include_empty: bool,
) -> tuple[list[Sample], Counter]:
    samples: list[Sample] = []
    totals: Counter = Counter()
    for json_path in sorted(root.rglob("*.json")):
        relative_parts = json_path.relative_to(root).parts
        if not relative_parts or relative_parts[-2].lower() not in {"images", "raw_images"}:
            continue
        image_path = find_sibling_image(json_path)
        if image_path is None:
            totals["missing_image"] += 1
            continue
        sample, stats = parse_json_sample(
            json_path,
            image_path,
            root,
            min_confidence,
            include_empty,
        )
        totals.update(stats)
        if sample is not None:
            samples.append(sample)
    return samples, totals


def split_by_group(
    samples: list[Sample],
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[str]]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group].append(sample)

    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    target = max(1, int(len(samples) * val_ratio))
    val_groups: list[str] = []
    val_count = 0
    for group in groups:
        if val_count >= target:
            break
        val_groups.append(group)
        val_count += len(grouped[group])

    val_set = set(val_groups)
    train = [sample for sample in samples if sample.group not in val_set]
    val = [sample for sample in samples if sample.group in val_set]
    return train, val, val_groups


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def reset_output(output: Path, root: Path) -> None:
    output = output.resolve()
    root = root.resolve()
    if output == root or root not in output.parents:
        raise ValueError(f"Refusing to reset output outside data root: {output}")
    if not output.name.startswith("dataset_yolov8n"):
        raise ValueError(f"Refusing to reset unexpected output directory: {output}")
    if output.exists():
        shutil.rmtree(output)


def write_dataset(
    train: list[Sample],
    val: list[Sample],
    output: Path,
    val_groups: list[str],
    totals: Counter,
    min_confidence: float,
    seed: int,
) -> dict:
    link_modes = Counter()
    instance_counts = Counter()

    for split, items in (("train", train), ("val", val)):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for sample in items:
            relative_token = safe_token(sample.relative_dir.replace("\\", "__"))
            stem = f"{relative_token}__{sample.image_path.stem}"
            image_destination = image_dir / f"{stem}{sample.image_path.suffix.lower()}"
            label_destination = label_dir / f"{stem}.txt"
            link_modes[link_or_copy(sample.image_path, image_destination)] += 1
            label_destination.write_text("\n".join(sample.label_lines), encoding="utf-8")
            instance_counts[split] += sample.instances

    yaml_path = output / "parking_yolov8.yaml"
    yaml_path.write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: Parking\n",
        encoding="utf-8",
    )

    summary = {
        "source_root": str(output.parent),
        "output": str(output),
        "classes": [LABEL_NAME],
        "min_confidence": min_confidence,
        "seed": seed,
        "val_groups": val_groups,
        "train_images": len(train),
        "val_images": len(val),
        "train_instances": instance_counts["train"],
        "val_instances": instance_counts["val"],
        "train_negative_images": sum(item.is_negative for item in train),
        "val_negative_images": sum(item.is_negative for item in val),
        "link_modes": dict(link_modes),
        "annotation_stats": dict(totals),
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLOv8n-seg data from IImages")
    parser.add_argument("--data-root", default=r"D:\Aiparking\IImages")
    parser.add_argument("--output", default=r"D:\Aiparking\IImages\dataset_yolov8n_quad_v1")
    parser.add_argument("--min-conf", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--exclude-empty", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if not root.exists():
        raise SystemExit(f"Data root does not exist: {root}")

    samples, totals = discover_samples(root, args.min_conf, not args.exclude_empty)
    if not samples:
        raise SystemExit("No usable image/JSON pairs found.")

    train, val, val_groups = split_by_group(samples, args.val_ratio, args.seed)
    reset_output(output, root)
    summary = write_dataset(train, val, output, val_groups, totals, args.min_conf, args.seed)

    print("IImages dataset built")
    print(f"source: {root}")
    print(f"output: {output}")
    print(f"usable pairs: {len(samples)}")
    print(f"train images: {summary['train_images']}")
    print(f"val images: {summary['val_images']}")
    print(f"train instances: {summary['train_instances']}")
    print(f"val instances: {summary['val_instances']}")
    print(f"train negatives: {summary['train_negative_images']}")
    print(f"val negatives: {summary['val_negative_images']}")
    print(f"validation groups: {', '.join(val_groups)}")
    print(f"link modes: {summary['link_modes']}")
    print(f"yaml: {output / 'parking_yolov8.yaml'}")
    print(f"summary: {output / 'build_summary.json'}")


if __name__ == "__main__":
    main()
