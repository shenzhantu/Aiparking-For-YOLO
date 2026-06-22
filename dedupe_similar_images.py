"""
筛选静态/近重复素材，保留不太相似的代表样本用于人工检查。

默认只扫描:
    D:\\Aiparking\\image backcup\\Photo\\images*

输出到:
    D:\\Aiparking\\Aiparking For YOLO\\dedupe_preview\\unique_images

脚本不会删除或移动原始素材。默认使用硬链接节省磁盘空间；如果硬链接失败，
会自动退回到普通复制。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageStat


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"D:\Aiparking\image backcup\Photo")
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "dedupe_preview"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    source_name: str


@dataclass(frozen=True)
class ImageSignature:
    dhash: int
    luma_bucket: int


@dataclass(frozen=True)
class SelectedItem:
    item: ImageItem
    signature: ImageSignature
    group_id: int


@dataclass(frozen=True)
class DuplicateItem:
    item: ImageItem
    kept_path: Path
    group_id: int
    distance: int


@dataclass(frozen=True)
class FailedItem:
    item: ImageItem
    error: str


@dataclass
class DedupeResult:
    total_images: int
    selected: list[SelectedItem]
    duplicates: list[DuplicateItem]
    duplicate_groups: dict[int, list[DuplicateItem]]
    failed: list[FailedItem]


@dataclass(frozen=True)
class DedupeConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    threshold: int = 6
    copy_mode: str = "hardlink"
    overwrite: bool = True


class BKTreeNode:
    def __init__(self, selected: SelectedItem):
        self.selected = selected
        self.children: dict[int, BKTreeNode] = {}


class BKTree:
    def __init__(self) -> None:
        self.root: BKTreeNode | None = None

    def add(self, selected: SelectedItem) -> None:
        if self.root is None:
            self.root = BKTreeNode(selected)
            return

        node = self.root
        while True:
            distance = hamming_distance(selected.signature.dhash, node.selected.signature.dhash)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKTreeNode(selected)
                return
            node = child

    def query(self, signature: ImageSignature, threshold: int) -> list[SelectedItem]:
        if self.root is None:
            return []

        matches: list[SelectedItem] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(signature.dhash, node.selected.signature.dhash)
            if distance <= threshold:
                matches.append(node.selected)

            low = distance - threshold
            high = distance + threshold
            for edge_distance, child in node.children.items():
                if low <= edge_distance <= high:
                    stack.append(child)
        return matches


def discover_source_dirs(data_root: Path, selected_names: list[str] | None = None) -> list[Path]:
    if selected_names:
        dirs = [data_root / name for name in selected_names]
    else:
        dirs = sorted(
            [path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("images")],
            key=lambda path: path.name,
        )
    return [path for path in dirs if path.exists() and path.is_dir()]


def find_image_files(source_dirs: Iterable[Path]) -> list[ImageItem]:
    items: list[ImageItem] = []
    for source_dir in source_dirs:
        for path in sorted(source_dir.iterdir(), key=lambda p: p.name):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(ImageItem(path=path, source_name=source_dir.name))
    return items


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    with Image.open(image_path) as image:
        gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(gray.tobytes())

    value = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            value = (value << 1) | int(left > right)
    return value


def compute_luma_bucket(image_path: Path, bucket_size: int = 8) -> int:
    with Image.open(image_path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        mean_luma = ImageStat.Stat(gray).mean[0]
    return int(mean_luma // bucket_size)


def compute_signature(image_path: Path) -> ImageSignature:
    hash_size = 8
    with Image.open(image_path) as image:
        gray_hash = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        hash_pixels = list(gray_hash.tobytes())
        gray_luma = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        mean_luma = ImageStat.Stat(gray_luma).mean[0]

    value = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = hash_pixels[row_start + col]
            right = hash_pixels[row_start + col + 1]
            value = (value << 1) | int(left > right)

    return ImageSignature(dhash=value, luma_bucket=int(mean_luma // 8))


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def signature_distance(first: ImageSignature, second: ImageSignature) -> int:
    return hamming_distance(first.dhash, second.dhash) + abs(first.luma_bucket - second.luma_bucket)


def select_unique_images(
    items: list[ImageItem],
    threshold: int,
    progress: Callable[[int, int], None] | None = None,
) -> DedupeResult:
    selected: list[SelectedItem] = []
    duplicates: list[DuplicateItem] = []
    duplicate_groups: dict[int, list[DuplicateItem]] = {}
    failed: list[FailedItem] = []
    index = BKTree()

    total = len(items)
    for item_index, item in enumerate(items, start=1):
        if progress:
            progress(item_index, total)
        try:
            signature = compute_signature(item.path)
        except (OSError, ValueError) as exc:
            failed.append(FailedItem(item=item, error=str(exc)))
            continue

        best_match: tuple[SelectedItem, int] | None = None
        for kept in index.query(signature, threshold):
            distance = signature_distance(signature, kept.signature)
            if distance <= threshold and (best_match is None or distance < best_match[1]):
                best_match = (kept, distance)

        if best_match is None:
            kept_item = SelectedItem(item=item, signature=signature, group_id=len(selected) + 1)
            selected.append(kept_item)
            index.add(kept_item)
            continue

        kept, distance = best_match
        duplicate = DuplicateItem(
            item=item,
            kept_path=kept.item.path,
            group_id=kept.group_id,
            distance=distance,
        )
        duplicates.append(duplicate)
        duplicate_groups.setdefault(kept.group_id, []).append(duplicate)

    return DedupeResult(
        total_images=len(items),
        selected=selected,
        duplicates=duplicates,
        duplicate_groups=duplicate_groups,
        failed=failed,
    )


def relative_output_path(item: ImageItem) -> Path:
    return Path(item.source_name) / item.path.name


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    unique_dir = output_dir / "unique_images"
    reports_dir = output_dir / "reports"
    if overwrite:
        for path in (unique_dir, reports_dir):
            if path.exists():
                shutil.rmtree(path)
    unique_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_selected_items(result: DedupeResult, config: DedupeConfig) -> dict[str, int]:
    ensure_clean_output(config.output_dir, config.overwrite)
    unique_dir = config.output_dir / "unique_images"

    copied_images = 0
    copied_json = 0
    for selected in result.selected:
        dst = unique_dir / relative_output_path(selected.item)
        link_or_copy(selected.item.path, dst, config.copy_mode)
        copied_images += 1

        json_src = selected.item.path.with_suffix(".json")
        if json_src.exists():
            link_or_copy(json_src, dst.with_suffix(".json"), config.copy_mode)
            copied_json += 1

    return {"copied_images": copied_images, "copied_json": copied_json}


def write_reports(result: DedupeResult, config: DedupeConfig, copy_summary: dict[str, int]) -> None:
    reports_dir = config.output_dir / "reports"
    summary = {
        "data_root": str(config.data_root),
        "output_dir": str(config.output_dir),
        "threshold": config.threshold,
        "copy_mode": config.copy_mode,
        "total_images": result.total_images,
        "selected_images": len(result.selected),
        "filtered_similar_images": len(result.duplicates),
        "duplicate_groups": len(result.duplicate_groups),
        "failed_images": len(result.failed),
        **copy_summary,
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (reports_dir / "groups.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["status", "group_id", "source", "path", "kept_path", "distance"])
        for selected in result.selected:
            writer.writerow(
                [
                    "kept",
                    selected.group_id,
                    selected.item.source_name,
                    str(selected.item.path),
                    str(selected.item.path),
                    0,
                ]
            )
        for duplicate in result.duplicates:
            writer.writerow(
                [
                    "filtered",
                    duplicate.group_id,
                    duplicate.item.source_name,
                    str(duplicate.item.path),
                    str(duplicate.kept_path),
                    duplicate.distance,
                ]
            )

    with (reports_dir / "failed_images.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "path", "error"])
        for failed in result.failed:
            writer.writerow([failed.item.source_name, str(failed.item.path), failed.error])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛掉静态近重复图片，保留不太相似的代表样本。")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="默认: D:\\Aiparking\\image backcup\\Photo")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="默认: 项目目录\\dedupe_preview")
    parser.add_argument("--source", action="append", help="只扫描指定子目录名，可重复传入，如 --source images（6）")
    parser.add_argument("--threshold", type=int, default=6, help="相似阈值，越大筛得越狠，默认 6")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink", help="默认 hardlink，节省空间")
    parser.add_argument("--dry-run", action="store_true", help="只生成统计，不复制图片")
    parser.add_argument("--no-overwrite", action="store_true", help="不清空上一次 dedupe_preview 输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DedupeConfig(
        data_root=Path(args.data_root),
        output_dir=Path(args.output),
        threshold=args.threshold,
        copy_mode=args.copy_mode,
        overwrite=not args.no_overwrite,
    )

    source_dirs = discover_source_dirs(config.data_root, args.source)
    if not source_dirs:
        raise SystemExit(f"没有找到可扫描目录: {config.data_root}")

    print("扫描目录:")
    for source_dir in source_dirs:
        print(f"  - {source_dir}")

    items = find_image_files(source_dirs)
    print(f"找到图片: {len(items)} 张")
    print(f"相似阈值: {config.threshold}")

    result = select_unique_images(items, threshold=config.threshold)
    copy_summary = {"copied_images": 0, "copied_json": 0}
    if not args.dry_run:
        copy_summary = copy_selected_items(result, config)
    else:
        ensure_clean_output(config.output_dir, config.overwrite)

    write_reports(result, config, copy_summary)

    print("\n完成筛选")
    print(f"保留代表样本: {len(result.selected)} 张")
    print(f"筛掉近重复: {len(result.duplicates)} 张")
    print(f"近重复组数: {len(result.duplicate_groups)} 组")
    print(f"坏图/无法读取: {len(result.failed)} 张")
    print(f"输出目录: {config.output_dir / 'unique_images'}")
    print(f"报告目录: {config.output_dir / 'reports'}")


if __name__ == "__main__":
    main()
