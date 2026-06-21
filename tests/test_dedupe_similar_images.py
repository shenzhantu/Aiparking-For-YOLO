import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dedupe_similar_images import (
    DedupeConfig,
    compute_dhash,
    copy_selected_items,
    find_image_files,
    select_unique_images,
    write_reports,
)


class DedupeSimilarImagesTests(unittest.TestCase):
    def write_image(self, path: Path, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (32, 32), color).save(path)

    def test_dhash_matches_identical_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.jpg"
            second = root / "b.jpg"
            self.write_image(first, (120, 120, 120))
            self.write_image(second, (120, 120, 120))

            self.assertEqual(compute_dhash(first), compute_dhash(second))

    def test_select_unique_images_keeps_one_representative_per_similar_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "images"
            source.mkdir()
            self.write_image(source / "frame_0001.jpg", (20, 20, 20))
            self.write_image(source / "frame_0002.jpg", (20, 20, 20))
            self.write_image(source / "frame_0003.jpg", (240, 240, 240))

            items = find_image_files([source])
            result = select_unique_images(items, threshold=0)

            self.assertEqual(result.total_images, 3)
            self.assertEqual(len(result.selected), 2)
            self.assertEqual(len(result.duplicate_groups), 1)

    def test_copy_selected_items_also_copies_matching_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "images"
            output = root / "output"
            source.mkdir()
            image = source / "frame_0001.jpg"
            label = source / "frame_0001.json"
            self.write_image(image, (20, 20, 20))
            label.write_text(json.dumps({"imagePath": image.name}), encoding="utf-8")

            result = select_unique_images(find_image_files([source]), threshold=0)
            summary = copy_selected_items(
                result,
                DedupeConfig(data_root=root, output_dir=output, copy_mode="copy"),
            )

            selected_image = output / "unique_images" / "images" / image.name
            selected_label = selected_image.with_suffix(".json")
            self.assertEqual(summary["copied_images"], 1)
            self.assertTrue(selected_image.exists())
            self.assertTrue(selected_label.exists())

    def test_corrupt_images_are_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "images"
            output = root / "output"
            source.mkdir()
            self.write_image(source / "frame_0001.jpg", (20, 20, 20))
            (source / "broken.jpg").write_bytes(b"not a real image")

            result = select_unique_images(find_image_files([source]), threshold=0)
            copy_summary = copy_selected_items(
                result,
                DedupeConfig(data_root=root, output_dir=output, copy_mode="copy"),
            )
            write_reports(
                result,
                DedupeConfig(data_root=root, output_dir=output, copy_mode="copy"),
                copy_summary,
            )

            self.assertEqual(result.total_images, 2)
            self.assertEqual(len(result.selected), 1)
            self.assertEqual(len(result.failed), 1)
            self.assertTrue((output / "reports" / "failed_images.csv").exists())


if __name__ == "__main__":
    unittest.main()
