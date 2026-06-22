import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from select_premium_images import (
    Detection,
    bucket_for_detections,
    collect_images_recursive,
    create_label_json,
    shape_for_detection,
)


class SelectPremiumImagesTests(unittest.TestCase):
    def write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (80, 120, 160)).save(path)

    def test_bucket_for_detections_splits_high_uncertain_and_empty(self):
        self.assertEqual(bucket_for_detections([], 0.9), "missed_or_empty")
        self.assertEqual(
            bucket_for_detections([Detection(0, 0.89, [[0, 0], [1, 0], [1, 1]])], 0.9),
            "uncertain",
        )
        self.assertEqual(
            bucket_for_detections([Detection(0, 0.91, [[0, 0], [1, 0], [1, 1]])], 0.9),
            "high_conf",
        )

    def test_barrier_detection_is_written_as_rectangle(self):
        shape = shape_for_detection(
            Detection(1, 0.95, [[10, 20], [30, 22], [35, 50], [8, 48]])
        )

        self.assertEqual(shape["label"], "barrier")
        self.assertEqual(shape["shape_type"], "rectangle")
        self.assertEqual(shape["points"], [[8, 20], [35, 50]])

    def test_create_label_json_uses_image_filename(self):
        data = create_label_json(
            Path("frame_0001.jpg"),
            [Detection(0, 0.95, [[0, 0], [10, 0], [10, 10]])],
            width=100,
            height=80,
        )

        self.assertEqual(data["imagePath"], "frame_0001.jpg")
        self.assertEqual(data["imageWidth"], 100)
        self.assertEqual(data["imageHeight"], 80)
        self.assertEqual(data["shapes"][0]["label"], "Parking")

    def test_collect_images_recursive_finds_nested_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_image(root / "images" / "frame_0001.jpg")
            self.write_image(root / "sample_review" / "preview.png")
            (root / "manifest.json").write_text(json.dumps({}), encoding="utf-8")

            images = collect_images_recursive(root)

        self.assertEqual([path.name for path in images], ["frame_0001.jpg", "preview.png"])


if __name__ == "__main__":
    unittest.main()
