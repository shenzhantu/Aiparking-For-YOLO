import json
import tempfile
import unittest
from pathlib import Path

from build_yolov8_dataset import (
    SourceConfig,
    convert_points_to_polygon,
    extract_yolo_segments,
    iter_weighted_items,
    item_repeat_count,
)


class BuildYolov8DatasetTests(unittest.TestCase):
    def test_rectangle_and_cuboid_shapes_become_four_point_polygons(self):
        self.assertEqual(
            convert_points_to_polygon([[10, 20], [30, 50]], "rectangle"),
            [[10.0, 20.0], [30.0, 20.0], [30.0, 50.0], [10.0, 50.0]],
        )
        self.assertEqual(
            convert_points_to_polygon([[30, 50], [10, 20]], "cuboid"),
            [[10.0, 20.0], [30.0, 20.0], [30.0, 50.0], [10.0, 50.0]],
        )

    def test_extract_segments_filters_low_confidence_and_unknown_labels(self):
        data = {
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "Parking",
                    "score": 0.39,
                    "shape_type": "polygon",
                    "points": [[0, 0], [10, 0], [10, 10]],
                },
                {
                    "label": "barrier",
                    "score": 0.4,
                    "shape_type": "rectangle",
                    "points": [[10, 20], [30, 50]],
                },
                {
                    "label": "d",
                    "shape_type": "polygon",
                    "points": [[0, 0], [1, 0], [1, 1]],
                },
            ],
        }

        segments, stats = extract_yolo_segments(data, {"Parking": 0, "barrier": 1}, 0.4)

        self.assertEqual(len(segments), 1)
        self.assertTrue(segments[0].startswith("1 "))
        self.assertEqual(stats["low_confidence"], 1)
        self.assertEqual(stats["unknown_label"], 1)

    def test_iter_weighted_items_repeats_newer_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "images_new"
            source_dir.mkdir()
            image_path = source_dir / "frame_0001.jpg"
            json_path = source_dir / "frame_0001.json"
            image_path.write_bytes(b"fake")
            json_path.write_text(
                json.dumps({"imagePath": image_path.name, "shapes": []}),
                encoding="utf-8",
            )

            items = list(iter_weighted_items([SourceConfig(source_dir, 3)]))

        self.assertEqual(len(items), 3)
        self.assertEqual([item.repeat_index for item in items], [0, 1, 2])

    def test_barrier_class_boost_multiplies_training_repeats(self):
        item = type(
            "FakeItem",
            (),
            {
                "source": SourceConfig(Path("images（4）"), 2),
                "class_counts": {1: 3},
            },
        )()

        self.assertEqual(item_repeat_count(item, {1: 4}), 8)


if __name__ == "__main__":
    unittest.main()
