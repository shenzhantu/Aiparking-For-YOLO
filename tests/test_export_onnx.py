import unittest
from pathlib import Path

from export_onnx import output_path_for_size


class ExportOnnxTests(unittest.TestCase):
    def test_output_path_for_default_and_lightweight_sizes(self):
        model = Path("models") / "best.pt"

        self.assertEqual(output_path_for_size(model, 640), Path("models") / "best.onnx")
        self.assertEqual(output_path_for_size(model, 512), Path("models") / "best_512.onnx")


if __name__ == "__main__":
    unittest.main()
