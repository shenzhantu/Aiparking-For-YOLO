import tempfile
import unittest
from pathlib import Path
from unittest import mock

from train import build_train_kwargs, read_results_csv_without_polars


class TrainPolarsFallbackTests(unittest.TestCase):
    def test_reads_results_csv_with_standard_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "results.csv"
            csv_path.write_text(
                "epoch,metrics/mAP50(M)\n1,0.54802\n",
                encoding="utf-8",
            )
            trainer = mock.Mock()
            trainer.csv = csv_path

            result = read_results_csv_without_polars(trainer)

        self.assertEqual(result, {"epoch": ["1"], "metrics/mAP50(M)": ["0.54802"]})

    def test_training_disables_ultralytics_plots_to_avoid_polars_crash(self):
        args = mock.Mock(
            epochs=3,
            imgsz=512,
            batch=2,
            project="runs",
            name="test",
            patience=1,
            save_period=1,
            workers=0,
        )

        kwargs = build_train_kwargs(args, Path("data.yaml"), "cpu")

        self.assertFalse(kwargs["plots"])


if __name__ == "__main__":
    unittest.main()
