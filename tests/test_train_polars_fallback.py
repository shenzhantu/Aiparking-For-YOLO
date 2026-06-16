import tempfile
import unittest
from pathlib import Path
from unittest import mock

from train import read_results_csv_without_polars


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


if __name__ == "__main__":
    unittest.main()
