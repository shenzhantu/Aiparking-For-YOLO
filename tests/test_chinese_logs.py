import csv
import json
import tempfile
import unittest
from pathlib import Path

from monitor_training import append_snapshot
from write_iteration_log import write_log


class ChineseLogTests(unittest.TestCase):
    def test_final_iteration_log_uses_chinese_headings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_summary = root / "build_summary.json"
            results_csv = root / "results.csv"
            prediction_target = root / "images5"
            output = root / "final.md"
            run_log = root / "run.log"
            monitor_log = root / "monitor.md"
            model = root / "best.pt"
            onnx = root / "best.onnx"

            prediction_target.mkdir()
            (prediction_target / "frame_0001.jpg").write_bytes(b"fake")
            (prediction_target / "frame_0001.json").write_text(
                json.dumps(
                    {
                        "shapes": [
                            {"label": "Parking", "score": 0.88},
                            {"label": "barrier", "score": 0.66},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            dataset_summary.write_text(
                json.dumps(
                    {
                        "train_images": 10,
                        "val_images": 2,
                        "train_instances": {"Parking": 9, "barrier": 1},
                        "val_instances": {"Parking": 2},
                        "source_weighted_train_images": {"images（4）": 4},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with results_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["epoch", "metrics/mAP50(M)"])
                writer.writeheader()
                writer.writerow({"epoch": "3", "metrics/mAP50(M)": "0.75"})
            model.write_bytes(b"pt")
            onnx.write_bytes(b"onnx")

            write_log(
                output=output,
                dataset_summary=dataset_summary,
                results_csv=results_csv,
                prediction_target=prediction_target,
                model=model,
                onnx=onnx,
                run_log=run_log,
                monitor_log=monitor_log,
            )

            text = output.read_text(encoding="utf-8")
            self.assertIn("# 更新日志", text)
            self.assertIn("## v4.0", text)
            self.assertIn("### 📊 数据集", text)
            self.assertIn("### 📝 预测 images（5）/", text)
            self.assertNotIn("What Changed", text)

    def test_final_iteration_log_matches_changelog_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_summary = root / "build_summary.json"
            results_csv = root / "results.csv"
            prediction_target = root / "images5"
            output = root / "final.md"
            run_log = root / "run.log"
            monitor_log = root / "monitor.md"
            model = root / "best.pt"
            onnx = root / "best.onnx"

            prediction_target.mkdir()
            dataset_summary.write_text(
                json.dumps(
                    {
                        "train_images": 12628,
                        "val_images": 951,
                        "train_instances": {"Parking": 14620, "barrier": 878},
                        "val_instances": {"Parking": 1144, "barrier": 78},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with results_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "epoch",
                        "metrics/mAP50(B)",
                        "metrics/mAP50-95(B)",
                        "metrics/mAP50(M)",
                        "metrics/mAP50-95(M)",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": "29",
                        "metrics/mAP50(B)": "0.91",
                        "metrics/mAP50-95(B)": "0.65",
                        "metrics/mAP50(M)": "0.63",
                        "metrics/mAP50-95(M)": "0.46",
                    }
                )
            model.write_bytes(b"pt")
            onnx.write_bytes(b"onnx")

            write_log(
                output=output,
                dataset_summary=dataset_summary,
                results_csv=results_csv,
                prediction_target=prediction_target,
                model=model,
                onnx=onnx,
                run_log=run_log,
                monitor_log=monitor_log,
            )

            text = output.read_text(encoding="utf-8")
            self.assertIn("## v4.0", text)
            self.assertIn("### 📊 数据集", text)
            self.assertIn("| 类别 | 训练集 | 验证集 | 总计 |", text)
            self.assertIn("### 🎯 训练结果", text)
            self.assertIn("| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |", text)
            self.assertIn("### 📝 预测 images（5）/", text)

    def test_monitor_snapshot_uses_chinese_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor_log = root / "monitor.md"
            run_log = root / "run.log"
            results_csv = root / "results.csv"
            run_log.write_text("epoch 1\n", encoding="utf-8")

            append_snapshot(monitor_log, run_log, results_csv)

            text = monitor_log.read_text(encoding="utf-8")
            self.assertIn("显卡状态", text)
            self.assertIn("最新指标", text)
            self.assertIn("最近训练日志", text)


if __name__ == "__main__":
    unittest.main()
