"""
Train the AiParking segmentation model with YOLOv8.

Default dataset:
  D:\\Aiparking\\image backcup\\dataset_yolov8_weighted\\parking_yolov8.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = Path(r"D:\Aiparking\image backcup\dataset_yolov8_weighted\parking_yolov8.yaml")
DEFAULT_PROJECT = BASE_DIR / "runs"
DEFAULT_NAME = "parking_yolov8_seg"
DEFAULT_MODEL = "yolov8s-seg.pt"


def read_results_csv_without_polars(trainer) -> dict[str, list[str]]:
    """Read Ultralytics results.csv without importing polars."""
    csv_path = Path(trainer.csv)
    if not csv_path.exists():
        return {}

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return {}
        data = {field: [] for field in reader.fieldnames}
        for row in reader:
            for field in reader.fieldnames:
                data[field].append((row.get(field) or "").strip())
    return data


def install_results_csv_fallback() -> None:
    """Patch Ultralytics to survive polars import/runtime failures while saving."""
    original = BaseTrainer.read_results_csv

    def safe_read_results_csv(self):
        try:
            return original(self)
        except Exception as exc:
            print(f"polars failed while reading results.csv, using csv fallback: {exc}")
            return read_results_csv_without_polars(self)

    BaseTrainer.read_results_csv = safe_read_results_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for AiParking")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="YOLO dataset yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLOv8 segmentation checkpoint")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--save-period", type=int, default=20)
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export after training")
    return parser.parse_args()


def build_train_kwargs(args: argparse.Namespace, data_path: Path, device: str) -> dict:
    return {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "patience": args.patience,
        "save": True,
        "save_period": args.save_period,
        "workers": args.workers,
        "verbose": True,
        "plots": False,
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset yaml not found: {data_path}\n"
            "Run build_yolov8_dataset.py before training."
        )

    device = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("AiParking YOLOv8-seg training")
    print("=" * 60)
    print(f"data: {data_path}")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"epochs: {args.epochs}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch: {args.batch}")
    print(f"patience: {args.patience}")

    install_results_csv_fallback()
    model = YOLO(args.model)
    model.train(**build_train_kwargs(args, data_path, device))

    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("Training finished")
    print("=" * 60)
    print(f"best model: {best_model}")

    if not args.no_export and best_model.exists():
        print("\nExporting ONNX for deployment...")
        exported = YOLO(str(best_model)).export(
            format="onnx",
            imgsz=args.imgsz,
            opset=12,
            simplify=True,
            half=False,
        )
        print(f"onnx: {exported}")


if __name__ == "__main__":
    main()
