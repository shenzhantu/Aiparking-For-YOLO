"""Export deployment ONNX models from the current AiParking PyTorch checkpoint."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "best.pt"


def output_path_for_size(model_path: Path, imgsz: int) -> Path:
    if imgsz == 640:
        return model_path.with_suffix(".onnx")
    return model_path.with_name(f"{model_path.stem}_{imgsz}.onnx")


def export_one(model_path: Path, imgsz: int, opset: int, simplify: bool) -> Path:
    output_path = output_path_for_size(model_path, imgsz)
    default_output = model_path.with_suffix(".onnx")
    preserved_default = None
    if output_path != default_output and default_output.exists():
        preserved_default = default_output.with_suffix(".onnx.preserve")
        if preserved_default.exists():
            preserved_default.unlink()
        default_output.replace(preserved_default)

    print(f"加载模型: {model_path}")
    print(f"导出 ONNX: imgsz={imgsz}, opset={opset}, simplify={simplify}")
    try:
        exported = Path(
            YOLO(str(model_path)).export(
                format="onnx",
                opset=opset,
                imgsz=imgsz,
                simplify=simplify,
                half=False,
            )
        )
        if exported.resolve() != output_path.resolve():
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(exported), output_path)
    finally:
        if preserved_default and preserved_default.exists():
            preserved_default.replace(default_output)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"导出成功: {output_path} ({size_mb:.1f} MB)")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AiParking ONNX deployment models")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Input .pt checkpoint")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640], help="One or more export sizes")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--no-simplify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    for size in args.imgsz:
        export_one(model_path, size, args.opset, not args.no_simplify)


if __name__ == "__main__":
    main()
