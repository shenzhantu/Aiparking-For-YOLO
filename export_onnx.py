"""重新导出 ONNX 模型，确保与 X-AnyLabeling 兼容"""
import os
from ultralytics import YOLO

BASE_DIR = r"D:\Aiparking\Aiparking For YOLO"
model_path = os.path.join(BASE_DIR, "models", "best.pt")
output_path = os.path.join(BASE_DIR, "models", "best.onnx")

print(f"加载模型: {model_path}")
model = YOLO(model_path)

print("导出 ONNX (opset=12, imgsz=640)...")
model.export(
    format="onnx",
    opset=12,
    imgsz=640,
    simplify=True,
)

# ultralytics 会自动保存为 best.onnx，检查是否在正确位置
auto_output = model_path.replace(".pt", ".onnx")
if auto_output != output_path and os.path.exists(auto_output):
    import shutil
    shutil.move(auto_output, output_path)
    print(f"已移动到: {output_path}")

if os.path.exists(output_path):
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"导出成功! 文件大小: {size_mb:.1f} MB")
    print(f"路径: {output_path}")
else:
    print(f"导出失败，文件不存在: {output_path}")
