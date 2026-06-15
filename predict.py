"""
用训练好的 YOLO11-seg 模型自动标注图片，导出为 X-AnyLabeling 兼容的 JSON 格式
支持多边形分割标注（而非矩形框）

用法:
  python predict.py                           # 使用默认模型和阈值 0.25
  python predict.py --conf 0.3                # 调整置信度阈值
  python predict.py --overwrite               # 覆盖已有标注（包括手动标注）
  python predict.py --model path/to/best.pt   # 指定模型
"""

import json
import os
import glob
import argparse
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "runs", "parking_seg", "weights", "best.pt")
TARGET_DIR = os.path.join(BASE_DIR, "images")
CONF_THRESHOLD = 0.25  # 降低默认阈值，提高召回率
CLASS_NAMES = {0: "Parking", 1: "barrier"}


def masks_to_polygons(result):
    """
    从 YOLO segmentation 结果中提取多边形顶点。
    优先使用 result.masks（分割 mask），如果不可用则回退到 result.boxes（矩形框）。
    返回: list of dict, 每个包含 polygon_points, confidence, class_id
    """
    detections = []

    if result.masks is not None and result.masks.xy is not None:
        # 使用分割 mask 的多边形顶点
        for i, polygon in enumerate(result.masks.xy):
            # polygon 是 Nx2 的 numpy 数组，已经是原始图片坐标
            points = polygon.tolist()
            # 确保至少有3个点（多边形最少需要3个顶点）
            if len(points) < 3:
                continue
            conf = float(result.boxes.conf[i])
            cls_id = int(result.boxes.cls[i])
            detections.append({
                "points": points,
                "confidence": conf,
                "class_id": cls_id,
            })
    elif result.boxes is not None:
        # 回退：使用矩形框（转为4点多边形）
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append({
                "points": points,
                "confidence": conf,
                "class_id": cls_id,
            })

    return detections


def create_labelme_json(image_path, detections, img_w, img_h):
    """创建 X-AnyLabeling 兼容的 JSON（多边形分割格式）"""
    shapes = []
    for det in detections:
        label = CLASS_NAMES.get(det["class_id"], f"class_{det['class_id']}")
        # barrier 用 rectangle，Parking 用 polygon
        if label == "barrier":
            pts = det["points"]
            if len(pts) == 4:
                shape_type = "rectangle"
                # rectangle 格式: [左上角, 右下角]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                points = [[min(xs), min(ys)], [max(xs), max(ys)]]
            else:
                shape_type = "polygon"
                points = pts
        else:
            shape_type = "polygon"
            points = det["points"]
        shapes.append({
            "label": label,
            "score": det["confidence"],
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": shape_type,
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        })

    img_filename = os.path.basename(image_path)

    return {
        "version": "4.0.0-beta.7",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": img_filename,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
        "description": "",
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg 自动标注 → X-AnyLabeling JSON（多边形）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型路径")
    parser.add_argument("--target", default=TARGET_DIR, help="要标注的图片目录")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="置信度阈值 (默认 0.25)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有标注（包括手动标注）")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 模型不存在: {args.model}")
        print("请先运行 python train.py 训练模型")
        return

    print(f"加载模型: {args.model}")
    model = YOLO(args.model)

    # 检测模型类型
    model_type = "seg" if hasattr(model, "model") and "seg" in str(type(model.model)).lower() else "detect"
    print(f"模型类型: {model_type}")

    # 收集图片
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []
    for ext in img_extensions:
        all_images.extend(glob.glob(os.path.join(args.target, f"*{ext}")))
        all_images.extend(glob.glob(os.path.join(args.target, f"*{ext.upper()}")))
    all_images = list(set(all_images))

    if not all_images:
        print(f"错误: 在 {args.target} 中未找到图片")
        return

    # 过滤已有标注
    to_process = []
    skipped = 0
    for img_path in all_images:
        json_path = os.path.splitext(img_path)[0] + ".json"
        if os.path.exists(json_path) and not args.overwrite:
            skipped += 1
        else:
            to_process.append(img_path)

    print(f"找到 {len(all_images)} 张图片")
    print(f"已有标注（跳过）: {skipped} 张")
    print(f"待标注: {len(to_process)} 张")
    print(f"置信度阈值: {args.conf}")

    if not to_process:
        print("没有需要标注的图片")
        return

    # 批量推理
    print("\n开始推理...")
    total_detections = 0
    processed = 0
    empty_count = 0  # 无检测结果的图片数

    batch_size = 32
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]

        results = model.predict(
            source=batch,
            conf=args.conf,
            device="0" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        for idx, result in enumerate(results):
            img_path = batch[idx]
            img_w, img_h = result.orig_shape[1], result.orig_shape[0]

            # 提取多边形检测结果
            detections = masks_to_polygons(result)

            if len(detections) == 0:
                # 无检测结果 → 不写 JSON（避免 X-AnyLabeling 显示假勾选）
                empty_count += 1
            else:
                # 有检测结果 → 保存 JSON
                labelme_json = create_labelme_json(img_path, detections, img_w, img_h)
                json_path = os.path.splitext(img_path)[0] + ".json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(labelme_json, f, ensure_ascii=False, indent=2)
                total_detections += len(detections)

            processed += 1

        print(f"  进度: {processed}/{len(to_process)}  "
              f"({total_detections} 个检测, {empty_count} 张空图)")

    print(f"\n完成!")
    print(f"处理: {processed} 张图片")
    print(f"检测: {total_detections} 个停车位（多边形标注）")
    print(f"无目标: {empty_count} 张（未写入 JSON）")
    print(f"标注已保存到: {args.target}")
    print(f"\n下一步:")
    print(f"  1. 用 X-AnyLabeling 打开 {args.target} 逐张审核")
    print(f"  2. 修正错误标注，补充漏检的目标")
    print(f"  3. 运行 python labelme2yolox.py 生成最终数据集")


if __name__ == "__main__":
    main()
