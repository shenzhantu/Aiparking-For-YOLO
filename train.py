"""
训练 YOLO11-seg 停车位分割模型（支持多边形标注）
用法: python train.py
"""

import json
import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def coco_seg_to_yolo_txt(json_path, img_dir, label_dir):
    """将 COCO JSON（含 segmentation）转为 YOLO 分割 TXT 格式

    YOLO 分割 TXT 格式: class_id x1 y1 x2 y2 ... xn yn (归一化坐标)
    """
    os.makedirs(label_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_info = {}
    for img in coco["images"]:
        img_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    converted = 0
    skipped = 0
    for img_id, info in img_info.items():
        w, h = info["width"], info["height"]
        txt_name = os.path.splitext(info["file_name"])[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_name)

        lines = []
        for ann in img_anns.get(img_id, []):
            cat_id = ann["category_id"] - 1  # COCO 1-indexed → YOLO 0-indexed

            # 优先使用 segmentation（多边形）
            seg = ann.get("segmentation")
            if seg and len(seg) > 0:
                # segmentation 格式: [[x1,y1,x2,y2,...,xn,yn]] 或多个多边形
                for polygon in seg:
                    if len(polygon) < 6:  # 至少 3 个点（6 个坐标）
                        continue
                    # 归一化坐标
                    coords = []
                    for i in range(0, len(polygon), 2):
                        x = polygon[i] / w
                        y = polygon[i + 1] / h
                        coords.append(f"{x:.6f}")
                        coords.append(f"{y:.6f}")
                    lines.append(f"{cat_id} " + " ".join(coords))
            else:
                # 回退到 bbox
                bbox = ann["bbox"]  # [x, y, w, h]
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                norm_w = bbox[2] / w
                norm_h = bbox[3] / h
                # bbox 转 4 点矩形多边形
                x1 = (bbox[0]) / w
                y1 = (bbox[1]) / h
                x2 = (bbox[0] + bbox[2]) / w
                y2 = (bbox[1] + bbox[3]) / h
                lines.append(f"{cat_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")

        if lines:
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            converted += 1
        else:
            skipped += 1

    return converted, skipped


def main():
    # Step 1: 转换 COCO JSON → YOLO 分割 TXT
    print("=" * 50)
    print("Step 1: 转换标注格式 (COCO JSON → YOLO 分割 TXT)")
    print("=" * 50)

    for split in ["train", "val"]:
        json_path = os.path.join(DATASET_DIR, f"annotations_{split}.json")
        img_dir = os.path.join(DATASET_DIR, "images", split)
        label_dir = os.path.join(DATASET_DIR, "labels", split)

        if not os.path.exists(json_path):
            print(f"错误: 找不到 {json_path}")
            return

        converted, skipped = coco_seg_to_yolo_txt(json_path, img_dir, label_dir)
        print(f"  {split}: 转换 {converted} 张图片, 跳过 {skipped} 张（无标注）")

    # Step 2: 训练 YOLO11-seg 分割模型
    print("\n" + "=" * 50)
    print("Step 2: 开始训练 YOLO11s-seg 分割模型")
    print("=" * 50)

    # 使用 YOLO11s-seg 预训练权重
    model_path = os.path.join(BASE_DIR, "weights", "yolo11s-seg.pt")
    if not os.path.exists(model_path):
        print(f"下载 YOLO11s-seg 预训练权重...")
        os.makedirs(os.path.join(BASE_DIR, "weights"), exist_ok=True)
        # ultralytics 会自动下载

    model = YOLO("yolo11s-seg.pt")  # 自动下载预训练权重

    results = model.train(
        data=os.path.join(BASE_DIR, "parking.yaml"),
        epochs=100,
        imgsz=640,
        batch=8,
        device="0" if torch.cuda.is_available() else "cpu",
        project=os.path.join(BASE_DIR, "runs"),
        name="parking_seg",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        workers=4,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)
    best_model = os.path.join(BASE_DIR, "runs", "parking_seg", "weights", "best.pt")
    print(f"最佳模型: {best_model}")
    print(f"\n下一步:")
    print(f"  1. 运行 python predict.py --model {best_model} 重新标注")
    print(f"  2. 用 X-AnyLabeling 审核多边形标注")
    print(f"  3. 将审核后的标注加入数据集，重新训练")


if __name__ == "__main__":
    main()
