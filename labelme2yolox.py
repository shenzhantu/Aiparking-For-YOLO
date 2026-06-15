"""
LabelMe / X-AnyLabeling → YOLOx COCO 格式转换脚本
兼容 LabelMe 和 X-AnyLabeling 的 JSON 标注格式
用法:
  python labelme2yolox.py                          # 默认处理 images（1）目录
  python labelme2yolox.py --input images           # 指定输入目录
  python labelme2yolox.py --input images（1） --ratio 0.9
"""

import json
import os
import glob
import shutil
import random
import argparse

# ============ 配置区 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "images（1）")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_RATIO = 0.8
SEED = 42
# ================================


def labelme_to_coco_bbox(points):
    """将多边形点转为 COCO bbox 格式 [x, y, width, height]"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def labelme_to_coco_segmentation(points):
    """将多边形点转为 COCO segmentation 格式 [x1,y1,x2,y2,...]"""
    seg = []
    for p in points:
        seg.extend(p)
    return seg


def convert(input_dir, output_dir, train_ratio, seed):
    random.seed(SEED)

    # 收集所有 JSON 标注文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not json_files:
        print(f"错误: 在 {input_dir} 中未找到 JSON 文件")
        return

    print(f"输入目录: {input_dir}")
    print(f"找到 {len(json_files)} 个标注文件")

    # 收集所有类别
    categories = {}
    cat_id = 1

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            label = shape["label"]
            if label not in categories:
                categories[label] = cat_id
                cat_id += 1

    print(f"类别: {categories}")

    # 打乱并分割训练/验证集
    random.shuffle(json_files)
    split_idx = int(len(json_files) * train_ratio)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    print(f"训练集: {len(train_files)} 张, 验证集: {len(val_files)} 张")

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": cid, "name": name} for name, cid in categories.items()]
        }

        ann_id = 1
        img_id = 1

        img_out_dir = os.path.join(output_dir, "images", split_name)
        os.makedirs(img_out_dir, exist_ok=True)

        for jf in split_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_filename = data.get("imagePath", os.path.basename(jf).replace(".json", ".jpg"))
            img_path = os.path.join(input_dir, img_filename)

            if not os.path.exists(img_path):
                print(f"  警告: 图片不存在，跳过: {img_path}")
                continue

            img_h = data.get("imageHeight", 0)
            img_w = data.get("imageWidth", 0)

            dst_img = os.path.join(img_out_dir, img_filename)
            shutil.copy2(img_path, dst_img)

            coco["images"].append({
                "id": img_id,
                "file_name": img_filename,
                "width": img_w,
                "height": img_h,
            })

            for shape in data.get("shapes", []):
                label = shape["label"]
                points = shape["points"]
                cid = categories[label]

                bbox = labelme_to_coco_bbox(points)
                seg = labelme_to_coco_segmentation(points)
                area = bbox[2] * bbox[3]

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": bbox,
                    "segmentation": [seg],
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

            img_id += 1

        out_json = os.path.join(output_dir, f"annotations_{split_name}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)

        print(f"  {split_name}: {len(coco['images'])} 张图, {len(coco['annotations'])} 个标注 → {out_json}")

    # 保存类别文件
    classes_file = os.path.join(output_dir, "classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        for name in sorted(categories, key=lambda x: categories[x]):
            f.write(name + "\n")

    print(f"\n完成! 数据集已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LabelMe/X-AnyLabeling → COCO 格式转换")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="输入标注目录")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--ratio", type=float, default=TRAIN_RATIO, help="训练集比例")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    args = parser.parse_args()

    convert(args.input, args.output, args.ratio, args.seed)
