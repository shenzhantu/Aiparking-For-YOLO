"""
将 X-AnyLabeling 的 JSON 标注（labelme 格式）转为 COCO 格式
用于将审核后的自动标注加入训练数据集

用法:
  python xanylabeling2coco.py --input images --output dataset/annotations_auto.json
  python xanylabeling2coco.py --input images --output dataset/annotations_auto.json --split train
"""

import json
import os
import glob
import argparse
from collections import defaultdict


def labelme_to_coco(json_dir, class_names=None, min_confidence=0.0):
    """将目录下所有 labelme JSON 转为 COCO 格式
    Args:
        min_confidence: 最低置信度阈值，低于此值的标注将被过滤（0.0=不过滤）
    """
    if class_names is None:
        class_names = ["Parking", "barrier"]  # 默认类别

    # COCO 格式结构
    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # 建立类别
    for i, name in enumerate(class_names):
        coco["categories"].append({
            "id": i + 1,  # COCO 1-indexed
            "name": name,
            "supercategory": "none",
        })

    # label → category_id 映射
    label_to_cat = {name: i + 1 for i, name in enumerate(class_names)}

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    print(f"扫描 {len(json_files)} 个 JSON 文件...")

    ann_id = 1
    img_id = 1
    processed = 0
    skipped = 0

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  跳过 {os.path.basename(jf)}: {e}")
            skipped += 1
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            skipped += 1
            continue

        # 图片信息
        img_filename = data.get("imagePath", os.path.basename(jf).replace(".json", ".jpg"))
        img_w = data.get("imageWidth", 0)
        img_h = data.get("imageHeight", 0)

        if img_w == 0 or img_h == 0:
            # 尝试从图片文件获取尺寸
            img_path = os.path.join(json_dir, img_filename)
            if os.path.exists(img_path):
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            else:
                print(f"  跳过 {os.path.basename(jf)}: 无法获取图片尺寸")
                skipped += 1
                continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": img_w,
            "height": img_h,
        })

        # 标注
        for shape in shapes:
            label = shape.get("label", "Parking")
            cat_id = label_to_cat.get(label)
            if cat_id is None:
                # 未知类别，添加到类别列表
                cat_id = len(class_names) + 1
                class_names.append(label)
                label_to_cat[label] = cat_id
                coco["categories"].append({
                    "id": cat_id,
                    "name": label,
                    "supercategory": "none",
                })

            # 置信度过滤
            score = shape.get("score", 1.0)
            if score is not None and score < min_confidence:
                continue

            points = shape.get("points", [])
            shape_type = shape.get("shape_type", "polygon")

            # 矩形/立方体 → 转为4点多边形
            if shape_type in ("rectangle", "cuboid") and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            if len(points) < 3:
                continue

            # 多边形 → segmentation 格式（flat list of x,y）
            seg = []
            for p in points:
                seg.extend([p[0], p[1]])

            # 计算 bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": [seg],
                "bbox": [x_min, y_min, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1
        processed += 1

    print(f"处理: {processed} 张图片, 跳过: {skipped} 张")
    print(f"标注: {len(coco['annotations'])} 个")
    print(f"类别: {[c['name'] for c in coco['categories']]}")

    return coco


def merge_coco(coco1, coco2):
    """合并两个 COCO 数据集"""
    merged = {
        "images": list(coco1["images"]),
        "annotations": list(coco1["annotations"]),
        "categories": list(coco1["categories"]),
    }

    # 合并类别（去重）
    existing_cats = {c["name"]: c["id"] for c in merged["categories"]}
    cat_id_map = {}
    for cat in coco2["categories"]:
        if cat["name"] in existing_cats:
            cat_id_map[cat["id"]] = existing_cats[cat["name"]]
        else:
            new_id = max(existing_cats.values()) + 1
            existing_cats[cat["name"]] = new_id
            merged["categories"].append({**cat, "id": new_id})
            cat_id_map[cat["id"]] = new_id

    # 合并图片（去重，按文件名）
    existing_imgs = {img["file_name"]: img["id"] for img in merged["images"]}
    img_id_map = {}
    next_img_id = max(img["id"] for img in merged["images"]) + 1 if merged["images"] else 1

    for img in coco2["images"]:
        if img["file_name"] in existing_imgs:
            img_id_map[img["id"]] = existing_imgs[img["file_name"]]
        else:
            new_img = {**img, "id": next_img_id}
            merged["images"].append(new_img)
            img_id_map[img["id"]] = next_img_id
            next_img_id += 1

    # 合并标注
    next_ann_id = max(ann["id"] for ann in merged["annotations"]) + 1 if merged["annotations"] else 1

    for ann in coco2["annotations"]:
        new_ann = {
            **ann,
            "id": next_ann_id,
            "image_id": img_id_map[ann["image_id"]],
            "category_id": cat_id_map[ann["category_id"]],
        }
        merged["annotations"].append(new_ann)
        next_ann_id += 1

    return merged


def main():
    parser = argparse.ArgumentParser(description="X-AnyLabeling JSON → COCO 格式")
    parser.add_argument("--input", required=True, help="标注 JSON 所在目录")
    parser.add_argument("--output", required=True, help="输出 COCO JSON 路径")
    parser.add_argument("--merge", help="合并已有的 COCO JSON（可选）")
    parser.add_argument("--classes", nargs="+", default=["Parking", "barrier"], help="类别列表")
    parser.add_argument("--min-conf", type=float, default=0.0, help="最低置信度阈值（默认 0.0，不过滤）")
    args = parser.parse_args()

    coco = labelme_to_coco(args.input, args.classes, min_confidence=args.min_conf)

    if args.merge and os.path.exists(args.merge):
        print(f"\n合并已有数据集: {args.merge}")
        with open(args.merge, "r", encoding="utf-8") as f:
            existing = json.load(f)
        coco = merge_coco(existing, coco)
        print(f"合并后: {len(coco['images'])} 张图片, {len(coco['annotations'])} 个标注")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"\n已保存: {args.output}")
    print(f"  图片: {len(coco['images'])}")
    print(f"  标注: {len(coco['annotations'])}")
    print(f"  类别: {[c['name'] for c in coco['categories']]}")


if __name__ == "__main__":
    main()
