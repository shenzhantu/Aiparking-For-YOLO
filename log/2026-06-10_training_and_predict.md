# 2026-06-10 训练与预测日志

## 操作概要

1. **备份项目** → `D:\Aiparking\Aiparking For YOLObackup`（3.9G）
2. **合并审核后的标注到训练集** — 将 `images/` 目录下 6016 张已审核图片的标注转为 COCO 格式并合并
3. **重新训练 YOLO11s-seg 模型** — 使用扩展后的数据集（7432张图片）
4. **用新模型预测 `images（2）/`** — 1372 张新图片

---

## 训练数据

| 数据集 | 合并前 | 合并后 |
|--------|--------|--------|
| 训练集 | 1132图/995标注 | **6546图/7192标注** |
| 验证集 | 284图/247标注 | **886图/942标注** |
| **总计** | **1416图** | **7432图** |

来源：
- 原始手动标注数据：1416张（`dataset/images/`）
- 用户审核后的自动标注：6016张（`images/` 目录，含手动+审核后的自动标注）

## 训练配置

- 模型：YOLO11s-seg
- Epochs：100（完整训练，未触发 early stopping）
- Batch size：8
- 图片尺寸：640
- 置信度阈值：0.25
- 设备：RTX 4060 Laptop GPU（8GB VRAM）

## 训练结果

| 指标 | Box | Mask |
|------|-----|------|
| Precision | 0.972 | 0.971 |
| Recall | 0.960 | 0.946 |
| mAP50 | 0.986 | 0.978 |
| mAP50-95 | 0.945 | 0.856 |

模型保存位置：`runs/parking_seg/weights/best.pt`

## 预测结果（images（2）/）

| 指标 | 数值 |
|------|------|
| 处理图片 | 1372 |
| 检测到停车位 | 1118 |
| 无目标图片 | 331 |
| 检出率 | 81.5% |

## 修改的文件

- `predict.py`：DEFAULT_MODEL 更新为 `runs/parking_seg/weights/best.pt`
- `dataset/annotations_train.json`：合并了审核后的标注
- `dataset/annotations_val.json`：合并了审核后的标注
- `dataset/images/train/`：复制了新图片
- `dataset/images/val/`：复制了新图片

## 下一步

1. 用 X-AnyLabeling 打开 `images（2）/` 审核标注
2. 审核完成后运行 `python xanylabeling2coco.py` 合并到训练集
3. 重新训练模型
