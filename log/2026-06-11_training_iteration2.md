# 2026-06-11 第二轮训练与预测日志

## 操作概要

1. **备份项目** → `D:\Aiparking\Aiparking For YOLObackup`（覆盖旧备份）
2. **合并审核后的标注到训练集** — 将 `images/` + `images（2）/` 目录下审核后图片的标注转为 COCO 格式并合并
3. **重新训练 YOLO11s-seg 模型** — 使用扩展后的数据集（8672张图片）
4. **用新模型预测 `images（3）/`** — 2072 张新图片

---

## 训练数据

| 数据集 | 合并前 | 合并后 |
|--------|--------|--------|
| 训练集 | 6546图/7192标注 | **7662图/8901标注** |
| 验证集 | 886图/942标注 | **1010图/1137标注** |
| **总计** | **7432图** | **8672图** |

来源：
- 原始手动标注数据：1416张（`dataset/images/`）
- 第一轮审核后的标注：6016张（`images/` 目录）
- 第二轮新增审核标注：1240张（`images（2）/` 目录，含 Parking + barrier 标签）

新增类别：**barrier**（障碍物，rectangle/cuboid/polygon 形状）

## 训练配置

- 模型：YOLO11s-seg
- Epochs：88/100（EarlyStopping 在 epoch 68 最优，patience=20）
- Batch size：8
- 图片尺寸：640
- 置信度阈值：0.4（用户要求过滤低置信度）
- 设备：RTX 4060 Laptop GPU（8GB VRAM）
- 训练耗时：4.2 小时

## 训练结果

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|------|-----------|-------------|------------|---------------|
| **all** | 0.982 | 0.717 | 0.678 | 0.507 |
| Parking | 0.982 | 0.927 | 0.974 | 0.840 |
| barrier | 0.982 | 0.507 | 0.383 | 0.175 |

Parking 类表现优秀，barrier 类因标注数量较少（31张图/45个标注）Mask 性能较低。

模型保存位置：`runs/parking_seg/weights/best.pt`

## 预测结果（images（3）/）

| 指标 | 数值 |
|------|------|
| 处理图片 | 2072 |
| 检测到目标 | 1234 |
| 无目标图片 | 858 |
| 检出率 | 59.6%（置信度≥0.4） |

## 修改的文件

- `predict.py`：DEFAULT_MODEL 更新为 `runs/parking_seg/weights/best.pt`，添加 barrier 类别支持，barrier 使用 rectangle shape_type
- `xanylabeling2coco.py`：添加 barrier 默认类别、置信度过滤 `--min-conf`、矩形/立方体转多边形
- `parking.yaml`：添加 barrier 类别（index 1）
- `dataset/annotations_train.json`：合并了 images（2）/ 的标注
- `dataset/annotations_val.json`：合并了 images（2）/ 的标注
- `dataset/images/train/`：复制了新图片
- `dataset/images/val/`：复制了新图片

## 下一步

1. 用 X-AnyLabeling 打开 `images（3）/` 审核标注
2. 审核完成后运行 `python xanylabeling2coco.py` 合并到训练集
3. 重新训练模型
