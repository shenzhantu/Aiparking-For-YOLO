# Aiparking For YOLO

基于 YOLO 分割模型的停车位与障碍物识别项目，用于停车场图像的半自动标注、模型迭代训练和板端部署准备。

当前主线模型已从 YOLO11s-seg 切换到 **YOLOv8s-seg**，优先适配板端对 YOLOv8 的部署支持。

## 功能

- **停车位分割检测**：识别 `Parking`，输出多边形分割标注。
- **障碍物检测**：识别 `barrier`，用于道闸、立柱、障碍物等目标。
- **半自动标注流水线**：模型预标注 → X-AnyLabeling 人工审核 → 加入训练集 → 继续迭代。
- **GitHub 轻量化仓库**：训练素材、数据集、权重和 runs 输出不提交到 Git；代码、Markdown 日志和更新记录保留在仓库中。

## 快速开始

### 训练

```bash
python train.py
```

训练数据默认使用外部路径中的数据集配置，当前主要数据目录为：

```text
D:\Aiparking\image backcup
```

### 批量预测

```bash
python predict.py --target "D:\Aiparking\image backcup\images（5）" --conf 0.4
```

预测会生成 X-AnyLabeling 兼容的 JSON 标注文件。

### 标注审核流程

1. 使用 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 打开预测目录。
2. 人工检查并修正 JSON 标注。
3. 重新构建 YOLOv8 加权数据集。
4. 继续训练新模型。
5. 使用新模型预测下一批图片。

## 数据与模型管理

为了避免 GitHub 仓库过大，本项目不提交以下内容：

- 原始图片与标注数据集：`images*/`、`dataset*/`
- Ultralytics 训练输出：`runs/`
- PyTorch/ONNX/OM/Engine 权重：`*.pt`、`*.onnx`、`*.om`、`*.engine`
- 原始运行日志：`*.log`、`*.done`、临时状态 JSON

需要交付给板端同学的模型文件位于本地：

```text
D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.onnx
```

该文件建议通过 GitHub Release、网盘或其他制品分发方式共享，不直接提交到源码仓库。

## 更新记录

项目训练迭代记录见：

- [CHANGELOG.md](CHANGELOG.md)

每轮操作日志保存在：

```text
log/*.md
```

这些 Markdown 日志会保留在 GitHub 仓库中，便于追踪训练数据、模型效果和关键操作。

## 当前 v4.0 指标

详见 [CHANGELOG.md](CHANGELOG.md)。摘要：

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|------|-----------|-------------|------------|---------------|
| all | 0.904 | 0.671 | 0.673 | 0.504 |
| Parking | 0.967 | 0.874 | 0.963 | 0.816 |
| barrier | 0.841 | 0.468 | 0.383 | 0.193 |

## 目录结构

```text
.
├── train.py                    # 模型训练脚本
├── predict.py                  # 批量预测并生成 X-AnyLabeling JSON
├── build_yolov8_dataset.py     # 构建 YOLOv8 加权数据集
├── xanylabeling2coco.py        # X-AnyLabeling/LabelMe JSON 转 COCO
├── labelme2yolox.py            # LabelMe 转 YOLOX/COCO 数据集
├── cleanup_auto.py             # 清理错误自动标注
├── export_onnx.py              # ONNX 导出工具
├── check_onnx.py               # ONNX 检查工具
├── parking.yaml                # 数据集配置示例
├── CHANGELOG.md                # 训练迭代更新记录
├── log/                        # Markdown 操作日志
└── tests/                      # 单元测试
```

## 环境

- Windows 11
- Python 3.11
- PyTorch + CUDA
- Ultralytics YOLO
- RTX 4060 Laptop GPU (8GB VRAM)
