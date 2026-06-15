# Aiparking For YOLO

基于 YOLO11s-seg 的智能停车位检测与分割模型训练项目。

## 功能

- 🚗 **停车位分割检测**（Parking）— 多边形分割，精确勾勒停车位轮廓
- 🚧 **障碍物检测**（barrier）— 道闸/立柱等障碍物识别
- 🔄 **半自动标注流水线**：模型预标注 → 人工审核 → 加入训练集 → 迭代提升

## 快速开始

### 训练

```bash
python train.py
```

自动完成 COCO JSON → YOLO TXT 转换 + 模型训练。

### 预测

```bash
python predict.py --target images（4）
```

对指定目录图片进行批量推理，生成 X-AnyLabeling JSON 标注文件。

### 标注审核

1. 用 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 打开预测生成的 JSON
2. 人工修正标注
3. 运行 `python xanylabeling2coco.py` 合并到训练集
4. 重新训练

## 数据集配置

编辑 `parking.yaml`：

```yaml
path: ./dataset          # 数据集根目录
train: images/train       # 训练图片
val: images/val           # 验证图片
names:
  0: Parking
  1: barrier
```

## 模型性能（v3.0）

| 类别 | Mask mAP50 | Mask mAP50-95 |
|------|------------|---------------|
| Parking | 0.971 | 0.820 |
| barrier | 0.412 | 0.164 |

详细更新记录见 [CHANGELOG.md](CHANGELOG.md)。

## 目录结构

```
├── train.py                 # 训练脚本（转换 + 训练）
├── predict.py               # 批量推理脚本
├── xanylabeling2coco.py     # X-AnyLabeling → COCO 转换
├── labelme2yolox.py         # LabelMe → YOLO 转换
├── cleanup_auto.py          # 清理错误自动标注
├── export_onnx.py           # 导出 ONNX
├── check_onnx.py            # 校验 ONNX
├── parking.yaml             # 数据集配置
├── log/                     # 训练日志
├── models/                  # 导出模型
├── weights/                 # 预训练权重
└── runs/parking_seg/        # 训练输出
    ├── weights/             # 模型检查点
    ├── results.csv          # 训练指标曲线
    └── *.png                # 混淆矩阵/PR曲线/样本图
```

## 环境

- Python 3.x + PyTorch + Ultralytics
- RTX 4060 Laptop GPU (8GB VRAM)
- Windows 11
