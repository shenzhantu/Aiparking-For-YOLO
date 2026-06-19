# 第六轮训练日志：YOLOv8 最终 Teacher 模型

日期：2026-06-20

## 本轮目标

- 将人工审核后的 `D:\Aiparking\image backcup\images（6）` 作为高权重训练素材加入训练。
- 继续使用 YOLOv8s-seg 训练最终版旧模型，用作后续素材筛选和新轻量模型生成的 teacher。
- 暂不在本轮把 Parking 多边形强制转四边形，四边形输出留到下一阶段 YOLOv8-pose/轻量学生模型中处理。
- 训练前已备份项目到 `D:\Aiparking\Aiparking For YOLObackup`。

## 代码调整

- `build_yolov8_dataset.py` 新增 `--trusted-source` 参数。
- `images（6）` 被设为可信人工审核源：该目录中的标注不再因为旧模型遗留的 `score < 0.4` 被过滤。
- 修正训练集构建中的目录名处理，避免默认排除目录因为中文括号编码问题失效。
- 新增单元测试，确保可信源会保留人工审核后的低置信度标注。

## 训练数据

数据根目录：`D:\Aiparking\image backcup`

| 来源目录 | 基础权重 | 说明 |
|---|---:|---|
| `images` | 1 | 早期审核素材 |
| `images（1）` | 1 | 原始人工标注素材 |
| `images（2）` | 2 | 后续审核素材 |
| `images（3）` | 3 | 后续审核素材 |
| `images（4）` | 4 | 后续审核素材 |
| `images（6）` | 6 | 本轮人工审核 barrier 强化素材，可信源 |

额外规则：含 `barrier` 的图片训练重复倍率再乘 2。

生成训练集：`D:\Aiparking\image backcup\dataset_yolov8_weighted`

| 类别 | 训练实例 | 验证实例 | 总计 |
|---|---:|---:|---:|
| Parking | 16,664 | 1,171 | 17,835 |
| barrier | 9,064 | 160 | 9,224 |

训练图片数：15,906  
验证图片数：996

## 训练配置

| 项目 | 值 |
|---|---|
| 模型 | YOLOv8s-seg |
| 初始权重 | `models\best.pt` |
| 输入尺寸 | 640 |
| batch | 16 |
| epochs | 120 |
| patience | 25 |
| 训练设备 | RTX 4060 Laptop GPU |
| 训练输出 | `runs\parking_yolov8_teacher_final` |

训练过程按约 30 分钟间隔检查，确认 GPU 使用正常。第一次 batch=8 启动后预计耗时过长，已在早期停止并改为 batch=16 重新训练。

## 训练结果

训练在 epoch 51 后触发 EarlyStopping，最佳结果出现在 epoch 26。

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.981 | 0.826 | 0.909 | 0.745 |
| Parking | 0.979 | 0.894 | 0.976 | 0.836 |
| barrier | 0.984 | 0.758 | 0.842 | 0.654 |

## 对比上一轮

| 指标 | v5.0 | v6.0 | 变化 |
|---|---:|---:|---:|
| All Mask mAP50 | 0.756 | 0.909 | +20.2% |
| All Mask mAP50-95 | 0.571 | 0.745 | +30.5% |
| barrier Mask mAP50 | 0.533 | 0.842 | +58.0% |
| barrier Mask mAP50-95 | 0.288 | 0.654 | +127.1% |

## 输出文件

- `models\best.pt`
- `models\best.onnx`
- `models\best_512.onnx`
- `runs\parking_yolov8_teacher_final\weights\best.pt`
- `runs\parking_yolov8_teacher_final\weights\best.onnx`
- `runs\parking_yolov8_teacher_final\weights\best_512.onnx`

ONNX 校验：

| 文件 | 输入 | 输出 |
|---|---|---|
| `best.onnx` | `[1, 3, 640, 640]` | `[1, 38, 8400]`, `[1, 32, 160, 160]` |
| `best_512.onnx` | `[1, 3, 512, 512]` | `[1, 38, 5376]`, `[1, 32, 128, 128]` |

## 结论

本轮是旧 teacher 模型的最后强化训练。`images（6）` 的大量 barrier 人工标注显著改善了 barrier 分割能力，已经适合作为后续素材筛选模型。下一阶段建议用该 teacher 模型筛出高质量素材，再从零训练更轻量的板端模型：Parking 使用四点输出方案，barrier 使用轻量检测方案。
