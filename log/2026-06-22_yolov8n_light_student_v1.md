# 2026-06-22 YOLOv8n 轻量 Student 模型训练日志

## 目标

本轮目标是从现有素材中筛选更精炼、更不重复的高质量图片，训练一个适合板端部署的轻量 YOLOv8 模型。模型优先考虑低延迟和较高帧率，同时保留 `Parking` 与 `barrier` 两个类别。

## 训练前备份

- 已在训练前备份项目目录。
- 备份位置：`D:\Aiparking\Aiparking For YOLObackup`
- 备份排除：`.git`、`dedupe_preview`、`__pycache__`、`*.pyc`

## 素材筛选

使用旧 teacher 模型对 `D:\Aiparking\image backcup\Photo\images（7）\images` 进行筛选，并输出到：

`D:\Aiparking\Premium photo\images7_teacher_selected`

筛选结果：

| 项目 | 数量 |
|---|---:|
| 原始图片 | 3,802 |
| 哈希去重后保留 | 1,275 |
| 过滤相似/重复图片 | 2,527 |
| 高置信样本（>=0.9） | 334 |
| 不确定样本 | 727 |
| 空图/漏检样本 | 214 |
| 读取失败图片 | 0 |

## 轻量训练集

训练集位置：

`D:\Aiparking\image backcup\dataset_yolov8_lightweight_v1`

训练集规模：

| 划分 | 图片数 | Parking 实例 | barrier 实例 |
|---|---:|---:|---:|
| train | 3,306 | 3,368 | 1,006 |
| val | 294 | 318 | 50 |

本轮继续保留对 `barrier` 的过采样权重，但训练集整体规模明显小于此前 teacher 模型，目的是提高板端推理速度。

## 训练配置

- 模型：`yolov8n-seg.pt`
- 任务：实例分割
- 输入尺寸：512
- epoch：240
- batch：32
- 设备：RTX 4060 Laptop GPU / CUDA
- 输出目录：`D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8n_light_seg`

训练过程中持续监控 GPU 与 epoch 进度。训练实际跑满 240 epoch，总耗时约 2.7 小时。

## 训练结果

最佳模型来自 epoch 211。

| 指标 | 数值 |
|---|---:|
| all Box mAP50 | 0.9548 |
| all Box mAP50-95 | 0.7711 |
| all Mask mAP50 | 0.8033 |
| all Mask mAP50-95 | 0.6498 |

最终验证中，各类别表现如下：

| 类别 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|
| Parking | 0.944 | 0.810 |
| barrier | 0.656 | 0.484 |

结论：

- `Parking` 识别质量较稳定。
- `barrier` 比早期弱模型有改善，但验证集中只有 50 个 barrier 实例，后续仍需要更多不同角度、不同距离、不同光照下的障碍物样本。
- YOLOv8n-seg 参数量约 326 万，ONNX 仅约 13.2 MB，适合作为板端轻量候选模型。

## 导出文件

已生成并复制到 `models` 目录：

- `D:\Aiparking\Aiparking For YOLO\models\best_yolov8n_light.pt`
- `D:\Aiparking\Aiparking For YOLO\models\best_yolov8n_light_512.onnx`

ONNX 校验结果：

- 输入：`images = 1x3x512x512`
- 输出：`output0 = 1x38x5376`
- 输出：`output1 = 1x32x128x128`

## 代码更新

- 新增 `select_premium_images.py`：完成图片哈希去重、旧模型筛选、高置信/不确定/空图分桶。
- 优化 `dedupe_similar_images.py`：复用 BK-tree 去重流程，避免大规模图片 O(n²) 慢比较。
- 更新 `train.py`：关闭 Ultralytics 训练结束绘图，避免当前环境 `polars` CPU feature 检查异常阻断 ONNX 导出。
- 新增测试 `tests/test_select_premium_images.py`。
- 更新测试 `tests/test_train_polars_fallback.py`。

## 验证

- 单元测试：`python -m unittest discover -s tests`
- 结果：19 个测试全部通过。
- ONNX：`onnx.checker.check_model` 通过。

## 后续建议

1. 将 `models\best_yolov8n_light_512.onnx` 交给板端同学做实际 FPS 和精度测试。
2. 使用该轻量模型对新素材做筛选，而不是直接把所有旧素材继续堆进训练集。
3. 对 `barrier` 单独补充更密集、更丰富的人工审核样本，再考虑下一版轻量模型。
