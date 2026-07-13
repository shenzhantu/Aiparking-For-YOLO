# 2026-07-13 IImages 全新 YOLOv8n-seg 训练

## 训练状态

- 状态：已完成
- 本轮不执行相似图片去重
- 本轮不执行训练前备份（按本次明确要求）
- 训练模型：`YOLOv8n-seg`
- 输入尺寸：512
- 类别：`Parking`
- 标注形式：四边形标注转换为 YOLOv8-seg 分割标签
- 训练目标：约 6-8 小时，500 epochs 上限，patience=80
- 首次检查：第 1 轮，GPU 利用率约 81%，显存约 5.46GB，温度约 66°C

## 最终训练结果

- 实际训练轮数：344/500
- 停止原因：连续 80 轮没有超过最佳综合指标，触发 EarlyStopping
- 最佳模型：第 264 轮保存的 `best.pt`
- 实际训练时长：6.981 小时
- 训练设备：NVIDIA GeForce RTX 4060 Laptop GPU，CUDA
- 模型参数：约 3,258,259
- 计算量：约 11.3 GFLOPs

使用最佳模型在验证集上的最终指标：

| 指标 | Box | Mask |
|---|---:|---:|
| Precision | 0.982 | 0.981 |
| Recall | 0.942 | 0.941 |
| mAP50 | 0.987 | 0.986 |
| mAP50-95 | 0.893 | 0.836 |

验证速度约为 0.1ms 预处理、0.9ms 推理、1.9ms 后处理/图像（当前 PC 环境）。

## 部署文件

- PyTorch：`D:\Aiparking\Aiparking For YOLO\models\best_yolov8n_iimages_quad_v1.pt`
- ONNX：`D:\Aiparking\Aiparking For YOLO\models\best_yolov8n_iimages_quad_v1_512.onnx`
- ONNX 输入：`1x3x512x512`
- ONNX 输出：`output0=(1,37,5376)`、`output1=(1,32,128,128)`
- ONNX 检查：通过 `onnx.checker`

## 数据集

- 原始目录：`D:\Aiparking\IImages`
- 生成目录：`D:\Aiparking\IImages\dataset_yolov8n_quad_v1`
- 有效图片/JSON 配对：7471
- 训练集：5604 张
- 验证集：1867 张
- 训练实例：4448
- 验证实例：1814
- 训练负样本：1186
- 验证负样本：74
- 低于 0.4 的自动标注实例：过滤 39 个
- 空标注：作为负样本保留
- `Parking`/`parking`：统一为 `Parking`
- 派生的 `prelabel_quad` 目录：排除

## 监控

训练标准输出：`D:\Aiparking\Aiparking For YOLO\log\2026-07-13_iimages_yolov8n_train_stdout.log`

每小时监控日志：`D:\Aiparking\Aiparking For YOLO\log\2026-07-13_iimages_yolov8n_monitor.log`

训练结束后的 epoch、最佳指标、GPU 情况和 ONNX 导出结果已补充；本次代码、日志和部署模型已通过 HTTPS 推送到 GitHub。
