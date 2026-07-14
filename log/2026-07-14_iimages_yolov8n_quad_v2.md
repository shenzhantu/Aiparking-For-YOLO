# 2026-07-14 IImages YOLOv8n-seg v2 迭代训练

## 训练状态

- 状态：已完成
- 基础模型：`models\best_yolov8n_iimages_quad_v1.pt`（v1 微调）
- 训练模型：YOLOv8n-seg
- 输入尺寸：512
- 类别：`Parking` 单类别
- 标注形式：四边形 polygon
- 训练目标：epochs 300 上限，patience=60
- 数据策略：全量 IImages + New image 6 过采样 3 倍 + New image 5 过采样 2 倍
- 训练前备份：已执行 robocopy /MIR 到 `D:\Aiparking\Aiparking For YOLObackup`

## 最终训练结果

- 实际训练轮数：61/300
- 停止原因：连续 60 轮没有超过最佳综合指标，触发 EarlyStopping
- 最佳模型：第 1 轮保存的 `best.pt`
- 实际训练时长：10.883 小时
- 训练设备：NVIDIA GeForce RTX 4060 Laptop GPU，CUDA:0
- lr0：0.01（Ultralytics 默认，微调场景偏高）

使用最佳模型（epoch 1）在验证集上的最终指标：

| 指标 | Box | Mask |
|---|---:|---:|
| Precision | 0.847 | 0.848 |
| Recall | 0.998 | 0.999 |
| mAP50 | 0.971 | 0.972 |
| mAP50-95 | 0.855 | 0.838 |

验证速度约为 0.1ms 预处理、0.9ms 推理、1.6ms 后处理/图像。

### 关于 epoch 1 最佳的说明

本轮从 v1 已收敛权重微调，默认 lr0=0.01 对微调偏高，epoch 1 后模型指标持续小幅下降，
未能在后续轮次超过 epoch 1。建议后续微调时将 lr0 降至 0.001 或更低。
由于验证集与 v1 不同（本轮为 New image 4\Ground_004，v1 为 full_frame_ss 等），
指标不完全可比。v1 验证集指标（Box mAP50=0.987, Mask mAP50=0.986）仅供参考。

## 部署文件

- PyTorch：`D:\Aiparking\Aiparking For YOLO\models\best_yolov8n_iimages_quad_v2.pt`
- ONNX：`D:\Aiparking\Aiparking For YOLO\models\best_yolov8n_iimages_quad_v2_512.onnx`
- ONNX 输入：`1x3x512x512`，float32，BCHW
- ONNX 输出：`output0=(1,37,5376)`、`output1=(1,32,128,128)`
- ONNX opset：12，simplify=True，FP32
- ONNX 检查：通过 `onnx.checker`
- 结构与 v1 完全一致，板端配置无需改动

## 数据集

- 原始目录：`D:\Aiparking\IImages`
- 生成目录：`D:\Aiparking\IImages\dataset_yolov8n_quad_v2`
- 构建脚本：`build_iimages_yolov8_dataset.py`（新增 `--oversample` 参数）
- 有效图片/JSON 配对：10,145
- 训练集（含过采样）：15,010 张
- 验证集：1,939 张
- 训练实例：12,515
- 验证实例：1,646
- 训练负样本：2,568
- 验证负样本：295
- 验证集组：`New image 4\Ground_004`（按完整场景组隔离）
- 过采样配置：New image 6 ×3（额外 +5348 张）、New image 5 ×2（额外 +1456 张）
- 低于 0.4 的自动标注实例：过滤 39 个
- 全被过滤的图片：排除 32 张（不写成空标签）
- 空标注：作为负样本保留
- `Parking`/`parking`：统一为 `Parking`
- 派生的 `prelabel_quad` 目录：排除
- seed：20260714

## 训练参数

| 参数 | 值 |
|---|---|
| Python | `C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe` |
| 框架 | Ultralytics 8.4.21，PyTorch 2.5.1+cu121 |
| 基础模型 | `best_yolov8n_iimages_quad_v1.pt` |
| epochs | 300 |
| imgsz | 512 |
| batch | 32 |
| patience | 60 |
| workers | 4 |
| save_period | 25 |
| lr0 | 0.01（默认，偏高） |
| plots | False |
| device | cuda:0 |

## 监控

- 训练标准输出：`log\2026-07-14_iimages_train_stdout.log`
- 训练标准错误：`log\2026-07-14_iimages_train_stderr.log`（空，无错误）
- 每小时监控日志：`log\2026-07-14_iimages_monitor.log`
- 监督脚本：`monitor_iimages_training.ps1`（绑定训练 PID，每小时记录 epoch、mAP、GPU）

## 真实图片抽查

- 抽查来源：`D:\Aiparking\IImages\New image 6\images` 前 5 张
- 抽查结果：5 张全部检出停车位，0 张空检
- 推理脚本：`predict.py`，conf=0.4

## 代码变更

- `build_iimages_yolov8_dataset.py`：新增 `--oversample` 参数，支持按目录关键词对训练集样本过采样。
  过采样只在 train split 生效，val 不受影响。文件名加 `__os{N}` 后缀避免冲突。
- 新增单元测试通过情况：现有 19 个测试全部通过。

## 与 v1 对比

| 指标 | v1 (epoch 264) | v2 (epoch 1) | 说明 |
|---|---:|---:|---|
| Box mAP50 | 0.987 | 0.971 | 验证集不同，不完全可比 |
| Box mAP50-95 | 0.893 | 0.855 | 同上 |
| Mask mAP50 | 0.986 | 0.972 | 同上 |
| Mask mAP50-95 | 0.836 | 0.838 | 基本持平 |
| 训练时长 | 6.981h | 10.883h | 数据集更大 |
| 最佳轮次 | 264 | 1 | 微调 lr0 偏高 |

## 交接要点

- v2 部署文件结构同 v1，可直接替换板端模型文件。
- 若 New image 6 场景效果不佳，建议用 lr0=0.001 从 v1 重新微调。
- `--oversample` 参数已合入构建器，后续迭代可复用。
