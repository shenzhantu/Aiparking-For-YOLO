# 更新日志

本项目使用语义化版本号，基于训练迭代记录。

---

## v5.0 (2026-06-19) — 第五轮训练（YOLOv8，barrier 强化）

### 📊 数据集：14,686 张图片

| 类别 | 训练集 | 验证集 | 总计 |
|------|--------|--------|------|
| Parking | 15,694 | 1,152 | 16,846 |
| barrier | 2,562 | 91 | 2,653 |

> 本轮训练素材位于仓库外部 `D:\Aiparking\image backcup`；`images（6）/` 本轮只作为预测目标，不参与训练。
> 低置信度过滤规则保持为 `score < 0.4` 不进入训练；类别强化参数：`{'1': 3}`。

### 🎯 训练结果

> 当前记录来自第 `36` 轮训练后的 best.pt 验证结果。

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|------|-----------|-------------|------------|---------------|
| **all** | 0.974 | 0.786 | 0.756 | 0.571 |
| Parking | 0.986 | 0.919 | 0.980 | 0.855 |
| barrier | 0.962 | 0.653 | 0.533 | 0.288 |

### 📈 与 v4.0 对比

| 指标 | v4.0 | v5.0 当前 | 变化 |
|------|------|-----------|------|
| All Mask mAP50 | 0.673 | 0.756 | +12.3% |
| barrier Mask mAP50 | 0.383 | 0.533 | +39.2% |

### 🆕 功能更新

- 训练主模型从 YOLO11s-seg 切换为 YOLOv8s-seg，优先适配板端部署。
- 图片素材和生成数据集迁移到 `D:\Aiparking\image backcup`，避免 GitHub 仓库体积过大。
- 使用加权数据集：较新的审核素材拥有更高训练占比，降低早期低质量素材的影响。
- 对含 `barrier` 的训练样本进行额外过采样，集中增强障碍物识别。
- 保留 `Parking` 与 `barrier` 两类，并继续输出适合板端转换的 ONNX 模型。

### 📝 预测 images（6）/ — 300 张，JSON 92 个，检出率 30.7%

- 预测标签统计：`{'Parking': 95, 'barrier': 25}`
- 带 score 的自动标注数量：`120`

### 📦 输出文件

- PyTorch 模型：`D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.pt`，存在：`True`
- ONNX 模型：`D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.onnx`，存在：`True`
- 训练输出日志：`D:\Aiparking\Aiparking For YOLO\log\2026-06-18_235526_yolov8_barrier_train_stdout.log`
- 训练监控日志：`D:\Aiparking\Aiparking For YOLO\log\2026-06-18_235526_yolov8_barrier_monitor.md`

### 🧰 技术栈

- **模型**：YOLOv8s-seg
- **训练设备**：RTX 4060 Laptop GPU (8GB VRAM)
- **训练尺寸**：640
- **标注工具**：X-AnyLabeling
- **数据格式**：X-AnyLabeling JSON → YOLO Seg TXT
