# 更新日志

本项目使用语义化版本号，基于训练迭代记录。

---

## v4.0 (2026-06-16) — 第四轮训练（YOLOv8）

### 📊 数据集：13,579 张图片

| 类别 | 训练集 | 验证集 | 总计 |
|------|--------|--------|------|
| Parking | 14,620 | 1,144 | 15,764 |
| barrier | 878 | 78 | 956 |

> 本轮训练素材位于仓库外部 `D:\Aiparking\image backcup`；`images（5）/` 本轮只作为预测目标，不参与训练。
> 低置信度过滤规则保持为 `score < 0.4` 不进入训练；新素材权重更高，最高权重限制为 4。

### 🎯 训练结果

> 当前记录来自第 `29` 轮；如果训练尚未结束，这里是阶段指标，不是最终 best.pt 指标。

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|------|-----------|-------------|------------|---------------|
| **all** | 0.915 | 0.652 | 0.629 | 0.465 |
| Parking | — | — | — | — |
| barrier | — | — | — | — |

### 📈 与 v3.0 对比

| 指标 | v3.0 | v4.0 当前 | 变化 |
|------|------|-----------|------|
| All Mask mAP50 | 0.692 | 0.629 | -9.1% |
| barrier Mask mAP50 | 0.412 | — | — |

### 🆕 功能更新

- 训练主模型从 YOLO11s-seg 切换为 YOLOv8s-seg，优先适配板端部署。
- 图片素材和生成数据集迁移到 `D:\Aiparking\image backcup`，避免 GitHub 仓库体积过大。
- 使用加权数据集：较新的审核素材拥有更高训练占比，降低早期低质量素材的影响。
- 保留 `Parking` 与 `barrier` 两类；后续 barrier 仍需要更多含障碍物素材继续增强。

### 📝 预测 images（5）/ — 1,219 张，JSON 0 个，检出率 0.0%

- 预测标签统计：`{}`
- 带 score 的自动标注数量：`0`

### 📦 输出文件

- PyTorch 模型：`D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.pt`，存在：`True`
- ONNX 模型：`D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.onnx`，存在：`False`
- 训练输出日志：`D:\Aiparking\Aiparking For YOLO\log\2026-06-16_081855_yolov8_train_stdout.log`
- 训练监控日志：`D:\Aiparking\Aiparking For YOLO\log\2026-06-16_081855_yolov8_monitor.md`

### 🧰 技术栈

- **模型**：YOLOv8s-seg
- **训练设备**：RTX 4060 Laptop GPU (8GB VRAM)
- **训练尺寸**：640
- **标注工具**：X-AnyLabeling
- **数据格式**：X-AnyLabeling JSON → YOLO Seg TXT
