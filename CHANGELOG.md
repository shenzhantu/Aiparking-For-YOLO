# 更新日志

本项目使用语义化版本号记录模型训练、数据集迭代和部署文件变化。

---

## v9.0 (2026-07-14) - IImages v2 迭代：New image 6 过采样融合

### 训练数据

| 项目 | 数量 |
|---|---:|
| 有效图片/JSON 配对 | 10,145 |
| 训练集（含过采样） | 15,010 |
| 验证集 | 1,939 |
| 训练实例 | 12,515 |
| 验证实例 | 1,646 |
| 训练负样本 | 2,568 |
| 验证负样本 | 295 |

> 在 v1 基础上迭代，数据源为全量 `D:\Aiparking\IImages`，新增 `New image 6`（过采样 3 倍）和 `New image 5`（过采样 2 倍）。
> `build_iimages_yolov8_dataset.py` 新增 `--oversample` 参数支持按目录关键词过采样，仅在 train split 生效。
> 验证集按完整场景组 `New image 4\Ground_004` 隔离，seed=20260714。

### 训练结果

> 从 `models\best_yolov8n_iimages_quad_v1.pt` 微调，输入尺寸 512，batch 32，epochs 300 上限，patience 60。
> 实际训练 61 epochs，EarlyStopping 触发，最佳模型来自第 1 轮，耗时 10.883 小时。
> 注：微调默认 lr0=0.01 偏高，epoch 1 后指标持续小幅下降，建议后续微调降至 0.001。

| 指标 | Box | Mask |
|---|---:|---:|
| Precision | 0.847 | 0.848 |
| Recall | 0.998 | 0.999 |
| mAP50 | 0.971 | 0.972 |
| mAP50-95 | 0.855 | 0.838 |

### 部署文件

- `models\best_yolov8n_iimages_quad_v2.pt`，约 6.5 MB
- `models\best_yolov8n_iimages_quad_v2_512.onnx`，约 12.6 MB
- ONNX 输入：`1x3x512x512`
- ONNX 输出：`output0=(1,37,5376)`、`output1=(1,32,128,128)`
- 结构与 v1 完全一致，ONNX 检查通过

### 代码与日志

- `build_iimages_yolov8_dataset.py` 新增 `--oversample` 参数（按目录关键词对训练集样本过采样）。
- 新增中文训练日志：`log\2026-07-14_iimages_yolov8n_quad_v2.md`。
- 现有 19 个单元测试全部通过。

---

## v8.0 (2026-07-13) - IImages 场景专用轻量 YOLOv8n-seg 模型

### 训练数据

| 项目 | 数量 |
|---|---:|
| 原始图片 | 10,646 |
| 图片/JSON 有效配对 | 7,503 |
| 实际可用样本 | 7,471 |
| 训练集 | 5,604 |
| 验证集 | 1,867 |
| 训练实例 | 4,448 |
| 验证实例 | 1,814 |
| 训练负样本 | 1,186 |
| 验证负样本 | 74 |

> 本轮跳过相似图片去重和训练前备份，按要求直接使用 `D:\Aiparking\IImages` 中有同名 JSON 的素材。`Parking`/`parking` 已统一为 `Parking`，只训练 Parking 单类别；四边形标注保留并转换为 YOLOv8-seg 标签。

### 训练结果

> 使用 `yolov8n-seg.pt`，输入尺寸 512，batch 32，最多 500 epochs，patience 80。实际训练 344 epochs，耗时 6.981 小时，最佳模型来自第 264 轮。

| 指标 | Box | Mask |
|---|---:|---:|
| Precision | 0.982 | 0.981 |
| Recall | 0.942 | 0.941 |
| mAP50 | 0.987 | 0.986 |
| mAP50-95 | 0.893 | 0.836 |

### 部署文件

- `models\best_yolov8n_iimages_quad_v1.pt`，约 6.9 MB
- `models\best_yolov8n_iimages_quad_v1_512.onnx`，约 13.2 MB
- ONNX 输入：`1x3x512x512`
- ONNX 输出：`output0=(1,37,5376)`、`output1=(1,32,128,128)`
- ONNX 模型检查通过

### 代码与日志

- 新增 `build_iimages_yolov8_dataset.py`，支持递归读取 `images/raw_images`、负样本、标签归一化和四边形规范化。
- 新增 `monitor_iimages_training.ps1`，用于每小时记录训练轮次、指标和 GPU 状态。
- 新增中文训练日志：`log\2026-07-13_iimages_yolov8n_quad_v1.md`。

---

## v7.0 (2026-06-22) - 轻量 Student 模型（YOLOv8n-seg，512 输入）

### 素材筛选：images（7）旧模型高置信精选

| 项目 | 数量 |
|---|---:|
| 原始图片 | 3,802 |
| 哈希去重后保留 | 1,275 |
| 过滤相似/重复图片 | 2,527 |
| 高置信样本（>=0.9） | 334 |
| 不确定样本 | 727 |
| 空图/漏检样本 | 214 |

> 精选输出位于仓库外部 `D:\Aiparking\Premium photo\images7_teacher_selected`。
> 本次训练集位于仓库外部 `D:\Aiparking\image backcup\dataset_yolov8_lightweight_v1`，避免 GitHub 仓库继续膨胀。

### 轻量训练集

| 划分 | 图片数 | Parking 实例 | barrier 实例 |
|---|---:|---:|---:|
| train | 3,306 | 3,368 | 1,006 |
| val | 294 | 318 | 50 |

### 训练结果

> 使用 `yolov8n-seg.pt` 从零开始训练，输入尺寸 512，batch 32，训练 240 epoch。
> 最佳模型来自 epoch 211；训练结束后额外导出 512 输入 ONNX。

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | 0.9548 | 0.7711 | 0.8033 | 0.6498 |
| Parking | 0.916 | 0.865 | 0.956 | 0.810 |
| barrier | 0.938 | 0.682 | 0.718 | 0.484 |

### 部署文件

- 新增轻量 PyTorch 权重：`models\best_yolov8n_light.pt`（约 6.8 MB）
- 新增轻量 ONNX：`models\best_yolov8n_light_512.onnx`（约 13.2 MB）
- ONNX 输入：`1x3x512x512`
- ONNX 输出：`output0=(1,38,5376)`，`output1=(1,32,128,128)`

### 代码更新

- 新增 `select_premium_images.py`，用于图片哈希去重、旧模型筛选、高置信/不确定/空图分桶。
- 优化 `dedupe_similar_images.py`，复用 BK-tree 去重流程，避免大规模图片 O(n²) 比较过慢。
- 更新 `train.py`，关闭 Ultralytics 训练结束绘图，避免当前环境 `polars` CPU feature 检查异常阻断 ONNX 导出。
- 新增对应单元测试，当前测试数量为 19 个。

---

## v6.0 (2026-06-20) - 最终 Teacher 模型（YOLOv8，barrier 强化）

### 数据集：16,902 张加权训练/验证图片

| 类别 | 训练实例 | 验证实例 | 总计 |
|---|---:|---:|---:|
| Parking | 16,664 | 1,171 | 17,835 |
| barrier | 9,064 | 160 | 9,224 |

> 本轮训练素材位于仓库外部 `D:\Aiparking\image backcup`。
> `images（6）` 已人工审核，作为可信源加入训练，基础权重为 6。
> 含 `barrier` 的图片额外使用 2 倍过采样。

### 训练结果

> 使用 `models\best.pt` 继续微调，实际训练到 epoch 51 后 EarlyStopping，最佳模型来自 epoch 26。

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | 0.981 | 0.826 | 0.909 | 0.745 |
| Parking | 0.979 | 0.894 | 0.976 | 0.836 |
| barrier | 0.984 | 0.758 | 0.842 | 0.654 |

### 与 v5.0 对比

| 指标 | v5.0 | v6.0 | 变化 |
|---|---:|---:|---:|
| All Mask mAP50 | 0.756 | 0.909 | **+20.2%** |
| All Mask mAP50-95 | 0.571 | 0.745 | **+30.5%** |
| barrier Mask mAP50 | 0.533 | 0.842 | **+58.0%** |
| barrier Mask mAP50-95 | 0.288 | 0.654 | **+127.1%** |

### 功能更新

- 新增 `--trusted-source`，人工审核目录不再因旧模型残留的低 `score` 被过滤。
- 修复训练集构建中中文括号目录名处理问题。
- 使用 `images（6）` 强化 barrier，显著改善 barrier 分割质量。
- 保持 YOLOv8s-seg 作为最终 teacher，用于后续高质量素材筛选。
- 更新部署模型：`models\best.pt`、`models\best.onnx`、`models\best_512.onnx`。

---

## v5.0 (2026-06-19) - 第五轮训练（YOLOv8，barrier 初步强化）

### 数据集：14,686 张图片

| 类别 | 训练实例 | 验证实例 | 总计 |
|---|---:|---:|---:|
| Parking | 15,694 | 1,152 | 16,846 |
| barrier | 2,562 | 91 | 2,653 |

### 训练结果

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | 0.974 | 0.786 | 0.756 | 0.571 |
| Parking | 0.986 | 0.919 | 0.980 | 0.855 |
| barrier | 0.962 | 0.653 | 0.533 | 0.288 |

### 功能更新

- 新增 `--class-boost barrier:3`，针对 barrier 样本过采样。
- 修复多尺寸 ONNX 导出逻辑，避免 512 导出覆盖 640 标准模型。
- 仓库保留当前稳定模型：`models\best.pt`、`models\best.onnx`、`models\best_512.onnx`。

---

## v4.0 (2026-06-16) - 第四轮训练（切换 YOLOv8）

### 数据集：13,579 张图片

| 类别 | 训练实例 | 验证实例 | 总计 |
|---|---:|---:|---:|
| Parking | 14,620 | 1,144 | 15,764 |
| barrier | 878 | 78 | 956 |

### 训练结果

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | 0.904 | 0.671 | 0.673 | 0.504 |
| Parking | 0.967 | 0.874 | 0.963 | 0.816 |
| barrier | 0.841 | 0.468 | 0.383 | 0.193 |

### 功能更新

- 主训练模型从 YOLO11s-seg 切换为 YOLOv8s-seg，优先适配板端部署。
- 训练素材和生成数据集迁移到 `D:\Aiparking\image backcup`，避免 GitHub 仓库体积过大。
- 使用加权训练数据，降低早期低质量素材影响。

---

## v3.0 (2026-06-13) - 第三轮训练

### 数据集：9,968 张图片

| 类别 | 总实例 |
|---|---:|
| Parking | 10,895 |
| barrier | 460 |

### 训练结果

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | 0.975 | 0.697 | 0.692 | 0.492 |
| Parking | 0.980 | 0.905 | 0.971 | 0.820 |
| barrier | 0.971 | 0.489 | 0.412 | 0.164 |

---

## v2.0 (2026-06-11) - 第二轮训练

### 数据集：8,672 张图片

新增 `barrier` 类别，并支持 rectangle、cuboid、polygon 多种标注形状。

| 类别 | Box mAP50 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|
| **all** | 0.982 | 0.678 | 0.507 |
| Parking | 0.982 | 0.974 | 0.840 |
| barrier | 0.982 | 0.383 | 0.175 |

---

## v1.0 (2026-06-10) - 初始训练

### 数据集：7,432 张图片

原始人工标注 1,416 张，加上审核后的自动标注 6,016 张。

| 指标 | Box | Mask |
|---|---:|---:|
| Precision | 0.972 | 0.971 |
| Recall | 0.960 | 0.946 |
| mAP50 | 0.986 | 0.978 |
| mAP50-95 | 0.945 | 0.856 |

---

## 技术栈

- 模型：YOLOv8s-seg（当前最终 teacher 版本）
- 训练设备：RTX 4060 Laptop GPU (8GB VRAM)
- 标注工具：X-AnyLabeling
- 数据格式：X-AnyLabeling JSON -> YOLO Seg TXT -> PyTorch/ONNX
- 部署文件：`models\best.pt`、`models\best.onnx`、`models\best_512.onnx`
