# Aiparking For YOLO

基于 YOLOv8-seg 的停车场图像半自动标注与迭代训练项目。
识别类别：`Parking`（停车位）、`barrier`（障碍物/道闸）。

## 环境

| 项目 | 值 |
|------|-----|
| Python | `C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe` (3.11.9) |
| GPU | RTX 4060 Laptop (8GB VRAM), CUDA 12.1 |
| 框架 | ultralytics 8.4.21, PyTorch 2.5.1+cu121 |
| 标注工具 | X-AnyLabeling (输出 X-AnyLabeling JSON) |
| 项目根目录 | `D:\Aiparking\Aiparking For YOLO\` |

**重要**：所有 Python 命令必须使用 conda 环境绝对路径执行，不可用 `python` 或 `python3`。

## 关键目录

| 用途 | 路径 |
|------|------|
| 项目根目录 | `D:\Aiparking\Aiparking For YOLO\` |
| 训练素材（已审核） | `D:\Aiparking\image backcup\` |
| 已审核素材 | `D:\Aiparking\image backcup\images（N）\` |
| 原始未审核素材 | `D:\Aiparking\image backcup\Photo\images（N）\` |
| 生成的训练集 | `D:\Aiparking\image backcup\dataset_yolov8_weighted\` |
| Teacher 模型 | `models\best.pt` / `.onnx` / `_512.onnx` |
| Student 模型 | `models\best_yolov8n_light.pt` / `_512.onnx` |
| 训练输出 | `runs\parking_yolov8_seg\` |
| 日志目录 | `log\` |
| 项目备份 | `D:\Aiparking\Aiparking For YOLObackup\` |
| 精选素材 | `D:\Aiparking\Premium photo\` |

## 素材命名约定

- `images`（无括号）= 最早审核素材，训练权重 1
- `images（N）` = 第 N 批审核素材（**全角括号**），权重 min(N, 4)
- `Photo\images（N）` = 第 N 批未审核原始素材
- `images（5）` = **固定保留为独立预测验证集，永远不参与训练**
- `Premium photo\` = Teacher 模型精选/分桶输出

## 版本历史

| 版本 | 日期 | 说明 | Mask mAP50 |
|------|------|------|:---:|
| v1.0 | 06-10 | 初始训练 (YOLO11) | 0.978 |
| v2.0 | 06-11 | 新增 barrier 类 | 0.678 |
| v3.0 | 06-13 | 第三轮 | 0.692 |
| v4.0 | 06-16 | 切换 YOLOv8 | 0.673 |
| v5.0 | 06-19 | barrier 初步强化 | 0.756 |
| v6.0 | 06-20 | Teacher 最终版 | 0.909 |
| v7.0 | 06-22 | Student 轻量版 | 0.803 |

## 核心脚本

| 脚本 | 用途 |
|------|------|
| `build_yolov8_dataset.py` | X-AnyLabeling JSON → 加权 YOLO 数据集 |
| `train.py` | YOLOv8-seg 训练 + 自动 ONNX 导出 |
| `predict.py` | 推理 → X-AnyLabeling JSON 自动标注 |
| `monitor_training.py` | 训练过程 GPU/指标定期采样（每 30 分钟） |
| `write_iteration_log.py` | 生成中文 Markdown 训练日志 |
| `export_onnx.py` | 多尺寸 ONNX 导出 (640 + 512) |
| `select_premium_images.py` | 新素材去重 + Teacher 分桶筛选 |
| `dedupe_similar_images.py` | BK-tree 感知哈希去重引擎 |
| `finish_after_training.py` | 训练后编排（备用方案） |

---

## 标准训练迭代流程 (SOP)

每次迭代按以下 7 步执行。具体命令中 `<PLACEHOLDER>` 替换为实际值。

### 步骤 0：项目备份

```powershell
robocopy "D:\Aiparking\Aiparking For YOLO" "D:\Aiparking\Aiparking For YOLObackup" /MIR /XD .git dedupe_preview __pycache__ /XF *.pyc /NP /NJH /NJS
```

验证：检查备份目录存在且包含核心文件。

### 步骤 1：构建加权数据集

```powershell
& "C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" `
  "D:\Aiparking\Aiparking For YOLO\build_yolov8_dataset.py" `
  --data-root "D:\Aiparking\image backcup" `
  --output "D:\Aiparking\image backcup\dataset_yolov8_weighted" `
  --source <SOURCES> `
  --trusted-source <TRUSTED_SOURCES> `
  --exclude <EXCLUDES> `
  --class-boost <CLASS_BOOST> `
  --min-conf 0.4 `
  --val-ratio 0.15 `
  --seed 20260616
```

验证：检查 `build_summary.json` 和 `parking_yolov8.yaml` 已生成。

### 步骤 2：启动训练 + 监控

```powershell
# 终端 1: 训练（前台）
& "C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" `
  "D:\Aiparking\Aiparking For YOLO\train.py" `
  --data "D:\Aiparking\image backcup\dataset_yolov8_weighted\parking_yolov8.yaml" `
  --model <MODEL> `
  --epochs <EPOCHS> `
  --imgsz 640 --batch 8 --patience 35 --save-period 20
```

```powershell
# 终端 2: 监控（后台，每 30 分钟采样）
& "C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" `
  "D:\Aiparking\Aiparking For YOLO\monitor_training.py" `
  --run-log "<LOG_DIR>\train_stdout.log" `
  --results-csv "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\results.csv" `
  --monitor-log "<LOG_DIR>\monitor.md" `
  --done-file "<LOG_DIR>\train.done" `
  --interval 1800
```

验证：训练完成后检查 `weights/best.pt` 和 `weights/best.onnx` 存在。

### 步骤 3：多尺寸 ONNX 导出

```powershell
& "C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" `
  "D:\Aiparking\Aiparking For YOLO\export_onnx.py" `
  --model "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.pt" `
  --imgsz 640 512
```

验证：检查 `best.onnx` 和 `best_512.onnx` 存在。

### 步骤 4：复制模型到 models/

```powershell
Copy-Item "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.pt" "D:\Aiparking\Aiparking For YOLO\models\best.pt" -Force
Copy-Item "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.onnx" "D:\Aiparking\Aiparking For YOLO\models\best.onnx" -Force
Copy-Item "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best_512.onnx" "D:\Aiparking\Aiparking For YOLO\models\best_512.onnx" -Force
```

### 步骤 5：对新素材自动标注

```powershell
& "C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" `
  "D:\Aiparking\Aiparking For YOLO\predict.py" `
  --model "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.pt" `
  --target "<PREDICT_TARGET>" `
  --conf 0.4
```

注意：**不要**对已有人工审核的目录加 `--overwrite`。

### 步骤 6：写入日志

```powershell
& "C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe" `
  "D:\Aiparking\Aiparking For YOLO\write_iteration_log.py" `
  --output "<LOG_DIR>\<DATE>_<VERSION_DESC>.md" `
  --dataset-summary "D:\Aiparking\image backcup\dataset_yolov8_weighted\build_summary.json" `
  --results-csv "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\results.csv" `
  --prediction-target "<PREDICT_TARGET>" `
  --model "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.pt" `
  --onnx "D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.onnx" `
  --run-log "<LOG_DIR>\train_stdout.log" `
  --monitor-log "<LOG_DIR>\monitor.md" `
  --current-version "<VERSION>" `
  --current-title "<TITLE>" `
  --previous-version "<PREV_VERSION>" `
  --previous-all-mask-map50 <PREV_ALL_MASK_MAP50> `
  --previous-barrier-mask-map50 <PREV_BARRIER_MASK_MAP50>
```

### 步骤 7：Git 提交 + 推送 GitHub

```powershell
git -C "D:\Aiparking\Aiparking For YOLO" add log/*.md CHANGELOG.md README.md
git -C "D:\Aiparking\Aiparking For YOLO" commit -m "docs: add <VERSION> training log"
git -C "D:\Aiparking\Aiparking For YOLO" push org master
```

远程仓库：`org` → `git@github.com:shenzhantu/Aiparking-For-YOLO.git`

---

## 用户关键约束（不可违反）

1. ✅ **训练前必须备份**到 `D:\Aiparking\Aiparking For YOLObackup`
2. ❌ **绝不删除**任何已有标注 JSON 文件
3. 👀 **训练中必须启动 monitor_training.py** 进行监督
4. 📝 **每次迭代写 Markdown 日志**到 `log/` 目录
5. 🔒 **images（5）固定排除**，永远不参与训练
6. 🐍 **所有 Python 命令**使用 conda 环境绝对路径
7. 📤 **每次迭代后提交并推送**到 GitHub (`git push org master`)

## Git 注意事项

- 仓库仅跟踪代码、Markdown 日志、配置文件
- 大文件（模型权重、数据集、runs）由 `.gitignore` 排除
- `models/best*.pt` 和 `models/best*.onnx` 是例外——已显式放行
- 每次迭代后提交：日志 md、CHANGELOG.md、脚本变更
