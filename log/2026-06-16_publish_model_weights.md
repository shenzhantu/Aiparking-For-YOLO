# 2026-06-16 模型权重上传策略调整

## 背景

GitHub 仓库清理后，原策略是不提交 `.pt` 和 `.onnx` 模型文件，只保留代码、文档和 Markdown 日志。

本次根据项目实战需求调整：仓库需要直接携带当前稳定模型，便于控制端同学下载和转换板端格式。

## 本次操作

- 保留 `runs/`、`weights/`、训练中间权重和数据集忽略规则，避免仓库再次膨胀。
- 放行并提交当前稳定模型：
  - `models/best.pt`
  - `models/best.onnx`
- 将最新 YOLOv8s-seg 训练结果从 `runs/parking_yolov8_seg/weights/` 同步到 `models/`。
- 更新 `README.md`，说明仓库内模型文件用途。
- 更新 `CHANGELOG.md`，记录 v4.0 模型交付位置。

## 模型文件

| 文件 | 用途 | 大小 |
|------|------|------|
| `models/best.pt` | Ultralytics/PyTorch 继续训练或推理 | 23.9 MB |
| `models/best.onnx` | 板端部署中间格式，可继续转换 OM | 47.4 MB |

## 注意

- `runs/` 中的训练过程输出不会提交。
- `epoch*.pt` 等中间检查点不会提交。
- 如后续模型单文件超过 GitHub 100 MB 限制，应改用 Git LFS 或 GitHub Release。
