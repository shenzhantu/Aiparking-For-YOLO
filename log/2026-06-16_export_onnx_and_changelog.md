# 2026-06-16 ONNX 导出与 CHANGELOG 同步日志

## 本次操作

- 使用最新 YOLOv8s-seg 最佳权重导出 ONNX：
  - 输入：`D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.pt`
  - 输出：`D:\Aiparking\Aiparking For YOLO\runs\parking_yolov8_seg\weights\best.onnx`
- 将 `D:\Aiparking\Aiparking For YOLO\CHANGELOG.md` 同步到 `v4.0`。
- 按原有 CHANGELOG 风格补充：
  - 数据集规模
  - 训练结果
  - 与 v3.0 对比
  - 功能更新
  - images（5）预测结果
  - 技术栈更新
- 刷新本轮正式日志：
  - `D:\Aiparking\Aiparking For YOLO\log\2026-06-16_yolov8_iteration4.md`

## ONNX 验证

- ONNX 文件存在：是
- 文件大小：`47,376,329` 字节
- opset：`12`
- 输入：`images`，形状 `[1, 3, 640, 640]`
- 输出：
  - `output0`，形状 `[1, 38, 8400]`
  - `output1`，形状 `[1, 32, 160, 160]`

## 测试验证

- 单元测试：`7/7` 通过
- 验证命令：`python -m unittest discover -s tests -v`
