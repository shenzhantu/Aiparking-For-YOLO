# 2026-07-04 New image 3 相似素材筛选

## 本次目标

对新素材 `D:\Aiparking\New Images\New image 3` 进行静态/高度相似图片筛选，仅保留差异较明显的代表样本。

本次没有启动训练，也没有调用旧模型进行置信度筛选。

## 使用脚本

- `D:\Aiparking\Aiparking For YOLO\dedupe_similar_images.py`

## 输入与输出

- 输入目录：`D:\Aiparking\New Images\New image 3\images`
- 输出目录：`D:\Aiparking\New Images\New image 3 deduped\unique_images`
- 报告目录：`D:\Aiparking\New Images\New image 3 deduped\reports`

## 参数

- 相似阈值：`6`
- 输出方式：`copy`
- 原始素材：不删除、不移动

## 筛选结果

- 原始图片数：2547
- 保留代表样本：578
- 筛掉近重复图片：1969
- 近重复组数：319
- 坏图/无法读取：0
- 同名 JSON 一并复制：21

## 说明

第一次运行时以 `D:\Aiparking\New Images\New image 3` 作为直接图片层级，结果扫描到 0 张图片。检查后确认实际图片位于下一级 `images` 目录，因此重新以 `D:\Aiparking\New Images\New image 3\images` 作为扫描源完成筛选。

由于本次任务不是训练任务，也没有修改训练数据集或模型文件，所以没有执行训练前备份。
