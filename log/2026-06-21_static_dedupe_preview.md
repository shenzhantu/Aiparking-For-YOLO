# 2026-06-21 静态近重复素材筛选预览

## 本次目标

为后续“轻量高质量新模型”准备精炼素材，先从静态连拍/画面高度相似的历史素材中筛出不太相似的代表样本，供人工查看筛选效果。

## 新增脚本

- 新增 `dedupe_similar_images.py`
- 默认扫描目录：`D:\Aiparking\image backcup\Photo`
- 默认扫描子目录：`images`、`images（1）`、`images（2）`、`images（3）`、`images（4）`、`images（5）`、`images（6）`
- 默认输出目录：`D:\Aiparking\Aiparking For YOLO\dedupe_preview`
- 默认相似阈值：`6`
- 默认复制方式：`hardlink`，节省磁盘空间；如果硬链接失败会自动退回普通复制。

## 筛选逻辑

- 使用感知哈希 `dHash` 判断画面结构是否接近。
- 额外加入平均亮度分桶，降低纯色、低纹理或曝光差异导致的误合并。
- 使用 BK-tree 索引近邻哈希，避免 1.5 万张图片两两暴力比较。
- 每组近重复图片保留 1 张代表图。
- 如果代表图旁边存在同名 `.json` 标注文件，会同步保留。
- 不删除、不移动、不改写任何原始素材。
- 坏图不会中断流程，会写入 `failed_images.csv`。

## 本次运行结果

| 项目 | 数量 |
|---|---:|
| 扫描图片 | 15,564 |
| 保留代表图片 | 2,419 |
| 筛掉近重复图片 | 13,144 |
| 近重复组 | 1,145 |
| 保留 JSON 标注 | 1,729 |
| 无法读取坏图 | 1 |

## 输出文件

- 代表样本：`D:\Aiparking\Aiparking For YOLO\dedupe_preview\unique_images`
- 汇总报告：`D:\Aiparking\Aiparking For YOLO\dedupe_preview\reports\summary.json`
- 分组明细：`D:\Aiparking\Aiparking For YOLO\dedupe_preview\reports\groups.csv`
- 坏图报告：`D:\Aiparking\Aiparking For YOLO\dedupe_preview\reports\failed_images.csv`

## 坏图记录

- `D:\Aiparking\image backcup\Photo\images\camera_training_20260628_082207_000272.jpg`
- 错误：`image file is truncated (23 bytes not processed)`

## 验证

- 运行 `python -m unittest discover -s tests`
- 结果：14 个测试全部通过。

## 注意

`dedupe_preview/` 是本地预览产物，已加入 `.gitignore`，不会上传到 GitHub。后续如果筛选效果过于激进，可以把阈值降低到 `4`；如果仍保留太多静态近重复图，可以把阈值提高到 `8`。
