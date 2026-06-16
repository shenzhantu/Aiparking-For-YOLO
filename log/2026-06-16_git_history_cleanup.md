# 2026-06-16 Git 历史瘦身日志

## 背景

准备将项目同步到 GitHub 仓库：

```text
shenzhantu/Aiparking-For-YOLO
```

虽然训练素材已经移动到仓库外部，但 `.git` 目录仍接近 4GB。排查后确认，主要原因是旧提交历史中包含了训练权重、ONNX、Ultralytics `runs/` 输出和中间 checkpoint。

## 处理内容

- 更新 `.gitignore`：
  - 忽略 `images*/`、`dataset*/`
  - 忽略 `runs/`
  - 忽略 `*.pt`、`*.onnx`、`*.om`、`*.engine`
  - 忽略 raw runtime 日志：`*.log`、`*.done`、临时 JSON
  - 保留 `log/*.md`，确保训练日志和操作日志能出现在 GitHub
- 从 Git 索引中移除旧训练产物：
  - `runs/`
  - `weights/`
  - `models/best.pt`
  - `models/best.onnx`
  - 顶层预训练权重
- 重写 Git 历史，移除历史中已经提交过的大文件对象。
- 执行 `git gc --prune=now --aggressive` 压缩仓库对象。
- 重写 README，修复乱码，并说明数据/模型/日志管理策略。
- 更新 `CHANGELOG.md` 到 v4.0，保留项目更新记录。

## 备份

重写历史前已生成仓库外部备份：

```text
D:\Aiparking\git-history-backups\Aiparking-For-YOLO-before-history-cleanup-2026-06-16.bundle
```

## 清理结果

- 清理前 `.git` 目录约：`4.37 GB`
- 清理后 `.git` 目录约：`0.08 MB`
- Git pack 大小：`51.52 KiB`
- 当前历史最大对象：约 `36.7 KiB`

## 注意事项

本次操作重写了 Git 历史，因此本地 commit hash 已变化。同步 GitHub 时需要使用：

```bash
git push --force-with-lease org master
```

如远端已有其他人基于旧历史继续提交，推送前需要先确认协作状态。
