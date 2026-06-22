"""
写入本轮 YOLOv8 训练的中文 Markdown 日志。

日志风格参考项目根目录的 CHANGELOG.md：版本号、数据集表格、训练结果表格、
与上一轮对比、预测结果和技术栈。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def last_metrics(results_csv: Path) -> dict:
    if not results_csv.exists():
        return {}
    rows = list(csv.DictReader(results_csv.read_text(encoding="utf-8").splitlines()))
    return rows[-1] if rows else {}


def class_metrics_from_log(run_log: Path) -> dict[str, dict[str, str]]:
    if not run_log.exists():
        return {}
    raw = run_log.read_bytes()
    if raw.count(b"\x00") > max(8, len(raw) // 20):
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    metrics: dict[str, dict[str, str]] = {}
    pattern = re.compile(
        r"^\s*(all|Parking|barrier)\s+\d+\s+\d+\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        label = match.group(1)
        values = match.groups()[1:]
        metrics[label] = {
            "box_p": values[0],
            "box_r": values[1],
            "box_map50": values[2],
            "box_map5095": values[3],
            "mask_p": values[4],
            "mask_r": values[5],
            "mask_map50": values[6],
            "mask_map5095": values[7],
        }
    return metrics


def prediction_summary(target: Path) -> dict:
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"):
        images.extend(target.glob(pattern))
    json_files = list(target.glob("*.json"))
    labels = Counter()
    scored = 0
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for shape in data.get("shapes", []):
            labels[shape.get("label", "unknown")] += 1
            if shape.get("score") is not None:
                scored += 1

    image_count = len(set(images))
    detected_json = len(json_files)
    detection_rate = (detected_json / image_count * 100) if image_count else 0.0
    return {
        "images": image_count,
        "json": detected_json,
        "labels": dict(labels),
        "scored_shapes": scored,
        "detection_rate": detection_rate,
    }


def as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_metric(value: object) -> str:
    number = as_float(value)
    return "—" if number is None else f"{number:.3f}"


def fmt_class_metric(class_metrics: dict[str, dict[str, str]], label: str, key: str) -> str:
    return fmt_metric((class_metrics.get(label) or {}).get(key))


def fmt_class_or_metric(
    class_metrics: dict[str, dict[str, str]],
    label: str,
    key: str,
    fallback: object,
) -> str:
    value = (class_metrics.get(label) or {}).get(key)
    return fmt_metric(value if value is not None else fallback)


def fmt_change(current: object, previous: float) -> str:
    number = as_float(current)
    if number is None:
        return "—"
    delta = (number - previous) / previous * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def class_total(build: dict, class_name: str) -> int:
    train = build.get("train_instances", {}) or {}
    val = build.get("val_instances", {}) or {}
    return as_int(train.get(class_name)) + as_int(val.get(class_name))


def write_log(
    output: Path,
    dataset_summary: Path,
    results_csv: Path,
    prediction_target: Path,
    model: Path,
    onnx: Path,
    run_log: Path,
    monitor_log: Path,
    current_version: str,
    current_title: str,
    previous_version: str,
    previous_all_mask_map50: float,
    previous_barrier_mask_map50: float,
) -> None:
    build = load_json(dataset_summary)
    metrics = last_metrics(results_csv)
    class_metrics = class_metrics_from_log(run_log)
    predictions = prediction_summary(prediction_target)
    target_name = prediction_target.name
    train_instances = build.get("train_instances", {}) or {}
    val_instances = build.get("val_instances", {}) or {}
    class_boosts = build.get("class_boosts", {}) or {}
    total_images = as_int(build.get("train_images")) + as_int(build.get("val_images"))
    epoch = metrics.get("epoch", "—")
    all_mask_map50 = (class_metrics.get("all") or {}).get("mask_map50") or metrics.get("metrics/mAP50(M)")

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 更新日志",
        "",
        "本项目使用语义化版本号，基于训练迭代记录。",
        "",
        "---",
        "",
        f"## {current_version} ({datetime.now().strftime('%Y-%m-%d')}) — {current_title}",
        "",
        f"### 📊 数据集：{total_images:,} 张图片",
        "",
        "| 类别 | 训练集 | 验证集 | 总计 |",
        "|------|--------|--------|------|",
        f"| Parking | {as_int(train_instances.get('Parking')):,} | {as_int(val_instances.get('Parking')):,} | {class_total(build, 'Parking'):,} |",
        f"| barrier | {as_int(train_instances.get('barrier')):,} | {as_int(val_instances.get('barrier')):,} | {class_total(build, 'barrier'):,} |",
        "",
        f"> 本轮训练素材位于仓库外部 `D:\\Aiparking\\image backcup`；`{target_name}/` 本轮只作为预测目标，不参与训练。",
        f"> 低置信度过滤规则保持为 `score < 0.4` 不进入训练；类别强化参数：`{class_boosts}`。",
        "",
        "### 🎯 训练结果",
        "",
        f"> 当前记录来自第 `{epoch}` 轮训练后的 best.pt 验证结果。",
        "",
        "| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |",
        "|------|-----------|-------------|------------|---------------|",
        f"| **all** | {fmt_class_or_metric(class_metrics, 'all', 'box_map50', metrics.get('metrics/mAP50(B)'))} | {fmt_class_or_metric(class_metrics, 'all', 'box_map5095', metrics.get('metrics/mAP50-95(B)'))} | {fmt_class_or_metric(class_metrics, 'all', 'mask_map50', metrics.get('metrics/mAP50(M)'))} | {fmt_class_or_metric(class_metrics, 'all', 'mask_map5095', metrics.get('metrics/mAP50-95(M)'))} |",
        f"| Parking | {fmt_class_metric(class_metrics, 'Parking', 'box_map50')} | {fmt_class_metric(class_metrics, 'Parking', 'box_map5095')} | {fmt_class_metric(class_metrics, 'Parking', 'mask_map50')} | {fmt_class_metric(class_metrics, 'Parking', 'mask_map5095')} |",
        f"| barrier | {fmt_class_metric(class_metrics, 'barrier', 'box_map50')} | {fmt_class_metric(class_metrics, 'barrier', 'box_map5095')} | {fmt_class_metric(class_metrics, 'barrier', 'mask_map50')} | {fmt_class_metric(class_metrics, 'barrier', 'mask_map5095')} |",
        "",
        f"### 📈 与 {previous_version} 对比",
        "",
        f"| 指标 | {previous_version} | {current_version} 当前 | 变化 |",
        "|------|------|-----------|------|",
        f"| All Mask mAP50 | {previous_all_mask_map50:.3f} | {fmt_metric(all_mask_map50)} | {fmt_change(all_mask_map50, previous_all_mask_map50)} |",
        f"| barrier Mask mAP50 | {previous_barrier_mask_map50:.3f} | {fmt_class_metric(class_metrics, 'barrier', 'mask_map50')} | {fmt_change((class_metrics.get('barrier') or {}).get('mask_map50'), previous_barrier_mask_map50)} |",
        "",
        "### 🆕 功能更新",
        "",
        "- 训练主模型从 YOLO11s-seg 切换为 YOLOv8s-seg，优先适配板端部署。",
        "- 图片素材和生成数据集迁移到 `D:\\Aiparking\\image backcup`，避免 GitHub 仓库体积过大。",
        "- 使用加权数据集：较新的审核素材拥有更高训练占比，降低早期低质量素材的影响。",
        "- 对含 `barrier` 的训练样本进行额外过采样，集中增强障碍物识别。",
        "- 保留 `Parking` 与 `barrier` 两类，并继续输出适合板端转换的 ONNX 模型。",
        "",
        f"### 📝 预测 {target_name}/ — {predictions['images']:,} 张，JSON {predictions['json']:,} 个，检出率 {predictions['detection_rate']:.1f}%",
        "",
        f"- 预测标签统计：`{predictions['labels']}`",
        f"- 带 score 的自动标注数量：`{predictions['scored_shapes']}`",
        "",
        "### 📦 输出文件",
        "",
        f"- PyTorch 模型：`{model}`，存在：`{model.exists()}`",
        f"- ONNX 模型：`{onnx}`，存在：`{onnx.exists()}`",
        f"- 训练输出日志：`{run_log}`",
        f"- 训练监控日志：`{monitor_log}`",
        "",
        "### 🧰 技术栈",
        "",
        "- **模型**：YOLOv8s-seg",
        "- **训练设备**：RTX 4060 Laptop GPU (8GB VRAM)",
        "- **训练尺寸**：640",
        "- **标注工具**：X-AnyLabeling",
        "- **数据格式**：X-AnyLabeling JSON → YOLO Seg TXT",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="写入本轮训练 Markdown 日志")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--prediction-target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--monitor-log", required=True)
    parser.add_argument("--current-version", required=True, help="当前版本号，如 v8.0")
    parser.add_argument("--current-title", required=True, help="当前迭代标题")
    parser.add_argument("--previous-version", required=True, help="上一轮版本号，如 v7.0")
    parser.add_argument("--previous-all-mask-map50", type=float, required=True, help="上一轮 All Mask mAP50")
    parser.add_argument("--previous-barrier-mask-map50", type=float, required=True, help="上一轮 barrier Mask mAP50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    write_log(
        output=output,
        dataset_summary=Path(args.dataset_summary),
        results_csv=Path(args.results_csv),
        prediction_target=Path(args.prediction_target),
        model=Path(args.model),
        onnx=Path(args.onnx),
        run_log=Path(args.run_log),
        monitor_log=Path(args.monitor_log),
        current_version=args.current_version,
        current_title=args.current_title,
        previous_version=args.previous_version,
        previous_all_mask_map50=args.previous_all_mask_map50,
        previous_barrier_mask_map50=args.previous_barrier_mask_map50,
    )
    print(f"已写入日志：{output}")


if __name__ == "__main__":
    main()
