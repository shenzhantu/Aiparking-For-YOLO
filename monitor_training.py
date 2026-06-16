"""
Write periodic Chinese training monitor snapshots while a long YOLO run is active.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path


def read_last_metrics(results_csv: Path) -> str:
    if not results_csv.exists():
        return "results.csv 尚未生成"
    try:
        rows = list(csv.DictReader(results_csv.read_text(encoding="utf-8").splitlines()))
    except Exception as exc:
        return f"读取 results.csv 失败: {exc}"
    if not rows:
        return "results.csv 已存在，但还没有指标行"
    row = rows[-1]
    interesting = [
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
    ]
    parts = []
    for key in interesting:
        if key in row:
            parts.append(f"{key}={row[key].strip()}")
    return ", ".join(parts) if parts else str(row)


def read_gpu_status() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"nvidia-smi 不可用: {exc}"
    return output.strip()


def tail_text(path: Path, lines: int = 8) -> str:
    if not path.exists():
        return "训练日志尚未生成"
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"读取训练日志失败: {exc}"
    return "\n".join(content[-lines:])


def append_snapshot(monitor_log: Path, run_log: Path, results_csv: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    monitor_log.parent.mkdir(parents=True, exist_ok=True)
    with monitor_log.open("a", encoding="utf-8") as file:
        file.write(f"\n## {timestamp}\n\n")
        file.write(f"- 显卡状态：`{read_gpu_status()}`\n")
        file.write(f"- 最新指标：`{read_last_metrics(results_csv)}`\n\n")
        file.write("最近训练日志：\n\n")
        file.write("```text\n")
        file.write(tail_text(run_log))
        file.write("\n```\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor long YOLO training")
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--monitor-log", required=True)
    parser.add_argument("--done-file", required=True)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--interval", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_log = Path(args.run_log)
    monitor_log = Path(args.monitor_log)
    done_file = Path(args.done_file)
    results_csv = Path(args.results_csv)

    append_snapshot(monitor_log, run_log, results_csv)
    while not done_file.exists():
        time.sleep(args.interval)
        append_snapshot(monitor_log, run_log, results_csv)
    append_snapshot(monitor_log, run_log, results_csv)


if __name__ == "__main__":
    main()
