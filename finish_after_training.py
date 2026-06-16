"""
Wait for the active YOLOv8 training process, then run prediction and write the final log.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def process_exists(pid: int) -> bool:
    try:
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    except Exception:
        command = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
        ]
        return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def run_step(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8", errors="replace") as out:
        with stderr_path.open("a", encoding="utf-8", errors="replace") as err:
            out.write(f"\n===== {' '.join(command)} =====\n")
            out.flush()
            proc = subprocess.run(command, stdout=out, stderr=err, text=True)
            return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finish AiParking YOLOv8 iteration after training")
    parser.add_argument("--state", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--conf", default="0.4")
    parser.add_argument("--poll", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(args.project)
    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    train_pid = int(state["train_pid"])
    done_file = Path(state["done_file"])
    log_dir = project / "log"
    stamp = state["stamp"]
    finish_log = log_dir / f"{stamp}_yolov8_finish_stdout.log"
    finish_err = log_dir / f"{stamp}_yolov8_finish_stderr.log"
    model_path = project / "runs" / "parking_yolov8_seg" / "weights" / "best.pt"
    onnx_path = project / "runs" / "parking_yolov8_seg" / "weights" / "best.onnx"
    results_csv = project / "runs" / "parking_yolov8_seg" / "results.csv"
    final_log = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}_yolov8_iteration4.md"

    with finish_log.open("a", encoding="utf-8") as out:
        out.write(f"Waiting for training PID {train_pid} at {datetime.now()}\n")

    while process_exists(train_pid):
        time.sleep(args.poll)

    done_file.write_text(f"training finished at {datetime.now()}\n", encoding="utf-8")

    if model_path.exists():
        predict_code = run_step(
            [
                args.python,
                str(project / "predict.py"),
                "--model",
                str(model_path),
                "--target",
                args.target,
                "--conf",
                args.conf,
            ],
            finish_log,
            finish_err,
        )
    else:
        predict_code = 1
        with finish_err.open("a", encoding="utf-8") as err:
            err.write(f"best model not found: {model_path}\n")

    run_step(
        [
            args.python,
            str(project / "write_iteration_log.py"),
            "--output",
            str(final_log),
            "--dataset-summary",
            r"D:\Aiparking\image backcup\dataset_yolov8_weighted\build_summary.json",
            "--results-csv",
            str(results_csv),
            "--prediction-target",
            args.target,
            "--model",
            str(model_path),
            "--onnx",
            str(onnx_path),
            "--run-log",
            state["train_stdout"],
            "--monitor-log",
            state["monitor_log"],
        ],
        finish_log,
        finish_err,
    )

    if predict_code != 0:
        sys.exit(predict_code)


if __name__ == "__main__":
    main()
