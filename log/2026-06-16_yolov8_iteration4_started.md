# YOLOv8 Iteration 4 Started

## Status

- Started at: 2026-06-16 00:43:56 Asia/Shanghai
- Training process PID: `14188`
- Monitor process PID: `18132`
- Finisher process PID: `21412`
- Current model family: `YOLOv8s-seg`
- Training target: `D:\Aiparking\image backcup\dataset_yolov8_weighted\parking_yolov8.yaml`
- Prediction target after training: `D:\Aiparking\image backcup\images（5）`

## Backup

- Training-before-change backup was refreshed at `D:\Aiparking\Aiparking For YOLObackup`.
- The external source material remains under `D:\Aiparking\image backcup`.

## Code Changes

- Added `build_yolov8_dataset.py` to build an external weighted YOLOv8 segmentation dataset.
- Rewrote `train.py` to train `YOLOv8s-seg` and export ONNX.
- Rewrote `predict.py` to default to the YOLOv8 output model and annotate `images（5）`.
- Added `monitor_training.py` for periodic training snapshots.
- Added `write_iteration_log.py` for final Markdown summary after training and prediction.
- Updated `.gitignore` to avoid committing local image/dataset folders and intermediate checkpoints.
- Updated `run_train.bat` and `run_predict.bat`.

## Dataset Build

- Sources used:
  - `images`, weight 1
  - `images（1）`, weight 1
  - `images（2）`, weight 2
  - `images（3）`, weight 3
  - `images（4）`, weight 4
- `images（5）` is excluded from training because it has no JSON annotations yet.
- Confidence filter: labels with `score < 0.4` are excluded.
- Unknown labels such as `d` are ignored during dataset generation; original JSON files are not modified.
- Generated train images after weighting: 12628
- Generated validation images: 951
- Train instances: `Parking=14620`, `barrier=878`
- Validation instances: `Parking=1144`, `barrier=78`

## Verification

- Python compile check passed for the new/modified scripts.
- Unit tests passed: 3/3.
- CUDA check passed: PyTorch sees `NVIDIA GeForce RTX 4060 Laptop GPU`.
- First live training check showed GPU utilization around 69% and VRAM around 2.4GB.

## Active Logs

- Training stdout: `D:\Aiparking\Aiparking For YOLO\log\2026-06-16_004356_yolov8_train_stdout.log`
- Training stderr: `D:\Aiparking\Aiparking For YOLO\log\2026-06-16_004356_yolov8_train_stderr.log`
- Monitor log: `D:\Aiparking\Aiparking For YOLO\log\2026-06-16_004356_yolov8_monitor.md`
- State file: `D:\Aiparking\Aiparking For YOLO\log\current_yolov8_training_state.json`
- Finisher stdout: `D:\Aiparking\Aiparking For YOLO\log\2026-06-16_004356_yolov8_finish_stdout.log`
- Finisher stderr: `D:\Aiparking\Aiparking For YOLO\log\2026-06-16_004356_yolov8_finish_stderr.log`
