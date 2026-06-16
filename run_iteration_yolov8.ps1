$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe"
$LogDir = Join-Path $Project "log"
$RunName = "parking_yolov8_seg"
$RunDir = Join-Path $Project "runs\$RunName"
$DatasetDir = "D:\Aiparking\image backcup\dataset_yolov8_weighted"
$OpenParen = [char]0xFF08
$CloseParen = [char]0xFF09
$TargetDir = "D:\Aiparking\image backcup\images${OpenParen}5${CloseParen}"
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$RunLog = Join-Path $LogDir "${Stamp}_yolov8_iteration_run.log"
$MonitorLog = Join-Path $LogDir "${Stamp}_yolov8_iteration_monitor.md"
$DoneFile = Join-Path $LogDir "${Stamp}_yolov8_iteration.done"
$FinalLog = Join-Path $LogDir "$(Get-Date -Format "yyyy-MM-dd")_yolov8_iteration4.md"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (Test-Path -LiteralPath $DoneFile) {
    Remove-Item -LiteralPath $DoneFile -Force
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    "`n===== $Name =====" | Tee-Object -FilePath $RunLog -Append
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $RunLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

"AiParking YOLOv8 iteration started at $(Get-Date)" | Tee-Object -FilePath $RunLog -Append

Run-Step "Build weighted YOLOv8 dataset" @(
    (Join-Path $Project "build_yolov8_dataset.py"),
    "--data-root", "D:\Aiparking\image backcup",
    "--output", $DatasetDir,
    "--min-conf", "0.4",
    "--val-ratio", "0.15",
    "--seed", "20260616"
)

$ResultsCsv = Join-Path $RunDir "results.csv"
$monitorArgs = @(
    (Join-Path $Project "monitor_training.py"),
    "--run-log", $RunLog,
    "--monitor-log", $MonitorLog,
    "--done-file", $DoneFile,
    "--results-csv", $ResultsCsv,
    "--interval", "1800"
)
Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $monitorArgs

try {
    Run-Step "Train YOLOv8 segmentation model" @(
        (Join-Path $Project "train.py"),
        "--data", (Join-Path $DatasetDir "parking_yolov8.yaml"),
        "--model", "yolov8s-seg.pt",
        "--epochs", "180",
        "--imgsz", "640",
        "--batch", "8",
        "--patience", "35",
        "--project", (Join-Path $Project "runs"),
        "--name", $RunName,
        "--save-period", "20"
    )

    Run-Step "Predict images5" @(
        (Join-Path $Project "predict.py"),
        "--model", (Join-Path $RunDir "weights\best.pt"),
        "--target", $TargetDir,
        "--conf", "0.4"
    )
}
finally {
    New-Item -ItemType File -Force -Path $DoneFile | Out-Null
}

Run-Step "Write Markdown iteration log" @(
    (Join-Path $Project "write_iteration_log.py"),
    "--output", $FinalLog,
    "--dataset-summary", (Join-Path $DatasetDir "build_summary.json"),
    "--results-csv", $ResultsCsv,
    "--prediction-target", $TargetDir,
    "--model", (Join-Path $RunDir "weights\best.pt"),
    "--onnx", (Join-Path $RunDir "weights\best.onnx"),
    "--run-log", $RunLog,
    "--monitor-log", $MonitorLog
)

"AiParking YOLOv8 iteration finished at $(Get-Date)" | Tee-Object -FilePath $RunLog -Append
