param(
    [Parameter(Mandatory = $true)]
    [int]$TargetPid,
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [Parameter(Mandatory = $true)]
    [string]$MonitorLog,
    [int]$IntervalSeconds = 3600
)

$run = (Resolve-Path -LiteralPath $RunDirectory).Path
$log = [System.IO.Path]::GetFullPath($MonitorLog)
New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($log)) | Out-Null

function Write-MonitorLine([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
    Write-Output $line
}

Write-MonitorLine "monitor started; training pid=$TargetPid; interval=${IntervalSeconds}s"

while ($true) {
    $process = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    $csvPath = Join-Path $run "results.csv"
    $epochText = "results.csv not available"
    if (Test-Path -LiteralPath $csvPath) {
        try {
            $rows = @(Import-Csv -LiteralPath $csvPath)
            if ($rows.Count -gt 0) {
                $last = $rows[-1]
                $epochText = "epoch=$($last.epoch); box_mAP50=$($last.'metrics/mAP50(B)'); mask_mAP50=$($last.'metrics/mAP50(M)'); box_mAP50_95=$($last.'metrics/mAP50-95(B)'); mask_mAP50_95=$($last.'metrics/mAP50-95(M)')"
            }
        } catch {
            $epochText = "results.csv read failed: $($_.Exception.Message)"
        }
    }

    $gpuText = "nvidia-smi unavailable"
    try {
        $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpu) {
            $gpuText = "gpu=$gpu"
        }
    } catch {
        $gpuText = "nvidia-smi read failed"
    }

    if ($process) {
        Write-MonitorLine "$epochText; $gpuText; training is running"
        Start-Sleep -Seconds $IntervalSeconds
    } else {
        Write-MonitorLine "$epochText; $gpuText; training process ended"
        break
    }
}
