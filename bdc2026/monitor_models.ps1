$modelDir = "D:\stockclone\bdc2026\app\code\src\model"
$logFile = "D:\stockclone\bdc2026\app\code\train_output.log"
$expectedFiles = @(
    "lightgbm_model.pkl",
    "catboost_model.pkl",
    "xgboost_model.pkl",
    "patchtst_model.pth",
    "dlinear_model.pth",
    "tft_model.pth",
    "spectralm_model.pkl",
    "meta_model.pkl",
    "scaler.pkl"
)

function Get-Progress {
    $generated = @()
    if (Test-Path $modelDir) {
        $files = Get-ChildItem $modelDir -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
        $generated = $expectedFiles | Where-Object { $files -contains $_ }
    }
    $count = $generated.Count
    $percent = [math]::Round(($count / $expectedFiles.Count) * 100, 1)
    return @{ Count = $count; Percent = $percent; Generated = $generated }
}

function Check-TrainingComplete {
    if (Test-Path $logFile) {
        $lastLines = Get-Content $logFile -Tail 10 -Encoding utf8 -ErrorAction SilentlyContinue
        if ($lastLines -match "训练完成") {
            return $true
        }
    }
    return $false
}

$startTime = Get-Date
Write-Host "========================================"
Write-Host "  模型生成进度监控器已启动"
Write-Host "  开始时间: $startTime"
Write-Host "  目标目录: $modelDir"
Write-Host "  期望文件数: $($expectedFiles.Count)"
Write-Host "========================================"

while ($true) {
    $progress = Get-Progress
    $trainingComplete = Check-TrainingComplete
    $elapsed = (Get-Date) - $startTime

    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 运行时长: $($elapsed.ToString('hh\:mm\:ss'))"
    Write-Host "  已生成文件数: $($progress.Count) / $($expectedFiles.Count)"
    Write-Host "  进度: $($progress.Percent)%"

    if ($progress.Generated.Count -gt 0) {
        Write-Host "  已生成文件:"
        foreach ($f in $progress.Generated) {
            Write-Host "    + $f"
        }
    }

    $missing = $expectedFiles | Where-Object { $progress.Generated -notcontains $_ }
    if ($missing.Count -gt 0) {
        Write-Host "  待生成文件:"
        foreach ($f in $missing) {
            Write-Host "    - $f"
        }
    }

    if ($progress.Count -eq $expectedFiles.Count) {
        Write-Host ""
        Write-Host "========================================"
        Write-Host "  所有模型文件已生成！"
        Write-Host "  总耗时: $($elapsed.ToString('hh\:mm\:ss'))"
        Write-Host "========================================"
        break
    }

    if ($trainingComplete) {
        Write-Host ""
        Write-Host "========================================"
        Write-Host "  检测到训练日志包含'训练完成'"
        Write-Host "  进行最终检查..."
        Write-Host "  最终已生成: $($progress.Count) / $($expectedFiles.Count)"
        Write-Host "========================================"
        break
    }

    Write-Host "  等待60秒后再次检查..."
    Start-Sleep -Seconds 60
}
