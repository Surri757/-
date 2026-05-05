$logFile = "D:\stockclone\bdc2026\app\code\train_output.log"

# 等待日志文件存在
while (-not (Test-Path $logFile)) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 日志文件不存在，等待10秒后重试..."
    Start-Sleep -Seconds 10
}

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 开始监控训练日志: $logFile"

while ($true) {
    try {
        $lines = Get-Content $logFile -Tail 30 -Encoding utf8 -ErrorAction Stop
        Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ===== 最新30行日志 ====="
        $lines | ForEach-Object { Write-Host $_ }
        
        # 检查是否完成
        $content = Get-Content $logFile -Raw -Encoding utf8 -ErrorAction Stop
        if ($content -match "训练完成|训练结束") {
            Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ===== 检测到训练完成/结束，打印最后50行 ====="
            Get-Content $logFile -Tail 50 -Encoding utf8 | ForEach-Object { Write-Host $_ }
            Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 监控结束。"
            exit 0
        }
    }
    catch {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 读取失败: $_"
    }
    
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 等待30秒后继续监控..."
    Start-Sleep -Seconds 30
}
