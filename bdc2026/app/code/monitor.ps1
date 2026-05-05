[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$errorLog = "D:\stockclone\bdc2026\app\code\train_error.log"
$outputLog = "D:\stockclone\bdc2026\app\code\train_output.log"
$lastSize = 0
$checkCount = 0
$maxWaitSeconds = 2400  # 最多等待40分钟

function Check-Keywords {
    param([string]$line)
    $keywords = @("Error", "Exception", "Traceback", "error", "exception", "traceback")
    foreach ($kw in $keywords) {
        if ($line -match $kw) {
            return $true
        }
    }
    return $false
}

Write-Host "===== 错误日志监控器启动 ====="
Write-Host "监控文件: $errorLog"
Write-Host "训练日志: $outputLog"
Write-Host "开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "================================"

while ($checkCount * 60 -lt $maxWaitSeconds) {
    $checkCount++
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    
    # 检查训练是否完成
    if (Test-Path $outputLog) {
        $outputContent = Get-Content $outputLog -Raw -Encoding UTF8
        if ($outputContent -match "训练完成") {
            Write-Host "`n[$timestamp] 训练日志检测到'训练完成'，监控器退出。"
            exit 0
        }
    }
    
    # 检查错误日志
    if (Test-Path $errorLog) {
        $currentSize = (Get-Item $errorLog).Length
        if ($currentSize -gt $lastSize) {
            if ($lastSize -eq 0) {
                Write-Host "`n[$timestamp] 错误日志有内容 ($currentSize 字节):"
                $content = Get-Content $errorLog -Encoding UTF8
                foreach ($line in $content) {
                    if (Check-Keywords $line) {
                        Write-Host "  [警告] $line" -ForegroundColor Red
                    } else {
                        Write-Host "  $line"
                    }
                }
            } else {
                Write-Host "`n[$timestamp] 错误日志新增内容:"
                $fs = New-Object System.IO.FileStream($errorLog, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
                $fs.Seek($lastSize, [System.IO.SeekOrigin]::Begin) | Out-Null
                $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
                while ($null -ne ($line = $sr.ReadLine())) {
                    if (Check-Keywords $line) {
                        Write-Host "  [警告] $line" -ForegroundColor Red
                    } else {
                        Write-Host "  $line"
                    }
                }
                $sr.Close()
                $fs.Close()
            }
            $lastSize = $currentSize
        } else {
            Write-Host "[$timestamp] 第 $checkCount 次检查 - 无新增错误内容 (当前大小: $currentSize 字节)"
        }
    } else {
        Write-Host "[$timestamp] 第 $checkCount 次检查 - 错误日志不存在"
    }
    
    Start-Sleep -Seconds 60
}

Write-Host "`n监控器达到最大等待时间，自动退出。"
