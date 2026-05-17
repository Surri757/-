#!/bin/bash
set -e

find_python() {
    for p in /mnt/d/Miniconda3/python.exe /mnt/d/Miniconda3/python /d/Miniconda3/python; do
        [ -f "$p" ] && { echo "$p"; return; }
    done
    echo "python"
}
PYTHON=$(find_python)

echo "========================================"
echo "2026清华大学大数据挑战赛 - 训练主程序"
echo "========================================"

cd "$(dirname "$0")"
start_time=$(date +%s)

export PYTHONHASHSEED=42
echo "使用 Python: $PYTHON"
echo "开始训练..."
$PYTHON -u src/train.py

end_time=$(date +%s)
duration=$(( (end_time - start_time) / 60 ))

echo "========================================"
echo "训练完成! 总耗时: $duration 分钟"
echo "========================================"
