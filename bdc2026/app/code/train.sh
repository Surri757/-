#!/bin/bash

set -e

echo "=========================================="
echo "2026清华大学大数据挑战赛 - 训练主程序"
echo "=========================================="

cd "$(dirname "$0")"

start_time=$(date +%s)

echo "开始训练..."
python -u src/train.py

end_time=$(date +%s)
duration=$(( (end_time - start_time) / 60 ))

echo "=========================================="
echo "训练完成! 总耗时: $duration 分钟"
echo "=========================================="