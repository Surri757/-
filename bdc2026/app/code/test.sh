#!/bin/bash

set -e

echo "=========================================="
echo "2026清华大学大数据挑战赛 - 预测主程序"
echo "=========================================="

cd "$(dirname "$0")"

start_time=$(date +%s)

export PYTHONHASHSEED=42
echo "开始预测..."
python -u src/test.py

end_time=$(date +%s)
duration=$(( (end_time - start_time) / 60 ))

echo "=========================================="
echo "预测完成! 总耗时: $duration 分钟"
echo "=========================================="