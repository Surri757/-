#!/bin/bash

set -e

echo "=========================================="
echo "2026清华大学大数据挑战赛 - 环境初始化"
echo "=========================================="

echo "创建必要的目录..."
mkdir -p code/src/model
mkdir -p code/data
mkdir -p output
mkdir -p temp

echo "检查Python版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

echo "安装依赖包..."
pip install -q numpy pandas scikit-learn
pip install -q lightgbm catboost xgboost
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q scipy statsmodels matplotlib shap tqdm

echo "验证关键依赖..."
python -c "import numpy; import pandas; import sklearn; import lightgbm; import catboost; import xgboost; import torch; import scipy; import matplotlib; import shap; import tqdm; print('所有依赖包验证通过')"

echo "=========================================="
echo "环境初始化完成!"
echo "=========================================="