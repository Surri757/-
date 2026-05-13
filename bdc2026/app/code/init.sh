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

echo "安装依赖包（精确版本，来自 requirements.txt）..."
pip install -q numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.0
pip install -q lightgbm==4.3.0 catboost==1.2.5 xgboost==2.0.3
pip install -q torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q scipy==1.13.1 statsmodels==0.14.2 matplotlib==3.8.5 "shap>=0.44.0" tqdm

echo "验证关键依赖..."
python -c "import numpy; import pandas; import sklearn; import lightgbm; import catboost; import xgboost; import torch; import scipy; import matplotlib; import shap; import tqdm; print('所有依赖包验证通过')"

echo "=========================================="
echo "环境初始化完成!"
echo "=========================================="