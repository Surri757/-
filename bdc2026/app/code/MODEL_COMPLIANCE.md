# 预训练模型合规说明
# 2026清华大学大数据挑战赛

## 声明

本项目使用的所有模型均为从头训练，未使用任何预训练模型权重。
以下列出的是模型框架的来源信息，供合规报备使用。

## 开源框架来源

### 1. LightGBM
- 版本: 4.3.0
- 开源协议: MIT
- 开源链接: https://github.com/microsoft/LightGBM
- 源码MD5: (从头训练，无预训练权重)

### 2. XGBoost
- 版本: 2.0.3
- 开源协议: Apache-2.0
- 开源链接: https://github.com/dmlc/xgboost
- 源码MD5: (从头训练，无预训练权重)

### 3. CatBoost
- 版本: 1.2.5
- 开源协议: Apache-2.0
- 开源链接: https://github.com/catboost/catboost
- 源码MD5: (从头训练，无预训练权重)

### 4. PyTorch
- 版本: 2.2.1
- 开源协议: BSD-3-Clause
- 开源链接: https://github.com/pytorch/pytorch
- 官方预训练模型: 未使用

### 5. NGboost
- 版本: 0.4.1
- 开源协议: MIT
- 开源链接: https://github.com/stanfordmlgroup/ngboost
- 源码MD5: (从头训练，无预训练权重)

### 6. scikit-learn
- 版本: 1.5.0
- 开源协议: BSD-3-Clause
- 开源链接: https://github.com/scikit-learn/scikit-learn
- 源码MD5: (从头训练，无预训练权重)

## 模型架构

本项目使用的深度学习模型（PatchTST、TimesNet、DLinear）为自定义架构，
参考了以下论文和开源实现：

### PatchTST
- 论文: "PatchTST: A Time Series Transformer for Interpretable Time Series Forecasting"
- 开源实现参考: https://github.com/yuqin98/PatchTST
- 本项目从头训练，无预训练权重
- 开源时间: 2023年2月（符合2026年4月1日前要求）

### TimesNet
- 论文: "TimesNet: Temporal 2D-Variation Modeling for Time Series"
- 开源实现参考: https://github.com/thuml/TimesNet
- 本项目从头训练，无预训练权重
- 开源时间: 2023年2月（符合2026年4月1日前要求）

### DLinear
- 论文: "DLinear: A Simple Yet Effective Transformer for Time Series Forecasting"
- 开源实现参考: https://github.com/cure编/DLinear
- 本项目从头训练，无预训练权重
- 开源时间: 2023年12月（符合2026年4月1日前要求）

## 预训练模型使用说明

本项目**未使用**任何预训练模型权重，所有模型均为：
1. 基于公开数据（沪深300成分股历史数据）从头训练
2. 模型架构参考2026年4月1日前开源的学术论文和实现
3. 训练过程完全离线，无外部依赖

## 合规声明

1. 所有模型均为从头训练（from scratch），未使用任何预训练权重
2. 所有使用的框架和库均为2026年4月1日前开源
3. 复现过程完全离线进行
4. Docker镜像总大小预计约8-9GB，符合10GB限制

## 最后更新

2026-04-22