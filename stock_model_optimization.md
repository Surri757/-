# 股票量化模型优化资料汇总

## 概述

本文档汇总了用于提升股票量化模型性能（特别是排序准确率 final_score）相关的研究资料和技术文档。涵盖以下四大核心方向：

1. **Transformer 金融时间序列预测** - 6篇核心论文
2. **排序学习（Learning to Rank）** - 4篇核心论文
3. **PyTorch 训练加速优化** - 混合精度、梯度累积技术
4. **超参数调优策略** - XGBoost/LightGBM 调参方法

---

## 第一部分：Transformer 金融时间序列预测

### 核心论文

| 论文 | arXiv ID | 关键内容 |
|------|----------|----------|
| Financial Time Series Forecasting using CNN and Transformer | 2304.04912 | CNN+Transformer 组合捕捉局部和全局特征 |
| Autoformer: Long Range Forecasting with Auto-Correlation | 2106.13008 | Auto-Correlation 机制处理长序列依赖 |
| Temporal Fusion Transformer (TFT) | 1912.09363 | 可解释的多时间范围预测框架 |
| Informer: Beyond Efficient Transformer | 2012.07436 | ProbSparse 自注意力，O(L log L) 复杂度 |
| Cross-Modal Temporal Fusion | 2504.13522 | 跨模态融合（价格+新闻+市场指标） |
| Long-Range Transformers | 2109.12218 | 动态时空预测的长程依赖建模 |

### 模型架构选择建议

```
短期预测 → TFT (可解释性强)
长期预测 → Autoformer / Informer (高效处理长序列)
多模态融合 → Cross-Modal Temporal Fusion
```

### 优化方向

1. **CNN+Transformer** 组合捕捉局部+全局特征
2. **Auto-Correlation** 替代标准注意力机制
3. **ProbSparse** 减少计算复杂度
4. **门控机制** 提升模型可解释性

---

## 第二部分：排序学习（Learning to Rank）

### 核心论文

| 论文 | arXiv ID | 关键内容 |
|------|----------|----------|
| SortNet: Neural-Based Sorting | 2311.01864 | 神经网络排序算法 |
| NDCG Optimization for Deep Learning | 2202.12183 | 大规模 NDCG 随机优化 |
| ILMART: Interpretable LambdaMART | 2206.00473 | 可解释 LambdaMART 排序 |
| SpatialRank: Urban Event Ranking | 2310.00270 | 时空依赖的 NDCG 优化 |

### LambdaMART 算法原理

LambdaMART 结合了：
- **MART（梯度提升树）**：强大的非线性建模能力
- **LambdaRank**：基于度量感知的优化框架

核心思想：不直接优化排序指标，而是通过计算损失函数相对于模型输出的梯度来间接优化 NDCG。

### NDCG 优化策略

1. **直接 NDCG 优化**：直接在损失函数中使用 NDCG 梯度
2. **Pairwise 优化**：优化文档对的相对顺序
3. **Listwise 优化**：同时考虑整个列表的排序效果

### XGBoost rank:ndcg 使用

```python
import xgboost as xgb

params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@10',
    'eta': 0.05,
    'max_depth': 5
}

dtrain = xgb.DMatrix(features, label=returns, qid=query_groups)
model = xgb.train(params, dtrain, num_boost_round=500)
```

---

## 第三部分：PyTorch 训练加速优化

### 混合精度训练（AMP）

**核心配置**：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Channels Last 内存格式

配合 AMP 使用可获得 **22%+ 性能提升**：

```python
model = model.to(memory_format=torch.channels_last)
data = data.to(memory_format=torch.channels_last)
```

### 梯度累积

当显存受限时，使用累积步数模拟更大 batch size：

```python
accumulation_steps = 4

for step, (data, target) in enumerate(dataloader):
    with autocast():
        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (step + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### DataLoader 优化

```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,          # 根据 CPU 核心数调整
    pin_memory=True,       # 加速数据传输到 GPU
    persistent_workers=True
)
```

### 优化效果汇总

| 优化项 | 预期收益 |
|--------|----------|
| AMP (float16) | 30-50% 加速 |
| GradScaler | 防止梯度下溢 |
| Channels Last | 22%+ 加速 |
| 梯度累积 | 有效增大 batch size |

---

## 第四部分：超参数调优策略

### 调优方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| Grid Search | 保证全局最优 | 维度灾难 | 参数空间小 |
| Random Search | 高效（高维空间） | 可能错过最优 | 参数空间大 |
| Bayesian Optimization | 高效利用评估次数 | 需要初始探索 | 评估成本高 |

### XGBoost 核心参数调优顺序

1. **固定学习率**，调优 `n_estimators`（early stopping）
2. 调优 `max_depth` 和 `min_child_weight`
3. 调优 `subsample` 和 `colsample_bytree`
4. 调优正则化参数（`gamma`, `reg_alpha`, `reg_lambda`）
5. **降低学习率**，增加树数量

### Optuna 自动调优示例

```python
import optuna
import xgboost as xgb

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    return score.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 金融量化模型特有考虑

1. **时间序列交叉验证**：使用 Walk-Forward Validation
2. **过拟合风险**：金融数据噪声大，需要更强的正则化
3. **评估指标**：Sharpe Ratio、Max Drawdown、IC

---

## 综合优化建议

### 针对 StockRanker / 排序模型

1. **损失函数**：使用 `rank:ndcg` 目标函数，直接优化 NDCG
2. **模型架构**：考虑 CNN+Transformer 组合
3. **训练加速**：AMP + Channels Last + 梯度累积
4. **超参数调优**：使用 Optuna 自动化调优

### 推荐配置

```python
# 模型配置
model = model.to(memory_format=torch.channels_last)

# 优化器配置
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# AMP + 梯度累积
scaler = GradScaler()
accumulation_steps = 4

# XGBoost 排序学习
params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@10',
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

---

## 参考文献

详见 `reference.bib` 文件，包含以下分类：

- **Transformer 金融预测**：6篇 arXiv 论文
- **排序学习**：4篇 arXiv 论文
- **PyTorch 训练优化**：6条技术文档
- **排序损失函数优化**：4条技术文档
- **超参数调优**：6条技术文档

---

*文档生成时间：2026-05-09*
*工作空间：E:\stock*
