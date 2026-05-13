# 排序学习损失函数优化资料汇总

## 概述

本文档汇总了排序学习（Learning to Rank）中损失函数优化的相关资料，重点关注 LambdaMART、NDCG 优化和神经网络排序等关键技术的最新进展。

---

## 1. LambdaMART 算法详解

### 1.1 算法原理

LambdaMART 是排序学习领域的核心算法，结合了：
- **MART（梯度提升树）**：强大的非线性建模能力
- **LambdaRank**：基于度量感知的优化框架

**核心思想**：LambdaMART 不直接优化排序指标，而是通过计算损失函数相对于模型输出的梯度来间接优化 NDCG 等排序指标。

### 1.2 训练流程

```
1. 应用模型预测文档得分
2. 按得分对文档排序
3. 计算 NDCG 变化量 (ΔNDCG)
4. 根据 ΔNDCG 计算 Lambda 梯度
5. 使用梯度更新模型权重
6. 迭代直到收敛
```

### 1.3 关键特性

- **灵活性**：评估指标易于更换，只需调整梯度计算方式
- **高效性**：直接针对排序指标优化，比传统 pointwise 方法效果更好
- **适用性**：同时支持二值相关性和多级相关性评分

---

## 2. NDCG（Normalized Discounted Cumulative Gain）优化

### 2.1 NDCG 公式

```
DCG@k = Σ(i=1 to k) rel_i / log2(i+1)
NDCG@k = DCG@k / IDCG@k
```

其中：
- `rel_i`：第 i 位文档的相关性评分
- `IDCG`：理想 DCG（最佳排序下的 DCG 值）

### 2.2 优化策略

1. **直接 NDCG 优化**：直接在损失函数中使用 NDCG 梯度
2. **Pairwise 优化**：优化文档对的相对顺序，间接提升 NDCG
3. **Listwise 优化**：同时考虑整个列表的排序效果

---

## 3. XGBoost 排序学习实现

### 3.1 目标函数

```python
# XGBoost 排序学习目标
objective='rank:ndcg'
```

### 3.2 主要参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| objective | rank:ndcg / rank:pairwise / rank:map | rank:ndcg |
| eval_metric | ndcg@n / map / ndcg | ndcg@10 |
| lambdarank_num_pair_per_sample | 每样本的 pair 数量 | 10 |

### 3.3 使用示例

```python
import xgboost as xgb

# 创建 DMatrix，包含 query 维度
dtrain = xgb.DMatrix(train_data, label=train_label, qid=train_query_id)

# 设置参数
params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@10',
    'eta': 0.1,
    'max_depth': 6
}

# 训练
model = xgb.train(params, dtrain, num_boost_round=100)
```

---

## 4. 神经网络排序优化

### 4.1 排序损失函数类型

1. **Pointwise Loss**
   - MSE Loss：预测绝对分数
   - Cross-Entropy：多分类相关性评分

2. **Pairwise Loss**
   - Hinge Loss (RankSVM)
   - Log Loss (RankNet)

3. **Listwise Loss**
   - LambdaRank Loss
   - ListNet Loss
   - NDCG Loss

### 4.2 ListNet Loss

```python
def listnet_loss(y_true, y_pred):
    """
    ListNet Loss - 概率分布交叉熵
    """
    # 计算真实概率分布
    P_y = softmax(y_true)
    # 计算预测概率分布
    P_z = softmax(y_pred)
    # 交叉熵损失
    loss = -torch.sum(P_y * torch.log(P_z + 1e-10))
    return loss
```

### 4.3 NDCG Loss（直接优化）

```python
def ndcg_loss(y_true, y_pred, k=10):
    """
    直接优化 NDCG 的损失函数
    """
    # 计算 discounts
    discounts = 1 / torch.log2(torch.arange(2, k+2, device=y_pred.device).float())
    
    # 计算 DCG
    gains = y_pred  # 或使用 y_true
    dcg = torch.sum(gains[:k] * discounts)
    
    # 计算 IDCG
    ideal_gains, _ = torch.sort(y_true, descending=True)
    idcg = torch.sum(ideal_gains[:k] * discounts)
    
    # NDCG 损失（越小越好，所以取负值）
    ndcg = dcg / (idcg + 1e-10)
    return 1 - ndcg
```

---

## 5. 梯度下降优化技巧

### 5.1 学习率调度

| 方法 | 说明 |
|------|------|
| Step Decay | 固定周期降低学习率 |
| Cosine Annealing | 余弦退火调度 |
| Warmup | 预热后衰减 |

### 5.2 优化器选择

- **Adam**：自适应学习率，适合大多数场景
- **AdamW**：带权重衰减的 Adam，正则化更好
- **LAMB**：大batch训练效果好

### 5.3 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 6. 股票量化模型应用建议

### 6.1 适合的场景

- **StockRanker 风格排序**：对股票进行收益率排序预测
- **多因子选股**：因子组合的排序优化
- **滚动回测**：动态调整股票排序

### 6.2 实施建议

1. **使用 XGBoost/LightGBM**：内置 rank:ndcg 目标函数，开箱即用
2. **神经网络方案**：结合 Transformer 或 Deep & Wide 模型，使用 ListNet 或 LambdaRank Loss
3. **评估指标**：使用 NDCG@10 作为主要评估指标
4. **Query 设计**：以时间窗口或行业作为 query 分组

### 6.3 代码示例

```python
# 股票排序学习示例（使用 XGBoost）
import xgboost as xgb
import numpy as np

# 假设数据：每只股票的特征和未来收益
# features: [股票数, 特征维度]
# returns: [股票数,] - 未来收益
# query_groups: 每个时间窗口的股票数量

query_groups = [len(returns[i:i+window]) for i in range(0, len(returns), window)]

dtrain = xgb.DMatrix(
    features, 
    label=returns,
    qid=query_groups
)

params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@10',
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

model = xgb.train(params, dtrain, num_boost_round=500)
```

---

## 7. 参考资料

1. **LambdaMART Explained** - Shaped.ai: https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank
2. **XGBoost Learning to Rank** - 官方文档: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
3. **Learning to Rank Using an Ensemble** - Burges et al.: https://proceedings.mlr.press/v14/burges11a/burges11a.pdf
4. **ltr-lib Python库** - https://pypi.org/project/ltr-lib/

---

*文档生成时间：2026-05-09*
