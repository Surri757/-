# Stock Ranking Learning调研报告

## 1. 概述

股票排序学习(Stock Ranking Learning)是量化交易中的一种重要方法，目标是对一组股票进行排序，使排名靠前的股票在未来具有更好的收益表现。与传统的点对点(pointwise)预测不同，排序学习关注的是股票之间的相对顺序关系。

## 2. LightGBM Ranker在股票预测中的应用

### 2.1 LightGBM LambdaRank算法

**核心原理**：LambdaRank是LambdaMART算法的核心，将梯度提升树(MART)与指标感知优化相结合，通过直接优化NDCG等排序指标来学习排序。

**关键特点**：
- 使用`lambdarank`目标函数
- 支持按query/group分组
- 可直接优化NDCG@K指标
- 高效处理大规模数据

**股票预测应用**：
```python
from lightgbm import LGBMRanker

ranker = LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31
)

# 按日期分组股票
ranker.fit(X_train, y_train, group=date_groups)
predictions = ranker.predict(X_test)
```

### 2.2 Kaggle股票预测案例

**JPX东京证券交易所预测**：Kaggle上有专门的LightGBM Ranker入门教程，展示了如何用排序方法选择上涨股票。

**参考notebook**：https://www.kaggle.com/code/bturan19/lightgbm-ranker-introduction

### 2.3 关键参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| objective | lambdarank | 排序目标函数 |
| ndcg_eval_at | [5, 10, 20] | NDCG评估位置 |
| label_gain | [0,1,2,3,...] | 相关性等级增益 |
| lambdarank_truncation_level | 10-30 | 截断层级 |

## 3. XGBoost Learning to Rank

### 3.1 XGBoost排序目标函数

XGBoost通过三种排序目标函数实现学习排序：

1. **rank:ndcg** (默认) - 优化NDCG指标，基于LambdaMART
2. **rank:pairwise** - 优化成对排序损失
3. **rank:map** - 优化平均准确率

**文档参考**：https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html

### 3.2 股票选择应用

XGBoost的Pairwise Ranking通过采样大量股票对，最小化成对损失：

```python
import xgboost as xgb

params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@10',
    'eta': 0.05,
    'max_depth': 6
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# 需要设置query groups
xgb.train(params, dtrain, num_boost_round=500,
          evals=[(dtrain, 'train'), (dtest, 'test')])
```

### 3.3 LambdaMART算法详解

LambdaMART是Learning to Rank领域最成功的算法之一：
- 结合梯度下降和排序指标优化
- 每个样本的梯度由Lambda梯度表示
- 效率高，可扩展性强

## 4. 排序学习在金融领域的应用

### 4.1 ACT: Anti-Crosstalk股票排序

**最新研究(2024)**：ACT (Anti-Crosstalk Learning) 强调显式捕获股票间关系的重要性，使用Anti-Crosstalk机制减少股票特征间的干扰。

**论文**：https://arxiv.org/html/2604.20204v1

**关键洞察**：
- 跨截面股票排序需要考虑股票间的相关性
- 传统方法忽略股票间的交互作用
- Anti-Crosstalk机制可以减少特征干扰

### 4.2 学习到排序的Portfolio Selection

**研究来源**：Stock picking with machine learning (Ideas.repec.org)

**关键发现**：
- ML-based股票选择模型显著优于等权重组合
- 集成学习方法效果最好
- 特征选择对性能影响显著

## 5. NDCG优化技术

### 5.1 NDCG指标

NDCG (Normalized Discounted Cumulative Gain) 是排序评估的核心指标：

```
DCG@K = Σ(2^rel_i - 1) / log2(i + 1)
NDCG@K = DCG@K / IDCG@K
```

### 5.2 NDCG优化方法

**学术研究**：Large-scale Stochastic Optimization of NDCG Surrogates for Deep Learning with Provable Convergence

**关键策略**：
- 将NDCG代理优化表述为组合优化问题
- 开发具有可证明收敛保证的高效随机算法
- per-iteration复杂度与mini-batch大小成正比

### 5.3 股票排序的NDCG应用

```python
def ndcg_at_k(y_true, y_pred, k=10):
    """计算NDCG@K"""
    # 获取top-k预测
    top_k_idx = np.argsort(y_pred)[-k:]
    
    # 计算DCG
    dcg = 0
    for i, idx in enumerate(top_k_idx):
        rel = y_true[idx]
        dcg += (2**rel - 1) / np.log2(i + 2)
    
    # 计算IDCG
    ideal_sorted = np.sort(y_true)[::-1][:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) 
               for i, rel in enumerate(ideal_sorted))
    
    return dcg / idcg if idcg > 0 else 0
```

## 6. 特征工程最佳实践

### 6.1 股票排序关键特征

| 特征类别 | 示例特征 | 重要性 |
|----------|----------|--------|
| 价格动量 | return_1d, return_5d, return_20d | 高 |
| 技术指标 | RSI, MACD, Bollinger Bands | 高 |
| 波动率 | volatility_10, volatility_30, ATR | 高 |
| 成交量 | volume_ratio, OBV | 中 |
| 价值因子 | P/E, P/B, dividend_yield | 中 |
| 动量因子 | momentum_20, momentum_60 | 高 |

### 6.2 基于特征重要性的选择

从当前项目的feature_importance.csv可知，Top-5特征为：
1. CORD60 - 相关性指标
2. WVMA60 - 加权移动平均
3. VWAP0 - 成交量加权平均价
4. obv - 能量潮指标
5. volatility_10 - 10日波动率

### 6.3 特征工程建议

1. **滚动统计特征**：rolling mean, rolling std, rolling min/max
2. **相对排名特征**：cross-sectional rank within each day
3. **行业相对表现**：相对于行业的超额收益
4. **时间动量交互**：momentum × volatility

## 7. Listwise排序方法

### 7.1 Listwise vs Pairwise vs Pointwise

| 方法 | 粒度 | 优点 | 缺点 |
|------|------|------|------|
| Pointwise | 单个样本 | 简单直观 | 忽略样本间关系 |
| Pairwise | 样本对 | 考虑相对顺序 | 计算复杂度高 |
| Listwise | 整个列表 | 全局最优 | 实现复杂 |

### 7.2 股票排序的Listwise应用

对于股票排序，每个日期的股票列表作为一个query，优化整个列表的排序质量：

```python
class ListwiseStockRanker:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X_list, dates):
        """输入每天的股票列表"""
        predictions = []
        for date, stocks in zip(dates, X_list):
            # 预测当天所有股票的收益
            preds = self.model.predict(stocks)
            # 按预测收益排序
            ranked_indices = np.argsort(preds)[::-1]
            predictions.append((date, ranked_indices))
        return predictions
```

## 8. 集成排序策略

### 8.1 多模型集成

1. **LightGBM Ranker + XGBoost Ranker** - 加权平均
2. **Deep Learning + GBM** - Stacking
3. **不同排序目标** - rank:ndcg + rank:pairwise组合

### 8.2 集成实现示例

```python
# 多个排序模型预测
lgb_preds = lgb_ranker.predict(X_test)
xgb_preds = xgb_ranker.predict(X_test)
transformer_preds = transformer_model.predict(X_test)

# 加权集成
final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * transformer_preds

# 排序选股
selected_stocks = np.argsort(final_preds)[-top_k:]
```

## 9. 改进建议

### 9.1 短期改进（1-2周）

1. **实现LightGBM Ranker**：使用lambdarank目标函数
2. **添加Cross-sectional Rank特征**：每天股票相对于市场排名
3. **NDCG@10作为评估指标**：与当前MSE指标对比

### 9.2 中期改进（1个月）

1. **XGBoost Ranker集成**：与LightGBM Ranker加权平均
2. **特征工程增强**：添加波动率特征、行业相对表现
3. **损失函数改进**：考虑实现自定义NDCG优化损失

### 9.3 长期改进（2-3个月）

1. **ACT机制**：Anti-Crosstalk学习显式建模股票间关系
2. **端到端排序优化**：直接优化投资组合收益
3. **多任务学习**：同时预测收益和风险

## 10. 参考文献

1. LightGBM Ranker - https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
2. XGBoost Learning to Rank - https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html
3. LambdaMART Explained - https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank
4. ACT: Anti-Crosstalk Learning - https://arxiv.org/html/2604.20204v1
5. NDCG Optimization - https://arxiv.org/abs/2202.12183
6. LightGBM Ranker Stock Prediction - https://bozkurtturanyigit.medium.com/forecasting-many-stock-assets-with-one-ml-model-4e581f799588

---

*调研时间：2026-05-10*
*关键词：Stock Ranking, LambdaMART, NDCG, LightGBM, XGBoost*