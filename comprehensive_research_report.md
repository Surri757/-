# 股票预测模型改进综合调研报告

**目标**：将收益率从1.4044提升至5%以上  
**当前模型**：StockTransformer (序列长度60，特征维度197，学习率1e-5，Top-5权重2.0)  
**调研时间**：2026-05-10

---

## 一、调研概述

本次调研系统收集并分析了Kaggle金融预测竞赛冠军方案、Transformer金融时序预测SOTA方法、以及股票排序学习的最佳实践。调研覆盖了2020-2024年Jane Street、Two Sigma等顶级竞赛的获奖技术，以及学术界在时间序列预测、对比学习、排序优化等领域的最新进展。

---

## 二、Kaggle竞赛冠军方案总结

### 2.1 Jane Street Kaggle竞赛关键发现

**顶级方案技术架构**：
- **MLP (多层感知机)** 是主流选择，配合Autoencoder进行特征压缩
- **多任务学习 (Multitask Learning)** 用于同时预测多个响应值
- **集成方法**（多个模型的加权平均）显著提升性能

**特征工程关键发现**：
1. **波动率特征是核心**：ts_id与日波动率高度相关
2. **滚动统计特征**：rolling mean, rolling std, rolling min/max
3. **时间特征交叉**：不同时间范围的响应值随机组合减少过拟合

**损失函数优化**：
- 改进的交叉熵函数
- Utility score function加权
- 随机选择不同时间范围的响应值

### 2.2 LightGBM/XGBoost在金融预测中的应用

| 指标 | LightGBM表现 |
|------|-------------|
| 总回报率 | 394% |
| 优势 | 高效处理大规模数据 |
| 适用场景 | 特征重要性分析、基线对比模型 |

---

## 三、Transformer金融时序预测SOTA方法

### 3.1 时间序列Transformer架构 (2024-2025)

| 模型 | 来源 | 关键特点 |
|------|------|----------|
| **TimeMixer** | ICLR 2024 | 可分解多尺度混合，完全基于MLP，效率高 |
| **Toto** | Datadog 2024 | 可观测性数据优化，预训练+微调范式 |
| **Skip-Timeformer** | IJCAI 2024 | 跳跃时间交互，适合多周期量化策略 |
| **BasisFormer** | 2025 | 可学习可解释基底，预测提升11-15% |
| **Stockformer** | arXiv 2025 | 针对金融预测任务改进 |
| **Quantformer** | arXiv 2024 | 三频率(月/周/日)量化交易策略 |

### 3.2 位置编码改进方法

1. **动态位置编码 (Dynamic Position Encoding)**
   - 自适应计算位置信息
   - 克服静态正弦/可学习嵌入的局限
   - 特别适合非平稳金融数据

2. **低维绝对位置编码 (ldAPE)**
   - 解决传统位置编码在各向异性问题
   - 效果优于tAPE和Sin-APE

3. **时间特征作为位置编码**
   - past_time_features作为输入
   - 参考HuggingFace Time Series Transformer实现

### 3.3 对比学习在金融时序的应用

| 方法 | 来源 | 应用场景 |
|------|------|----------|
| **MF-CLR** | ICLR 2024 | 多频率对比学习，长短期预测 |
| **资产嵌入对比学习** | ACM ICAIF 2024 | 基于收益相似性捕获股票间关系 |
| **软对比学习** | ICLR 2024 | 解决相邻时间戳忽略内在相关性问题 |

**金融领域采样策略**：
- **正样本**：历史模式中具有相同次日价格方向的模式
- **负样本**：具有相反未来波动的模式

### 3.4 CrossAttention和注意力机制

1. **Gated Cross-Attention Mechanism (2024)**
   - 多模态股票预测的稳定融合
   - 门控机制平衡不同模态信息

2. **异构图注意力 (HGA-MT)**
   - 风险收益多任务学习
   - 股票关系建模（产业链、供应链）

3. **动态调整注意力权重**
   - 根据市场状态自适应
   - 高效融合主动和被动投资

---

## 四、股票排序学习 (Stock Ranking Learning)

### 4.1 LightGBM LambdaRank算法

**核心原理**：LambdaRank将梯度提升树(MART)与LambdaMART算法结合，通过直接优化NDCG等排序指标学习排序。

**关键参数配置**：
```python
from lightgbm import LGBMRanker

ranker = LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    ndcg_eval_at=[5, 10, 20],
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31
)
```

### 4.2 XGBoost Learning to Rank

**三种排序目标函数**：
- `rank:ndcg` (默认) - 优化NDCG指标，基于LambdaMART
- `rank:pairwise` - 优化成对排序损失
- `rank:map` - 优化平均准确率

### 4.3 ACT: Anti-Crosstalk Learning (2024)

**核心创新**：
- 显式捕获股票间关系的重要性
- Anti-Crosstalk机制减少股票特征间的干扰
- 跨截面股票排序需要考虑股票间的相关性

**论文**：https://arxiv.org/html/2604.20204v1

### 4.4 NDCG优化技术

**NDCG指标计算**：
```
DCG@K = Σ(2^rel_i - 1) / log2(i + 1)
NDCG@K = DCG@K / IDCG@K
```

**优化方法**：
- 大规模随机优化NDCG代理
- 具有可证明收敛保证的高效随机算法
- per-iteration复杂度与mini-batch大小成正比

---

## 五、特征工程最佳实践

### 5.1 当前模型Top-5重要特征

| 排名 | 特征名 | 说明 |
|------|--------|------|
| 1 | CORD60 | 相关性指标 |
| 2 | WVMA60 | 加权移动平均 |
| 3 | VWAP0 | 成交量加权平均价 |
| 4 | obv | 能量潮指标 |
| 5 | volatility_10 | 10日波动率 |

### 5.2 建议新增特征

| 特征类别 | 建议特征 | 来源 |
|----------|----------|------|
| 滚动统计 | rolling_mean_20, rolling_std_20, rolling_min/max | Jane Street冠军方案 |
| 波动率 | volatility_30, ATR, Bollinger Bands | 技术分析 |
| 动量 | momentum_5, momentum_20, return_1d/5d/20d | 量化实践 |
| 相对排名 | cross-sectional rank (日内股票相对排名) | Stock Ranking研究 |
| 行业相对 | 相对于行业的超额收益 | 因子投资 |

---

## 六、改进建议路线图

### 6.1 短期改进 (1-2周) - 可快速验证

| 改进方向 | 具体措施 | 预期效果 |
|----------|----------|----------|
| **特征工程增强** | 添加波动率特征、滚动统计特征 | 提升模型表达能力 |
| **评估指标** | 增加NDCG@10作为排序质量指标 | 更好地衡量选股能力 |
| **LightGBM Ranker基线** | 实现LightGBM LambdaRank作为对比 | 验证深度学习vs GBM |

### 6.2 中期改进 (1个月) - 显著提升性能

| 改进方向 | 具体措施 | 预期效果 |
|----------|----------|----------|
| **集成学习** | LightGBM Ranker + StockTransformer加权集成 | 稳定性+精度 |
| **损失函数优化** | 实现自定义NDCG优化损失 | 直接优化排序质量 |
| **特征工程增强** | 添加cross-sectional rank、行业相对表现 | 增强相对比较能力 |
| **XGBoost Ranker** | 实现XGBoost rank:ndcg，与LightGBM集成 | 进一步提升 |

### 6.3 长期改进 (2-3个月) - 突破性进展

| 改进方向 | 具体措施 | 预期效果 |
|----------|----------|----------|
| **ACT机制** | Anti-Crosstalk学习，显式建模股票间关系 | 捕获股票相关性 |
| **多任务学习** | 同时预测收益和风险，共享表征 | 泛化能力提升 |
| **对比学习预训练** | MF-CLR多频率对比学习 | 无监督学习市场模式 |
| **TimeMixer架构** | 引入多尺度混合机制 | 捕获不同时间尺度特征 |

---

## 七、关键参考文献

### 7.1 Kaggle竞赛方案
- scaomath/jane_street_kaggle - 深度学习 + 波动率特征
- andre-ye/jane_street_kaggle - Top 1% MLP方案
- arxiv:2501.07580 - LightGBM特征工程，394%回报率
- arxiv:2009.07701 - 滚动统计特征

### 7.2 Transformer架构
- TimeMixer (ICLR 2024) - https://github.com/kwuking/TimeMixer
- Stockformer (arXiv:2502.09625)
- Quantformer (arXiv:2404.00424)
- 对比学习: MF-CLR (ICLR 2024), 资产嵌入 (ACM ICAIF 2024)

### 7.3 排序学习
- LightGBM LGBMRanker API文档
- XGBoost Learning to Rank文档
- ACT: Anti-Crosstalk Learning (arXiv:2604.20204)
- NDCG优化 (arXiv:2202.12183)

---

## 八、结论

基于本次调研，我们明确了以下改进方向：

1. **特征工程是快速提升的关键**：添加波动率特征、滚动统计特征、cross-sectional rank等
2. **排序学习优于回归学习**：对于股票选择问题，LightGBM/XGBoost Ranker可能更适合
3. **集成策略有效**：将Transformer与GBM结合可以兼顾精度和稳定性
4. **损失函数定制化**：直接优化NDCG等排序指标而非MSE

**推荐实施顺序**：
1. 首先实现LightGBM Ranker作为快速基线验证
2. 然后增强特征工程（滚动统计、波动率特征）
3. 最后实施Transformer+GBM集成方案

---

*调研完成时间：2026-05-10*  
*数据来源：Kaggle、GitHub、arXiv、Semantic Scholar*