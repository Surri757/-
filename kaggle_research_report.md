# Kaggle金融预测竞赛冠军方案调研报告

## 1. Jane Street Kaggle竞赛概述

Jane Street举办的Kaggle市场预测竞赛是量化金融领域最具影响力的ML竞赛之一。竞赛目标是预测金融市场的responder（响应值），用于选择最优交易执行。

## 2. 2020年Jane Street竞赛关键发现

### 顶级方案技术特点：

1. **模型架构**：
   - MLP (多层感知机) 是主流选择
   - Autoencoder + Multitask MLP 组合
   - 集成方法（多个模型的加权平均）

2. **特征工程**：
   - 利用波动率相关特征
   - 交易数量(ts_id)与日波动率的相关性
   - 滚动统计特征（rolling statistics）

3. **损失函数优化**：
   - 改进的交叉熵函数
   - Utility score function 加权
   - 随机选择不同时间范围的响应值来减少过拟合

4. **Top 1%方案特点**（scaomath团队）：
   - 深度学习模型
   - 波动率特征识别
   - 与Two Sigma竞赛第五名方案类似的思路

## 3. LightGBM/XGBoost在金融预测中的应用

### 关键文献发现：

1. **arxiv:2501.07580** - "Assets Forecasting with Feature Engineering and Transformation"
   - LightGBM、XGBoost、CatBoost和AdaBoost的对比研究
   - LightGBM在预测中达到394%的总回报率

2. **arxiv:2009.07701** - "Learnings from Kaggle's Forecasting Competitions"
   - LightGBM配合基于滚动统计的特征工程
   - 时间序列特征工程最佳实践

3. **ResearchGate研究** - "Stock Price Prediction Based on XGBoost and LightGBM"
   - 基于Jane Street数据集
   - 处理缺失值和异常数据的策略

## 4. 深度学习在量化交易中的应用

### Semantic Scholar研究 (Zhang & Zohren):
- 时序动量策略和横截面动量策略的增强
- 端到端组合优化框架
- 预测信号生成

### Cambridge University Press:
- 深度学习在量化交易中的系统性方法
- 时间序列神经网络
- 投资组合优化
- 动量交易和波动率缩放

### 关键挑战：
- 过拟合问题
- 高计算成本
- 可解释性差

## 5. 改进建议

### 基于调研的模型改进方向：

1. **集成学习**：
   - 将当前StockTransformer与LightGBM/XGBoost集成
   - 权重平均或stacking策略

2. **特征工程增强**：
   - 添加波动率相关特征（基于Top 1%方案）
   - 滚动统计特征（rolling mean, rolling std）
   - 时间特征交叉

3. **损失函数优化**：
   - 实现自定义的utility score loss
   - 考虑时间范围的随机化

4. **多任务学习**：
   - 预测多个时间范围的响应值
   - 共享表征学习

5. **风险控制策略**：
   - 组合权重优化
   - Sharpe比率直接优化

## 6. 关键引用

| 来源 | 关键技术 |
|------|----------|
| scaomath/jane_street_kaggle | 深度学习 + 波动率特征 |
| andre-ye/jane_street_kaggle | Top 1% MLP方案 |
| arxiv:2501.07580 | LightGBM特征工程 |
| arxiv:2009.07701 | 滚动统计特征 |
| Semantic Scholar | 端到端组合优化 |

## 7. 下一步行动

1. **短期**：将LightGBM/XGBoost作为基线对比模型
2. **中期**：实现特征工程增强，特别是波动率特征
3. **长期**：构建集成模型，结合Transformer和GBM

---
*调研时间：2026-05-10*
*数据来源：Kaggle、GitHub、arXiv、Semantic Scholar*
