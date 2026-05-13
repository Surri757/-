# Learning to Rank 排序学习论文汇总

## 论文收集信息

本次搜索聚焦于排序学习（Learning to Rank）在量化选股/股票排序中的应用，关键词包括 Learning to Rank、stock ranking、StockRanker、LambdaMART、NDCG optimization 等。

---

## 论文1: SortNet: 基于神经网络的排序算法

- **作者**: Leonardo Rigutini, Tiziano Papini, Marco Maggini, Franco Scarselli
- **arXiv ID**: 2311.01864v1
- **PDF**: papers/sortnet_neural_ranking_2311.01864.pdf
- **摘要**: 本文提出 SortNet，一种基于神经网络的排序算法，使用神经网络作为比较器来排序对象。算法采用迭代训练过程，每次迭代添加最具信息量的训练样本。SortNet 在 LETOR 数据集上展示了良好的性能，适用于用户偏好不同的排序任务。
- **与本项目相关性**: 神经网络排序方法，可作为 StockRanker 改进的参考。

---

## 论文2: 大规模 NDCG 深度学习随机优化

- **作者**: Zi-Hao Qiu, Quanqi Hu, Yongjian Zhong, Lijun Zhang, Tianbao Yang
- **arXiv ID**: 2202.12183v5
- **PDF**: papers/ndcg_optimization_deep_learning_2202.12183.pdf
- **摘要**: 本文提出了一种原则性方法来优化 NDCG 及其 top-K 变体，将 NDCG 代理优化表述为一个组合优化问题，开发了具有可证明收敛保证的高效随机算法，per-iteration 复杂度与 mini-batch 大小成正比。实验表明方法在 NDCG 方面优于先前排序方法。
- **与本项目相关性**: NDCG 优化方法对 StockRanker 的损失函数优化有直接参考价值。

---

## 论文3: ILMART - 可解释的 LambdaMART 排序

- **作者**: Claudio Lucchese, Franco Maria Nardini, Salvatore Orlando, Raffaele Perego, Alberto Veneri
- **arXiv ID**: 2206.00473v1
- **PDF**: papers/ilmart_interpretable_lambdamart_2206.00473.pdf
- **摘要**: 本文提出 ILMART，一种基于 LambdaMART 的可解释排序解决方案，通过利用有限且受控数量的成对特征交互来训练有效且可解释的排序模型。在三个公开可用的 LtR 数据集上的实验表明，ILMART 在 nDCG 方面比当前最先进的可解释排序方法高出 8%。
- **与本项目相关性**: LambdaMART 的改进方法，对 StockRanker 的 Lambda 梯度优化有重要参考价值。

---

## 论文4: SpatialRank - 城市事件排序

- **作者**: Bang An, Xun Zhou, Yongjian Zhong, Tianbao Yang
- **arXiv ID**: 2310.00270v5
- **PDF**: papers/spatialrank_ndcg_urban_2310.00270.pdf
- **摘要**: 本文提出 SpatialRank，一种新颖的城市事件排序方法，通过自适应图卷积层动态学习位置间的时空依赖性，优化带有空间分量的混合 NDCG 损失。在三个真实数据集上的实验表明，SpatialRank 在 NDCG 方面优于最新方法达 12.7%。
- **与本项目相关性**: 时空依赖建模方法，可用于股票市场的时间序列依赖建模和排序优化。

---

## 搜索结果统计

- **搜索关键词**: Learning to Rank stock ranking StockRanker LambdaMART NDCG optimization
- **搜索结果数量**: 10 篇
- **精选下载论文**: 4 篇
- **添加到 reference.bib**: 4 篇

## 后续建议

1. 深入阅读 NDCG 优化论文（qiu2022ndcg）中的损失函数设计
2. 研究 LambdaMART 的改进方法（lucchese2022ilmart）并应用到 StockRanker
3. 探索时空依赖建模方法用于股票时间序列特征