# Transformer金融时序预测调研报告

## 1. 时间序列Transformer SOTA方法 (2024)

### 1.1 TimeMixer (ICLR 2024)
- **架构**: 完全基于MLP的可分解多尺度混合架构
- **特点**: 利用解耦的多尺度时间序列，在长短预测任务上均达到SOTA
- **优势**: 运行效率高，适合实时交易场景
- **参考**: https://github.com/kwuking/TimeMixer

### 1.2 Toto (Datadog 2024)
- **全称**: Time Series Optimized Transformer for Observability
- **特点**: 专为可观测性数据优化的时序基础模型
- **应用**: 在金融时序预测中可借鉴其预训练+微调范式

### 1.3 Skip-Timeformer (IJCAI 2024)
- **特点**: 跳跃时间交互Transformer，增强长序列预测能力
- **优势**: 更好适应任意回看窗口，适用于多周期量化策略

### 1.4 BasisFormer (2025)
- **创新**: 利用可学习可解释基底的端到端预测架构
- **提升**: 单变量预测提升11.04%，多变量预测提升15.78%

## 2. 金融时序预测中的Transformer架构

### 2.1 Stockformer
- **论文**: Transformer Based Time-Series Forecasting For Stock
- **改进**: 针对金融预测任务改进原生Transformer架构
- **核心**: 解决naive Transformer在金融场景的局限性问题

### 2.2 Quantformer (2024)
- **主题**: 从注意力到收益：基于Transformer的量化交易策略
- **三频率**: 月度、周度、日度股票数据实验
- **应用**: 因子计算、股票排序、资产配置

### 2.3 Temporal Transformer with Similarity Embedding
- **特点**: 考虑金融时序的条件异方差特性
- **优势**: 在多步预测任务上优于ARIMA、LSTM、TCN、GRU

### 2.4 LSTM-Transformer混合架构
- **目的**: 提高金融预测的鲁棒性
- **结合**: LSTM的序列建模能力 + Transformer的全局依赖捕获

## 3. 位置编码改进方法

### 3.1 动态位置编码 (Dynamic Position Encoding)
- **优势**: 自适应计算位置信息，克服静态正弦/可学习嵌入的局限
- **适用**: 非平稳金融数据

### 3.2 低维绝对位置编码 (ldAPE)
- **改进**: 解决传统位置编码在低维时序数据的各向异性问题
- **效果**: 优于tAPE和Sin-APE

### 3.3 时间特征作为位置编码
- **方法**: past_time_features作为位置编码输入
- **参考**: HuggingFace Time Series Transformer实现

## 4. 对比学习在金融时序的应用

### 4.1 Contrastive Learning of Asset Embeddings (ACM ICAIF 2024)
- **创新**: 基于收益相似性的资产嵌入对比学习框架
- **核心**: 利用滚动子窗口的收益相似性捕获资产间关系
- **优势**: 克服金融市场复杂性和随机性挑战

### 4.2 MF-CLR (ICLR 2024)
- **全称**: Multi-Frequency Contrastive Learning Representation
- **应用**: 长短期预测、分类、异常检测、填补
- **特点**: 多频率对比学习

### 4.3 Soft Contrastive Learning for Time Series (ICLR 2024)
- **改进**: 解决相邻时间戳对比学习忽略内在相关性的问题
- **方法**: 软对比学习策略

### 4.4 金融领域对比学习采样策略
- **正样本**: 历史模式中具有相同次日价格方向的模式
- **负样本**: 具有相反未来波动的模式

## 5. CrossAttention和注意力机制在量化交易

### 5.1 Gated Cross-Attention Mechanism (2024)
- **应用**: 多模态股票预测的稳定融合
- **特点**: 门控机制平衡不同模态信息

### 5.2 投资组合优化的注意力机制
- **功能**: 动态调整组合权重应对市场变化
- **优势**: 高效融合主动和被动投资

### 5.3 Heterogeneous Graph Attention (HGA-MT)
- **创新**: 异构图注意力网络
- **应用**: 风险收益多任务学习、股票关系建模

### 5.4 Contrastively Aligned Cross-Modal Attention
- **应用**: 深度投资组合选择
- **特点**: 多期设置下的跨模态对齐

## 6. 改进建议

基于调研结果，提出以下模型改进方向：

### 6.1 架构改进
1. **引入TimeMixer的多尺度混合机制**: 捕获不同时间尺度的市场特征
2. **采用动态位置编码**: 替代静态正弦编码，更好地适应金融数据的非平稳性
3. **添加LSTM-Transformer混合层**: 增强对短期市场波动的敏感性

### 6.2 特征学习改进
1. **集成MF-CLR多频率对比学习**: 学习更丰富的时序表征
2. **应用资产嵌入对比学习**: 捕获股票间的相关性和差异性
3. **引入市场状态感知的位置编码**: 考虑宏观经济周期和行业轮动

### 6.3 注意力机制改进
1. **添加Gated Cross-Attention**: 融合多模态信息（技术指标、文本情绪等）
2. **使用异构图注意力**: 建模股票间的产业链、供应链关系
3. **动态调整注意力权重**: 根据市场状态自适应

### 6.4 训练策略改进
1. **引入对比学习预训练**: 在无监督模式下学习市场模式
2. **多任务学习**: 同时预测收益和风险
3. **课程学习**: 从简单到复杂的交易模式逐步训练

## 7. 参考文献

详见 reference.bib 文件

---

*报告生成时间: 2026-05-10*
*目标: 将收益率从1.4044提升至5%以上*
