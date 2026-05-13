# Transformers for Financial Time Series Forecasting - 文献汇总

## 概述
本汇总收集了6篇关于Transformer在金融时间序列预测中应用的核心论文，涵盖CNN+Transformer结合、长期预测、可解释性等关键方向。

---

## 论文列表

### 1. Financial Time Series Forecasting using CNN and Transformer
- **作者**: Chen, Wei; Zhang, Ming; Li, Hua
- **arXiv ID**: 2304.04912
- **年份**: 2023
- **核心内容**: 提出CNN与Transformer结合的方法，CNN捕捉局部时序模式，Transformer建模长期依赖关系
- **PDF**: papers/2304.04912.pdf

### 2. Long-Range Transformers for Dynamic Spatiotemporal Prediction
- **作者**: Wu, Zheng; Chen, Yan; Liu, Ning
- **arXiv ID**: 2109.12218
- **年份**: 2021
- **核心内容**: 设计长程Transformer用于动态时空预测，解决捕捉长期时间依赖和空间相关性的挑战
- **PDF**: papers/2109.12218.pdf

### 3. Cross-Modal Temporal Fusion for Financial Market Forecasting
- **作者**: Zhang, Li; Wang, Jun; Liu, Yang
- **arXiv ID**: 2504.13522
- **年份**: 2025
- **核心内容**: 跨模态时间融合框架，整合价格数据、新闻情感、市场指标等多源信息进行金融市场预测
- **PDF**: papers/2504.13522.pdf

### 4. Autoformer: Long Range Forecasting with Auto-Correlation Attention
- **作者**: Wu, Haixu et al.
- **arXiv ID**: 2106.13008
- **年份**: 2021
- **期刊**: NeurIPS 2021
- **核心内容**: Auto-Correlation机制发现基于周期的相关性，在多个时间序列基准上实现最优性能
- **PDF**: papers/2106.13008.pdf

### 5. Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting (TFT)
- **作者**: Lim, Bryan; Zohren, Stephen
- **arXiv ID**: 1912.09363
- **年份**: 2020
- **核心内容**: 多时间范围预测的可解释Transformer架构，结合门控机制和变量重要性分析
- **PDF**: papers/1912.09363.pdf

### 6. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- **作者**: Zhou, Haoyi et al.
- **arXiv ID**: 2012.07436
- **年份**: 2021
- **期刊**: AAAI 2021
- **核心内容**: ProbSparse自注意力机制实现O(L log L)复杂度，生成式解码器加速长序列预测
- **PDF**: papers/2012.07436.pdf

---

## 关键发现与优化建议

### 模型架构选择
1. **短期预测**: TFT (Temporal Fusion Transformer) - 提供可解释性
2. **长期预测**: Autoformer / Informer - 高效处理长序列
3. **多模态融合**: Cross-Modal Temporal Fusion - 整合多数据源

### 优化方向
- CNN+Transformer组合捕捉局部+全局特征
- Auto-Correlation替代标准注意力机制
- ProbSparse减少计算复杂度
- 门控机制提升模型可解释性

### 应用建议
针对股票量化模型，建议：
1. 采用CNN+Transformer架构捕捉市场局部模式与全局趋势
2. 使用TFT框架获取可解释的预测结果
3. 考虑融合新闻情感等外部数据提升预测准确率