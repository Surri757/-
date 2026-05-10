# 2026大数据挑战赛 — 沪深300量化预测方案

## 环境配置

- Python版本：3.10.14
- PyTorch版本：2.2.1+cu121
- CUDA版本：12.1
- 硬件要求：i7-13650H + 16GB + RTX4060 8GB
- 训练时间：约0.45小时
- 预测时间：约2分钟

## 可复现性保证

所有随机种子固定为42（numpy/pandas/sklearn/lightgbm/xgboost/catboost/torch/cudnn/Python random/PYTHONHASHSEED）。

## 数据来源

1. **baostock**：沪深300成分股日线数据（OHLCV + PE/PB）
2. **AKShare**：宏观经济数据（CPI/PPI/PMI/M2/LPR/汇率）
3. 所有数据已下载到本地data目录，纯离线运行

## 架构总览

```
数据层 → 特征工程 → 7模型Stacking → 信号生成 → 多层筛选(cascade) → 组合优化 → result.csv
                                                      ↓
                                    Gate(动态+SSM) + 黑名单 + 复活赛
```

## 模型：7模型 Stacking 集成

| 模型 | 类型 | OOF IC | 集成权重 |
|------|------|--------|---------|
| XGBoost | GBDT | 0.422 | 32.0% |
| LightGBM | GBDT | 0.415 | 23.9% |
| TFT | LSTM+Attention | 0.253 | 19.8% |
| PatchTST | Patch+Transformer | 0.256 | 15.3% |
| DLinear | 线性趋势分解 | 0.257 | 9.1% |
| CatBoost | GBDT | 0.385 | 0%（被压缩） |
| **Stacking Ensemble** | IC优化加权 | **0.395** | — |

- 元模型：NNLS（MSE最小化） vs IC优化（Spearman最大化），选用IC更高的
- 目标：T+1开盘买入 → T+5开盘卖出（open-to-open）
- TFT 为自研轻量版（128K参数，2s/epoch），含 RevIN + LSTM Encoder/Decoder + Multi-head Attention + GRN 门控
- 损失函数：MSE + Pairwise Ranking 混合损失
- 样本权重：高收益样本非线性加权（alpha=3.0）

## 特征工程（~130维）

1. 量价基础特征：日涨跌幅、多周期收益率、振幅、量比、换手率
2. 技术指标：MA/MACD/RSI/KDJ/布林带/ATR/OBV/CCI/DMI
3. Alpha因子：Amihud非流动性、VWAP偏离、MAX效应、隔夜缺口、日内反转
4. 行为金融情绪因子（15个）：FOMO追逐强度、恐慌抛售指数、牛熊订单比等
5. 行业与宏观特征：行业PE/PB分位数、宏观指标变化
6. 时序统计特征：波动率聚类、收益率偏度/峰度、夏普比率
7. 截面特征：行业内收益率z-score、市值分位数、流动性分位数

## 推理流程：多层筛选 (Cascade Pipeline)

```
300只信号
  ├─ [永久黑名单] 84只压舱石（银行/石油/保险/运营商/基建/公用事业）→ 直接淘汰
  ├─ [动态活跃度] 30日成交量<历史60%的票 → 暂时拉黑（放量可复活）
  ├─ [因子画像分组] 波动率×动量×估值 → 6-8组
  ├─ [Stage 1 粗筛] 各组用不同模型打分 → z-score排名 → 收益下限0.8% → 保留top
  ├─ [Stage 2 精选] 换一套模型交叉验证 → z-score排名 → 收益下限0.6% → 保留top
  ├─ [复活赛] TFT+DLinear 救回被误杀的爆发票（最多5只）
  └─ [决赛] 全量7模型Stacking + 波动率折扣 + max-Sharpe组合优化 → 5只 → result.csv
```

## Gate 动态准入（含 SSM 谱状态机）

5项校验，阈值根据市场状态自适应调整：

| 市场状态 | 置信度门槛 | 最大持仓 | 风险距离 |
|---------|-----------|---------|---------|
| 牛 (bull) | 0.05 | 5 | 1.5% |
| 震荡 (sideways) | 0.10 | 5 | 2.0% |
| 熊 (bear) | 0.15 | 4 | 2.5% |
| 恐慌 (panic) | 0.20 | 3 | 3.0% |

- **SSM 谱状态机**：7维市场特征向量（趋势分/波动分/预测分/波动率/波动变化/Sharpe/正向率）→ 5隐状态（偏牛/震荡/弱熊/熊市/恐慌），进一步微调阈值
- 预测质量高时额外放宽门槛；趋势强时持仓上限浮动

## 组合优化

- 方法：max-Sharpe（scipy SLSQP约束优化）
- 约束：sum(w)=1.0（满仓）、w_i≥0、持仓≤5只
- 协方差矩阵：优先从60日历史价格估计（Ledoit-Wolf收缩），回退到模型分歧+常数相关
- 决赛阶段对低波票预测收益施加0.7x折扣，导向高活性选股

## 组合约束

- 股票数量：≤5只（硬上限）
- 总权重：=1.0（满仓）
- 单只权重：≤1.0
- 止损宽度：根据市场状态动态调整（牛1.5x / 恐慌0.7x）

## 训练流程

1. 加载数据并预处理（排除最近120个交易日防泄露）
2. 特征工程（~130个特征，tqdm进度条）
3. 时间序列滚动交叉验证（6折）
4. 训练6个基础模型（GBDT并行 + PyTorch串行，tqdm进度条）
5. 训练Stacking元模型（NNLS + IC优化）
6. SHAP因子重要性分析（LightGBM TreeExplainer → top-20 PNG+JSON）
7. 保存所有模型到model目录

## 输出格式

- `result.csv`：UTF-8，stock_id为6位补零，含stock_id和weight两列，权重和=1.0
- `review_YYYYMMDD_HHMMSS.txt`：可读复盘报告
- `review_YYYYMMDD_HHMMSS.json`：结构化复盘数据
- `shap_importance.png` / `shap_importance.json`：SHAP因子重要性

## 用法

```bash
# 训练（自动拉数据 + 训练模型 + SHAP）
python train.py

# 预测（默认多层筛选 cascade）
python test.py                  # 多层筛选+复活赛
python test.py --mode simple    # 单层流水线（快速baseline）
```

## Docker镜像

- 基础镜像：nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
- 预计大小：8-9GB（≤10GB）
- 构建：docker build -t bdc2026 .
- 运行：docker-compose up

## 创新点

1. **多层因子画像筛选**：按波动率×动量×估值分组，不同组用不同模型组合，组内z-score跨组可比
2. **复活赛机制**：DL模型（TFT+DLinear）专救被树模型误判的爆发票
3. **动态活跃度过滤**：30日量比<60%暂时拉黑，放量自动复活
4. **SSM谱状态机**：图聚类识别5种市场微观隐状态，动态调节Gate阈值
5. **轻量TFT**：128K参数的自研Temporal Fusion Transformer，2s/epoch
6. **波动率折扣**：决赛阶段低波票预测收益×0.7，护航高活性选股

## 最后更新

2026-05-11
