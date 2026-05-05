# 2026大数据挑战赛-总冠军方案代码说明

## 环境配置

- Python版本：3.10.14
- PyTorch版本：2.2.1+cu121
- CUDA版本：12.1
- cuDNN版本：8.x
- 依赖包：见requirements.txt
- 硬件要求：i7-13650H+16GB+RTX4060 8GB
- 训练时间：约0.35小时
- 预测时间：约1分钟

## 可复现性保证

### 随机种子固定（全部设为42）

| 库 | 种子设置 |
|---|---------|
| numpy | np.random.seed(42) |
| pandas | pd.random.seed(42) |
| sklearn | check_random_state(42) |
| lightgbm | lgb.seed(42) |
| xgboost | xgb.set_random_seed(42) |
| catboost | cb.set_random_seed(42) |
| torch | torch.manual_seed(42) |
| cudnn | cudnn.deterministic=True |
| Python random | random.seed(42) |
| 环境变量 | PYTHONHASHSEED=42 |

## 数据来源

1. **baostock**：沪深300成分股日线数据， https://baostock.com
2. **AKShare**：宏观经济数据（CPI/PPI/PMI/M2/LPR/汇率）， https://akshare.akfamily.xyz
3. **申万宏源**：行业分类映射

所有数据均已下载到本地data目录，纯离线运行。

## 架构：分层量化交易系统

```
信号层 (signals.py) → Gate层 (gate.py) → 执行层 (executor.py) → 风控层 (risk_manager.py) → 复盘层 (review.py)
```

### 信号层
- 6模型Stacking集成：LightGBM + CatBoost + XGBoost + PatchTST + TimesNet + DLinear
- 元模型：IC优化加权（IC=0.376）
- 目标：T+1开盘买入 → T+5开盘卖出（open-to-open）
- ATR自适应止损 + 预测置信度估算

### Gate层（5项校验）
1. 白名单（可选，默认全市场）
2. 单只股票唯一持仓
3. 风险距离 ≥ 2%（止损不能太近）
4. 信号置信度 ≥ 0.1
5. 最大持仓 ≤ 5只

### 风控层 — 以损定仓
- 仓位 = max_loss_per_position / (入场价 - 止损价) / 入场价 × 置信度
- 每笔最大亏损：总资金2%
- 止损/止盈触发 + 保护失败强制平仓

### 执行层
- 模拟市价单 + 双边保护单（止损+止盈）
- 默认滑点 5bps
- 交易所状态验证

### 复盘层
- Gate拒绝记录 + 执行成交记录 + 风险事件记录 + 滑点记录
- 业绩归因（按股票/权重/收益率）
- 输出 .txt 和 .json 双格式报告

## 特征工程（~111个特征）

1. **量价基础特征**：日涨跌幅、多周期收益率、振幅、量比、换手率等
2. **技术指标**：MA、MACD、RSI、KDJ、布林带、ATR、OBV、CCI、DMI
3. **Alpha因子**：资金流、VWAP偏离、Amihud非流动性、MAX效应、隔夜缺口、日内反转、VPT量价趋势、EOM、量价背离
4. **行业与宏观特征**：行业PE/PB分位数、宏观指标变化等
5. **时序统计特征**：波动率聚类、收益率偏度/峰度、夏普比率等
6. **截面特征**：行业内收益率z-score、市值分位数、流动性分位数

## 组合约束

- 股票数量：≤5只
- 单只股票权重：≤0.30
- 权重由以损定仓自动计算，总仓位 ≤ 1.0
- 止损宽度根据市场状态动态调整（牛1.5x / 恐慌0.7x）

## 训练流程

1. 加载原始数据并预处理（排除最近120个交易日防泄露）
2. 特征工程（~111个特征）
3. 时间序列滚动交叉验证（6折）
4. 训练6个基础模型（GBDT用MAE/reg:squarederror，PyTorch用RankingMSE混合损失）
5. 训练Stacking元模型（NNLS + IC优化加权）
6. 保存所有模型到model目录

## 推理流程

1. 加载最新数据和模型
2. SignalGenerator 生成交易信号
3. LiveGate 5项准入校验
4. RiskManager 以损定仓计算权重
5. ExecutionSimulator 模拟执行
6. ReviewLayer 生成复盘报告 + result.csv

## 输出格式

- `result.csv`：UTF-8无BOM，stock_id为6位补零，含stock_id和weight两列
- `review_YYYYMMDD_HHMMSS.txt`：可读复盘报告
- `review_YYYYMMDD_HHMMSS.json`：结构化复盘数据

## Docker镜像

- 基础镜像：nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
- 预计大小：8-9GB（符合≤10GB要求）
- 构建命令：docker build -t bdc2026 .
- 运行命令：docker-compose up

## 其他注意事项

- 所有随机种子已固定为42，确保完全可复现
- 复现全程无需联网
- 所有中间文件会自动清理，不占用额外空间

## 最后更新

2026-05-05
