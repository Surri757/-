# 2026大数据挑战赛-总冠军方案代码说明

## 环境配置

- Python版本：3.10.14
- PyTorch版本：2.2.1+cu121
- CUDA版本：12.1
- cuDNN版本：8.x
- 依赖包：见requirements.txt
- 硬件要求：i7-13650H+16GB+RTX4060 8GB
- 训练时间：约7.2小时（预留0.5小时缓冲）
- 预测时间：约2.8分钟

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
2. **Tushare免费版**：资金流和融资融券数据， https://tushare.pro
3. **国家统计局**：宏观经济数据， https://www.stats.gov.cn
4. **申万宏源**：行业指数数据， https://www.swsindex.com

所有数据均已下载到本地data目录，纯离线运行。

## 算法

### 整体思路

采用"多源数据融合+冠军级多因子体系+7模型Stacking集成+Black-Litterman带不确定性加权组合优化"的终极方案

### 创新点

1. 融合8大类120个特征的多因子体系，覆盖量价、资金流、行业、宏观等所有维度
2. 引入NGboost量化预测不确定性，大幅提升组合鲁棒性
3. 使用PatchTST+TimesNet新一代时序模型，捕捉股价多周期波动
4. 时间序列滚动Stacking集成，充分挖掘模型互补性
5. Black-Litterman带不确定性加权的组合优化，解决传统方法对预测误差敏感的问题

### 模型结构

- 基础模型：NGboost, LightGBM, CatBoost, XGBoost, PatchTST, TimesNet, DLinear
- 元模型：Ridge回归（alpha=1.0）

### 模型开源信息

详见 [MODEL_COMPLIANCE.md](app/code/MODEL_COMPLIANCE.md)

所有模型均为从头训练，未使用任何预训练权重。

## 特征工程（共120个特征）

1. **量价基础特征（25个）**：日涨跌幅、5/10/20/60/120日涨跌幅、振幅、量比、换手率等
2. **技术指标特征（35个）**：MA、MACD、RSI、KDJ、布林带、ATR、OBV、威廉指标、CCI、ROC、DMI、BIAS
3. **资金流特征（20个）**：大单/中单/小单净流入占比、资金流趋势等
4. **行业与宏观特征（15个）**：个股相对行业涨跌幅、宏观指标变化等
5. **时序统计特征（15个）**：收益率标准差、偏度、峰度、最大回撤、夏普比率等
6. **估值特征（5个）**：PE分位、PB分位、PEG等
7. **流动性特征（3个）**：日均成交额、日均换手率、买卖价差
8. **事件特征（2个）**：成分股调整事件、财报发布事件

## 组合优化约束

- 股票数量：1-5只
- 单只股票权重：≤0.3
- 同一申万一级行业权重：≤0.4
- 权重和：≤1
- 组合波动率：≤沪深300指数波动率的1.2倍
- 剔除ST股、停牌股、日均成交额<1亿的流动性差股票

## 训练流程

1. 加载原始数据并预处理（数据对齐、缺失值填充、极端值剔除）
2. 生成8大类120个特征
3. 时间序列滚动交叉验证（窗口6年，步长1个月）
4. 训练7个基础模型
5. 训练Stacking元模型
6. 保存所有模型到model目录

## 推理流程

1. 加载最新数据
2. 生成预测特征
3. 加载所有模型进行预测
4. Stacking集成得到最终预测收益率和不确定性
5. Black-Litterman模型进行组合优化
6. 生成符合要求的result.csv到output目录

## Docker镜像

- 基础镜像：nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
- 预计大小：8-9GB（符合≤10GB要求）
- 构建命令：docker build -t bdc2026 .
- 运行命令：docker-compose up

## 其他注意事项

- 所有随机种子已固定为42，确保完全可复现
- 复现全程无需联网
- 所有中间文件会自动清理，不占用额外空间
- 结果输出格式：UTF-8无BOM，stock_id为6位数字补前导零

## 最后更新

2026-04-22