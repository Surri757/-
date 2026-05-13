# 2026大数据挑战赛 — 沪深300量化预测方案

## 环境配置

### 硬件
- CPU：i7-13650H（14核20线程）
- 内存：16GB
- GPU：NVIDIA GeForce RTX 4060 Laptop 8GB（Ada Lovelace, Compute Capability 8.9）
- 存储：50GB

### 系统
- OS：Ubuntu 22.04（Docker 基础镜像 `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`）
- Python：3.10.14
- CUDA：12.1
- cuDNN：8

### Python 依赖（精确版本，跨机器复现必须一致）

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
lightgbm==4.3.0
catboost==1.2.5
xgboost==2.0.3
torch==2.2.1
torchvision==0.17.1
torchaudio==2.2.1
scipy==1.13.1
statsmodels==0.14.2
matplotlib==3.8.5
shap>=0.44.0
tqdm
```

- 训练时间：约0.58小时（35分钟）
- 预测时间：约2分钟
- **跨机器复现前提：请确保上述 Python 依赖版本完全一致，否则 cuDNN/cuBLAS 底层实现和算法行为差异可能导致数值偏差**

## 可复现性保证

**同一份代码 + 固定 seed=42 + 相同软件环境 → 任意机器上训练过程和推理结果位级一致。**

### 确定性措施

| 层级 | 措施 | 说明 |
|------|------|------|
| Python | `PYTHONHASHSEED=42` | 消除 dict/set 遍历顺序不确定性 |
| NumPy/random | `np.random.seed(42)` + `random.seed(42)` | 所有随机数生成器固定 |
| LightGBM | `random_state=42` | GBDT 分裂点固定 |
| XGBoost | `random_state=42` + `tree_method='hist'` | GPU hist 在固定 seed 下可复现 |
| CatBoost | `random_state=42` + **`task_type='CPU'`** | **GPU 原子操作不可控，必须 CPU** |
| PyTorch | `torch.manual_seed(42)` + `cuda.manual_seed_all(42)` | 权重初始化固定 |
| cuDNN | `cudnn.deterministic=True` + `cudnn.benchmark=False` | 禁用启发式算法选择 |
| cuBLAS | `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 固定矩阵乘法累加顺序（import torch 前设置） |
| CUDA 算子 | `torch.use_deterministic_algorithms(True, warn_only=True)` | 强制 scatter/index_add 等走确定性路径 |
| sklearn PCA | `svd_solver='full'` | 禁用 randomized SVD，避免非确定性 |
| sklearn KMeans | `random_state=42` + `n_init=10` | 聚类中心固定 |
| DataLoader | `shuffle=False` + `num_workers=0` | 数据顺序固定，无子进程随机 |
| 数据 | 本地 CSV 缓存，离线运行 | 相同文件 → 相同输入 |

### 跨机器一致性前提

- 同一份代码和 `data/` 目录下的 CSV 缓存文件
- 相同的 PyTorch/CUDA/cuDNN 版本（见[环境配置](#环境配置)）
- 相同的 GPU 架构（Ada Lovelace / RTX 4060）

满足以上条件时，训练过程（loss 曲线、模型权重）和推理结果（result.csv）与提交结果位级一致。

## 数据来源

1. **baostock**：沪深300成分股日线数据（OHLCV + PE/PB），动态365天窗口
2. **AKShare**：宏观经济数据（CPI/PPI/PMI/M2/LPR/汇率），7天缓存有效期
3. 数据已缓存至本地 `data/` 目录，纯离线运行
4. 防信息泄露：排除最近30个交易日（5日预测仅需5日缓冲，30日留足余量）

## 架构总览

```
数据层(365天动态窗口,排除30天)
  → 特征工程(~130维)
    → 7模型Stacking集成
      → 信号生成(横截面增强)
        → Gate动态准入(SSM谱状态机)
          → Cascade多层筛选(黑名单→活跃度→分组→Stage1→Stage2→复活)
            → 组合优化(max-Sharpe)
              → result.csv
```

## 模型：7模型 Stacking 集成

| 模型 | 类型 | 参数量 | OOF IC | 集成权重 | 角色 |
|------|------|--------|--------|---------|------|
| XGBoost | GBDT | 500树×6层 | 0.419 | 主力 | 排序信号 |
| LightGBM | GBDT | 500树×6层 | 0.394 | 主力 | 排序信号 |
| CatBoost | GBDT (CPU) | 500树×6层 | 0.333 | 辅助 | 排序信号（CPU训练保证可复现） |
| PatchTST | Patch+Transformer | ~500K | 0.258 | 补充 | 时序模式 |
| DLinear | 线性趋势分解 | ~50K | 0.354 | 补充 | 复活赛裁判 |
| TFT | LSTM+Attention+GRN | ~128K | 0.297 | 补充 | 复活赛裁判 |
| SpectralM | HOAT+VME+SSM+AKRR | — | — | 谱信号 | 市场状态感知 |
| **Stacking Ensemble** | IC优化加权 | — | **0.453** | — | 最终决策 |

### 关键训练配置

- 目标：T+1开盘买入 → T+5开盘卖出（open-to-open）
- 元模型：NNLS（MSE最小化） vs IC优化（Spearman最大化），选用IC更高的
- 损失函数：MSE，GBDT用RMSE/regression，样本权重alpha=2.0
- 批量大小：DL 512，ML全量
- 验证策略：20%时间切分（按日期排序的最后20%作验证集）
- 交叉验证：股票分组4折（每折训练225只股票全量时序，验证75只未见股票）

### GBDT 超参数

| 参数 | LightGBM | CatBoost | XGBoost |
|------|----------|----------|---------|
| n_estimators | 500 | 500 | 500 |
| learning_rate | 0.02 | 0.02 | 0.02 |
| max_depth | 6 | 6 | 6 |
| subsample | 0.7 | 0.7 | 0.7 |
| colsample | 0.7 | — | 0.7 |
| objective | regression | RMSE | reg:squarederror |
| early_stopping | 30 | 30 | 30 |

### DL 超参数

| 参数 | PatchTST | DLinear | TFT |
|------|----------|---------|-----|
| d_model | 128 | — | 64 |
| n_heads | 8 | — | 4 |
| e_layers | 3 | — | — |
| lstm_hidden | — | — | 64 |
| epochs | 60 | 60 | 60 |
| patience | 12 | 12 | 12 |
| lr | 1e-3 | 1e-3 | 1e-3 |

## 特征工程（~130维 + 波动率聚类 = 131列）

1. **量价基础**（24维）：日涨跌幅、多周期收益率、振幅、量比、换手率、量价交叉
2. **技术指标**（35维）：MA/MACD/RSI/KDJ/布林带/ATR/OBV/Williams%R/CCI/ROC/DMI
3. **Alpha因子**（9维）：Chaikin资金流、VWAP偏离、Amihud非流动性、MAX效应、隔夜缺口、VPT、EOM、量价背离、Range-Volume
4. **行为金融情绪**（15维）：FOMO突破、追涨强度、换手率加速、恐慌抛售、成交量拐点、恐慌指数、牛熊比、上下影线、开盘情绪、动量加速、量价共振、锚定偏差、彩票偏好、极端收益
5. **行业与宏观**（15维）：行业收益/PE/PB/换手率、CPI/PPI/PMI/M2/LPR/汇率及变化率
6. **时序统计**（15维）：滚动std/skew/kurtosis、最大回撤、Sharpe比率、波动率变化
7. **估值**（3维）：PE/PB分位数、PEG
8. **流动性**（3维）：日均成交额、平均换手率、买卖价差
9. **事件**（1维）：财报季标识

## 推理流程：多层筛选 (Cascade Pipeline)

```
300只信号
  ├─ [特征缓存] 预计算所有股票特征矩阵（避免各阶段重复计算）
  ├─ [永久黑名单] 87只压舱石（银行/石油/保险/运营商/基建/公用事业）→ 直接淘汰
  ├─ [动态活跃度] 30日成交量<历史60%的票 → 暂时拉黑（放量可复活）
  ├─ [因子画像分组] 波动率×动量×估值 → 6-8组
  ├─ [Stage 1 粗筛] 各组用不同模型打分 → z-score排名 → 收益下限0.8% → 保留~50%
  ├─ [Stage 2 精选] 换一套模型交叉验证 → z-score排名 → 收益下限0.6% → 保留~30%
  ├─ [复活赛] TFT+DLinear 救回被误杀的爆发票（最多5只）
  └─ [决赛] 全量7模型Stacking + 波动率折扣 + max-Sharpe组合优化 → 5只 → result.csv
```

### 分组模型配置

| 分组 | Stage 1 模型 | Stage 2 模型 |
|------|-------------|-------------|
| highvol_momentum | TFT + PatchTST | CatBoost + DLinear |
| highvol_reversal | TFT + XGBoost | LightGBM + TFT |
| midvol_growth | LightGBM + DLinear | TFT + XGBoost |
| midvol_mixed | CatBoost + LightGBM | DLinear + XGBoost |
| midvol_value | CatBoost + LightGBM | XGBoost + DLinear |
| lowvol_growth | TFT + CatBoost | DLinear + LightGBM |
| lowvol_value | LightGBM + XGBoost | CatBoost + DLinear |

## Gate 动态准入（含 SSM 谱状态机）

5项校验，阈值根据市场状态自适应调整：

| 市场状态 | 置信度门槛 | 最大持仓 | 风险距离 |
|---------|-----------|---------|---------|
| 牛 (bull) | 0.05 | 5 | 1.5% |
| 震荡 (sideways) | 0.10 | 5 | 2.0% |
| 中性 (neutral) | 0.10 | 5 | 2.0% |
| 熊 (bear) | 0.15 | 4 | 2.5% |
| 恐慌 (panic) | 0.20 | 3 | 3.0% |

- 市场状态检测为共享模块级函数 `detect_market_regime()`，Gate 和主流程共用
- 基于5日/20日/60日收益率 + 波动率变化 + 预测质量综合判定
- **SSM 谱状态机**：7维市场特征向量 → 5隐状态，基于预定义状态中心的马氏距离分配
- 预测质量高时额外放宽门槛；趋势强时持仓上限浮动

## SpectralM 模块

自研谱分析模块，四个子组件组合：

- **HOAT** (高阶自相关张量)：提取偏度/峰度/交叉矩特征，PCA降维至8维
- **VME** (波动率流形嵌入)：局部协方差特征分解，对数特征值+主向量+PCA降维至6维
- **SSM** (谱状态机)：自适应相似度(余弦×RBF) → 图拉普拉斯 → 谱聚类 → 5隐状态+软概率
- **AKRR** (自适应核岭回归)：每个隐状态下独立KRR模型，马氏-RBF核，Nyström加速(>3000样本)

接口兼容 Stacking ensemble，训练时逐股票提取特征后合并训练，推理时单股票预测。

## 组合优化

- 方法：`pred_weighted` 信念加权（w_i ∝ exp(μ_i / T)，温度自适应）
- 备选：max-Sharpe / min-variance / max-return-risk-budget（scipy SLSQP约束优化）
- 约束：sum(w)=1.0（满仓）、w_i≥0、持仓≤5只、单只≤1.0
- 协方差矩阵：优先从60日历史价格估计（Ledoit-Wolf收缩），回退到模型分歧+常数相关
- 决赛阶段对低波票预测收益施加0.7x折扣，导向高活性选股
- 迭代投影确保 clip+renormalize 后所有权重 ≤ max_single

## 组合约束

- 股票数量：≤5只（硬上限）
- 总权重：=1.0（满仓）
- 单只权重：≤1.0
- 止损宽度：根据市场状态动态调整（牛1.5x / 恐慌0.7x）

## 训练流程

1. 加载365天动态数据，排除最近30个交易日
2. 并行特征工程（300只股票，ThreadPoolExecutor，tqdm进度条）
3. 全局 StandardScaler 标准化（130特征 + 波动率聚类 = 131列）
4. 训练3个GBDT模型（GPU可用时串行避免显存竞争，无GPU时并行）+ SpectralM
5. 串行训练3个PyTorch模型（per-stock序列，避免跨股票边界污染）
6. 股票分组4折交叉验证生成OOF预测
7. 训练Stacking元模型（NNLS vs IC优化，选IC更高的）
8. SHAP因子重要性分析（LightGBM TreeExplainer → top-20 PNG+JSON）
9. 保存所有模型到 `model/` 目录

## 输出格式

- `result.csv`：UTF-8，stock_id为6位补零，含stock_id和weight两列，权重和=1.0
- `review_YYYYMMDD_HHMMSS.txt`：可读复盘报告（Gate拒绝/执行/风险事件/最终组合）
- `review_YYYYMMDD_HHMMSS.json`：结构化复盘数据
- `shap_importance.png` / `shap_importance.json`：SHAP因子重要性

## 用法

```bash
# 环境初始化（安装依赖）
bash init.sh

# 训练（拉数据 + 训练模型 + SHAP）
bash train.sh

# 预测
bash test.sh                         # 多层筛选+复活赛（默认）
python src/test.py --mode simple     # 单层流水线（快速baseline），需先 export PYTHONHASHSEED=42

# Docker
docker build -t bdc2026 .
docker-compose up
```

> `train.sh` / `test.sh` 内已设置 `export PYTHONHASHSEED=42`，直接 `python src/train.py` 会丢失此环境变量，建议始终通过 shell 脚本运行。

## 项目结构

```
bdc2026/
├── Dockerfile                    # CUDA 12.1 + Python 3.10 + 全部依赖
├── docker-compose.yml            # GPU容器编排
├── app/
│   ├── readme.md                 # 本文档
│   ├── code/
│   │   ├── requirements.txt      # Python依赖
│   │   ├── init.sh               # 环境初始化
│   │   ├── train.sh              # 训练入口
│   │   ├── test.sh               # 预测入口
│   │   ├── data/                 # 缓存数据
│   │   │   ├── stock_data.csv    # 沪深300日线
│   │   │   ├── index_data.csv    # 指数数据
│   │   │   ├── industry_cache.csv
│   │   │   └── macro_cache.csv
│   │   ├── src/
│   │   │   ├── train.py          # 训练主程序（1267行）
│   │   │   ├── test.py           # 预测主程序
│   │   │   ├── data_fetcher.py   # 多源数据加载
│   │   │   ├── featurework.py    # 特征工程（9大类~130维）
│   │   │   ├── spectral_m.py     # SpectralM模块
│   │   │   ├── tft_model.py      # 轻量TFT模型定义
│   │   │   ├── cascade_pipeline.py  # 多层筛选流水线
│   │   │   ├── gate.py           # 动态准入校验层
│   │   │   ├── signals.py        # 信号生成（含横截面增强）
│   │   │   ├── portfolio_optimizer.py  # 组合优化
│   │   │   ├── risk_manager.py   # 风控管理
│   │   │   ├── executor.py       # 执行模拟
│   │   │   ├── review.py         # 复盘报告
│   │   │   ├── shap_analysis.py  # SHAP分析
│   │   │   └── model/            # 保存的模型文件（9个）
│   │   └── output/               # 输出（result.csv + 复盘 + SHAP）
│   └── output/                   # Docker挂载同目录
└── .gitignore
```

## Docker镜像

- 基础镜像：nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
- 预计大小：8-9GB（≤10GB）
- 构建：`docker build -t bdc2026 .`
- 运行：`docker-compose up`（自动执行 init → train → test）
- 离线运行：依赖包在Dockerfile中安装，数据通过COPY打包，无需外网

## 创新点

1. **SpectralM 谱分析模块**：HOAT高阶矩 + VME波动率流形 + SSM谱状态机 + AKRR核岭回归，从收益率分布和波动率结构中提取市场微观状态信号
2. **股票分组交叉验证**：按股票（非时间）划分CV折，每折训练全量时序、验证未见股票，真实反映模型泛化能力
3. **多层因子画像筛选**：波动率×动量×估值分组，每组用不同模型组合，组内z-score跨组可比
4. **复活赛机制**：TFT + DLinear 专救被树模型误判的爆发票
5. **动态活跃度过滤**：30日量比<60%暂时拉黑，放量自动复活
6. **SSM谱状态机**：图聚类识别5种市场微观隐状态，动态调节Gate阈值
7. **轻量TFT**：128K参数自研Temporal Fusion Transformer，RevIN + LSTM Encoder/Decoder + Multi-head Attention + GRN
8. **特征缓存优化**：Cascade管道一次性预计算所有股票特征，各阶段复用，避免~1200次重复计算
9. **迭代投影权重**：组合优化clip+renormalize循环投影，确保约束严格满足

## 最后更新

2026-05-13
