"""
训练主程序 - 7模型 Stacking 集成
"""
import os
import sys
import pickle
import random
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import gc
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ---- 配置 ----
RANDOM_SEED = 42
SEQ_LEN = 60
PRED_HORIZON = 5
N_FOLDS = 4
MAX_THREADS = min(multiprocessing.cpu_count() or 4, 16)

# 模型超参数
GBDT_N_ESTIMATORS = 500
GBDT_LR = 0.02
GBDT_MAX_DEPTH = 6
GBDT_EARLY_STOP = 30
GBDT_SUBSAMPLE = 0.7
GBDT_COLSAMPLE = 0.7
PT_EPOCHS = 60
PT_PATIENCE = 12
PT_LR = 1e-3
D_MODEL = 128
N_HEADS = 8
E_LAYERS = 3
PATCH_LEN = 16
STRIDE = 8
N_VOL_CLUSTERS = 3
SAMPLE_WEIGHT_ALPHA = 2.0
VAL_SPLIT = 0.20

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # code/
APP_DIR = os.path.dirname(BASE_DIR)                                             # app/
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
OUTPUT_DIR = os.path.join(APP_DIR, 'output')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger('train')


def set_all_seeds(seed=RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        import lightgbm as lgb
        lgb.seed(seed)
    except Exception:
        pass
    try:
        import xgboost as xgb
        xgb.set_random_seed(seed)
    except Exception:
        pass
    try:
        import catboost as cb
        cb.set_random_seed(seed)
    except Exception:
        pass


set_all_seeds()

# ---- 导入 ----
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

torch_available = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    torch_available = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("使用CPU训练")
except Exception as e:
    logger.warning(f"PyTorch不可用: {e}")
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    device = 'cpu'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from featurework import FeatureEngineering
from data_fetcher import load_all_data, EXCLUDE_LAST_TRADING_DAYS
from tft_model import TFTModel
from spectral_m import SpectralMEnsemble


# ============================================================
# 真实模型架构实现
# ============================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022)"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self.stdev + self.mean
            return x


class PatchTSTModel(nn.Module):
    """PatchTST: 通道独立的Patch化时序Transformer (Nie et al., 2023)"""
    def __init__(self, seq_len=SEQ_LEN, pred_len=1, d_model=D_MODEL,
                 n_heads=N_HEADS, e_layers=E_LAYERS, patch_len=PATCH_LEN,
                 stride=STRIDE, n_features=100, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.d_model = d_model

        # RevIN 归一化
        self.revin = RevIN(n_features, affine=True)

        # Patch投影
        self.patch_embed = nn.Linear(patch_len * n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)

        # 输出头
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len)
        )

    def forward(self, x):
        # x: (B, seq_len, n_features)
        B = x.size(0)
        x = self.revin(x, 'norm')

        # Patch: (B, n_patches, patch_len * n_features)
        x = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, n_features, patch_len)
        x = x.permute(0, 1, 3, 2).reshape(B, self.n_patches, -1)

        x = self.patch_embed(x)  # (B, n_patches, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, d_model)
        x = x + self.pos_embed[:, :self.n_patches + 1, :]
        x = self.dropout(x)

        # Transformer
        x = self.encoder(x)
        x = self.norm(x)

        # 用CLS token预测
        x = self.head(x[:, 0, :])  # (B, pred_len)
        return x.squeeze(-1)


class DLinearModel(nn.Module):
    """DLinear: 趋势-季节分解线性模型 + 可学习特征加权 (Zeng et al., 2023)"""
    def __init__(self, seq_len=SEQ_LEN, pred_len=1, n_features=100, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.n_features = n_features

        # RevIN
        self.revin = RevIN(n_features, affine=True)

        # 移动平均核 (不可训练)
        self.ma_kernel = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        # 趋势分支: 每个特征一个线性层
        self.trend_linear = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(n_features)
        ])
        # 季节分支: 每个特征一个线性层
        self.seasonal_linear = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(n_features)
        ])

        # 可学习特征融合权重 (替代等权sum)
        self.feature_weight = nn.Linear(n_features, 1)

    def forward(self, x):
        # x: (B, seq_len, n_features)
        x = self.revin(x, 'norm')

        # 移动平均提取趋势
        x_t = x.permute(0, 2, 1)  # (B, n_features, seq_len)
        trend = self.ma_kernel(x_t)  # (B, n_features, seq_len)
        seasonal = x_t - trend

        # 各特征独立线性映射
        trend_out = torch.stack([
            self.trend_linear[i](trend[:, i, :]) for i in range(self.n_features)
        ], dim=1)  # (B, n_features, pred_len)

        seasonal_out = torch.stack([
            self.seasonal_linear[i](seasonal[:, i, :]) for i in range(self.n_features)
        ], dim=1)  # (B, n_features, pred_len)

        combined = trend_out + seasonal_out  # (B, n_features, pred_len)
        # 可学习加权融合: (B, n_features, 1) → (B, 1, 1)
        out = self.feature_weight(combined.transpose(1, 2)).squeeze(-1)  # (B, pred_len)
        return out.squeeze(-1)


class RankingMSELoss(nn.Module):
    """MSE + Pairwise Ranking混合损失 (用于排序导向训练)"""
    def __init__(self, alpha=0.3, margin=0.01):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        # Pairwise ranking loss within batch
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        mask = (target_diff.abs() > 1e-6).float()
        if mask.sum() < 1:
            return mse_loss
        rank_loss = torch.clamp(self.margin - torch.sign(target_diff) * pred_diff, min=0)
        rank_loss = (rank_loss * mask).sum() / (mask.sum() + 1e-8)
        return mse_loss + self.alpha * rank_loss


class SequenceDataset:
    """时序数据集 (支持样本权重)"""
    def __init__(self, X, y=None, seq_len=SEQ_LEN, sample_weight=None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32) if y is not None else None
        self.seq_len = seq_len
        self.sample_weight = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]
        if self.y is not None:
            y = self.y[idx + self.seq_len]
            w = self.sample_weight[idx + self.seq_len] if self.sample_weight is not None else 1.0
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)
        return torch.from_numpy(x)


# ============================================================
# 波动率聚类 & 非对称权重
# ============================================================

def compute_volatility_cluster(close_prices, n_clusters=N_VOL_CLUSTERS):
    """基于60日历史波动率将股票分为低/中/高波动簇（固定阈值）"""
    if isinstance(close_prices, np.ndarray):
        close_prices = pd.Series(close_prices.flatten())
    returns = close_prices.pct_change().dropna()
    if len(returns) < 60:
        return 1  # 数据不足，默认中波动
    vol_60d = returns.rolling(60).std().iloc[-1]
    if pd.isna(vol_60d) or vol_60d <= 0:
        return 1
    # A股典型日波动率: 低<0.018, 中0.018~0.03, 高>0.03
    if vol_60d < 0.018:
        return 0
    elif vol_60d < 0.03:
        return 1
    else:
        return 2


def compute_sample_weights(targets, alpha=SAMPLE_WEIGHT_ALPHA):
    """非对称加权：高收益样本获得更高训练权重"""
    targets = np.asarray(targets)
    target_std = np.std(targets) + 1e-8
    # 正收益且超过0.5个标准差的样本获得额外权重
    weights = np.ones_like(targets)
    high_return_mask = targets > (0.5 * target_std)
    weights[high_return_mask] = 1.0 + alpha * targets[high_return_mask] / target_std
    return np.clip(weights, 1.0, 5.0)


# ============================================================
# Stacking 集成
# ============================================================

class WeightedEnsemble:
    """可序列化的加权集成模型"""
    def __init__(self, weights):
        self.weights = np.asarray(weights)
    def predict(self, X):
        return X @ self.weights


class StackingEnsemble:
    """时间序列滚动Stacking集成"""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.early_stop_counts = {}

    def create_base_models(self):
        """创建7个基础模型（LightGBM使用lambdarank排序目标）"""
        set_all_seeds()
        device_type = 'GPU' if (torch_available and torch is not None and torch.cuda.is_available()) else 'CPU'

        # GBDT模型: 排序导向的损失函数
        self.base_models = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                max_depth=GBDT_MAX_DEPTH, num_leaves=31,
                objective='regression',  # MSE 配合排序损失效果更好
                random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
                subsample=GBDT_SUBSAMPLE, colsample_bytree=GBDT_COLSAMPLE,
                subsample_freq=1,
                reg_alpha=0.1, reg_lambda=0.5, min_child_samples=20,
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                depth=GBDT_MAX_DEPTH, random_state=RANDOM_SEED,
                loss_function='RMSE',
                verbose=0, task_type='GPU',
                early_stopping_rounds=GBDT_EARLY_STOP,
                l2_leaf_reg=3, border_count=128,
                boosting_type='Plain', bootstrap_type='Bernoulli',
                subsample=GBDT_SUBSAMPLE,
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                max_depth=GBDT_MAX_DEPTH, random_state=RANDOM_SEED,
                objective='reg:squarederror',
                tree_method='hist', device='cuda' if 'GPU' in device_type else 'cpu',
                early_stopping_rounds=GBDT_EARLY_STOP,
                subsample=GBDT_SUBSAMPLE, colsample_bytree=GBDT_COLSAMPLE,
                reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
            )
        }

        if torch_available and torch is not None:
            n_feat = 100  # 将在训练时根据实际数据调整
            self.base_models['patchtst'] = PatchTSTModel(
                seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                n_heads=N_HEADS, e_layers=E_LAYERS
            )
            self.base_models['spectralm'] = SpectralMEnsemble(
                hoat_window=60, vme_window=60, n_states=5
            )
            self.base_models['dlinear'] = DLinearModel(
                seq_len=SEQ_LEN, n_features=n_feat
            )
            self.base_models['tft'] = TFTModel(
                seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                n_heads=4, lstm_hidden=64, dropout=0.1
            )
            logger.info("7模型方案: LightGBM + CatBoost + XGBoost + SpectralM + PatchTST + DLinear + TFT")
        else:
            logger.info("3模型方案: LightGBM + CatBoost + XGBoost (PyTorch不可用)")

    def train_base_models(self, X_train, y_train, sample_weight=None, dates_train=None,
                           all_X_by_stock=None, scaler=None, exclude_days=EXCLUDE_LAST_TRADING_DAYS):
        """训练所有基础模型

        ML模型: 使用全量数据训练 (日期已排序, 验证集取最后10%时间窗口)
        DL模型: 使用 per-stock 序列训练 (避免跨股票边界污染)
        """
        set_all_seeds()
        logger.info("开始训练基础模型...")

        # ── 日期感知验证集划分 (取最后10%时间) ──
        val_cutoff = None
        if dates_train is not None:
            unique_dates = np.unique(dates_train)
            val_cutoff = unique_dates[int(len(unique_dates) * (1 - VAL_SPLIT))]
            val_mask = dates_train >= val_cutoff
            train_mask = ~val_mask
            X_tr, y_tr = X_train[train_mask], y_train[train_mask]
            X_val, y_val = X_train[val_mask], y_train[val_mask]
            sw_tr = sample_weight[train_mask] if sample_weight is not None else None
            sw_val = sample_weight[val_mask] if sample_weight is not None else None
        else:
            val_size = int(len(X_train) * VAL_SPLIT)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
            sw_tr = sample_weight[:-val_size] if sample_weight is not None else None
            sw_val = sample_weight[-val_size:] if sample_weight is not None else None

        logger.info(f"训练集: {len(X_tr)} 样本, 验证集: {len(X_val)} 样本"
                    f"({'时间截止: ' + str(val_cutoff) if val_cutoff is not None else ''})")

        # 分离ML和PyTorch模型
        ml_models = {k: v for k, v in self.base_models.items()
                     if k in ['lightgbm', 'catboost', 'xgboost']}
        pt_models = {k: v for k, v in self.base_models.items()
                     if k not in ['lightgbm', 'catboost', 'xgboost', 'spectralm']}

        # 并行训练ML模型
        def _train_ml(item):
            name, model = item
            logger.info(f"训练 {name}...")
            try:
                if name == 'lightgbm':
                    # 用RMSE早停 + 非对称样本权重
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        sample_weight=sw_tr,
                        callbacks=[lgb.early_stopping(GBDT_EARLY_STOP, verbose=False),
                                   lgb.log_evaluation(0)]
                    )
                elif name == 'catboost':
                    model.fit(X_tr, y_tr,
                             eval_set=[(X_val, y_val)],
                             sample_weight=sw_tr,
                             use_best_model=True, verbose=0)
                elif name == 'xgboost':
                    model.fit(X_tr, y_tr,
                             eval_set=[(X_val, y_val)],
                             sample_weight=sw_tr,
                             verbose=False)
                # 验证集评估
                preds = model.predict(X_val)
                mse = np.mean((preds - y_val) ** 2)
                ic, _ = spearmanr(preds, y_val)
                logger.info(f"  {name} val MSE={mse:.6f}, IC={ic:.4f}")
            except Exception as e:
                logger.error(f"  {name} 训练失败: {e}")
            return name, model

        # 串行训练避免 GPU 模型并行时显存死锁 (XGBoost/CatBoost 共用 CUDA)
        gpu_available = torch_available and torch is not None and torch.cuda.is_available()
        n_workers = 1 if gpu_available else min(len(ml_models), MAX_THREADS)
        logger.info(f"训练 {len(ml_models)} 个ML模型 (workers={n_workers})...")
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(tqdm(ex.map(_train_ml, ml_models.items()),
                              total=len(ml_models), desc="GBDT训练", unit="model"))
        for name, model in results:
            self.base_models[name] = model

        # 训练 SpectralM (AKRR) — 需要 per-stock 收益率数据
        if 'spectralm' in self.base_models:
            logger.info("训练 spectralm (HOAT+VME+SSM+AKRR)...")
            try:
                # 从训练数据的 close 价格计算收益率 (与主训练一致, 排除相同天数)
                from data_fetcher import load_all_data
                df_all, _, _ = load_all_data(exclude_last_trading_days=exclude_days)
                returns_dict = {}
                targets_dict = {}
                for sid in df_all['stock_id'].unique():
                    sub = df_all[df_all['stock_id'] == sid].sort_values('date').set_index('date')
                    if len(sub) < 100:
                        continue
                    close = sub['close'].values
                    open_p = sub['open'].values
                    rets = np.diff(np.log(close + 1e-10))
                    # 目标: T+1开盘买入 → T+5开盘卖出收益率
                    tgt = open_p[5:] / open_p[:-5] - 1
                    # 对齐
                    min_len = min(len(rets), len(tgt) - 5)
                    returns_dict[sid] = rets[-min_len-5:-5] if min_len > 0 else rets
                    targets_dict[sid] = tgt[-min_len:] if min_len > 0 else tgt
                preds = self.base_models['spectralm'].fit(returns_dict, targets_dict)
                logger.info(f"  spectralm 训练完成 ({len(preds)} 只股票有预测)")
            except Exception as e:
                logger.warning(f"  spectralm 训练失败: {e}, 从ensemble移除")
                del self.base_models['spectralm']

        # 串行训练PyTorch模型 (per-stock 序列, 避免跨股票边界)
        if torch_available and torch is not None and pt_models:
            logger.info(f"串行训练 {len(pt_models)} 个PyTorch模型 (per-stock 序列)...")

            # ── 构建 per-stock 序列 ──
            dl_X, dl_y, dl_w = [], [], []
            if all_X_by_stock and scaler is not None:
                for stock_id, (stock_X_raw, stock_y_raw) in all_X_by_stock.items():
                    if len(stock_X_raw) <= SEQ_LEN:
                        continue
                    stock_X = scaler.transform(stock_X_raw)
                    # 每只股票最后10%作为验证, 前面作为训练 (通过后续split处理)
                    for i in range(len(stock_X) - SEQ_LEN):
                        dl_X.append(stock_X[i:i + SEQ_LEN])
                        dl_y.append(stock_y_raw[i + SEQ_LEN])
                        dl_w.append(1.0)
                if len(dl_X) > 0:
                    dl_X = np.array(dl_X, dtype=np.float32)
                    dl_y = np.array(dl_y, dtype=np.float32)
                    dl_w = np.array(dl_w, dtype=np.float32)
                    # 时间感知分割: 后10% 序列做验证
                    n_val = max(1, int(len(dl_X) * VAL_SPLIT))
                    dl_X_tr, dl_X_val = dl_X[:-n_val], dl_X[-n_val:]
                    dl_y_tr, dl_y_val = dl_y[:-n_val], dl_y[-n_val:]
                    dl_w_tr = dl_w[:-n_val]
                    logger.info(f"  DL序列数据: {len(dl_X)} 条 (训练{len(dl_X_tr)}, 验证{len(dl_X_val)})")
                else:
                    dl_X_tr = dl_X_val = dl_y_tr = dl_y_val = dl_w_tr = None
                    logger.warning("  DL序列数据为空, 跳过所有PyTorch模型")
            else:
                dl_X_tr = dl_X_val = dl_y_tr = dl_y_val = dl_w_tr = None

            for name, model in pt_models.items():
                logger.info(f"训练 {name}...")
                try:
                    if dl_X_tr is None or len(dl_X_tr) < 100:
                        logger.warning(f"  {name} 序列数据不足，跳过")
                        continue

                    n_feat = dl_X_tr.shape[2]  # (n_seq, seq_len, n_features)
                    if name == 'patchtst':
                        model = PatchTSTModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                                             n_heads=N_HEADS, e_layers=E_LAYERS)
                    elif name == 'dlinear':
                        model = DLinearModel(seq_len=SEQ_LEN, n_features=n_feat)
                    elif name == 'tft':
                        model = TFTModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                                         n_heads=4, lstm_hidden=64, dropout=0.1)
                    model = model.to(device)

                    # 从 pre-built 序列创建 DataLoader
                    class PrebuiltSeqDataset(Dataset):
                        def __init__(self, X, y, weights=None):
                            self.X = torch.from_numpy(X)
                            self.y = torch.from_numpy(y)
                            self.w = torch.from_numpy(weights) if weights is not None else torch.ones(len(y))
                        def __len__(self):
                            return len(self.X)
                        def __getitem__(self, idx):
                            return self.X[idx], self.y[idx], self.w[idx]

                    train_ds = PrebuiltSeqDataset(dl_X_tr, dl_y_tr, dl_w_tr)
                    val_ds = PrebuiltSeqDataset(dl_X_val, dl_y_val)

                    if len(train_ds) < 100:
                        logger.warning(f"  {name} 序列数据不足，跳过")
                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False,
                                             num_workers=0, pin_memory=True)
                    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                                           num_workers=0, pin_memory=True)

                    criterion = nn.MSELoss()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=PT_LR, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=10, T_mult=2
                    )
                    scaler_amp = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

                    best_val_loss = float('inf')
                    best_val_ic = -float('inf')
                    patience_counter = 0
                    best_state = None

                    epoch_pbar = tqdm(range(PT_EPOCHS), desc=f"  {name}", leave=False, unit="ep")
                    for epoch in epoch_pbar:
                        model.train()
                        train_loss = 0
                        for batch_data in train_loader:
                            batch_X, batch_y = batch_data[0], batch_data[1]
                            batch_w = batch_data[2] if len(batch_data) > 2 else torch.ones_like(batch_y)
                            batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
                            optimizer.zero_grad()
                            if scaler_amp:
                                with torch.cuda.amp.autocast():
                                    out = model(batch_X)
                                    loss = criterion(out, batch_y) * batch_w.mean()
                                scaler_amp.scale(loss).backward()
                                scaler_amp.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                scaler_amp.step(optimizer)
                                scaler_amp.update()
                            else:
                                out = model(batch_X)
                                loss = criterion(out, batch_y) * batch_w.mean()
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                            train_loss += loss.item()

                        scheduler.step()

                        # 验证（同时计算Loss和IC）
                        model.eval()
                        val_loss = 0
                        val_preds_list = []
                        val_targets_list = []
                        with torch.no_grad():
                            for batch_data in val_loader:
                                batch_X, batch_y = batch_data[0], batch_data[1]
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                if scaler_amp:
                                    with torch.cuda.amp.autocast():
                                        out = model(batch_X)
                                        val_loss += criterion(out, batch_y).item()
                                else:
                                    out = model(batch_X)
                                    val_loss += criterion(out, batch_y).item()
                                val_preds_list.extend(out.cpu().numpy().flatten().tolist())
                                val_targets_list.extend(batch_y.cpu().numpy().flatten().tolist())
                        avg_val_loss = val_loss / max(len(val_loader), 1)
                        avg_train_loss = train_loss / max(len(train_loader), 1)

                        # 计算验证集IC
                        val_ic = 0.0
                        if len(val_preds_list) > 10:
                            val_ic = np.corrcoef(val_preds_list, val_targets_list)[0, 1]
                            val_ic = val_ic if np.isfinite(val_ic) else 0.0

                        epoch_pbar.set_postfix(
                            train=f"{avg_train_loss:.4f}",
                            val_loss=f"{avg_val_loss:.4f}",
                            val_ic=f"{val_ic:.4f}",
                            best_ic=f"{best_val_ic:.4f}"
                        )
                        if (epoch + 1) % 5 == 0 or epoch == 0:
                            logger.info(f"  {name} Epoch {epoch+1}/{PT_EPOCHS}, "
                                       f"train={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                                       f"val_ic={val_ic:.4f}")

                        # 以IC为主、Loss为辅的早停策略
                        is_better = (val_ic > best_val_ic + 0.001 or
                                    (abs(val_ic - best_val_ic) < 0.001 and avg_val_loss < best_val_loss * 0.999))
                        if is_better:
                            best_val_loss = min(best_val_loss, avg_val_loss)
                            best_val_ic = max(best_val_ic, val_ic)
                            patience_counter = 0
                            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        else:
                            patience_counter += 1
                            if patience_counter >= PT_PATIENCE:
                                logger.info(f"  {name} 早停 @ epoch {epoch+1} (best_ic={best_val_ic:.4f})")
                                break

                    # 恢复最佳权重
                    if best_state is not None:
                        model.load_state_dict(best_state)
                        del best_state

                    model.eval()
                    self.base_models[name] = model.cpu()
                    logger.info(f"  {name} 训练完成 (最佳 val_loss={best_val_loss:.6f}, val_ic={best_val_ic:.4f})")

                    del train_loader, val_loader, train_ds, val_ds
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                except Exception as e:
                    logger.error(f"  {name} 训练失败: {e}")
                    import traceback
                    traceback.print_exc()

        logger.info("基础模型训练完成")

    def generate_oof_predictions(self, X, y, n_folds=N_FOLDS, sample_weight=None, dates=None,
                                  stock_ids_per_row=None, all_X_by_stock=None):
        """股票分组交叉验证生成OOF预测

        按股票(而非时间)划分fold: 每折训练~75%股票, 验证~25%股票。
        每只股票在训练和验证中都保留完整时序, 确保DL模型的per-stock序列质量。
        """
        set_all_seeds()
        n_samples = len(X)
        oof_preds = np.zeros((n_samples, len(self.base_models)))

        if stock_ids_per_row is None:
            logger.warning("无 stock_ids_per_row, 回退到时间序列CV")
            return self._generate_oof_temporal(X, y, n_folds, sample_weight, dates, stock_ids_per_row)

        # 按股票分组
        unique_sids = np.unique(stock_ids_per_row)
        n_stocks = len(unique_sids)
        if n_stocks < n_folds * 2:
            logger.warning(f"股票数({n_stocks})不足, 回退到时间序列CV")
            return self._generate_oof_temporal(X, y, n_folds, sample_weight, dates, stock_ids_per_row)

        rng = np.random.RandomState(RANDOM_SEED)
        shuffled = rng.permutation(unique_sids)
        fold_stocks = np.array_split(shuffled, n_folds)

        logger.info(f"股票分组交叉验证 ({n_folds} 折, {n_stocks} 只股票, {n_samples} 样本)")

        for fold_idx in tqdm(range(n_folds), desc="CV Fold", unit="fold"):
            val_stocks = set(fold_stocks[fold_idx])
            train_stocks = list(set(unique_sids) - val_stocks)

            # 构建训练/验证的 row-level mask
            val_mask = np.isin(stock_ids_per_row, list(val_stocks))
            train_mask = ~val_mask

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            if len(train_idx) < 500 or len(val_idx) < 100:
                logger.warning(f"  Fold {fold_idx+1} 数据不足, 跳过")
                continue

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            val_sids = stock_ids_per_row[val_idx]

            logger.info(f"  Fold {fold_idx+1}/{n_folds}: train={len(train_idx)}({len(train_stocks)}股), val={len(val_idx)}({len(val_stocks)}股)")

            fold_preds = np.zeros((len(val_idx), len(self.base_models)))
            n_feat = X_tr.shape[1]

            for model_idx, name in enumerate(self.base_models.keys()):
                try:
                    model = self._create_fresh_model(name, n_feat)
                    if model is None:
                        continue

                    if name in ['lightgbm', 'catboost', 'xgboost']:
                        sw_tr = sample_weight[train_idx] if sample_weight is not None else None
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                        fold_preds[:, model_idx] = model.predict(X_val)

                    elif name == 'spectralm':
                        pass  # SpectralM 在主训练流程中完成, CV 跳过

                    elif torch_available and torch is not None:
                        model = model.to(device)
                        fold_preds[:, model_idx] = self._train_predict_dl_fold(
                            model, name, X_tr, y_tr, stock_ids_per_row[train_idx],
                            X_val, val_sids
                        )

                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"  Fold {fold_idx+1} {name} 失败: {e}")

            oof_preds[val_idx, :] = fold_preds

            if torch_available and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return oof_preds

    def _create_fresh_model(self, name, n_feat):
        dt = 'GPU' if (torch_available and torch is not None and torch.cuda.is_available()) else 'CPU'
        if name == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                max_depth=GBDT_MAX_DEPTH, num_leaves=31,
                objective='regression', random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
                subsample=GBDT_SUBSAMPLE, colsample_bytree=GBDT_COLSAMPLE,
                subsample_freq=1, reg_alpha=0.1, reg_lambda=0.5, min_child_samples=20,
            )
        elif name == 'catboost':
            return cb.CatBoostRegressor(
                iterations=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                depth=GBDT_MAX_DEPTH, random_state=RANDOM_SEED,
                loss_function='RMSE', verbose=0, task_type=dt, boosting_type='Plain',
                bootstrap_type='Bernoulli', subsample=GBDT_SUBSAMPLE,
            )
        elif name == 'xgboost':
            dt_xgb = 'cuda' if 'GPU' in dt else 'cpu'
            return xgb.XGBRegressor(
                n_estimators=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                max_depth=GBDT_MAX_DEPTH, random_state=RANDOM_SEED,
                objective='reg:squarederror', tree_method='hist', device=dt_xgb,
                subsample=GBDT_SUBSAMPLE, colsample_bytree=GBDT_COLSAMPLE,
                reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
            )
        elif name == 'patchtst':
            return PatchTSTModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                                n_heads=N_HEADS, e_layers=E_LAYERS)
        elif name == 'spectralm':
            return SpectralMEnsemble(hoat_window=60, vme_window=60, n_states=5)
        elif name == 'dlinear':
            return DLinearModel(seq_len=SEQ_LEN, n_features=n_feat)
        elif name == 'tft':
            return TFTModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                            n_heads=4, lstm_hidden=64, dropout=0.1)
        return None

    def _train_predict_dl_fold(self, model, name, X_tr, y_tr, train_sids, X_val, val_sids):
        """在一个 CV fold 内训练DL模型并预测 (per-stock 序列, 与主训练一致)"""
        # 训练序列: 从 fold 训练集的 per-stock 数据构建
        dl_X, dl_y = [], []
        for sid in np.unique(train_sids):
            mask = train_sids == sid
            stock_X = X_tr[mask]
            stock_y = y_tr[mask]
            if len(stock_X) <= SEQ_LEN:
                continue
            for i in range(len(stock_X) - SEQ_LEN):
                dl_X.append(stock_X[i:i + SEQ_LEN])
                dl_y.append(stock_y[i + SEQ_LEN])

        if len(dl_X) < 100:
            return np.zeros(len(X_val))

        dl_X = np.array(dl_X, dtype=np.float32)
        dl_y = np.array(dl_y, dtype=np.float32)
        n_val = max(1, int(len(dl_X) * VAL_SPLIT))

        class _DS(Dataset):
            def __init__(self, X_seq, y_seq):
                self.X = torch.from_numpy(X_seq)
                self.y = torch.from_numpy(y_seq)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        tr_loader = DataLoader(_DS(dl_X[:-n_val], dl_y[:-n_val]),
                             batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
        vl_loader = DataLoader(_DS(dl_X[-n_val:], dl_y[-n_val:]),
                             batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=PT_LR, weight_decay=1e-4)
        amp_scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        best_loss = float('inf')
        pat = 0
        for epoch in range(max(15, PT_EPOCHS // 3)):
            model.train()
            for batch_data in tr_loader:
                bx, by = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                if amp_scaler:
                    with torch.cuda.amp.autocast():
                        loss = criterion(model(bx), by)
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    loss = criterion(model(bx), by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            model.eval()
            vl = sum(criterion(model(batch_data[0].to(device)), batch_data[1].to(device)).item()
                    for batch_data in vl_loader) / max(len(vl_loader), 1)
            if vl < best_loss * 0.999:
                best_loss = vl
                pat = 0
            else:
                pat += 1
                if pat >= 8:
                    break

        # 预测: per-stock 序列 → 映射回 flat
        preds_col = np.zeros(len(X_val))
        model.eval()
        for sid in np.unique(val_sids):
            mask = val_sids == sid
            local_idx = np.where(mask)[0]
            stock_X = X_val[mask]
            if len(stock_X) <= SEQ_LEN:
                continue
            seqs = np.array([stock_X[i-SEQ_LEN:i] for i in range(SEQ_LEN, len(stock_X)+1)], dtype=np.float32)
            if len(seqs) == 0:
                continue
            pred_dl = DataLoader(_DS(seqs, np.zeros(len(seqs))),
                               batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
            with torch.no_grad():
                preds = []
                for batch_data in pred_dl:
                    preds.extend(model(batch_data[0].to(device)).cpu().numpy().flatten())
            preds = np.array(preds)
            for j, p in enumerate(preds):
                if SEQ_LEN + j < len(local_idx):
                    preds_col[local_idx[SEQ_LEN + j]] = p

        return preds_col

    def _generate_oof_temporal(self, X, y, n_folds, sample_weight, dates, stock_ids_per_row):
        """回退方案: 时间序列CV"""
        n_samples = len(X)
        fold_size = n_samples // n_folds
        oof_preds = np.zeros((n_samples, len(self.base_models)))

        logger.info(f"时间序列CV ({n_folds} 折, 每折{fold_size}样本)")
        for fold_idx in range(n_folds):
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples
            if val_start < 300:
                continue
            train_idx = list(range(0, val_start))
            val_idx = list(range(val_start, val_end))
            X_tr, y_tr = X[train_idx], y[train_idx]
            logger.info(f"  Fold {fold_idx+1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}")

            for model_idx, name in enumerate(self.base_models.keys()):
                if name not in ['lightgbm', 'catboost', 'xgboost']:
                    continue
                try:
                    model = self._create_fresh_model(name, X_tr.shape[1])
                    if model:
                        sw = sample_weight[train_idx] if sample_weight is not None else None
                        model.fit(X_tr, y_tr, sample_weight=sw)
                        oof_preds[val_idx, model_idx] = model.predict(X[val_idx])
                except Exception:
                    pass

        return oof_preds

    def train_meta_model(self, X_meta, y_meta):
        """训练Stacking元模型 (NNLS非负权重 + IC优化)"""
        from scipy.optimize import nnls, minimize
        set_all_seeds()
        logger.info("训练Stacking元模型...")

        n_models = X_meta.shape[1]
        model_names = list(self.base_models.keys())

        # 方案1: NNLS (MSE最小化, 非负权重)
        w_nnls, residual = nnls(X_meta, y_meta)
        w_nnls = w_nnls / (w_nnls.sum() + 1e-10)

        # 方案2: IC最大化 (SLSQP, 非负权重, 和为1)
        def neg_ic(w, X, y):
            w = np.maximum(w, 0)
            w = w / (w.sum() + 1e-10)
            pred = X @ w
            ic = np.corrcoef(pred, y)[0, 1]
            return -ic if np.isfinite(ic) else 1.0

        try:
            result = minimize(
                neg_ic, x0=np.ones(n_models) / n_models,
                args=(X_meta, y_meta),
                method='SLSQP',
                bounds=[(0, 1)] * n_models,
                constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                options={'maxiter': 500}
            )
            w_ic = np.maximum(result.x, 0)
            w_ic = w_ic / (w_ic.sum() + 1e-10)
        except Exception as e:
            logger.warning(f"IC优化失败: {e}, 使用NNLS权重")
            w_ic = w_nnls.copy()

        # 比较两种方案
        pred_nnls = X_meta @ w_nnls
        pred_ic = X_meta @ w_ic

        ic_nnls = np.corrcoef(pred_nnls, y_meta)[0, 1]
        ic_ic = np.corrcoef(pred_ic, y_meta)[0, 1]
        mse_nnls = np.mean((pred_nnls - y_meta) ** 2)
        mse_ic = np.mean((pred_ic - y_meta) ** 2)

        logger.info(f"NNLS权重: IC={ic_nnls:.4f}, MSE={mse_nnls:.6f}")
        logger.info(f"IC优化权重: IC={ic_ic:.4f}, MSE={mse_ic:.6f}")

        # 选择IC更高的方案（比赛关注排序能力）
        if ic_ic > ic_nnls:
            best_weights = w_ic
            best_method = 'IC优化'
        else:
            best_weights = w_nnls
            best_method = 'NNLS'

        logger.info(f"选用{best_method}权重 (IC={max(ic_ic, ic_nnls):.4f})")

        # 打印各模型权重
        for i, (name, w) in enumerate(zip(model_names, best_weights)):
            logger.info(f"  {name}: {w:.4f} ({'有效' if w > 0.05 else '被压缩'})")

        # 保存为可调用的模型包装器
        self.meta_model = WeightedEnsemble(best_weights)

        # 评估
        preds = self.meta_model.predict(X_meta)
        mse = np.mean((preds - y_meta) ** 2)
        ic, _ = spearmanr(preds, y_meta)
        logger.info(f"元模型 OOF MSE={mse:.6f}, IC={ic:.4f}")

    def save_models(self):
        """保存所有模型"""
        for name, model in self.base_models.items():
            if name in ['lightgbm', 'catboost', 'xgboost', 'spectralm']:
                with open(os.path.join(self.model_dir, f'{name}_model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            else:
                torch.save(model.state_dict(), os.path.join(self.model_dir, f'{name}_model.pth'))

        if self.meta_model:
            # 确保 pickle 在其他脚本加载时能找到正确模块路径
            self.meta_model.__class__.__module__ = 'train'
            with open(os.path.join(self.model_dir, 'meta_model.pkl'), 'wb') as f:
                pickle.dump(self.meta_model, f)

        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"模型已保存到 {self.model_dir}")


# ============================================================
# 主训练流程
# ============================================================

def main():
    set_all_seeds()
    logger.info("=" * 60)
    logger.info("股价预测模型 - 训练主程序")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    start_time = datetime.now()

    # 加载数据
    df, industry_df, macro_df = load_all_data(exclude_last_trading_days=EXCLUDE_LAST_TRADING_DAYS)
    logger.info(f"数据加载完成: {len(df)} 条记录, {df['stock_id'].nunique()} 只股票")

    # 特征工程
    fe = FeatureEngineering()
    stock_ids = sorted(df['stock_id'].unique())
    logger.info(f"开始特征工程 ({len(stock_ids)} 只股票)...")

    all_X, all_y, all_dates, all_clusters = [], [], [], []
    all_X_by_stock = {}   # 保留 per-stock 数据供 DL 模型训练

    def process_stock(stock_id):
        stock_df = df[df['stock_id'] == stock_id].copy()
        stock_df = stock_df.sort_values('date').set_index('date')

        features = fe.build_all_features(stock_df, industry_df, macro_df)
        features = fe.remove_outliers(features)

        close = stock_df['close']
        open_p = stock_df['open']
        # T+1开盘买入 → T+5开盘卖出的实际收益率
        target = open_p.shift(-PRED_HORIZON) / (open_p.shift(-1) + 1e-8) - 1

        # 波动率聚类（基于60日历史波动率）
        cluster_id = compute_volatility_cluster(close)

        valid_idx = features.dropna().index.intersection(target.dropna().index)
        if len(valid_idx) > 100:
            X_arr = features.loc[valid_idx].fillna(0).values
            y_arr = target.loc[valid_idx].values
            dates_arr = valid_idx.values.astype('datetime64[D]').astype(np.int64)
            cluster_arr = np.full(len(valid_idx), cluster_id, dtype=np.float32)
            return X_arr, y_arr, dates_arr, cluster_arr
        return None, None, None, None

    # 并行处理股票特征
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        results = list(tqdm(ex.map(process_stock, stock_ids),
                           total=len(stock_ids), desc="特征工程", unit="stock"))

    for stock_id, res in zip(stock_ids, results):
        if res[0] is not None:
            # 添加波动率聚类特征列 (对齐 scaler 的 130 维)
            stock_X_with_cluster = np.column_stack([res[0], res[3][:1].repeat(len(res[0]))])
            all_X.append(stock_X_with_cluster)
            all_y.append(res[1])
            all_dates.append(res[2])
            all_clusters.append(res[3])
            all_X_by_stock[stock_id] = (stock_X_with_cluster, res[1])  # (X, y) per stock

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    dates_all = np.concatenate(all_dates)
    cluster_all = np.concatenate(all_clusters)

    logger.info(f"特征矩阵: {X.shape}, 目标: {y.shape}")
    logger.info(f"波动率聚类分布: 低={np.sum(cluster_all==0)}, 中={np.sum(cluster_all==1)}, 高={np.sum(cluster_all==2)}")

    # 非对称样本权重（高收益样本获得更高训练权重）
    sample_weight = compute_sample_weights(y)
    logger.info(f"样本权重: mean={sample_weight.mean():.2f}, "
                f"高权重占比={np.mean(sample_weight > 1.5):.1%}")

    # 构建 stock_id 追踪数组 (在排序前 — 股票数据仍按 stock_id 分组)
    _stock_ids_flat = []
    for i, arr in enumerate(all_X):
        _stock_ids_flat.append(np.full(len(arr), i, dtype=np.int32))
    stock_ids_per_row = np.concatenate(_stock_ids_flat)

    # 按日期排序 → 修复 train/val split 仅覆盖字母序末尾股票的偏差
    date_order = np.argsort(dates_all)
    X = X[date_order]
    y = y[date_order]
    dates_all = dates_all[date_order]
    sample_weight = sample_weight[date_order]
    stock_ids_per_row = stock_ids_per_row[date_order]
    logger.info(f"数据已按日期排序: {pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(dates_all[0]))} ~ "
                f"{pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(dates_all[-1]))}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练
    stacking = StackingEnsemble()
    stacking.scaler = scaler
    stacking.create_base_models()
    stacking.train_base_models(X_scaled, y, sample_weight=sample_weight, dates_train=dates_all,
                               all_X_by_stock=all_X_by_stock, scaler=scaler)

    # OOF 交叉验证预测
    logger.info("生成OOF预测...")
    oof_preds = stacking.generate_oof_predictions(X_scaled, y,
                                                   sample_weight=sample_weight,
                                                   dates=dates_all,
                                                   stock_ids_per_row=stock_ids_per_row,
                                                   all_X_by_stock=all_X_by_stock)

    # 训练元模型
    stacking.train_meta_model(oof_preds, y)

    # 模型评估
    logger.info("\n" + "=" * 60)
    logger.info("模型评估结果")
    logger.info("=" * 60)

    for i, (name, model) in enumerate(stacking.base_models.items()):
        mask = oof_preds[:, i] != 0
        if mask.sum() > 100:
            logger.info(f"\n{name}:")
            evaluate_predictions(y[mask], oof_preds[mask, i])

    # 元模型预测
    meta_preds = stacking.meta_model.predict(oof_preds)
    logger.info(f"\nStacking Ensemble:")
    evaluate_predictions(y, meta_preds)

    # 早停统计
    if stacking.early_stop_counts:
        logger.info("\n早停统计:")
        for m, c in stacking.early_stop_counts.items():
            logger.info(f"  {m}: {c} 次")

    # 保存模型
    stacking.save_models()

    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    logger.info("=" * 60)
    logger.info(f"训练完成! 总耗时: {elapsed:.2f} 小时")
    logger.info("=" * 60)

    # ── SHAP 因子重要性分析 (LightGBM) ──
    try:
        from shap_analysis import run_shap_analysis
        # 采样背景数据
        n_bg = min(2000, len(X_scaled))
        rng = np.random.RandomState(RANDOM_SEED)
        bg_indices = rng.choice(len(X_scaled), n_bg, replace=False)
        X_bg = X_scaled[bg_indices]

        # 获取特征名：用第一只股票的特征列
        first_stock = sorted(df['stock_id'].unique())[0]
        first_df = df[df['stock_id'] == first_stock].sort_values('date').set_index('date')
        first_features = fe.build_all_features(first_df, industry_df, macro_df)
        feature_names = list(first_features.columns) + ['volatility_cluster']
        # 截断或填充到 X 的列数
        if len(feature_names) < X_bg.shape[1]:
            feature_names += [f'feature_{i}' for i in range(len(feature_names), X_bg.shape[1])]
        feature_names = feature_names[:X_bg.shape[1]]

        lgb_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
        shap_result = run_shap_analysis(lgb_path, X_bg, feature_names, OUTPUT_DIR)
        logger.info(f"SHAP分析完成, top-3: {', '.join(f['feature'] + '=' + str(round(f['shap_importance'], 4)) for f in shap_result[:3])}")
    except Exception as e:
        logger.warning(f"SHAP分析跳过: {e}")

    return True


# 内联评估函数（避免循环import）
def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 10:
        return
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)
    ic, _ = spearmanr(y_true, y_pred)
    logger.info(f"  MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.4f}, IC={ic:.4f}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
