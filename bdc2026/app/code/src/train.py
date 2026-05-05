"""
训练主程序 - 6模型 Stacking 集成 (修复版)
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

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ---- 配置 ----
RANDOM_SEED = 42
SEQ_LEN = 60
PRED_HORIZON = 5
N_FOLDS = 6
MAX_THREADS = min(multiprocessing.cpu_count() or 4, 16)

# 模型超参数
GBDT_N_ESTIMATORS = 200
GBDT_LR = 0.05
GBDT_MAX_DEPTH = 6
GBDT_EARLY_STOP = 20
PT_EPOCHS = 50
PT_PATIENCE = 10
PT_LR = 1e-3
D_MODEL = 128
N_HEADS = 8
E_LAYERS = 3
PATCH_LEN = 16
STRIDE = 8
N_VOL_CLUSTERS = 3       # 波动率聚类数
SAMPLE_WEIGHT_ALPHA = 3.0  # 非对称权重: 高收益样本加权系数

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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
from data_fetcher import load_all_data
from monitor import TrainingMonitor


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


class TimesNetModel(nn.Module):
    """TimesNet: 多周期时序分解网络 (Wu et al., 2023)"""
    def __init__(self, seq_len=SEQ_LEN, pred_len=1, d_model=D_MODEL,
                 e_layers=2, n_features=100, top_k=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.d_model = d_model
        self.n_features = n_features

        # RevIN
        self.revin = RevIN(n_features, affine=True)

        # FFT周期发现 → 2D卷积
        self.conv_blocks = nn.ModuleList()
        for i in range(e_layers):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(n_features, d_model, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Conv2d(d_model, n_features, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(n_features),
            ))

        # 输出
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(seq_len * n_features, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len)
        )

    def _fft_periods(self, x):
        """FFT找出top-k周期 + 固定备用周期"""
        fixed_periods = [5, 10, 20]
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs().mean(dim=0).mean(dim=-1)
        amp[0] = 0
        freq_len = amp.size(0)
        _, top_indices = torch.topk(amp[1:min(freq_len - 1, 31)], min(self.top_k, freq_len - 2))
        fft_periods = []
        for idx in top_indices:
            period = self.seq_len / (idx.item() + 1)
            period = max(2, min(int(round(period)), self.seq_len // 2))
            fft_periods.append(period)
        # 合并FFT发现的周期和固定备用周期
        all_periods = list(dict.fromkeys(fft_periods + fixed_periods))
        return all_periods[:self.top_k]

    def forward(self, x):
        B, T, C = x.shape
        x = self.revin(x, 'norm')

        periods = self._fft_periods(x)
        if not periods:
            periods = [7, 14, 30]

        # 对每个周期做2D reshape + 2D conv
        outputs = []
        for period in periods:
            # Padding使长度可被period整除
            pad_len = (period - T % period) % period
            if pad_len > 0:
                x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            else:
                x_pad = x
            # Reshape: (B, period, T//period, C) → (B, C, period, T//period)
            x_2d = x_pad.reshape(B, -1, period, C).permute(0, 3, 2, 1)

            # 2D conv blocks
            for conv_block in self.conv_blocks:
                x_2d = conv_block(x_2d) + x_2d  # 残差

            # 转回1D
            x_1d = x_2d.permute(0, 3, 2, 1).reshape(B, -1, C)
            x_1d = x_1d[:, :T, :]  # 截断回原始长度
            outputs.append(x_1d)

        # 聚合多周期结果
        x = torch.stack(outputs, dim=-1).mean(dim=-1)  # (B, T, C)

        # Flatten + predict
        x = x.reshape(B, -1)
        x = self.head(x)
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
        """创建6个基础模型（LightGBM使用lambdarank排序目标）"""
        set_all_seeds()
        device_type = 'GPU' if (torch_available and torch is not None and torch.cuda.is_available()) else 'CPU'

        # GBDT模型: 排序导向的损失函数
        self.base_models = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                max_depth=GBDT_MAX_DEPTH, num_leaves=31,
                objective='mae',  # MAE比MSE更鲁棒，排序效果更好
                random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                depth=GBDT_MAX_DEPTH, random_state=RANDOM_SEED,
                loss_function='MAE',  # MAE for better ranking
                verbose=0, task_type=device_type,
                early_stopping_rounds=GBDT_EARLY_STOP,
                l2_leaf_reg=3, border_count=128,
                boosting_type='Plain'
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                max_depth=GBDT_MAX_DEPTH, random_state=RANDOM_SEED,
                objective='reg:squarederror',
                tree_method='hist', device='cuda' if 'GPU' in device_type else 'cpu',
                early_stopping_rounds=GBDT_EARLY_STOP,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0
            )
        }

        if torch_available and torch is not None:
            n_feat = 100  # 将在训练时根据实际数据调整
            self.base_models['patchtst'] = PatchTSTModel(
                seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                n_heads=N_HEADS, e_layers=E_LAYERS
            )
            self.base_models['timesnet'] = TimesNetModel(
                seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL, e_layers=2
            )
            self.base_models['dlinear'] = DLinearModel(
                seq_len=SEQ_LEN, n_features=n_feat
            )
            logger.info("6模型方案: LightGBM + CatBoost + XGBoost + PatchTST + TimesNet + DLinear")
        else:
            logger.info("3模型方案: LightGBM + CatBoost + XGBoost (PyTorch不可用)")

    def train_base_models(self, X_train, y_train, sample_weight=None, dates_train=None, monitor=None):
        """训练所有基础模型（支持排序目标和样本权重）"""
        set_all_seeds()
        logger.info("开始训练基础模型...")

        val_size = int(len(X_train) * 0.1)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]

        sw_tr = sample_weight[:-val_size] if sample_weight is not None else None
        sw_val = sample_weight[-val_size:] if sample_weight is not None else None
        dt_tr = dates_train[:-val_size] if dates_train is not None else None
        dt_val = dates_train[-val_size:] if dates_train is not None else None

        logger.info(f"训练集: {len(X_tr)} 样本, 验证集: {len(X_val)} 样本")

        # 分离ML和PyTorch模型
        ml_models = {k: v for k, v in self.base_models.items()
                     if k in ['lightgbm', 'catboost', 'xgboost']}
        pt_models = {k: v for k, v in self.base_models.items()
                     if k not in ['lightgbm', 'catboost', 'xgboost']}

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

        logger.info(f"并行训练 {len(ml_models)} 个ML模型...")
        with ThreadPoolExecutor(max_workers=min(len(ml_models), MAX_THREADS)) as ex:
            results = list(ex.map(_train_ml, ml_models.items()))
        for name, model in results:
            self.base_models[name] = model

        # 串行训练PyTorch模型
        if torch_available and torch is not None and pt_models:
            logger.info(f"串行训练 {len(pt_models)} 个PyTorch模型...")
            for name, model in pt_models.items():
                logger.info(f"训练 {name}...")
                try:
                    # 根据实际特征数重建模型
                    n_feat = X_tr.shape[1]
                    if name == 'patchtst':
                        model = PatchTSTModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                                             n_heads=N_HEADS, e_layers=E_LAYERS)
                    elif name == 'timesnet':
                        model = TimesNetModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL, e_layers=2)
                    elif name == 'dlinear':
                        model = DLinearModel(seq_len=SEQ_LEN, n_features=n_feat)
                    model = model.to(device)

                    train_ds = SequenceDataset(X_tr, y_tr, SEQ_LEN, sw_tr)
                    val_ds = SequenceDataset(X_val, y_val, SEQ_LEN)

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

                    for epoch in range(PT_EPOCHS):
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

                        if (epoch + 1) % 5 == 0 or epoch == 0:
                            logger.info(f"  {name} Epoch {epoch+1}/{PT_EPOCHS}, "
                                       f"train={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                                       f"val_ic={val_ic:.4f}")

                        # 记录到监控器
                        if monitor is not None:
                            monitor.log_pt_epoch(name, epoch + 1, avg_train_loss, avg_val_loss, val_ic)

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
                    if 'best_state' in dir():
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

    def generate_oof_predictions(self, X, y, n_folds=N_FOLDS, sample_weight=None, dates=None, monitor=None):
        """时间序列滚动交叉验证生成OOF预测（支持排序目标和样本权重）"""
        set_all_seeds()
        n_samples = len(X)
        fold_size = n_samples // n_folds
        oof_preds = np.zeros((n_samples, len(self.base_models)))

        def create_fresh_model(name, n_feat):
            if name == 'lightgbm':
                return lgb.LGBMRegressor(
                    n_estimators=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                    max_depth=GBDT_MAX_DEPTH, num_leaves=31,
                    objective='mae', random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
                    subsample=0.8, colsample_bytree=0.8
                )
            elif name == 'catboost':
                dt = 'GPU' if (torch_available and torch is not None and torch.cuda.is_available()) else 'CPU'
                return cb.CatBoostRegressor(
                    iterations=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                    depth=GBDT_MAX_DEPTH, random_state=RANDOM_SEED,
                    loss_function='MAE', verbose=0, task_type=dt, boosting_type='Plain'
                )
            elif name == 'xgboost':
                dt = 'cuda' if (torch_available and torch is not None and torch.cuda.is_available()) else 'cpu'
                return xgb.XGBRegressor(
                    n_estimators=GBDT_N_ESTIMATORS, learning_rate=GBDT_LR,
                    max_depth=GBDT_MAX_DEPTH, random_state=RANDOM_SEED,
                    objective='reg:squarederror', tree_method='hist', device=dt
                )
            elif name == 'patchtst':
                return PatchTSTModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                                    n_heads=N_HEADS, e_layers=E_LAYERS)
            elif name == 'timesnet':
                return TimesNetModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL, e_layers=2)
            elif name == 'dlinear':
                return DLinearModel(seq_len=SEQ_LEN, n_features=n_feat)
            return None

        logger.info(f"时间序列滚动交叉验证 ({n_folds} 折)...")
        min_train = SEQ_LEN * 2  # 最小训练样本数
        for fold_idx in range(n_folds):
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples

            if val_start < min_train:
                logger.warning(f"  Fold {fold_idx+1} 训练集不足(需要>{min_train}, 当前{val_start})，跳过")
                continue

            train_idx = list(range(0, val_start))
            val_idx = list(range(val_start, val_end))

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            logger.info(f"  Fold {fold_idx+1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}")

            fold_preds = np.zeros((len(val_idx), len(self.base_models)))
            n_feat = X_tr.shape[1]

            for model_idx, name in enumerate(self.base_models.keys()):
                try:
                    model = create_fresh_model(name, n_feat)
                    if model is None:
                        continue

                    if name in ['lightgbm', 'catboost', 'xgboost']:
                        sw_tr = sample_weight[train_idx] if sample_weight is not None else None
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                        fold_preds[:, model_idx] = model.predict(X_val)
                    elif torch_available and torch is not None:
                        model = model.to(device)
                        sw_fold = sample_weight[train_idx] if sample_weight is not None else None
                        train_ds = SequenceDataset(X_tr, y_tr, SEQ_LEN, sw_fold)
                        if len(train_ds) < 100:
                            continue

                        val_ds_size = max(1, int(len(train_ds) * 0.1))
                        train_sub = torch.utils.data.Subset(train_ds, range(len(train_ds) - val_ds_size))
                        val_sub = torch.utils.data.Subset(train_ds, range(len(train_ds) - val_ds_size, len(train_ds)))

                        tr_loader = DataLoader(train_sub, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
                        vl_loader = DataLoader(val_sub, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

                        criterion = RankingMSELoss(alpha=0.2)
                        optimizer = torch.optim.Adam(model.parameters(), lr=PT_LR)
                        scaler_amp = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

                        best_loss = float('inf')
                        pat = 0
                        pt_cv_epochs = max(15, PT_EPOCHS // 3)  # CV中足够epoch保证质量
                        for epoch in range(pt_cv_epochs):
                            model.train()
                            for batch_data in tr_loader:
                                bx, by = batch_data[0], batch_data[1]
                                bw = batch_data[2] if len(batch_data) > 2 else torch.ones_like(by)
                                bx, by, bw = bx.to(device), by.to(device), bw.to(device)
                                optimizer.zero_grad()
                                if scaler_amp:
                                    with torch.cuda.amp.autocast():
                                        out = model(bx)
                                        loss = criterion(out, by) * bw.mean()
                                    scaler_amp.scale(loss).backward()
                                    scaler_amp.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    scaler_amp.step(optimizer)
                                    scaler_amp.update()
                                else:
                                    loss = criterion(model(bx), by) * bw.mean()
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

                        # 预测
                        val_ds = SequenceDataset(X_val, seq_len=SEQ_LEN)
                        if len(val_ds) > 0:
                            vl_pred = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
                            model.eval()
                            preds = []
                            with torch.no_grad():
                                for bx in vl_pred:
                                    preds.extend(model(bx.to(device)).cpu().numpy().flatten())
                            preds = np.array(preds)
                            start = len(val_idx) - len(preds)
                            if start >= 0:
                                fold_preds[start:, model_idx] = preds
                            else:
                                fold_preds[:, model_idx] = preds[:len(val_idx)]

                        del model, tr_loader, vl_loader, train_ds
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"  Fold {fold_idx+1} {name} 失败: {e}")

            oof_preds[val_idx, :] = fold_preds

            # 记录CV折信息到监控器
            if monitor is not None and len(val_idx) > 10:
                model_ics = {}
                y_fold = y[val_idx]
                for mi, mname in enumerate(self.base_models.keys()):
                    mask = fold_preds[:, mi] != 0
                    if mask.sum() > 10:
                        ic = np.corrcoef(y_fold[mask], fold_preds[mask, mi])[0, 1]
                        model_ics[mname] = float(ic) if np.isfinite(ic) else 0.0
                monitor.log_cv_fold(fold_idx + 1, len(train_idx), len(val_idx), model_ics)

            if torch_available and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

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
            if name in ['lightgbm', 'catboost', 'xgboost']:
                with open(os.path.join(self.model_dir, f'{name}_model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            else:
                torch.save(model.state_dict(), os.path.join(self.model_dir, f'{name}_model.pth'))

        if self.meta_model:
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

    # 初始化监控器
    monitor = TrainingMonitor()

    # 加载数据
    df, industry_df, macro_df = load_all_data(exclude_last_trading_days=120)
    logger.info(f"数据加载完成: {len(df)} 条记录, {df['stock_id'].nunique()} 只股票")

    # 特征工程
    fe = FeatureEngineering()
    stock_ids = sorted(df['stock_id'].unique())
    logger.info(f"开始特征工程 ({len(stock_ids)} 只股票)...")

    all_X, all_y, all_dates, all_clusters = [], [], [], []

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
        results = list(ex.map(process_stock, stock_ids))

    for res in results:
        if res[0] is not None:
            all_X.append(res[0])
            all_y.append(res[1])
            all_dates.append(res[2])
            all_clusters.append(res[3])

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    dates_all = np.concatenate(all_dates)
    cluster_all = np.concatenate(all_clusters)

    # 添加波动率聚类特征
    X = np.column_stack([X, cluster_all])

    logger.info(f"特征矩阵: {X.shape}, 目标: {y.shape}")
    logger.info(f"波动率聚类分布: 低={np.sum(cluster_all==0)}, 中={np.sum(cluster_all==1)}, 高={np.sum(cluster_all==2)}")

    # 非对称样本权重（高收益样本获得更高训练权重）
    sample_weight = compute_sample_weights(y)
    logger.info(f"样本权重: mean={sample_weight.mean():.2f}, "
                f"高权重占比={np.mean(sample_weight > 1.5):.1%}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练
    stacking = StackingEnsemble()
    stacking.scaler = scaler
    stacking.create_base_models()
    stacking.train_base_models(X_scaled, y, sample_weight=sample_weight, dates_train=dates_all, monitor=monitor)

    # OOF 交叉验证预测
    logger.info("生成OOF预测...")
    oof_preds = stacking.generate_oof_predictions(X_scaled, y,
                                                   sample_weight=sample_weight,
                                                   dates=dates_all, monitor=monitor)

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
            metrics = evaluate_predictions(y[mask], oof_preds[mask, i], return_metrics=True)
            monitor.log_base_metrics(name, metrics)

    # 元模型预测
    meta_preds = stacking.meta_model.predict(oof_preds)
    logger.info(f"\nStacking Ensemble:")
    metrics = evaluate_predictions(y, meta_preds, return_metrics=True)
    monitor.log_base_metrics('Stacking', metrics)

    # 元模型权重记录
    if stacking.meta_model is not None and hasattr(stacking.meta_model, 'weights'):
        monitor.log_meta_weights(list(stacking.base_models.keys()),
                                 stacking.meta_model.weights)

    # LightGBM特征重要性
    if 'lightgbm' in stacking.base_models:
        try:
            lgb_model = stacking.base_models['lightgbm']
            fi = lgb_model.feature_importances_
            if fi is not None and len(fi) > 0:
                # 用占位特征名（实际使用时应从featurework获取）
                feat_names = [f'feat_{j}' for j in range(len(fi))]
                monitor.log_feature_importance(feat_names, fi)
        except Exception:
            pass

    # 早停统计
    if stacking.early_stop_counts:
        logger.info("\n早停统计:")
        for m, c in stacking.early_stop_counts.items():
            logger.info(f"  {m}: {c} 次")

    # 保存模型
    stacking.save_models()

    # 生成可视化仪表盘
    monitor.generate_dashboard()

    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    logger.info("=" * 60)
    logger.info(f"训练完成! 总耗时: {elapsed:.2f} 小时")
    logger.info("=" * 60)

    return True


# 内联评估函数（避免循环import）
def evaluate_predictions(y_true, y_pred, return_metrics=False):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 10:
        if return_metrics:
            return {'MSE': 0, 'MAE': 0, 'R2': 0, 'IC': 0}
        return
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)
    ic, _ = spearmanr(y_true, y_pred)
    logger.info(f"  MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.4f}, IC={ic:.4f}")
    if return_metrics:
        return {'MSE': float(mse), 'MAE': float(mae), 'R2': float(r2), 'IC': float(ic)}


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
