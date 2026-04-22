"""
训练主程序 - 7模型Stacking集成
2026清华大学大数据挑战赛冠军方案
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from featurework import FeatureEngineering
from data_fetcher import load_all_data


def set_all_seeds(seed=42):
    """设置所有随机种子确保可复现性"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        import lightgbm as lgb
        lgb.seed(seed)
    except:
        pass

    try:
        import xgboost as xgb
        xgb.set_random_seed(seed)
    except:
        pass

    try:
        import catboost as cb
        cb.set_random_seed(seed)
    except:
        pass

    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except:
        pass


class PatchTSTModel(nn.Module):
    """PatchTST模型 - 时序SOTA模型"""
    def __init__(self, seq_len=60, pred_len=1, d_model=64, n_heads=4, e_layers=3, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        self.project = nn.Linear(patch_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True),
            num_layers=e_layers
        )
        self.fc = nn.Linear(d_model * self.n_patches, pred_len)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unfold(1, self.patch_len, self.stride)
        x = x.permute(0, 2, 1)
        x = self.project(x)
        x = self.encoder(x)
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class TimesNetModel(nn.Module):
    """TimesNet模型 - 多周期波动捕捉"""
    def __init__(self, seq_len=60, pred_len=1, d_model=64, e_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.decoder = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        return self.decoder(x)


class DLinearModel(nn.Module):
    """DLinear模型 - 极简泛化王"""
    def __init__(self, seq_len=60, pred_len=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = nn.Linear(seq_len, seq_len)
        self.fc = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = x.squeeze(-1)
        seasonal = self.decomposition(x)
        return self.fc(seasonal).unsqueeze(-1)


class NGboostModel:
    """NGboost模型 - 量化预测不确定性"""
    def __init__(self, n_estimators=600, learning_rate=0.04, max_depth=7):
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        self.model = NGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            Dist=Normal,
            random_state=42,
            verbose=False
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_std(self, X):
        return self.model.pred_dist(X).std()


class StackingEnsemble:
    """时间序列滚动Stacking集成"""

    def __init__(self, model_dir='../model'):
        self.model_dir = model_dir
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_base_models(self):
        """创建7个基础模型"""
        set_all_seeds(42)

        self.base_models = {
            'ngboost': NGboostModel(n_estimators=600, learning_rate=0.04, max_depth=7),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000, learning_rate=0.03, max_depth=8, num_leaves=64,
                random_state=42, verbose=-1, device='gpu' if torch.cuda.is_available() else 'cpu',
                early_stopping_rounds=50, n_jobs=-1
            ),
            'catboost': cb.CatBoostRegressor(
                n_estimators=800, learning_rate=0.04, max_depth=7,
                random_state=42, verbose=0, task_type='GPU' if torch.cuda.is_available() else 'CPU',
                early_stopping_rounds=50
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=800, learning_rate=0.04, max_depth=7,
                random_state=42, tree_method='hist' if torch.cuda.is_available() else 'hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                early_stopping_rounds=50
            ),
            'patchtst': PatchTSTModel(seq_len=60, pred_len=1, d_model=64, n_heads=4, e_layers=3).to(self.device),
            'timesnet': TimesNetModel(seq_len=60, pred_len=1, d_model=64, e_layers=2).to(self.device),
            'dlinear': DLinearModel(seq_len=60, pred_len=1).to(self.device)
        }

    def time_series_split(self, X, y, window_size=6, step=1):
        """时间序列滚动划分"""
        train_windows = []
        n_samples = len(X)

        for start_idx in range(0, n_samples - window_size * 30, step * 30):
            end_idx = start_idx + window_size * 30
            train_windows.append((start_idx, end_idx))

        return train_windows

    def prepare_sequential_data(self, X, y, seq_len=60):
        """准备序列数据"""
        X_seq = []
        y_seq = []

        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def train_base_models(self, X_train, y_train):
        """训练所有基础模型"""
        set_all_seeds(42)
        print("开始训练基础模型...")

        # 划分验证集用于早停
        val_size = int(len(X_train) * 0.1)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train_subset = X_train[:-val_size]
        y_train_subset = y_train[:-val_size]

        for name, model in self.base_models.items():
            print(f"训练模型: {name}")
            if name == 'ngboost':
                model.fit(X_train_subset, y_train_subset)
            elif name == 'lightgbm':
                model.fit(X_train_subset, y_train_subset, 
                         eval_set=[(X_val, y_val)],
                         eval_metric='mse')
            elif name == 'catboost':
                model.fit(X_train_subset, y_train_subset, 
                         eval_set=[(X_val, y_val)],
                         use_best_model=True)
            elif name == 'xgboost':
                model.fit(X_train_subset, y_train_subset, 
                         eval_set=[(X_val, y_val)],
                         eval_metric='rmse')
            else:
                self.train_pytorch_model(model, X_train, y_train, epochs=50)

        print("基础模型训练完成")

    def train_pytorch_model(self, model, X_train, y_train, epochs=50, batch_size=128):
        """训练PyTorch模型"""
        set_all_seeds(42)
        X_seq, y_seq = self.prepare_sequential_data(X_train, y_train)

        # 划分验证集
        val_size = int(len(X_seq) * 0.1)
        X_val, y_val = X_seq[-val_size:], y_seq[-val_size:]
        X_train_subset, y_train_subset = X_seq[:-val_size], y_seq[:-val_size]

        X_tensor = torch.FloatTensor(X_train_subset).to(self.device)
        y_tensor = torch.FloatTensor(y_train_subset).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # 验证
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = criterion(val_output.squeeze(), y_val_tensor).item()
            model.train()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(loader):.6f}, Val Loss: {val_loss:.6f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    def generate_oof_predictions(self, X, y, n_folds=6):
        """生成交叉验证预测"""
        set_all_seeds(42)
        n_samples = len(X)
        fold_size = n_samples // n_folds

        oof_preds = np.zeros((n_samples, len(self.base_models)))

        for fold_idx in range(n_folds):
            print(f"交叉验证 Fold {fold_idx + 1}/{n_folds}")
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples

            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
            val_idx = list(range(val_start, val_end))

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            for model_idx, (name, model) in enumerate(self.base_models.items()):
                if name in ['ngboost', 'lightgbm', 'catboost', 'xgboost']:
                    model.fit(X_tr, y_tr)
                    oof_preds[val_idx, model_idx] = model.predict(X_val)
                else:
                    self.train_pytorch_model(model, X_tr, y_tr, epochs=30)
                    X_seq, _ = self.prepare_sequential_data(X_val, np.zeros(len(X_val)))
                    X_tensor = torch.FloatTensor(X_seq).to(self.device)
                    model.eval()
                    with torch.no_grad():
                        preds = model(X_tensor).cpu().numpy().flatten()
                    oof_preds[val_idx[:len(preds)], model_idx] = preds

        return oof_preds

    def train_meta_model(self, X_meta, y_meta):
        """训练Stacking元模型"""
        set_all_seeds(42)
        print("训练Stacking元模型...")
        self.meta_model = Ridge(alpha=1.0, random_state=42)
        self.meta_model.fit(X_meta, y_meta)
        print("元模型训练完成")

    def save_models(self):
        """保存所有模型"""
        os.makedirs(self.model_dir, exist_ok=True)

        for name, model in self.base_models.items():
            if name in ['ngboost', 'lightgbm', 'catboost', 'xgboost']:
                with open(os.path.join(self.model_dir, f'{name}_model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            else:
                torch.save(model.state_dict(), os.path.join(self.model_dir, f'{name}_model.pth'))

        if self.meta_model:
            with open(os.path.join(self.model_dir, 'meta_model.pkl'), 'wb') as f:
                pickle.dump(self.meta_model, f)

        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"所有模型已保存到 {self.model_dir}")


def download_baostock_data():
    """从baostock下载沪深300成分股历史数据"""
    try:
        import baostock as bs
    except ImportError:
        print("baostock未安装，跳过数据下载")
        return None

    print("从baostock下载沪深300成分股数据...")

    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock登录失败: {lg.error_msg}")
        return None

    hs300 = bs.query_hs300_stocks()
    stock_codes = []
    while (hs300.error_code == '0') and (hs300.next()):
        stock_codes.append(hs300.get_row_data()[1])

    bs.logout()

    if not stock_codes:
        print("未获取到沪深300成分股列表")
        return None

    all_data = []
    start_date = '2014-01-01'
    end_date = '2026-04-18'

    bs.login()
    for i, code in enumerate(stock_codes[:50]):
        code = code.replace('.SH', '').replace('.SZ', '')
        if code.startswith('6'):
            bs_code = f"sh.{code}"
        else:
            bs_code = f"sz.{code}"

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turnover_rate,pe,pb",
            start_date=start_date,
            end_date=end_date,
            frequency='d'
        )

        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            all_data.append({
                'stock_id': code,
                'date': row[0],
                'open': float(row[1]) if row[1] else 0,
                'high': float(row[2]) if row[2] else 0,
                'low': float(row[3]) if row[3] else 0,
                'close': float(row[4]) if row[4] else 0,
                'volume': int(row[5]) if row[5] else 0,
                'amount': float(row[6]) if row[6] else 0,
                'turnover_rate': float(row[7]) if row[7] else 0,
                'pe': float(row[8]) if row[8] else 0,
                'pb': float(row[9]) if row[9] else 0,
            })

        if (i + 1) % 10 == 0:
            print(f"已下载 {i+1}/{len(stock_codes[:50])} 只股票")

    bs.logout()

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"成功下载 {len(df)} 条记录")
        return df

    return None


def load_data(data_dir='../data'):
    """加载数据"""
    print("加载数据...")

    train_path = os.path.join(data_dir, 'train.csv')

    if os.path.exists(train_path):
        print("使用本地训练数据...")
        df = pd.read_csv(train_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['stock_id', 'date'])
        print(f"成功加载本地数据: {len(df)} 条记录")
        return df
    else:
        print("本地训练数据不存在，尝试从baostock下载...")
        df_baostock = download_baostock_data()
        if df_baostock is not None:
            df_baostock.to_csv(train_path, index=False)
            print(f"数据已保存到 {train_path}")
            return df_baostock
        print("错误：无法获取baostock数据")
        raise Exception("无法获取数据，请确保网络连接正常")


def generate_simulated_data():
    """生成模拟数据进行测试"""
    np.random.seed(42)
    dates = pd.date_range('2014-01-01', '2026-04-18', freq='B')

    stocks = ['000001', '000002', '600000', '600001', '600016', '600019', '600028', '600030', '600036', '600048']
    data_list = []

    for stock_id in stocks:
        price = 100
        prices = []
        for _ in range(len(dates)):
            price *= (1 + np.random.randn() * 0.02)
            prices.append(price)

        df_stock = pd.DataFrame({
            'stock_id': stock_id,
            'date': dates,
            'open': np.array(prices) * (1 + np.random.randn(len(dates)) * 0.005),
            'high': np.array(prices) * (1 + abs(np.random.randn(len(dates)) * 0.01)),
            'low': np.array(prices) * (1 - abs(np.random.randn(len(dates)) * 0.01)),
            'close': prices,
            'volume': np.random.randint(1000000, 2000000, len(dates)),
            'amount': np.random.randint(100000000, 2000000000, len(dates)),
            'turnover_rate': np.random.rand(len(dates)) * 5,
            'pe': np.random.rand(len(dates)) * 30 + 10,
            'pb': np.random.rand(len(dates)) * 2 + 0.5,
        })
        data_list.append(df_stock)

    return pd.concat(data_list, ignore_index=True)


def main():
    """主训练流程"""
    set_all_seeds(42)
    print("="*60)
    print("2026清华大学大数据挑战赛 - 训练主程序")
    print("="*60)

    start_time = datetime.now()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'data')
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

    df, industry_df, macro_df = load_all_data(data_dir)
    print(f"数据加载完成，共 {len(df)} 条记录")

    fe = FeatureEngineering()
    print("开始特征工程...")

    stock_ids = df['stock_id'].unique()
    all_X = []
    all_y = []

    for stock_id in stock_ids:
        stock_df = df[df['stock_id'] == stock_id].copy()
        stock_df = stock_df.sort_values('date')
        stock_df = stock_df.set_index('date')

        features = fe.build_all_features(stock_df, industry_df, macro_df)
        features = fe.remove_outliers(features)

        close = stock_df['close']
        target = close.shift(-5) / close - 1

        valid_idx = features.dropna().index.intersection(target.dropna().index)
        if len(valid_idx) > 100:
            X_stock = features.loc[valid_idx].fillna(0).values
            y_stock = target.loc[valid_idx].values

            all_X.append(X_stock)
            all_y.append(y_stock)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    stacking = StackingEnsemble(model_dir=model_dir)
    stacking.scaler = scaler
    stacking.create_base_models()

    stacking.train_base_models(X_scaled, y)

    print("生成交叉验证预测用于元模型训练...")
    oof_preds = stacking.generate_oof_predictions(X_scaled, y)

    stacking.train_meta_model(oof_preds, y)

    stacking.save_models()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600

    print("="*60)
    print(f"训练完成! 总耗时: {duration:.2f} 小时")
    print("="*60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)