"""
股票排序模型训练脚本 - V4版本
基于V2成功经验优化: final_score=0.141347
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import json
import multiprocessing as mp
import joblib
from config import config

# AMP混合精度
from torch.cuda.amp import autocast, GradScaler


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


feature_columns_map = {
    '39': ['instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
           'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
           'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j',
           'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20',
           'return_1', 'return_5', 'return_10', 'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
               'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
               'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
               'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
               'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60',
               'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
               'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60',
               'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60',
               'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
               'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
               'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
               'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
               'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
               'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
               'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
               'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60',
               'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
               'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
               'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j',
               'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20',
               'return_1', 'return_5', 'return_10', 'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}


def engineer_features_158plus39(df):
    """158+39特征工程"""
    df = df.sort_values('日期').copy()

    # 基础技术指标
    df['sma_5'] = df['收盘'].rolling(5).mean()
    df['sma_20'] = df['收盘'].rolling(20).mean()
    df['ema_12'] = df['收盘'].ewm(span=12).mean()
    df['ema_26'] = df['收盘'].ewm(span=26).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # RSI
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 成交量变化
    df['volume_change'] = df['成交量'].pct_change()
    df['obv'] = (np.sign(df['收盘'].diff()) * df['成交量']).cumsum()
    df['volume_ma_5'] = df['成交量'].rolling(5).mean()
    df['volume_ma_20'] = df['成交量'].rolling(20).mean()
    df['volume_ratio'] = df['成交量'] / (df['volume_ma_20'] + 1e-12)

    # KDJ
    low_min = df['最低'].rolling(9).min()
    high_max = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low_min) / (high_max - low_min + 1e-12) * 100
    df['kdj_k'] = rsv.ewm(span=3).mean()
    df['kdj_d'] = df['kdj_k'].ewm(span=3).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # Bollinger Bands
    df['boll_mid'] = df['收盘'].rolling(20).mean()
    df['boll_std'] = df['收盘'].rolling(20).std()

    # ATR
    high_low = df['最高'] - df['最低']
    high_close = np.abs(df['最高'] - df['收盘'].shift())
    low_close = np.abs(df['最低'] - df['收盘'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    # EMA 60
    df['ema_60'] = df['收盘'].ewm(span=60).mean()

    # 波动率
    df['volatility_10'] = df['收盘'].pct_change().rolling(10).std()
    df['volatility_20'] = df['收盘'].pct_change().rolling(20).std()

    # 收益率
    df['return_1'] = df['收盘'].pct_change(1)
    df['return_5'] = df['收盘'].pct_change(5)
    df['return_10'] = df['收盘'].pct_change(10)

    # 价差
    df['high_low_spread'] = df['最高'] - df['最低']
    df['open_close_spread'] = df['收盘'] - df['开盘']
    df['high_close_spread'] = df['最高'] - df['收盘']
    df['low_close_spread'] = df['收盘'] - df['最低']

    return df


def engineer_features_39(df):
    """39特征工程 - 基础版本"""
    return engineer_features_158plus39(df)


def _build_label_and_clean(df, drop_small_open=True):
    """构建标签并清洗"""
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        df = df[df['open_t1'] > 1e-4]

    df['label'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)
    df = df.dropna(subset=['label'])
    df.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return df


def preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    """通用预处理"""
    feature_num = config.get('feature_num', '158+39')
    feature_engineer = engineer_features_158plus39 if feature_num == '158+39' else engineer_features_39
    feature_columns = feature_columns_map.get(feature_num, feature_columns_map['158+39'])

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空")

    num_processes = min(2, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def create_ranking_dataset(df, features, sequence_length):
    """创建排序数据集"""
    sequences, targets, relevances, stock_indices = [], [], [], []

    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)
    dates = df['日期'].unique()

    print(f"  [DEBUG] create_ranking_dataset: df rows={len(df)}, unique dates={len(dates)}, sequence_length={sequence_length}")

    for i in tqdm(range(sequence_length, len(dates)), desc="创建数据集"):
        window_end = dates[i]
        window_start = dates[i - sequence_length]

        window_data = df[(df['日期'] >= window_start) & (df['日期'] < window_end)]

        if len(window_data['股票代码'].unique()) < 10:
            continue

        # Only use numeric features for pivot (exclude 'instrument' which is stock index)
        numeric_features_for_pivot = [f for f in features if f != 'instrument']
        # For each feature, create a pivot table, then combine them
        feature_data_list = []
        stock_columns = None
        for feat in numeric_features_for_pivot:
            feat_pivot = window_data.pivot(index='日期', columns='股票代码', values=feat)
            if feat_pivot.shape[0] >= sequence_length and feat_pivot.shape[1] >= 10:
                if stock_columns is None:
                    stock_columns = feat_pivot.columns.tolist()
                feature_data_list.append(feat_pivot.values)

        if len(feature_data_list) == 0 or stock_columns is None:
            continue

        # Stack features: shape [seq_len, num_stocks, num_features]
        seq = np.stack(feature_data_list, axis=-1)
        label_data = df[df['日期'] == window_end].set_index('股票代码')['label']

        valid_stocks = [s for s in stock_columns if s in label_data.index]
        if len(valid_stocks) < 10:
            continue

        valid_indices = [stock_columns.index(s) for s in valid_stocks]
        seq = seq[:, valid_indices, :]
        labels = np.array([label_data[s] for s in valid_stocks])

        _, sorted_idx = torch.sort(torch.tensor(labels), descending=True)
        relevance = torch.zeros(len(labels), dtype=torch.long)
        relevance[sorted_idx] = torch.arange(len(labels), 0, -1, dtype=torch.long)

        stock_idx = [stockid2idx[s] for s in valid_stocks]

        sequences.append(seq)
        targets.append(labels)
        relevances.append(relevance.cpu().numpy())
        stock_indices.append(stock_idx)

    print(f"  [DEBUG] create_ranking_dataset: created {len(sequences)} samples")
    # Return as list to handle varying num_stocks - collate_fn handles padding
    return (sequences, np.array(targets, dtype=object), np.array(relevances, dtype=object), np.array(stock_indices, dtype=object))


class RankingDataset(Dataset):
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'relevance': torch.LongTensor(self.relevance_scores[idx]),
            'stock_indices': torch.LongTensor(self.stock_indices[idx])
        }


def collate_fn(batch):
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]

    # sequences shape: (seq_len, num_stocks, features) -> transpose to (num_stocks, seq_len, features)
    transposed = [np.transpose(seq, (1, 0, 2)) for seq in sequences]

    max_stocks = max(seq.shape[0] for seq in transposed)
    seq_len = transposed[0].shape[1]
    feature_dim = transposed[0].shape[2]

    padded_sequences, padded_targets, padded_relevance, padded_stock_indices, masks = [], [], [], [], []

    for seq, tgt, rel, stock_idx in zip(transposed, targets, relevance, stock_indices):
        num_stocks = seq.shape[0]

        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq = np.pad(seq, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            tgt = np.pad(tgt, (0, pad_size), mode='constant')
            rel = np.pad(rel, (0, pad_size), mode='constant', constant_values=-1)
            stock_idx = np.pad(stock_idx, (0, pad_size), mode='constant', constant_values=-1)

        mask = np.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    return {
        'sequences': torch.FloatTensor(np.array(padded_sequences)),
        'targets': torch.FloatTensor(np.array(padded_targets)),
        'relevance': torch.LongTensor(np.array(padded_relevance)),
        'stock_indices': torch.LongTensor(np.array(padded_stock_indices)),
        'masks': torch.FloatTensor(np.array(masks))
    }


class WeightedRankingLoss(nn.Module):
    def __init__(self, temperature=1.0, k=5, weight_factor=2.0, pairwise_weight=1, base_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight

    def listwise_loss(self, y_pred, y_true, weights):
        # Use log_softmax for numerical stability (avoids log(softmax(~0)) = NaN)
        log_pred_probs = F.log_softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)
        weighted_ce = -(target_probs * log_pred_probs * weights)
        return (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()

    def pairwise_loss(self, y_pred, y_true, weights):
        batch_size, num_items = y_pred.size()
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        weighted_loss = pairwise_loss * mask * weight_matrix
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        return (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()

    def forward(self, y_pred, y_true):
        # Force float32 to avoid NaN with AMP float16
        y_pred = y_pred.float()
        y_true = y_true.float()

        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        _, top_indices = torch.topk(y_true, k, dim=1)
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor

        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        return listwise + self.pairwise_weight * pairwise


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    metrics = {'pred_return_sum': 0, 'max_return_sum': 0, 'random_return_sum': 0,
               'ratio_pred': 0, 'ratio_random': 0, 'final_score': 0}
    count = 0

    for i in range(y_pred.size(0)):
        mask = masks[i].cpu().numpy()
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < k:
            continue

        valid_pred = y_pred[i][valid_indices].cpu().numpy()
        valid_true = y_true[i][valid_indices].cpu().numpy()

        pred_top_idx = np.argsort(valid_pred)[::-1][:k]
        true_top_idx = np.argsort(valid_true)[::-1][:k]

        pred_return = valid_true[pred_top_idx].sum()
        max_return = valid_true[true_top_idx].sum()
        random_return = k * valid_true.mean()

        ratio_pred = pred_return / (max_return + 1e-12) if abs(max_return) > 1e-9 else 0
        ratio_random = random_return / (max_return + 1e-12) if abs(max_return) > 1e-9 else 0
        denominator = max_return - random_return
        final_score = (pred_return - random_return) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0

        metrics['pred_return_sum'] += pred_return
        metrics['max_return_sum'] += max_return
        metrics['random_return_sum'] += random_return
        metrics['ratio_pred'] += ratio_pred
        metrics['ratio_random'] += ratio_random
        metrics['final_score'] += final_score
        count += 1

    if count > 0:
        for key in metrics:
            metrics[key] /= count
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    accumulation_loss = None

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)

        with autocast(enabled=scaler is not None):
            # Replace NaN in input features with 0 (stocks with incomplete history)
            sequences = torch.nan_to_num(sequences, nan=0.0)

            outputs = model(sequences)
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            batch_loss = None
            batch_size = sequences.size(0)

            for i in range(batch_size):
                mask = masks[i].cpu().numpy()
                valid_indices = np.where(mask > 0)[0]
                if len(valid_indices) < 2:
                    continue

                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]

                # Normalize targets for numerical stability
                t_mean = valid_true.mean()
                t_std = valid_true.std().clamp(min=1e-8)
                norm_true = (valid_true - t_mean) / t_std

                loss = criterion(valid_pred.unsqueeze(0), norm_true.unsqueeze(0))
                batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                batch_loss = batch_loss / batch_size

        if accumulation_steps > 1:
            batch_loss = batch_loss / accumulation_steps
            accumulation_loss = batch_loss if accumulation_loss is None else accumulation_loss + batch_loss
        else:
            accumulation_loss = batch_loss

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.scale(accumulation_loss).backward()
                if config.get('max_grad_norm', 5.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                accumulation_loss.backward()
                if config.get('max_grad_norm', 5.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
            optimizer.zero_grad()
            accumulation_loss = None

        if accumulation_loss is not None:
            total_loss += accumulation_loss.item()

        with torch.no_grad():
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

    if accumulation_loss is not None:
        if scaler is not None:
            scaler.scale(accumulation_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            accumulation_loss.backward()
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    return avg_loss, total_metrics


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            sequences = torch.nan_to_num(sequences, nan=0.0)

            outputs = model(sequences)
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            batch_loss = None
            batch_size = sequences.size(0)

            for i in range(batch_size):
                mask = masks[i].cpu().numpy()
                valid_indices = np.where(mask > 0)[0]
                if len(valid_indices) < 2:
                    continue

                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]

                # Normalize targets for numerical stability
                t_mean = valid_true.mean()
                t_std = valid_true.std().clamp(min=1e-8)
                norm_true = (valid_true - t_mean) / t_std

                loss = criterion(valid_pred.unsqueeze(0), norm_true.unsqueeze(0))
                batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()

            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    return avg_loss, total_metrics


def split_train_val_by_last_month(df, sequence_length):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围: {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围: {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    data_path = config.get('data_path', './data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file, encoding='utf-8')
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    all_stock_ids = full_df['股票代码'].unique()
    global stockid2idx
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    # 2. 特征工程
    train_data, features = preprocess_common(train_df, stockid2idx, desc="训练集特征工程")
    val_data, _ = preprocess_common(val_df, stockid2idx, desc="验证集特征工程")

    # 3. 标准化
    scaler = StandardScaler()
    print(f"Before dropna: train_data shape = {train_data.shape}, features count = {len(features)}")
    print(f"Feature columns sample: {features[:5]}")
    print(f"train_data columns sample: {list(train_data.columns[:10])}")

    # 只对数值型特征列进行标准化
    numeric_features = [f for f in features if f in train_data.columns]
    print(f"Numeric features count: {len(numeric_features)}")

    if len(numeric_features) == 0:
        print("ERROR: No matching features found!")
        return None

    train_data[numeric_features] = train_data[numeric_features].replace([np.inf, -np.inf], np.nan)
    val_data[numeric_features] = val_data[numeric_features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=numeric_features)
    val_data = val_data.dropna(subset=numeric_features)
    train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
    val_data[numeric_features] = scaler.transform(val_data[numeric_features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # 4. 创建数据集
    print("创建训练集...")
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset(
        train_data, features, config['sequence_length']
    )
    print("创建验证集...")
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset(
        val_data, features, config['sequence_length']
    )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")

    # 5. DataLoader
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 6. 模型
    # Note: 'instrument' column is excluded from numeric features, so use len(numeric_features)
    numeric_feature_count = len([f for f in features if f != 'instrument'])
    from model import StockTransformer
    model = StockTransformer(input_dim=numeric_feature_count, config=config, num_stocks=num_stocks).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5, temperature=config.get('temperature', 0.1), weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'], base_weight=config.get('base_weight', 1.0)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 2e-5))

    # Warmup + Cosine schedule
    warmup_epochs = config.get('warmup_epochs', 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, config['num_epochs'] - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if config.get('use_amp', True) and device.type == 'cuda' else None
    accumulation_steps = config.get('accumulation_steps', 1)

    # 8. 训练
    best_score = -float('inf')
    best_epoch = -1

    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps)
        print(f"Train Loss: {train_loss:.4f}")
        for k, v in train_metrics.items():
            print(f"Train {k}: {v:.4f}")

        eval_loss, eval_metrics = evaluate_epoch(model, val_loader, criterion, device)
        print(f"Eval Loss: {eval_loss:.4f}")
        for k, v in eval_metrics.items():
            print(f"Eval {k}: {v:.4f}")

        scheduler.step()

        current_final_score = eval_metrics.get('final_score', 0.0)
        if current_final_score > best_score:
            best_score = current_final_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"保存最佳模型 - final score: {best_score:.4f}")

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")
    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")

    return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n训练完成！最佳 final score: {best_score:.4f}")