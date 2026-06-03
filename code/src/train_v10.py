"""
V4 Training Pipeline — Dual-Head Model with Portfolio-Optimized Loss

Core innovations:
  1. Dual-head model: ranking scores + predicted returns
  2. Portfolio-optimized loss: directly targets competition metric R = Σ(w_i × r_i)
  3. Cross-sectional rank features: each feature's percentile within date
  4. Improved evaluation: tracks portfolio return alongside ranking metrics
  5. GBDT ensemble support: separate CatBoost/LightGBM model for fusion
"""

import os, sys, json, time, warnings, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import joblib
import multiprocessing as mp

warnings.filterwarnings('ignore')

from config_v10 import config
from model_v4 import DualHeadTransformer
from utils import engineer_features_158plus39, create_ranking_dataset_vectorized
from losses_v4 import PortfolioOptimizedLoss
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from train import (
    set_seed, _build_label_and_clean, split_train_val_by_last_month,
    feature_cloums_map, feature_engineer_func_map,
    RankingDataset, collate_fn, calculate_ranking_metrics,
)


# ── Gradient Checkpointing Patch ──────────────────────────────

_original_forward = DualHeadTransformer.forward_features

def _ckpt_forward_features(self, src):
    B, N, L, F = src.size()
    x = src.view(B * N, L, F)
    x = self.input_proj(x)
    x = self.pos_encoder(x)
    for layer in self.temporal_encoder.layers:
        x = torch_checkpoint(layer, x, use_reentrant=False)
    x = self.feature_attention(x)
    x = x.view(B, N, -1)
    x = self.cross_stock_attention(x)
    x = x.view(B * N, -1)
    x = self.shared_layers(x)
    return x, B, N

if config.get('use_gradient_checkpointing'):
    DualHeadTransformer.forward_features = _ckpt_forward_features


# ── EMA ──────────────────────────────────────────────────────

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup.clear()


# ── Warmup Cosine Scheduler ──────────────────────────────────

def create_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr_ratio=0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Cross-Sectional Rank Features ────────────────────────────

def add_cross_sectional_features(data, feature_cols):
    """
    Add cross-sectional percentile rank for each feature within each date.
    This captures a stock's RELATIVE position vs peers — critical for ranking.
    """
    data = data.copy()
    cs_cols = []
    for col in tqdm(feature_cols, desc='CS-Rank'):
        if col == 'instrument':
            continue
        cs_col = f'{col}_csr'
        data[cs_col] = data.groupby('日期')[col].rank(pct=True)
        cs_cols.append(cs_col)

    data[cs_cols] = data[cs_cols].fillna(0.5)
    return data, cs_cols


# ── Preprocessing ────────────────────────────────────────────

def preprocess_data(df, stockid2idx, is_train=True, add_cs=True):
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    # Filter stocks with insufficient data (< 60 rows)
    counts = df.groupby('股票代码').size()
    valid_stocks = counts[counts >= config['sequence_length']].index
    df = df[df['股票代码'].isin(valid_stocks)]

    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    # Sequential FE (no mp.Pool - CUDA safe)
    processed_list = [feature_engineer(g) for g in tqdm(groups, desc="FE")]
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    drop_small = is_train
    processed = _build_label_and_clean(processed, drop_small_open=drop_small)

    # Determine base feature columns (without instrument)
    base_features = [c for c in feature_columns if c != 'instrument']

    # Add cross-sectional rank features
    extra_cols = []
    if add_cs and config.get('use_cs_rank_features'):
        processed, extra_cols = add_cross_sectional_features(processed, base_features)
        print(f"Added {len(extra_cols)} cross-sectional rank features")

    # Full feature list (instrument will be removed before model input)
    all_features = feature_columns.copy()
    all_features.extend(extra_cols)

    return processed, all_features


# ── Training Epoch ───────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device, epoch,
                writer, scaler=None, accumulation_steps=1, ema=None):
    model.train()
    total_loss = 0.0
    total_components = {}
    total_metrics = {}
    num_batches = 0
    accum_loss = None

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Train E{epoch+1}")):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)

        with autocast(enabled=scaler is not None):
            rank_scores, pred_returns = model(sequences)

            # Apply mask
            masked_rank = rank_scores * masks + (1 - masks) * (-1e9)

            # Compute multi-task loss
            total_batch_loss, component_losses = criterion(
                masked_rank, pred_returns, targets, masks)

        if accumulation_steps > 1:
            total_batch_loss = total_batch_loss / accumulation_steps
            accum_loss = accum_loss + total_batch_loss if accum_loss is not None else total_batch_loss
        else:
            accum_loss = total_batch_loss

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.scale(accum_loss).backward()
                if config.get('max_grad_norm', 5.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                accum_loss.backward()
                if config.get('max_grad_norm', 5.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update()
            accum_loss = None

        # Track loss
        if accum_loss is not None:
            total_loss += accum_loss.item()
        elif total_batch_loss is not None:
            total_loss += total_batch_loss.item()

        # Track components
        for k, v in component_losses.items():
            total_components[k] = total_components.get(k, 0) + (v if isinstance(v, float) else v)

        # Track ranking metrics (on ranking head)
        with torch.no_grad():
            mets = calculate_ranking_metrics(masked_rank, targets, masks, k=5)
            for k, v in mets.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

        num_batches += 1

    # Handle remaining accumulated gradient
    if accum_loss is not None:
        if scaler is not None:
            scaler.scale(accum_loss).backward()
            if config.get('max_grad_norm', 5.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            accum_loss.backward()
            if config.get('max_grad_norm', 5.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_components:
        total_components[k] /= num_batches
    for k in total_metrics:
        total_metrics[k] /= num_batches

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/final_score', total_metrics.get('final_score', 0), epoch)
        for k, v in total_components.items():
            writer.add_scalar(f'train/loss_{k}', v, epoch)

    return avg_loss, total_metrics, total_components


# ── Evaluation Epoch ─────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device, epoch, writer, ema=None):
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    total_loss = 0.0
    total_components = {}
    total_metrics = {}
    num_batches = 0

    # Track portfolio returns across all eval batches
    all_portfolio_returns = []
    all_pred_returns = []
    all_actual_returns = []

    for batch in tqdm(dataloader, desc=f"Eval E{epoch+1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)

        rank_scores, pred_returns = model(sequences)
        masked_rank = rank_scores * masks + (1 - masks) * (-1e9)

        total_batch_loss, component_losses = criterion(
            masked_rank, pred_returns, targets, masks)

        total_loss += total_batch_loss.item()

        for k, v in component_losses.items():
            total_components[k] = total_components.get(k, 0) + (v if isinstance(v, float) else v)

        mets = calculate_ranking_metrics(masked_rank, targets, masks, k=5)
        for k, v in mets.items():
            total_metrics[k] = total_metrics.get(k, 0) + v

        # Track portfolio returns for actual competition metric
        for b in range(sequences.size(0)):
            m = masks[b].bool()
            if m.sum() < 5:
                continue
            scores_b = masked_rank[b]
            rets_b = targets[b]
            k = min(5, m.sum().item())
            _, top_idx = torch.topk(scores_b, k)
            selected_rets = rets_b[top_idx]
            weights = F.softmax(selected_rets / 0.01, dim=0)
            port_ret = (weights * selected_rets).sum().item()
            all_portfolio_returns.append(port_ret)
            all_pred_returns.append(selected_rets.mean().item())
            all_actual_returns.append(rets_b[m].mean().item())

        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_components:
        total_components[k] /= num_batches
    for k in total_metrics:
        total_metrics[k] /= num_batches

    # Portfolio-level metrics (what the competition actually cares about)
    if all_portfolio_returns:
        port_returns = np.array(all_portfolio_returns)
        total_metrics['port_mean_return'] = float(port_returns.mean())
        total_metrics['port_std_return'] = float(port_returns.std())
        total_metrics['port_sharpe'] = float(port_returns.mean() / (port_returns.std() + 1e-8))
        total_metrics['port_win_rate'] = float((port_returns > 0).mean())
        total_metrics['port_pos_return'] = float(np.median(all_pred_returns))

    if writer:
        writer.add_scalar('eval/loss', avg_loss, epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, epoch)

    if ema is not None:
        ema.restore()

    return avg_loss, total_metrics, total_components


# ── Main ─────────────────────────────────────────────────────

def main():
    set_seed(42)
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    device = torch.device('cuda' if torch.cuda.is_available()
                          else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Data ─────────────────────────────────────────────
    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    val_months = config.get("val_months", 12)
    df_copy = full_df.copy()
    df_copy['日期'] = pd.to_datetime(df_copy['日期'])
    df_copy = df_copy.sort_values(['日期', '股票代码']).reset_index(drop=True)
    last_date = df_copy['日期'].max()
    val_start = (last_date - pd.DateOffset(months=val_months)).normalize()
    val_context_start = val_start - pd.tseries.offsets.BDay(config['sequence_length'] - 1)
    train_df = df_copy[df_copy['日期'] < val_start].copy()
    val_df = df_copy[df_copy['日期'] >= val_context_start].copy()
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    print(f"Train: {train_df['日期'].min()} to {train_df['日期'].max()}")
    print(f"Val:   {val_df['日期'].min()} to {val_df['日期'].max()}")
    print(f"Val start date: {val_start.date()}")

    all_stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
    num_stocks = len(stockid2idx)
    print(f"Stocks: {num_stocks}")

    # Feature engineering
    train_data, feature_columns = preprocess_data(train_df, stockid2idx, is_train=True)
    val_data, _ = preprocess_data(val_df, stockid2idx, is_train=False)

    features = [c for c in feature_columns if c != 'instrument']
    print(f"Features: {len(features)} (base 197 + cs-rank)")

    # Standardize
    scaler = StandardScaler()
    for d in [train_data, val_data]:
        d[features] = d[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # Build ranking datasets
    train_seqs, train_tgts, train_rel, train_ids = create_ranking_dataset_vectorized(
        train_data, features, config['sequence_length'])
    val_seqs, val_tgts, val_rel, val_ids = create_ranking_dataset_vectorized(
        val_data, features, config['sequence_length'],
        min_window_end_date=val_start.strftime('%Y-%m-%d'))

    print(f"Train samples: {len(train_seqs)}, Val samples: {len(val_seqs)}")

    # DataLoaders
    train_ds = RankingDataset(train_seqs, train_tgts, train_rel, train_ids)
    val_ds = RankingDataset(val_seqs, val_tgts, val_rel, val_ids)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # ── Model ────────────────────────────────────────────
    model = DualHeadTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # ── Loss ─────────────────────────────────────────────
    criterion = PortfolioOptimizedLoss(config)

    # ── Optimizer + Scheduler ────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                   weight_decay=config.get('weight_decay', 1e-5))
    scheduler = create_scheduler(optimizer, config.get('warmup_epochs', 5),
                                  config['num_epochs'], config['learning_rate'],
                                  min_lr_ratio=config.get('min_lr_ratio', 0.05))

    scaler_amp = GradScaler() if config.get('use_amp') and device.type == 'cuda' else None
    acc_steps = config.get('accumulation_steps', 1)
    ema = EMAModel(model, decay=config.get('ema_decay', 0.999)) if config.get('use_ema') else None

    # ── Train ────────────────────────────────────────────
    best_score = -float('inf')
    best_port_return = -float('inf')
    best_epoch = -1
    no_improve_count = 0
    patience = config.get('early_stopping_patience', 0)  # 0 = disabled
    min_delta = config.get('early_stopping_min_delta', 1e-4)
    es_criterion = config.get('early_stopping_criterion', 'val_final_score')  # 'val_final_score' | 'train_loss'
    best_train_loss = float('inf')
    print(f"[EARLY-STOP] criterion={es_criterion} patience={patience} min_delta={min_delta}")

    for epoch in range(config['num_epochs']):
        lr = scheduler.get_last_lr()[0]
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']} (LR: {lr:.2e})")
        print(f"{'='*60}")

        # Update criterion epoch (for portfolio warmup scheduling)
        criterion.set_epoch(epoch)

        train_loss, train_mets, train_comp = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            scaler=scaler_amp, accumulation_steps=acc_steps, ema=ema)

        print(f"Train Loss: {train_loss:.4f} | final_score: {train_mets.get('final_score', 0):.4f}")
        if train_comp:
            comp_str = ' '.join(f'{k}={v:.4f}' for k, v in train_comp.items())
            print(f"  Components: {comp_str}")

        eval_loss, eval_mets, eval_comp = eval_epoch(
            model, val_loader, criterion, device, epoch, writer, ema=ema)

        print(f"Eval  Loss: {eval_loss:.4f} | final_score: {eval_mets.get('final_score', 0):.4f}")
        if eval_comp:
            comp_str = ' '.join(f'{k}={v:.4f}' for k, v in eval_comp.items())
            print(f"  Components: {comp_str}")

        # Portfolio metrics are what the competition cares about
        port_ret = eval_mets.get('port_mean_return', 0)
        port_sharpe = eval_mets.get('port_sharpe', 0)
        port_win = eval_mets.get('port_win_rate', 0)
        print(f"  Portfolio: mean_ret={port_ret:.4f}, sharpe={port_sharpe:.2f}, win_rate={port_win:.2%}")

        scheduler.step()
        if writer:
            writer.add_scalar('lr', lr, epoch)

        # ★ A 方案:用比赛指标 final_score 选 best (不是 port_ret)
        #   + early stopping: criterion 可选 'val_final_score'(默认) 或 'train_loss'
        #     - 'val_final_score': val final_score 连续 patience epoch 涨不到 min_delta 就停
        #     - 'train_loss':       train loss 连续 patience epoch 降不到 min_delta 就停
        #                          (按用户偏好:学习是否完成看 train loss,不靠 noisy val 信号)
        current = eval_mets.get('final_score', 0)
        if current > best_score + min_delta:
            best_score = current
            best_port_return = port_ret
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_ema.pth'))
                ema.restore()
            print(f"  => Saved best (final_score={current:.4f}, port_ret={port_ret:.4f}) @ epoch {best_epoch}")

        # 早停判定:按 es_criterion 决定用哪个信号
        if es_criterion == 'train_loss':
            # train loss 下降 < min_delta 视为 plateau
            if train_loss < best_train_loss - min_delta:
                best_train_loss = train_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            if patience > 0:
                print(f"  [tloss] no_improve={no_improve_count}/{patience} (best train_loss={best_train_loss:.4f}, val best final_score={best_score:.4f} @ epoch {best_epoch})")
                if no_improve_count >= patience:
                    print(f"\n*** Early stopping (train_loss plateau) @ epoch {epoch+1}, best val final_score={best_score:.4f} @ epoch {best_epoch}, best train_loss={best_train_loss:.4f} ***")
                    break
        else:
            # 原有:val final_score plateau
            if current <= best_score + min_delta:
                no_improve_count += 1
            else:
                no_improve_count = 0
            if patience > 0:
                print(f"  no_improve={no_improve_count}/{patience} (best final_score={best_score:.4f} @ epoch {best_epoch})")
                if no_improve_count >= patience:
                    print(f"\n*** Early stopping @ epoch {epoch+1}, best final_score={best_score:.4f} @ epoch {best_epoch} ***")
                    break

    print(f"\n{'='*60}")
    print(f"V4 Done! Best epoch={best_epoch}")
    print(f"Best final_score: {best_score:.4f}")
    print(f"Best portfolio return: {best_port_return:.4f}")

    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best combined_score: {best_score:.6f}\n")
        f.write(f"Best portfolio_return: {best_port_return:.6f}\n")
        f.write("V4: Dual-head model + portfolio-optimized loss + cs-rank features\n")

    writer.close()
    return best_score


if __name__ == '__main__':
    # mp.set_start_method removed - CUDA conflict
    best = main()
    print(f"\n########## V10 Complete! Best: {best:.4f} ##########")
