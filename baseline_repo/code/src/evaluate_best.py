"""
评估最佳模型 - 输出选股、权重、收益率
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import joblib
import multiprocessing as mp
import train_v4 as train_module
from model import StockTransformer
from train_v4 import (
    engineer_features_39, feature_columns_map,
    create_ranking_dataset, collate_fn, RankingDataset,
    split_train_val_by_last_month, preprocess_common
)
from config import config
from torch.utils.data import DataLoader


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    set_seed(42)

    model_dir = './model/60_158+39_v4b'
    data_path = 'data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载配置和模型
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        saved_config = json.load(f)

    # 2. 加载数据
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file, encoding='utf-8')
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    idx2stockid = {idx: sid for sid, idx in stockid2idx.items()}
    num_stocks = len(stockid2idx)

    # Set global for train_v4 module
    train_module.stockid2idx = stockid2idx

    # 3. 特征工程
    train_data, features = preprocess_common(train_df, stockid2idx, desc="训练集特征工程")
    val_data, _ = preprocess_common(val_df, stockid2idx, desc="验证集特征工程")

    # 4. 标准化
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    numeric_features = [f for f in features if f in train_data.columns]
    val_data[numeric_features] = val_data[numeric_features].replace([np.inf, -np.inf], np.nan)
    val_data = val_data.dropna(subset=numeric_features)
    val_data[numeric_features] = scaler.transform(val_data[numeric_features])

    # 5. 创建验证集
    print("创建验证集...")
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset(
        val_data, features, config['sequence_length']
    )
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 6. 加载模型
    feature_num = len([f for f in features if f != 'instrument'])
    model = StockTransformer(feature_num, config, num_stocks).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=device))
    model.eval()
    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 7. 推理并收集结果
    # 获取验证集对应的日期
    val_dates_raw = val_data['日期'].unique()
    val_dates_raw = sorted(val_dates_raw)
    sequence_length = config['sequence_length']

    # 验证集样本对应的日期（window_end）
    val_sample_dates = val_dates_raw[sequence_length:]
    # 只取有样本的日期
    val_sample_dates = val_sample_dates[:len(val_sequences)]

    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="推理")):
            sequences = batch['sequences'].to(device)
            targets = batch['targets']
            masks = batch['masks']
            stock_indices = batch['stock_indices']

            sequences = torch.nan_to_num(sequences, nan=0.0)
            outputs = model(sequences)

            batch_size = sequences.size(0)
            for i in range(batch_size):
                mask_np = masks[i].numpy()
                valid_indices = np.where(mask_np > 0)[0]

                if len(valid_indices) < 5:
                    continue

                pred_scores = outputs[i][valid_indices].cpu().numpy()
                true_returns = targets[i][valid_indices].numpy()
                stock_ids = stock_indices[i][valid_indices].numpy()

                # Top-5 by predicted score
                top5_pred_idx = np.argsort(pred_scores)[::-1][:5]

                # Top-5 by actual return (optimal)
                top5_true_idx = np.argsort(true_returns)[::-1][:5]

                # Random 5 (expected)
                random_return = 5 * true_returns.mean()

                date = val_sample_dates[batch_idx] if batch_idx < len(val_sample_dates) else f"sample_{batch_idx}"

                pred_return = true_returns[top5_pred_idx].sum()
                max_return = true_returns[top5_true_idx].sum()

                result = {
                    'date': date,
                    'num_stocks': len(valid_indices),
                    'pred_return_sum': pred_return,
                    'max_return_sum': max_return,
                    'random_return_sum': random_return,
                    'top5_picks': [],
                    'top5_optimal': [],
                }

                for rank, idx in enumerate(top5_pred_idx):
                    stock_code = idx2stockid.get(stock_ids[idx], str(stock_ids[idx]))
                    result['top5_picks'].append({
                        'rank': rank + 1,
                        'stock_code': stock_code,
                        'pred_score': float(pred_scores[idx]),
                        'actual_return_5d': float(true_returns[idx]),
                    })

                for rank, idx in enumerate(top5_true_idx):
                    stock_code = idx2stockid.get(stock_ids[idx], str(stock_ids[idx]))
                    result['top5_optimal'].append({
                        'rank': rank + 1,
                        'stock_code': stock_code,
                        'actual_return_5d': float(true_returns[idx]),
                    })

                all_results.append(result)

    # 8. 汇总输出
    print("\n" + "=" * 80)
    print("                        最佳模型选股评估报告")
    print("=" * 80)
    print(f"模型: {model_dir}/best_model.pth")
    print(f"最佳 epoch: 67, Best final_score: 0.3022")
    print(f"验证集样本数: {len(all_results)}")
    print(f"选股策略: 每个交易日选5只预测得分最高的股票，等权重持有")
    print(f"收益率计算: 5日前瞻收益率 = (第5日开盘价 - 第1日开盘价) / 第1日开盘价")
    print("=" * 80)

    # 逐日明细
    print(f"\n{'日期':<14} {'模型Top5收益':>12} {'最优Top5收益':>12} {'随机收益':>10} {'超额收益':>10}")
    print("-" * 60)

    total_pred_return = 0
    total_max_return = 0
    total_random_return = 0

    for r in all_results:
        excess = r['pred_return_sum'] - r['random_return_sum']
        print(f"{str(r['date']):<14} {r['pred_return_sum']:>12.4f} {r['max_return_sum']:>12.4f} {r['random_return_sum']:>10.4f} {excess:>10.4f}")
        total_pred_return += r['pred_return_sum']
        total_max_return += r['max_return_sum']
        total_random_return += r['random_return_sum']

    n = len(all_results)
    avg_pred = total_pred_return / n
    avg_max = total_max_return / n
    avg_random = total_random_return / n
    avg_excess = avg_pred - avg_random

    # 计算 final_score
    denominator = avg_max - avg_random
    if abs(denominator) > 1e-6:
        final_score = (avg_pred - avg_random) / denominator
    else:
        final_score = 0

    print("-" * 60)
    print(f"{'平均':<14} {avg_pred:>12.4f} {avg_max:>12.4f} {avg_random:>10.4f} {avg_excess:>10.4f}")

    # 单股平均收益
    avg_pred_per_stock = avg_pred / 5
    avg_max_per_stock = avg_max / 5
    avg_random_per_stock = avg_random / 5

    print(f"\n{'=' * 80}")
    print(f"                          收益率汇总")
    print(f"{'=' * 80}")
    print(f"  模型Top-5组合平均5日收益率:   {avg_pred:>10.4f}  ({avg_pred * 100:.2f}%)")
    print(f"  模型单股平均5日收益率:         {avg_pred_per_stock:>10.4f}  ({avg_pred_per_stock * 100:.2f}%)")
    print(f"  最优Top-5组合平均5日收益率:    {avg_max:>10.4f}  ({avg_max * 100:.2f}%)")
    print(f"  随机Top-5组合平均5日收益率:    {avg_random:>10.4f}  ({avg_random * 100:.2f}%)")
    print(f"  超额收益(模型-随机):          {avg_excess:>10.4f}  ({avg_excess * 100:.2f}%)")
    print(f"  Final Score:                   {final_score:>10.4f}")
    print(f"  模型/最优比率:                 {avg_pred / avg_max if avg_max != 0 else 0:>10.4f}  ({avg_pred / avg_max * 100 if avg_max != 0 else 0:.2f}%)")

    # 选股频率统计
    print(f"\n{'=' * 80}")
    print(f"                          选股频率统计 (Top-5)")
    print(f"{'=' * 80}")

    from collections import Counter
    pick_counter = Counter()
    for r in all_results:
        for pick in r['top5_picks']:
            pick_counter[pick['stock_code']] += 1

    print(f"  被选中次数  股票代码    被选中占比")
    print(f"  {'-' * 40}")
    for stock, count in pick_counter.most_common(20):
        pct = count / n * 100
        print(f"  {count:>6}      {stock:<12} {pct:>6.1f}%")

    # 每个日期的详细选股
    print(f"\n{'=' * 80}")
    print(f"                      逐日选股明细")
    print(f"{'=' * 80}")

    for r in all_results:
        print(f"\n日期: {r['date']}  (可选股票数: {r['num_stocks']})")
        print(f"  模型选股 (Top-5):")
        for pick in r['top5_picks']:
            ret_pct = pick['actual_return_5d'] * 100
            print(f"    #{pick['rank']} {pick['stock_code']:<12} 得分={pick['pred_score']:.4f}  实际5日收益={ret_pct:>+.2f}%")
        print(f"  组合5日收益: {r['pred_return_sum'] * 100:>+.2f}%")
        print(f"  最优组合5日收益: {r['max_return_sum'] * 100:>+.2f}%")


if __name__ == '__main__':
    main()
