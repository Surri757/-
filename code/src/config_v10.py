"""
V10 Final Config — Expanded Data + CS Features + Balanced Validation

Timeline: 2019-01 to 2026-05 (7.4 years, 508K rows)
Validation: 6 months (balance between V4's 4m overfit and V5's 12m weak signal)
Features: CS features ON (real test improvement ~50%)
"""
config = {
    'sequence_length': 60, 'feature_num': '158+39',
    'data_path': './data', 'val_months': 6,
    'use_cs_rank_features': True,
    'd_model': 192, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 384,
    'dropout': 0.15, 'batch_size': 1, 'num_epochs': 60,
    'learning_rate': 1e-5, 'max_grad_norm': 5.0, 'weight_decay': 2e-5,
    'accumulation_steps': 8, 'use_amp': True, 'use_gradient_checkpointing': True,
    'warmup_epochs': 5, 'min_lr_ratio': 0.1,
    'use_ema': True, 'ema_decay': 0.999,
    'loss_type': 'portfolio',
    'reg_weight': 1.0, 'ic_weight': 0.5, 'portfolio_weight': 0.25,
    'listmle_weight': 0.3, 'huber_beta': 0.01, 'portfolio_temp': 0.1,
    'portfolio_warmup_epochs': 10, 'topk_k': 5,
    'output_dir': './model/60_158+39_v10',
}
