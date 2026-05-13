# 配置参数 - V4 优化版
# 基于V2成功经验优化：final_score=0.141347
# 核心改动：适度增加模型容量，更激进的top_k权重，增强正则化

sequence_length = 60
feature_num = '39'
config = {
    'sequence_length': sequence_length,
    'd_model': 64,         # 适配4GB显存（V2=64，V4原=96）
    'nhead': 4,            # 4头注意力（V2=2）
    'num_layers': 2,       # 2层Transformer（V2=2，V4原=3）
    'dim_feedforward': 256,
    'batch_size': 1,
    'num_epochs': 80,
    'learning_rate': 1e-4,  # 提高学习率加速收敛

    'dropout': 0.15,       # 适度增加dropout防过拟合

    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    # 损失函数权重 - 更激进地强调Top-5
    'pairwise_weight': 2,
    'base_weight': 0.3,
    'top5_weight': 5.0,    # 提高Top5权重（V2=3.5）
    'top10_weight': 2.0,

    'output_dir': './model/60_158+39_v4b',
    'data_path': 'data',
    'use_selected_features': False,

    'use_amp': False,
    'accumulation_steps': 16,

    'scheduler_type': 'cosine',
    'warmup_epochs': 5,

    'weight_decay': 3e-5,
    'label_smoothing': 0.0,
    'temperature': 1.0,
}