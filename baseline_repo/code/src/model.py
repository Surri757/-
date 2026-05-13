import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StockTransformer(nn.Module):
    """股票排序Transformer模型"""
    def __init__(self, input_dim, config, num_stocks):
        super(StockTransformer, self).__init__()
        self.config = config
        self.num_stocks = num_stocks

        d_model = config.get('d_model', 256)
        nhead = config.get('nhead', 8)
        num_layers = config.get('num_layers', 4)
        dim_feedforward = config.get('dim_feedforward', 1024)
        dropout = config.get('dropout', 0.1)
        self.sequence_length = config.get('sequence_length', 60)

        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=self.sequence_length)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层 - 为每个股票输出一个分数
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        x: [batch, num_stocks, seq_len, features]
        """
        batch_size, num_stocks, seq_len, features = x.size()

        # 合并batch和stock维度进行处理
        x = x.view(batch_size * num_stocks, seq_len, features)

        # 输入投影和位置编码
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 池化序列表示
        x = x.mean(dim=1)  # [batch*num_stocks, d_model]

        # 恢复到batch和stock维度
        x = x.view(batch_size, num_stocks, -1)

        # 输出每个股票的分数
        scores = self.fc_out(x).squeeze(-1)  # [batch, num_stocks]

        return scores


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)