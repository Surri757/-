"""
Temporal Fusion Transformer (TFT) — speed-optimized.
Linear projection replaces expensive VSN. ~50K params, fast training.
Interface: (B, seq_len, n_features) -> (B,).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """Gated Residual Network"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm2(x + residual)


class TFTModel(nn.Module):
    """TFT — speed-optimized for 130 features on consumer GPU.

    Architecture:
      Input projection -> LSTM Encoder -> LSTM Decoder ->
      Multi-head Attention -> GRN -> Temporal pool -> Output

    Args:
        seq_len: input sequence length (default 60)
        n_features: number of input features
        d_model: hidden dimension (default 64)
        n_heads: attention heads (default 2)
        lstm_hidden: LSTM hidden size (default 48)
        dropout: dropout rate (default 0.1)
    """
    def __init__(self, seq_len=60, n_features=100, d_model=64,
                 n_heads=2, lstm_hidden=48, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input projection + feature gate
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(
            d_model, lstm_hidden, batch_first=True, bidirectional=False
        )

        # LSTM Decoder
        self.lstm_decoder = nn.LSTM(
            d_model, lstm_hidden, batch_first=True, bidirectional=False
        )
        self.encoder_proj = nn.Linear(lstm_hidden, d_model)
        self.decoder_proj = nn.Linear(lstm_hidden, d_model)

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Post-attention
        self.post_attn_grn = GRN(d_model, dropout)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # Output
        self.output_grn = GRN(d_model, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Instance normalization
        self.gamma = nn.Parameter(torch.ones(1, 1, n_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, n_features))

    def forward(self, x):
        # x: (B, seq_len, n_features)
        # Instance normalization
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - mean) / std * self.gamma + self.beta

        # Input projection -> d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # LSTM Encoder
        enc_out, (h_n, c_n) = self.lstm_encoder(x)  # (B, T, H)

        # LSTM Decoder
        dec_out, _ = self.lstm_decoder(x, (h_n, c_n))  # (B, T, H)

        # Project to d_model
        enc_d = self.encoder_proj(enc_out)    # (B, T, d_model)
        dec_d = self.decoder_proj(dec_out)    # (B, T, d_model)

        # Multi-head Attention
        attn_out, _ = self.attention(dec_d, enc_d, enc_d)
        attn_out = self.post_attn_norm(attn_out + dec_d)

        # GRN + pool + output
        grn_out = self.post_attn_grn(attn_out)
        pooled = grn_out[:, -8:, :].mean(dim=1) + grn_out.max(dim=1).values * 0.5
        pooled = self.output_grn(pooled.unsqueeze(1)).squeeze(1)

        return self.head(pooled).squeeze(-1)
