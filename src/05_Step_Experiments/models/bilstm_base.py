"""
bilstm_base.py
==============
Bi-LSTM 単体モデル。

研究計画 1-4 の比較実験用。
CNN や線形投影層を通さず、入力特徴量(RMS等)を直接 Bi-LSTM に入力する。
アテンション層（nn.MultiheadAttention）の切り替えに対応。
"""
import torch
import torch.nn as nn


class BiLSTMBase(nn.Module):
    """
    Bi-LSTM 単体モデル（CNN・入力線形層なし）。

    Args:
        input_dim: 入力特徴量の次元数（直接LSTMの入力となる）。
        output_dim: 出力次元数。
        hidden_dim: Bi-LSTM の隠れ層次元数。
        num_layers: Bi-LSTM の層数。
        dropout: Bi-LSTM のドロップアウト率。
        use_attention: Attention層 (nn.MultiheadAttention) を使用するかどうか。
        num_heads: MultiheadAttention のヘッド数。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        use_attention=False,
        num_heads=4,
    ):
        super().__init__()

        self.use_attention = use_attention
        embed_dim = hidden_dim * 2

        # --- Bi-LSTM ---
        # 入力次元(input_dim)を直接LSTMで受け取る
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Attention層 ---
        if self.use_attention:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                )
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout,
            )

        # --- 全結合層 (出力投影層) ---
        self.output_proj = nn.Linear(embed_dim, output_dim)

        # --- 出力活性化関数 ---
        self.output_activation = nn.Tanh()

    def forward(self, x, **kwargs):
        """Args:
            x: (Batch, Seq, input_dim)
        Returns:
            x: (Batch, Seq, output_dim)
        """
        # 入力 x を直接 Bi-LSTM に渡す (状態 hn, cn は _ で捨てる)
        x, _ = self.lstm(x)  # (B, S, hidden_dim * 2)

        # Attention層 (残差接続付き)
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)  # (B, S, hidden_dim * 2)
            x = x + attn_out  # 残差接続 (Residual Connection)

        # 出力層を通す
        x = self.output_proj(x)  # (B, S, output_dim)
        x = self.output_activation(x)

        return x