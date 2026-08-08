"""
bilstm_base.py
==============
Bi-LSTM 単体モデル。

研究計画 1-4 の比較実験用。
CNN や線形投影層を通さず、入力特徴量(RMS等)を直接 Bi-LSTM に入力する。
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
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        # --- Bi-LSTM ---
        # 入力次元(input_dim)を直接LSTMで受け取る
        # 層数は上部で定義した NUM_LSTM_LAYERS を参照する
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- 出力投影層 ---
        # Bi-LSTM の出力は hidden_dim * 2 (順方向 + 逆方向)
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

        # --- 出力活性化関数 ---
        self.output_activation = nn.Tanh()

    def forward(self, x, **kwargs):
        """
        Args:
            x: (Batch, Seq, input_dim)
        Returns:
            out: (Batch, Seq, output_dim)
        """
        # 入力 x を直接 Bi-LSTM に渡す
        x, _ = self.lstm(x)  # (B, S, hidden_dim * 2)

        # 出力層を通す
        x = self.output_proj(x)  # (B, S, output_dim)
        x = self.output_activation(x)

        return x
