"""
cnn_bilstm.py
=============
CNN + Bi-LSTM モデル。

研究計画のメインモデル。
1-d Convolution で入力を高次元化した後、Bi-LSTM で時系列の文脈を学習する。

構成:
    [入力 (input_dim次元)]
           ↓
    ───ConvBlock───
    [1-dConv(out_channels, kernel_size, padding)] ── 高次元化
           ↓
    [BatchNorm1d] ── 学習の安定化と高速化
           ↓
    [ReLU] ── 非線形性を追加
           ↓
    ───ConvBlock───
    [Bi-LSTM (双方向、num_layers層、hidden_dim次元)] ── 時系列の文脈を学習
           ↓
    [Linear] ── hidden_dim*2 を output_dim へ低次元化
           ↓
    [tanh] ── 出力値を -1～1 の範囲に制限 (正規化範囲)
           ↓
    [出力 (output_dim次元)]
"""

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    CNN + Bi-LSTM モデル。

    Args:
        input_dim: 入力特徴量の次元数。config から動的に決定される。
        output_dim: 出力次元数（推定対象パラメータ数）。
        conv_channels: 1-d Conv の出力チャンネル数。
        kernel_size: 1-d Conv のカーネルサイズ。
        hidden_dim: Bi-LSTM の隠れ層次元数。
        num_layers: Bi-LSTM の層数。
        dropout: Bi-LSTM のドロップアウト率。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        conv_channels=64,
        kernel_size=9,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        # --- ConvBlock ---
        # padding を (kernel_size - 1) // 2 にすることで、
        # 入力と出力の時間長を同一に保つ (same padding)
        padding = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                input_dim, conv_channels, kernel_size, padding=padding, padding_mode='reflect', bias=False
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            # --- 2層目以降 ---
            # conv_blockを増やしたい場合は以下をコピペ
            # nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=padding, padding_mode='reflect', bias=False),
            # nn.BatchNorm1d(conv_channels),
            # nn.ReLU(),
        )

        # --- Bi-LSTM ---
        self.lstm = nn.LSTM(
            input_size=conv_channels,
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
        # Conv1d は (Batch, Channels, Seq) を期待するため転置
        x = x.permute(0, 2, 1)  # (B, input_dim, S)
        x = self.conv_block(x)  # (B, conv_channels, S)
        x = x.permute(0, 2, 1)  # (B, S, conv_channels)

        # Bi-LSTM
        x, _ = self.lstm(x)  # (B, S, hidden_dim * 2)

        # 出力投影
        x = self.output_proj(x)  # (B, S, output_dim)
        x = self.output_activation(x)  # (B, S, output_dim)

        return x
