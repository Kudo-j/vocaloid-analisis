"""
cnn_bilstm.py
=============
CNN + Bi-LSTM モデル。

研究計画のメインモデル。
1-d Convolution で入力を高次元化した後、Bi-LSTM で時系列の文脈を学習する。
オプションで Bi-LSTM の直後に Attention 層 (nn.MultiheadAttention) を挿入可能。

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
    ───Bi-LSTM Block───
    [Bi-LSTM (双方向、num_layers層、hidden_dim次元)] ── 時系列の文脈を学習
           ↓
    ───Attention Block (任意 / 残差接続付き)───
    [nn.MultiheadAttention] ── 長期依存・重要フレームの集約 (x + Attn(x))
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
    CNN + Bi-LSTM モデル (Attention オプション付き)。

    Args:
        input_dim: 入力特徴量の次元数。config から動的に決定される。
        output_dim: 出力次元数（推定対象パラメータ数）。
        conv_channels: 1-d Conv の出力チャンネル数。
        kernel_size: 1-d Conv のカーネルサイズ。
        hidden_dim: Bi-LSTM の隠れ層次元数。
        num_layers: Bi-LSTM の層数。
        dropout: Bi-LSTM / Attention のドロップアウト率。
        use_attention: Attention層 (nn.MultiheadAttention) を使用するかどうか。
        num_heads: MultiheadAttention のヘッド数。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        conv_channels=64,
        kernel_size=9,
        num_conv_layers=1,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3,
        use_attention=False,
        num_heads=4,
    ):
        super().__init__()

        self.use_attention = use_attention
        embed_dim = hidden_dim * 2

        # --- ConvBlock ---
        # padding を (kernel_size - 1) // 2 にすることで、
        # 入力と出力の時間長を同一に保つ (same padding)
        padding = (kernel_size - 1) // 2
        conv_layers_list = []
        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else conv_channels
            conv_layers_list.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        conv_channels,
                        kernel_size,
                        padding=padding,
                        padding_mode="reflect",
                        bias=False,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(),
                ]
            )
        self.conv_block = nn.Sequential(*conv_layers_list)

        # --- Bi-LSTM ---
        self.lstm = nn.LSTM(
            input_size=conv_channels,
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

        # --- 出力投影層 ---
        # Bi-LSTM / Attention の出力は hidden_dim * 2 (順方向 + 逆方向)
        self.output_proj = nn.Linear(embed_dim, output_dim)

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

        # Attention層 (残差接続付き)
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)  # (B, S, hidden_dim * 2)
            x = x + attn_out  # 残差接続 (Residual Connection)

        # 出力投影
        x = self.output_proj(x)  # (B, S, output_dim)
        x = self.output_activation(x)  # (B, S, output_dim)

        return x


if __name__ == "__main__":
    print("=== CNNBiLSTM 動作確認テスト ===")
    dummy_input = torch.randn(2, 100, 32)  # (Batch=2, Seq=100, Dim=32)
    print(f"入力テンソル形状: {dummy_input.shape}")

    for layers in [1, 2, 3]:
        model = CNNBiLSTM(
            input_dim=32,
            output_dim=3,
            conv_channels=64,
            kernel_size=19,
            num_conv_layers=layers,
            hidden_dim=64,
            num_layers=1,
        )
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        print(f"num_conv_layers={layers} 出力テンソル形状: {out.shape}")
        assert out.shape == (2, 100, 3), f"num_conv_layers={layers} の出力形状が不正です"

    print("すべての動作テストが正常に完了しました。")


