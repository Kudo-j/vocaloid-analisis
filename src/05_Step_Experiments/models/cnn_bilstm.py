"""
cnn_bilstm.py
=============
CNN + Bi-LSTM モデル。

研究計画のメインモデル。
1-d Convolution で入力を高次元化した後、Bi-LSTM で時系列の文脈を学習する。
オプションで Bi-LSTM の直後に双方向ゲート層 (DirectionalGate) を挿入可能。

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
    ───Directional Gate Block (任意 / 残差接続なし)───
    [DirectionalGate] ── 順逆の相補的スケーリング (gate*fwd, (1-gate)*bwd)
           ↓
    [Linear] ── hidden_dim*2 を output_dim へ低次元化
           ↓
    [tanh] ── 出力値を -1～1 の範囲に制限 (正規化範囲)
           ↓
    [出力 (output_dim次元)]
"""

import torch
import torch.nn as nn

try:
    from models.directional_gate import DirectionalGate
except ModuleNotFoundError:
    from directional_gate import DirectionalGate


class CNNBiLSTM(nn.Module):
    """
    CNN + Bi-LSTM モデル (DirectionalGate オプション付き)。

    Args:
        input_dim: 入力特徴量の次元数。config から動的に決定される。
        output_dim: 出力次元数（推定対象パラメータ数）。
        conv_channels: 1-d Conv の出力チャンネル数。
        kernel_size: 1-d Conv のカーネルサイズ。
        num_conv_layers: 1-d Conv の層数。
        hidden_dim: Bi-LSTM の隠れ層次元数。
        num_layers: Bi-LSTM の層数。
        dropout: Bi-LSTM のドロップアウト率。
        use_gate: 双方向ゲート層 (DirectionalGate) を使用するかどうか。
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
        use_gate=False,
        **kwargs,
    ):
        super().__init__()

        self.use_gate = use_gate
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

        # --- 双方向ゲート層 (DirectionalGate) ---
        if self.use_gate:
            self.directional_gate = DirectionalGate(hidden_dim=hidden_dim)

        # --- 出力投影層 ---
        # Bi-LSTM / Gate の出力は hidden_dim * 2 (順方向 + 逆方向)
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

        # 双方向ゲート層 (残差接続なし)
        if self.use_gate:
            x = self.directional_gate(x)  # (B, S, hidden_dim * 2)

        # 出力投影
        x = self.output_proj(x)  # (B, S, output_dim)
        x = self.output_activation(x)  # (B, S, output_dim)

        return x


if __name__ == "__main__":
    print("=== CNNBiLSTM 動作確認テスト ===")
    dummy_input = torch.randn(2, 100, 32)  # (Batch=2, Seq=100, Dim=32)
    print(f"入力テンソル形状: {dummy_input.shape}")

    for use_gate in [False, True]:
        for layers in [1, 2]:
            model = CNNBiLSTM(
                input_dim=32,
                output_dim=3,
                conv_channels=64,
                kernel_size=19,
                num_conv_layers=layers,
                hidden_dim=64,
                num_layers=1,
                use_gate=use_gate,
            )
            model.eval()
            with torch.no_grad():
                out = model(dummy_input)
            print(f"use_gate={use_gate}, num_conv_layers={layers} 出力テンソル形状: {out.shape}")
            assert out.shape == (2, 100, 3), "出力形状が不正です"

    print("すべての動作テストが正常に完了しました。")


