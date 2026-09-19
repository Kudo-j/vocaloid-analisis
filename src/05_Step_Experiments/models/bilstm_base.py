"""
bilstm_base.py
==============
Bi-LSTM 単体モデル。

研究計画 1-4 の比較実験用。
CNN や線形投影層を通さず、入力特徴量(RMS等)を直接 Bi-LSTM に入力する。
双方向ゲート層 (DirectionalGate) の切り替えに対応。
"""
import torch
import torch.nn as nn

try:
    from models.directional_gate import DirectionalGate
except ModuleNotFoundError:
    from directional_gate import DirectionalGate


class BiLSTMBase(nn.Module):
    """
    Bi-LSTM 単体モデル（CNN・入力線形層なし、DirectionalGate オプション付き）。

    Args:
        input_dim: 入力特徴量の次元数（直接LSTMの入力となる）。
        output_dim: 出力次元数。
        hidden_dim: Bi-LSTM の隠れ層次元数。
        num_layers: Bi-LSTM の層数。
        dropout: Bi-LSTM のドロップアウト率。
        use_gate: 双方向ゲート層 (DirectionalGate) を使用するかどうか。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        use_gate=False,
        **kwargs,
    ):
        super().__init__()

        self.use_gate = use_gate
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

        # --- 双方向ゲート層 (DirectionalGate) ---
        if self.use_gate:
            self.directional_gate = DirectionalGate(hidden_dim=hidden_dim)

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

        # 双方向ゲート層 (残差接続なし)
        if self.use_gate:
            x = self.directional_gate(x)  # (B, S, hidden_dim * 2)

        # 出力層を通す
        x = self.output_proj(x)  # (B, S, output_dim)
        x = self.output_activation(x)

        return x


if __name__ == "__main__":
    print("=== BiLSTMBase 動作確認テスト ===")
    dummy_input = torch.randn(2, 100, 32)
    for use_gate in [False, True]:
        model = BiLSTMBase(
            input_dim=32,
            output_dim=3,
            hidden_dim=64,
            num_layers=2,
            use_gate=use_gate,
        )
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        print(f"use_gate={use_gate} 出力形状: {out.shape}")
        assert out.shape == (2, 100, 3), "出力形状が不正です"
    print("BiLSTMBase 動作テスト正常完了。")