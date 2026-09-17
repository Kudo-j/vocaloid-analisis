"""
lstm_base.py
============
単方向 LSTM（順方向 / 逆方向）モデル。

研究計画の比較実験（順方向単体、逆方向単体の精度評価）用。
CNN や入力線形投影層を通さず、入力特徴量を直接単方向 LSTM に入力する。
direction パラメータ ("forward" または "backward") により、順方向・逆方向を切り替え可能。
逆方向処理では、入力テンソルの時間軸を反転させて LSTM に入力し、
LSTM の出力を再度時間軸反転することで、時系列の順序を整えて後続層（Attention、Linear）へ渡す。
"""

import torch
import torch.nn as nn


class LSTMBase(nn.Module):
    """
    単方向 LSTM 単体モデル（CNN・入力線形層なし、順方向 / 逆方向対応）。

    Args:
        input_dim: 入力特徴量の次元数（直接LSTMの入力となる）。
        output_dim: 出力次元数。
        hidden_dim: LSTM の隠れ層次元数。
        num_layers: LSTM の層数。
        dropout: LSTM のドロップアウト率。
        direction: "forward" (順方向) または "backward" (逆方向)。
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
        direction="forward",
        use_attention=False,
        num_heads=4,
    ):
        super().__init__()

        direction = direction.lower()
        if direction not in ("forward", "backward"):
            raise ValueError(
                f"direction は 'forward' または 'backward' を指定してください: {direction}"
            )

        self.direction = direction
        self.use_attention = use_attention
        embed_dim = hidden_dim

        # --- 単方向 LSTM ---
        # bidirectional=False で単方向処理を行う
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
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
        # 逆方向の場合、時間軸 (dim=1) を反転させて入力する
        if self.direction == "backward":
            x = torch.flip(x, dims=[1])

        # LSTM に渡す (状態 hn, cn は _ で捨てる)
        x, _ = self.lstm(x)  # (B, S, hidden_dim)

        # 逆方向の場合、時間軸 (dim=1) を再反転して本来の時系列順序に戻す
        if self.direction == "backward":
            x = torch.flip(x, dims=[1])

        # Attention層 (残差接続付き)
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)  # (B, S, hidden_dim)
            x = x + attn_out  # 残差接続 (Residual Connection)

        # 出力層を通す
        x = self.output_proj(x)  # (B, S, output_dim)
        x = self.output_activation(x)

        return x


# ==========================================
# 動作確認用 Config 領域
# ==========================================
CONFIG = {
    "batch_size": 2,
    "seq_len": 100,
    "input_dim": 32,
    "output_dim": 3,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "use_attention": True,
    "num_heads": 4,
}


if __name__ == "__main__":
    print("=== LSTMBase 動作確認テスト ===")
    dummy_input = torch.randn(
        CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["input_dim"]
    )
    print(f"入力テンソル形状: {dummy_input.shape}")

    # 順方向モデルのテスト
    print("\n--- 順方向 (forward) テスト ---")
    model_fwd = LSTMBase(
        input_dim=CONFIG["input_dim"],
        output_dim=CONFIG["output_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        direction="forward",
        use_attention=CONFIG["use_attention"],
        num_heads=CONFIG["num_heads"],
    )
    model_fwd.eval()
    with torch.no_grad():
        out_fwd = model_fwd(dummy_input)
    print(f"順方向 出力テンソル形状: {out_fwd.shape}")
    assert out_fwd.shape == (
        CONFIG["batch_size"],
        CONFIG["seq_len"],
        CONFIG["output_dim"],
    ), "順方向の出力形状が不正です"

    # 逆方向モデルのテスト
    print("\n--- 逆方向 (backward) テスト ---")
    model_bwd = LSTMBase(
        input_dim=CONFIG["input_dim"],
        output_dim=CONFIG["output_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        direction="backward",
        use_attention=CONFIG["use_attention"],
        num_heads=CONFIG["num_heads"],
    )
    model_bwd.eval()
    with torch.no_grad():
        out_bwd = model_bwd(dummy_input)
    print(f"逆方向 出力テンソル形状: {out_bwd.shape}")
    assert out_bwd.shape == (
        CONFIG["batch_size"],
        CONFIG["seq_len"],
        CONFIG["output_dim"],
    ), "逆方向の出力形状が不正です"

    print("\nすべての動作テストが正常に完了しました。")
