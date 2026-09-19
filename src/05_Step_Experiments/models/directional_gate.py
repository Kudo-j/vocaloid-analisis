"""
directional_gate.py
===================
Bi-LSTM の順方向・逆方向の出力を相補的にスケーリングする双方向ゲート機構。

各フレームの隠れ状態からスカラーのゲート値 g (0.0～1.0) を計算し、
順方向成分を g 倍、逆方向成分を (1.0 - g) 倍にスケーリングして結合する。
これにより、特定のフレーム（ノート先頭など）で順方向の影響を抑制し、
逆方向の出力を優先的に活用する学習を可能にする。
"""

import torch
import torch.nn as nn


class DirectionalGate(nn.Module):
    """
    Bi-LSTM 双方向ゲート層。

    Args:
        hidden_dim: Bi-LSTM の片方向あたりの隠れ層次元数。
                    入力テンソルの最終次元は hidden_dim * 2 である必要があります。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # hidden_dim * 2 (順方向 + 逆方向) からフレームごとのスカラー重み (0.0～1.0) を計算
        self.gate_fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

        # 初期バイアスをゼロ初期化 (初期状態では g ≈ 0.5 で順逆均等に流れる)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq_len, hidden_dim * 2) - Bi-LSTM の出力テンソル
        Returns:
            fused: (Batch, Seq_len, hidden_dim * 2) - ゲート適用後のテンソル
        """
        # 順方向と逆方向に分割: それぞれ (Batch, Seq_len, hidden_dim)
        h_fwd, h_bwd = torch.chunk(x, chunks=2, dim=-1)

        # ゲート値の計算: (Batch, Seq_len, 1)
        gate = self.sigmoid(self.gate_fc(x))

        # 相補的スケーリング (ブロードキャスト乗算)
        h_fwd_gated = gate * h_fwd
        h_bwd_gated = (1.0 - gate) * h_bwd

        # 結合して元の形状 (Batch, Seq_len, hidden_dim * 2) を維持
        fused = torch.cat([h_fwd_gated, h_bwd_gated], dim=-1)
        return fused


if __name__ == "__main__":
    print("=== DirectionalGate 動作確認テスト ===")
    batch_size = 2
    seq_len = 10
    hidden_dim = 64

    gate_layer = DirectionalGate(hidden_dim=hidden_dim)
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim * 2)

    output = gate_layer(dummy_input)
    print(f"入力形状: {dummy_input.shape}")
    print(f"出力形状: {output.shape}")

    assert output.shape == dummy_input.shape, "出力テンソルの形状が一致しません"
    print("DirectionalGate 動作テスト成功。")
