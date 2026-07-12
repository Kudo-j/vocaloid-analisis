import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Transformer本体は「どのフレームが1番目で、どのフレームが100番目か」わからない
    順序の情報を埋め込むための位置のタイムスタンプを計算する
    標準的なTransformerのサイン/コサインPositional Encoding
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 50000,
    ):
        # モデルの起動時に一度だけ実行
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransformerModel(nn.Module):
    """
    CQT+メタデータを入力とし、動的パラメータを予測するTransformerモデル
    Encoder単体構造 (Decoderなし):
      - エンコーダ: 入力特徴量(X)から直接出力シーケンス(Y)を生成
    """

    def __init__(
        self,
        num_x_features: int,
        num_y_features: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        ma_kernel_size: int = 15,
    ):
        """
        Args:
            num_x_features (int): 入力特徴量 (X) の次元数
            num_y_features (int): 出力特徴量 (Y) の次元数
            d_model (int): Transformerの内部次元数 (埋め込み次元)
            nhead (int): Multi-head Attentionのヘッド数
            num_encoder_layers (int): Transformer Encoder Layerの層数
            dim_feedforward (int): FFNの内部次元数
            dropout (float): ドロップアウト率
            ma_kernel_size (int): 移動平均のカーネルサイズ
        """
        super().__init__()
        self.d_model = d_model
        self.num_y_features = num_y_features

        # ========== エンコーダ側 ==========
        # 1. 入力埋め込み層
        self.input_embed = nn.Linear(num_x_features, d_model)

        # 2. エンコーダ用位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 層正規化
        self.encoder_layer_norm = nn.LayerNorm(d_model)

        # 3. Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        # ========== 出力層 ==========
        # 4. 出力層
        self.output_linear = nn.Linear(d_model, num_y_features)

        self.moving_avg = nn.AvgPool1d(
            kernel_size=ma_kernel_size, stride=1, padding=(ma_kernel_size - 1) // 2
        )

        # 5. 出力活性化関数
        self.output_activation = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor = None,  # trainスクリプト互換用（使わない）
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,  # trainスクリプト互換用
    ) -> torch.Tensor:
        """
        Args:
            src: (Batch, Seq_Len, Features)
        """
        src_embed = self.input_embed(src) * math.sqrt(self.d_model)
        src_pe = self.pos_encoder(src_embed)
        src_norm = self.encoder_layer_norm(src_pe)
        memory = self.transformer_encoder(
            src_norm, src_key_padding_mask=src_key_padding_mask
        )

        # 出力層: (B, S, D_model) -> (B, S, F_out)
        out_lin = self.output_linear(memory)
        
        # 移動平均計算: AvgPool1dは (Batch, Channel, Length) を期待するため転置が必要
        out_lin_transposed = out_lin.transpose(1, 2)
        out_avg_transposed = self.moving_avg(out_lin_transposed)
        out_avg = out_avg_transposed.transpose(1, 2)
        
        # 最後に Tanh を通す
        output = self.output_activation(out_avg)

        return output

    def inference(
        self,
        src: torch.Tensor,
        max_len: int,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        推論用メソッド (エンコーダのみなので一括で生成)
        """
        return self.forward(src, src_key_padding_mask=src_key_padding_mask)
