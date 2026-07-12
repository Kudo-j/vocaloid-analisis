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
        self.dropout = nn.Dropout(p=dropout)  # 引数からDropout設定

        position = torch.arange(max_len).unsqueeze(1)
        # [0, 1, 2, ..., 4999]（形状: [5000]） という位置のシーケンスを作成
        # .unsqueeze(1):これを[[0], [1], [2], ..., [4999]]（形状: [5000, 1]）に
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        # pe (Positional Encoding) を格納するための、中身がすべて0のテンソル（テーブル）を作成
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # peの偶数番目の次元（0::2 は 0, 2, 4, ...列目）にsinの値を一括で代入
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe テーブルの奇数番目の次元（1::2 は 1, 3, 5, ...列目）にcosの値を一括で代入
        pe = pe.transpose(0, 1)  # (1, max_len, d_model)
        # このモデルは batch_first=True（形状: [バッチサイズ, フレーム数, 次元数]）で動作
        # pe の形状を [50000, 1, 512] から [1, 50000, 512] に変更

        self.register_buffer("pe", pe)
        # pe をこのモデルの重みではないがモデルの不可欠な部品として登録するためのPyTorchの命令
        # pe は学習はしないが、モデルを保存・ロード・GPU移動するときは、
        # 他の重みと常に一緒に行動させてほしいとPyTorchに伝えるための命令

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        # pe分を足して位置情報付加
        return x


class TransformerModel(nn.Module):
    """
    CQT+メタデータを入力とし、動的パラメータを予測するTransformerモデル
    Encoder-Decoder構造:
      - エンコーダ: 入力特徴量(X)を文脈表現(memory)に変換
      - デコーダ: memoryを参照しながら出力シーケンス(Y)を自己回帰的に生成
    """

    def __init__(
        self,
        num_x_features: int,
        num_y_features: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
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
            num_decoder_layers (int): Transformer Decoder Layerの層数
            dim_feedforward (int): FFNの内部次元数
            dropout (float): ドロップアウト率
            ma_kernel_size (int): 移動平均のカーネルサイズ
        """
        super().__init__()
        self.d_model = d_model
        self.num_y_features = num_y_features

        # ========== エンコーダ側 ==========
        # 1. 入力埋め込み層
        #    入力特徴量 (num_x_features) を d_model に射影する
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
            batch_first=True,  # (Batch, Seq, Feature) の形式を受け入れる
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        # ========== デコーダ側 ==========
        # 4. デコーダ入力埋め込み層
        #    出力特徴量 (num_y_features) を d_model に射影する
        self.decoder_embed = nn.Linear(num_y_features, d_model)

        # 5. デコーダ用位置エンコーディング
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # デコーダ用層正規化
        self.decoder_layer_norm = nn.LayerNorm(d_model)

        # 6. Transformerデコーダ
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers
        )

        # ========== 出力層 ==========
        # 7. 出力層
        #    d_model から出力特徴量 (num_y_features) に射影する
        self.output_linear = nn.Linear(d_model, num_y_features)

        self.moving_avg = nn.AvgPool1d(
            kernel_size=ma_kernel_size, stride=1, padding=(ma_kernel_size - 1) // 2
        )

        # 8. 出力活性化関数
        self.output_activation = nn.Tanh()

        # 重みの初期化 (オプションだが、パフォーマンスに寄与することがある)
        self._init_weights()

    def _init_weights(self):
        # Xavier初期化を適用
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        """
        因果マスク（上三角がTrue）を生成する。
        デコーダが未来のフレームを参照しないようにするためのマスク。
        Args:
            sz: シーケンス長
            device: デバイス
        Returns:
            (sz, sz) の因果マスク。True=マスクされる位置。
        """
        # Booleanマスク: True = マスク, False = 通過
        mask = torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )
        return mask

    def _encode(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        エンコーダ部分を実行し、memory を返す。
        """
        src_embed = self.input_embed(src) * math.sqrt(self.d_model)
        src_pe = self.pos_encoder(src_embed)
        src_norm = self.encoder_layer_norm(src_pe)
        memory = self.transformer_encoder(
            src_norm, src_key_padding_mask=src_key_padding_mask
        )
        return memory

    def _decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        デコーダ部分のみを実行する。Scheduled Sampling 用ヘルパー。

        Args:
            tgt: デコーダ入力, shape (B, T, num_y_features)
            memory: エンコーダ出力, shape (B, S, d_model)
            tgt_key_padding_mask: (B, T) True=無視
            memory_key_padding_mask: (B, S) True=無視
        Returns:
            出力, shape (B, T, num_y_features)
        """
        tgt_embed = self.decoder_embed(tgt) * math.sqrt(self.d_model)
        tgt_pe = self.pos_decoder(tgt_embed)
        tgt_norm = self.decoder_layer_norm(tgt_pe)

        tgt_seq_len = tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, tgt.device)

        decoder_out = self.transformer_decoder(
            tgt_norm,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        out_lin = self.output_linear(decoder_out)
        out_avg = self.moving_avg(out_lin)
        output = self.output_activation(out_avg)
        return output

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        学習時のforward（Teacher Forcing）。

        Args:
            src (torch.Tensor):
                エンコーダ入力シーケンス,
                shape (batch_size, src_seq_len, num_x_features)
            tgt (torch.Tensor):
                デコーダ入力シーケンス（正解Yを1フレーム右シフトしたもの）,
                shape (batch_size, tgt_seq_len, num_y_features)
            src_key_padding_mask (torch.Tensor, optional):
                エンコーダ側パディングマスク,
                shape (batch_size, src_seq_len). True=無視.
            tgt_key_padding_mask (torch.Tensor, optional):
                デコーダ側パディングマスク,
                shape (batch_size, tgt_seq_len). True=無視.
        Returns:
            torch.Tensor:
                出力シーケンス,
                shape (batch_size, tgt_seq_len, num_y_features)
        """
        # 1. エンコーダ
        memory = self._encode(src, src_key_padding_mask)

        # 2. デコーダ入力の埋め込み
        tgt_embed = self.decoder_embed(tgt) * math.sqrt(self.d_model)
        tgt_pe = self.pos_decoder(tgt_embed)
        tgt_norm = self.decoder_layer_norm(tgt_pe)

        # 3. 因果マスク生成（未来のフレームを見ないようにする）
        tgt_seq_len = tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, tgt.device)

        # 4. Transformerデコーダ
        decoder_out = self.transformer_decoder(
            tgt_norm,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # 5. 出力層: (B, S, D_model) -> (B, S, F_out)
        out_lin = self.output_linear(decoder_out)
        # 移動平均計算
        out_avg = self.moving_avg(out_lin)
        # 6. 最後に Tanh を通す
        output = self.output_activation(out_avg)

        return output

    def inference(
        self,
        src: torch.Tensor,
        max_len: int,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        推論時の自己回帰生成。
        1フレームずつデコーダで生成し、出力を次のステップの入力にフィードバックする。

        Args:
            src (torch.Tensor):
                エンコーダ入力, shape (batch_size, src_seq_len, num_x_features)
            max_len (int):
                生成する最大フレーム数
            src_key_padding_mask (torch.Tensor, optional):
                エンコーダ側パディングマスク
        Returns:
            torch.Tensor:
                生成された出力, shape (batch_size, max_len, num_y_features)
        """
        batch_size = src.size(0)
        device = src.device

        # エンコーダを1回だけ実行
        memory = self._encode(src, src_key_padding_mask)

        # 初期デコーダ入力: ゼロベクトル (開始トークン相当)
        # shape: (batch_size, 1, num_y_features)
        decoder_input = torch.zeros(batch_size, 1, self.num_y_features, device=device)

        outputs = []

        for t in range(max_len):
            # デコーダ入力の埋め込み
            tgt_embed = self.decoder_embed(decoder_input) * math.sqrt(self.d_model)
            tgt_pe = self.pos_decoder(tgt_embed)
            tgt_norm = self.decoder_layer_norm(tgt_pe)

            # 因果マスク
            tgt_len = decoder_input.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)

            # デコーダ実行
            decoder_out = self.transformer_decoder(
                tgt_norm,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # 最新フレームの出力のみ取得
            last_out = decoder_out[:, -1:, :]  # (B, 1, d_model)
            out_lin = self.output_linear(last_out)  # (B, 1, num_y_features)
            out_val = self.output_activation(out_lin)

            outputs.append(out_val)

            # 次ステップのデコーダ入力に追加
            decoder_input = torch.cat([decoder_input, out_val], dim=1)

        # 全フレームを結合: (B, max_len, num_y_features)
        result = torch.cat(outputs, dim=1)

        # 移動平均でスムージング
        result = self.moving_avg(result)

        return result
