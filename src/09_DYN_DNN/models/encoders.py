"""
encoders.py — エンコーダ・ブランチ定義

ブランチ（結合前の次元調整・特徴抽出）:
  - Linear_Branch: 全結合層のみで次元調整を行う軽量ブランチ
  - CNN_Branch: 1次元CNNで周波数方向の特徴を抽出してから次元調整するブランチ

メインモデル（結合後の時系列学習）:
  - BiLSTM_Encoder: 双方向LSTM
  - Transformer_Encoder: Transformer Encoder

出力ヘッド:
  - DYN_Predictor: 時間軸方向の平滑化フィルタ付き推定ヘッド
"""
import torch
import torch.nn as nn
import math


# =============================================================================
# ブランチ用モジュール（結合前: 次元調整・空間特徴抽出）
# =============================================================================
class Linear_Branch(nn.Module):
    """
    全結合層のみで入力次元を hidden_dim に変換するブランチ。
    RMSやVSQXなど、周波数構造を持たない特徴量向け。
    時系列（Time）方向の学習は行わない。

    入力: (Batch, Time, input_dim)
    出力: (Batch, Time, hidden_dim)
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class CNN_Branch(nn.Module):
    """
    1次元CNNで周波数方向の局所的な特徴を抽出し、hidden_dim に変換するブランチ。
    Melスペクトログラムなど、周波数次元を持つ特徴量向け。
    時系列（Time）方向の学習は行わない。

    入力: (Batch, Time, Freq)
    出力: (Batch, Time, hidden_dim)

    内部処理:
      1. (B, T, Freq) -> (B*T, 1, Freq) に reshape して
         周波数方向に Conv1d を適用
      2. CNNの出力を Flatten して Linear で hidden_dim に変換
    """
    def __init__(self, input_dim, hidden_dim, num_filters=32,
                 kernel_size=5, num_layers=2, dropout=0.3):
        super().__init__()

        # CNN層を積み上げる
        layers = []
        in_channels = 1
        for i in range(num_layers):
            out_channels = num_filters * (2 ** i)  # フィルタ数を段階的に増やす
            padding = kernel_size // 2
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # CNN出力次元を計算（padding=same なので周波数次元は変わらない）
        cnn_out_dim = in_channels * input_dim

        # 全結合層で hidden_dim に変換
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: (Batch, Time, Freq)
        B, T, F = x.shape

        # (B*T, 1, Freq) に reshape して周波数方向に畳み込み
        x = x.reshape(B * T, 1, F)
        x = self.cnn(x)                  # (B*T, C_out, Freq)
        x = x.reshape(B * T, -1)         # (B*T, C_out * Freq)
        x = self.fc(x)                   # (B*T, hidden_dim)
        x = x.reshape(B, T, -1)          # (B, T, hidden_dim)
        return x


# =============================================================================
# メインモデル用モジュール（結合後: 時系列学習）
# =============================================================================
class BiLSTM_Encoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        # x: (Batch, Seq, Features)
        out, _ = self.lstm(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim=80, d_model=128, nhead=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.out_dim = d_model

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out


# =============================================================================
# 出力ヘッド
# =============================================================================
class DYN_Predictor(nn.Module):
    """時間軸方向の平滑化フィルタ付き DYN 推定ヘッド"""
    def __init__(self, input_dim, output_dim=1, dropout=0.3, kernel_size=7):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 時間軸方向に平滑化を行う1次元畳み込み層
        padding = kernel_size // 2
        self.smooth_conv = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        
        self.net2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (Batch, Time, enc_dim)
        out = self.net1(x)                # (Batch, Time, 64)
        
        # Conv1dは入力形状として (Batch, Channel, Time) を要求するため転置
        out = out.transpose(1, 2)         # (Batch, 64, Time)
        out = self.smooth_conv(out)       # 平滑化フィルタの適用
        out = torch.relu(out)
        out = out.transpose(1, 2)         # (Batch, Time, 64) に戻す
        
        out = self.net2(out)              # (Batch, Time, 1)
        return out
