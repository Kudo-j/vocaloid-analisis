import torch
import torch.nn as nn
import math

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
# Phase 1: Profile抽出器
# 「DYNが変化したとき、各周波数帯がどの程度影響を受けるか」のテンプレートを推定する
# =============================================================================
class ProfileExtractor(nn.Module):
    def __init__(self, input_dim, output_dim=80, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x: (Batch, Time, enc_dim)
        return self.net(x)  # (Batch, Time, 80)

# =============================================================================
# Phase 2: DYN推定器
# Encoder特徴量からDYNを直接推定する（平滑化フィルタ付き）
# =============================================================================
class DYN_Predictor(nn.Module):
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

# =============================================================================
# メインモデル: DisentangleNet
# 2段階学習構成:
#   Step 1: Encoder + ProfileExtractor を学習（forward_step1）
#   Step 2: DYN_Predictor を学習、Encoder + ProfileExtractor は凍結（forward）
# =============================================================================
class DisentangleNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_type = config.ENCODER_TYPE.lower()
        
        # 指定された入力特徴量の合計次元数を計算
        self.input_dim = sum(config.FEATURE_DIMS[f] for f in getattr(config, "INPUT_FEATURES", ["mel"]))
        
        if self.encoder_type == "bilstm":
            self.encoder = BiLSTM_Encoder(
                input_dim=self.input_dim,
                hidden_dim=config.LSTM_HIDDEN_SIZE,
                num_layers=config.LSTM_NUM_LAYERS,
                dropout=config.DROPOUT
            )
        elif self.encoder_type == "transformer":
            self.encoder = Transformer_Encoder(
                input_dim=self.input_dim,
                d_model=config.TF_D_MODEL,
                nhead=config.TF_NHEAD,
                num_layers=config.TF_NUM_LAYERS,
                dropout=config.DROPOUT
            )
        else:
            raise ValueError(f"Unknown ENCODER_TYPE: {self.encoder_type}")

        # Profile抽出器: Encoder特徴量 → 変化プロファイル
        self.profile_extractor = ProfileExtractor(
            input_dim=self.encoder.out_dim,
            output_dim=self.input_dim,
            dropout=config.DROPOUT
        )
        
        # DYN推定器: Encoder特徴量 → DYN (1次元)
        kernel_size = getattr(config, "SMOOTHING_KERNEL_SIZE", 7)
        self.dyn_predictor = DYN_Predictor(
            input_dim=self.encoder.out_dim,
            output_dim=1,
            dropout=config.DROPOUT,
            kernel_size=kernel_size
        )

    def forward_step1(self, mel_tuned, true_dyn):
        """
        Step 1 学習用: Profile抽出器の独立学習
        正解DYNを使い、「Profile × DYN = 真の差分」を学習させる
        
        Args:
            mel_tuned: (B, T, 80)
            true_dyn:  (B, T, 1)
        Returns:
            delta_pred: (B, T, 80) 予測差分
            profile:    (B, T, 80) 周波数変化プロファイル
        """
        enc_out = self.encoder(mel_tuned)
        profile = self.profile_extractor(enc_out)      # (B, T, 80)
        delta_pred = profile * true_dyn                # (B, T, 80) * (B, T, 1) → (B, T, 80)
        return delta_pred, profile

    def forward(self, mel_tuned):
        """
        Step 2 学習 / 推論用: DYN推定 → Profile × DYN で Untuned逆算
        
        Args:
            mel_tuned: (B, T, 80)
        Returns:
            mel_untuned_pred: (B, T, 80) 逆算された未調声メルスペクトログラム
            dyn_pred:         (B, T, 1) 推定DYN
        """
        enc_out = self.encoder(mel_tuned)
        profile = self.profile_extractor(enc_out)      # (B, T, 80)
        dyn_pred = self.dyn_predictor(enc_out)         # (B, T, 1)
        delta_pred = profile * dyn_pred                # (B, T, 80)
        mel_untuned_pred = mel_tuned - delta_pred
        return mel_untuned_pred, dyn_pred

    def freeze_step1_modules(self):
        """Step 1 で学習済みの Encoder + ProfileExtractor を凍結する"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.profile_extractor.parameters():
            param.requires_grad = False
        
        # 凍結した/していないパラメータ数を表示
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Frozen params: {frozen:,} | Trainable params: {trainable:,}")

# =============================================================================
# ベースラインモデル: BaselineSingleNet
# 1種類あるいは複数種類の入力特徴量（結合済）から直接DYNを推定するシンプルなモデル
# =============================================================================
class BaselineSingleNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_type = config.ENCODER_TYPE.lower()
        
        # 指定された入力特徴量の合計次元数を計算
        self.input_dim = sum(config.FEATURE_DIMS[f] for f in getattr(config, "INPUT_FEATURES", ["mel"]))
        
        if self.encoder_type == "bilstm":
            self.encoder = BiLSTM_Encoder(
                input_dim=self.input_dim,
                hidden_dim=config.LSTM_HIDDEN_SIZE,
                num_layers=config.LSTM_NUM_LAYERS,
                dropout=config.DROPOUT
            )
        elif self.encoder_type == "transformer":
            self.encoder = Transformer_Encoder(
                input_dim=self.input_dim,
                d_model=config.TF_D_MODEL,
                nhead=config.TF_NHEAD,
                num_layers=config.TF_NUM_LAYERS,
                dropout=config.DROPOUT
            )
        else:
            raise ValueError(f"Unknown ENCODER_TYPE: {self.encoder_type}")

        kernel_size = getattr(config, "SMOOTHING_KERNEL_SIZE", 7)
        self.dyn_predictor = DYN_Predictor(
            input_dim=self.encoder.out_dim,
            output_dim=1,
            dropout=config.DROPOUT,
            kernel_size=kernel_size
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, input_dim) 結合済みの特徴量
        Returns:
            dyn_pred: (B, T, 1) 推定DYN
        """
        enc_out = self.encoder(x)
        dyn_pred = self.dyn_predictor(enc_out)
        return dyn_pred

def get_model(config):
    """
    設定の TRAINING_MODE に応じてモデルを切り替える Factory 関数
    """
    mode = getattr(config, "TRAINING_MODE", "disentangle_step2")
    if mode in ["disentangle_step1", "disentangle_step2"]:
        return DisentangleNet(config)
    elif mode == "baseline_single":
        return BaselineSingleNet(config)
    else:
        raise ValueError(f"Unknown TRAINING_MODE: {mode}")
