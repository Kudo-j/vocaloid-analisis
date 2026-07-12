"""
model_factory.py — configからモデルをinstantiateするFactory
全モデルは dict を受け取り dict を返す統一インターフェースを持つ。
  入力: {"feature_name": (B, T, D), ...}
  出力: {"dyn_pred": (B, T, 1), ...}

設計方針 (リファクタリング後):
  - 各音響特徴量は独立したブランチ（Linear or CNN）で次元調整・空間特徴抽出される
  - ブランチのタイプは config.FEATURE_BRANCH_TYPES で特徴量ごとに指定可能
  - VSQXパラメータも独立した Linear ブランチで処理される
  - 全ブランチの出力を結合(concat)した後、メインモデル（BiLSTM or Transformer）で
    時系列学習を行い、DYN_Predictor で最終出力を得る
  - メインモデルの種類は config.MAIN_MODEL_TYPE で切り替え可能
"""
import torch
import torch.nn as nn
from models.encoders import (
    Linear_Branch,
    CNN_Branch,
    BiLSTM_Encoder,
    Transformer_Encoder,
    DYN_Predictor,
)


# =============================================================================
# 共通ヘルパー関数
# =============================================================================
def _get_feature_extractor(config):
    """feature_extractor を動的にimportして返す"""
    import sys
    if hasattr(config, "UTIL_DIR") and config.UTIL_DIR not in sys.path:
        sys.path.insert(0, config.UTIL_DIR)
    from feature_extractor import AudioFeatureExtractor
    return AudioFeatureExtractor()


def _get_branch_hidden_size(config, feature_name):
    """
    指定された特徴量のブランチ出力次元数を決定する。
    FEATURE_HIDDEN_SIZES に明示的な指定があればそれを使い、
    なければ BRANCH_HIDDEN_SIZE にフォールバックする。
    """
    sizes = getattr(config, "FEATURE_HIDDEN_SIZES", {})
    fallback = getattr(config, "BRANCH_HIDDEN_SIZE", 32)
    return sizes.get(feature_name, fallback)


def _get_vsqx_hidden_size(config):
    """VSQXブランチの出力次元数を決定する"""
    vsqx_h = getattr(config, "VSQX_HIDDEN_SIZE", 0)
    if vsqx_h > 0:
        return vsqx_h
    return getattr(config, "BRANCH_HIDDEN_SIZE", 32)


def _get_branch_type(config, feature_name):
    """
    指定された特徴量のブランチタイプを決定する。
    FEATURE_BRANCH_TYPES に明示的な指定があればそれを使い、
    なければ "linear" にフォールバックする。
    """
    types = getattr(config, "FEATURE_BRANCH_TYPES", {})
    return types.get(feature_name, "linear")


def _build_single_branch(config, feature_name, input_dim):
    """
    1つの特徴量に対するブランチモジュールを生成する。
    config.FEATURE_BRANCH_TYPES の指定に応じて Linear_Branch または CNN_Branch を返す。
    """
    hidden_dim = _get_branch_hidden_size(config, feature_name)
    dropout = getattr(config, "DROPOUT", 0.3)
    branch_type = _get_branch_type(config, feature_name)

    if branch_type == "cnn":
        num_filters = getattr(config, "CNN_NUM_FILTERS", 32)
        kernel_size = getattr(config, "CNN_KERNEL_SIZE", 5)
        num_layers = getattr(config, "CNN_NUM_LAYERS", 2)
        branch = CNN_Branch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif branch_type == "linear":
        branch = Linear_Branch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown branch type '{branch_type}' for feature '{feature_name}'. "
            f"Valid types: 'linear', 'cnn'"
        )

    return branch


def _build_feature_branches(config, extractor):
    """
    config.INPUT_FEATURES の各特徴量に対してブランチを生成する。
    ブランチのタイプは config.FEATURE_BRANCH_TYPES で特徴量ごとに指定可能。

    Returns:
        branches: nn.ModuleDict  {feature_name: Linear_Branch or CNN_Branch}
        total_out_dim: int  全ブランチの出力次元数の合計
    """
    features = getattr(config, "INPUT_FEATURES", ["rms"])

    branches = nn.ModuleDict()
    total_out_dim = 0

    for feat_name in features:
        input_dim = extractor.get_feature_dim(feat_name)
        branch = _build_single_branch(config, feat_name, input_dim)
        branches[feat_name] = branch
        total_out_dim += branch.out_dim

    return branches, total_out_dim


def _build_vsqx_branch(config):
    """
    INPUT_VSQX_PARAMS が設定されていればVSQX用の Linear_Branch を生成する。

    Returns:
        branch: Linear_Branch or None
        out_dim: int  (None の場合は 0)
    """
    vsqx_params = getattr(config, "INPUT_VSQX_PARAMS", [])
    if not vsqx_params:
        return None, 0

    vsqx_dim = len(vsqx_params)
    hidden_dim = _get_vsqx_hidden_size(config)
    dropout = getattr(config, "DROPOUT", 0.3)

    branch = Linear_Branch(
        input_dim=vsqx_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    return branch, branch.out_dim


def _build_main_model(config, input_dim):
    """
    結合後の時系列学習モデルを config.MAIN_MODEL_TYPE に基づいて生成する。

    Returns:
        model: BiLSTM_Encoder or Transformer_Encoder
    """
    main_type = getattr(config, "MAIN_MODEL_TYPE", "bilstm")
    num_layers = getattr(config, "LSTM_NUM_LAYERS", 2)
    dropout = getattr(config, "DROPOUT", 0.3)

    if main_type == "bilstm":
        hidden_dim = getattr(config, "LSTM_HIDDEN_SIZE", 64)
        model = BiLSTM_Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif main_type == "transformer":
        d_model = getattr(config, "TRANSFORMER_D_MODEL", 128)
        nhead = getattr(config, "TRANSFORMER_NHEAD", 4)
        model = Transformer_Encoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown MAIN_MODEL_TYPE: '{main_type}'. "
            f"Valid types: 'bilstm', 'transformer'"
        )

    return model


def _encode_features(branches, inputs, suffix=""):
    """
    全ブランチのforwardを実行し、出力をリストで返す。
    suffix が指定された場合、入力キーは "{feat_name}{suffix}" になる。
    例: suffix="_tuned" → inputs["rms_tuned"]
    """
    outputs = []
    for feat_name, branch in branches.items():
        key = f"{feat_name}{suffix}" if suffix else feat_name
        x = inputs[key]          # (B, T, D)
        out = branch(x)          # (B, T, hidden_dim)
        outputs.append(out)
    return outputs


# =============================================================================
# SmallRMSNet: RMS（1次元）向け軽量モデル（実験A用）
# 後方互換のため残すが、内部は新しいブランチ + メインモデル機構を使用
# =============================================================================
class SmallRMSNet(nn.Module):
    """
    RMS (1次元) 入力からDYNを推定する軽量モデル。
    内部では共通のブランチ + メインモデル機構を利用する。
    """
    def __init__(self, config):
        super().__init__()
        dropout = getattr(config, "DROPOUT", 0.3)
        kernel_size = getattr(config, "SMOOTHING_KERNEL_SIZE", 15)

        extractor = _get_feature_extractor(config)
        self.input_features = getattr(config, "INPUT_FEATURES", ["rms"])

        # 音響特徴量ブランチ（特徴量ごとに独立）
        self.branches, fusion_dim = _build_feature_branches(config, extractor)

        # VSQXブランチ（オプション）
        self.branch_vsqx, vsqx_out_dim = _build_vsqx_branch(config)
        fusion_dim += vsqx_out_dim

        # メインモデル（結合後の時系列学習）
        self.main_model = _build_main_model(config, fusion_dim)

        self.predictor = DYN_Predictor(
            input_dim=self.main_model.out_dim,
            output_dim=1,
            dropout=dropout,
            kernel_size=kernel_size,
        )

    def forward(self, inputs):
        """
        Args:
            inputs: dict {"rms": (B, T, 1), "vsqx": (B, T, V) (optional), ...}
        Returns:
            dict {"dyn_pred": (B, T, 1)}
        """
        parts = _encode_features(self.branches, inputs)

        if self.branch_vsqx is not None and "vsqx" in inputs:
            parts.append(self.branch_vsqx(inputs["vsqx"]))

        fused = torch.cat(parts, dim=-1)
        main_out = self.main_model(fused)
        dyn_pred = self.predictor(main_out)
        return {"dyn_pred": dyn_pred}


# =============================================================================
# BaselineSingleNet: 任意の入力特徴量をブランチ別に処理してDYNを推定
# =============================================================================
class BaselineSingleNet(nn.Module):
    """
    configで指定された入力特徴量を、それぞれ独立したブランチで次元調整し、
    結合後にメインモデルで時系列学習してDYNを推定するモデル。
    """
    def __init__(self, config):
        super().__init__()
        dropout = getattr(config, "DROPOUT", 0.3)
        kernel_size = getattr(config, "SMOOTHING_KERNEL_SIZE", 7)

        extractor = _get_feature_extractor(config)
        self.input_features = getattr(config, "INPUT_FEATURES", ["mel"])

        # 音響特徴量ブランチ（特徴量ごとに独立）
        self.branches, fusion_dim = _build_feature_branches(config, extractor)

        # VSQXブランチ（オプション）
        self.branch_vsqx, vsqx_out_dim = _build_vsqx_branch(config)
        fusion_dim += vsqx_out_dim

        # メインモデル（結合後の時系列学習）
        self.main_model = _build_main_model(config, fusion_dim)

        self.predictor = DYN_Predictor(
            input_dim=self.main_model.out_dim,
            output_dim=1,
            dropout=dropout,
            kernel_size=kernel_size,
        )

    def forward(self, inputs):
        """
        Args:
            inputs: dict {feature_name: (B, T, D), ..., "vsqx": (B, T, V) (optional)}
        Returns:
            dict {"dyn_pred": (B, T, 1)}
        """
        parts = _encode_features(self.branches, inputs)

        if self.branch_vsqx is not None and "vsqx" in inputs:
            parts.append(self.branch_vsqx(inputs["vsqx"]))

        fused = torch.cat(parts, dim=-1)
        main_out = self.main_model(fused)
        dyn_pred = self.predictor(main_out)
        return {"dyn_pred": dyn_pred}


# =============================================================================
# DiffNet: Tuned / Untuned / 差分 の3セット×特徴量ブランチ（実験B用）
# =============================================================================
class DiffNet(nn.Module):
    """
    Tuned音声特徴量・Untuned音声特徴量・その差分（Tuned - Untuned）を
    それぞれ独立したブランチで次元調整・特徴抽出し、結合後に
    メインモデル（BiLSTM or Transformer）で時系列学習してDYNを推定する。

    構造例 (INPUT_FEATURES = ["mel", "rms"], MAIN_MODEL_TYPE = "bilstm"):

      mel_tuned   --[CNN_Branch]----> (B,T,64) ┐
      rms_tuned   --[Linear_Branch]-> (B,T,16) ┤
      mel_untuned --[CNN_Branch]----> (B,T,64) ┤
      rms_untuned --[Linear_Branch]-> (B,T,16) ┼-> concat -> BiLSTM -> DYN_Predictor
      mel_diff    --[CNN_Branch]----> (B,T,64) ┤
      rms_diff    --[Linear_Branch]-> (B,T,16) ┤
      vsqx        --[Linear_Branch]-> (B,T,32) ┘
    """
    def __init__(self, config):
        super().__init__()
        dropout = getattr(config, "DROPOUT", 0.3)
        kernel_size = getattr(config, "SMOOTHING_KERNEL_SIZE", 15)

        extractor = _get_feature_extractor(config)
        self.input_features = getattr(config, "INPUT_FEATURES", ["rms"])

        # Tuned / Untuned / Diff それぞれに独立したブランチセットを生成
        self.branches_tuned, dim_per_set = _build_feature_branches(config, extractor)
        self.branches_untuned, _ = _build_feature_branches(config, extractor)
        self.branches_diff, _ = _build_feature_branches(config, extractor)

        # 3セットの合計
        fusion_dim = dim_per_set * 3

        # VSQXブランチ（オプション）
        self.branch_vsqx, vsqx_out_dim = _build_vsqx_branch(config)
        fusion_dim += vsqx_out_dim

        # メインモデル（結合後の時系列学習）
        self.main_model = _build_main_model(config, fusion_dim)

        self.predictor = DYN_Predictor(
            input_dim=self.main_model.out_dim,
            output_dim=1,
            dropout=dropout,
            kernel_size=kernel_size,
        )

    def forward(self, inputs):
        """
        Args:
            inputs: dict {
                "{feat}_tuned":   (B, T, D),
                "{feat}_untuned": (B, T, D),
                "vsqx": (B, T, V),  # optional
            }
        Returns:
            dict {"dyn_pred": (B, T, 1)}
        """
        # Tuned / Untuned の各ブランチを実行
        parts_t = _encode_features(self.branches_tuned, inputs, suffix="_tuned")
        parts_ut = _encode_features(self.branches_untuned, inputs, suffix="_untuned")

        # Diff ブランチ: 各特徴量の差分を計算してから encode
        parts_d = []
        for feat_name, branch in self.branches_diff.items():
            x_t = inputs[f"{feat_name}_tuned"]
            x_ut = inputs[f"{feat_name}_untuned"]
            x_diff = x_t - x_ut
            parts_d.append(branch(x_diff))

        parts = parts_t + parts_ut + parts_d

        if self.branch_vsqx is not None and "vsqx" in inputs:
            parts.append(self.branch_vsqx(inputs["vsqx"]))

        fused = torch.cat(parts, dim=-1)
        main_out = self.main_model(fused)
        dyn_pred = self.predictor(main_out)
        return {"dyn_pred": dyn_pred}


# =============================================================================
# Factory 関数
# =============================================================================
def get_model(config):
    """
    設定の MODEL_TYPE に応じてモデルを切り替える Factory 関数。
    全モデルは dict → dict のインターフェースを持つ。
    """
    model_type = getattr(config, "MODEL_TYPE", "baseline_single")

    if model_type == "small_rms":
        model = SmallRMSNet(config)
    elif model_type == "baseline_single":
        model = BaselineSingleNet(config)
    elif model_type == "diff_net":
        model = DiffNet(config)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

    # パラメータ数の表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type} | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # アーキテクチャの概要表示
    main_type = getattr(config, "MAIN_MODEL_TYPE", "bilstm")
    branch_types = getattr(config, "FEATURE_BRANCH_TYPES", {})
    features = getattr(config, "INPUT_FEATURES", ["rms"])
    branch_summary = {f: branch_types.get(f, "linear") for f in features}
    print(f"  Branches: {branch_summary}")
    print(f"  Main Model: {main_type}")

    return model
