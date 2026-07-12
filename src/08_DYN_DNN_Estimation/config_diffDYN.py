import os

# スクリプトの場所を基準にしたプロジェクトルート
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

# =============================================================================
# ===== データパス設定 =====
# =============================================================================
# パラメータ有り・無しのWAV・VSQXディレクトリ
TRAIN_TUNED_WAV_DIR = os.path.join(_PROJECT_ROOT, "data", "wav_ALL")  # パラメータ有wav
TRAIN_TUNED_VSQX_DIR = os.path.join(
    _PROJECT_ROOT, "data", "use_vsqx", "got", "aug"
)  # パラメータ有vsqx
TRAIN_UNTUNED_WAV_DIR = os.path.join(
    _PROJECT_ROOT, "data", "wav_ALL"
)  # パラメータ無しwav
TRAIN_UNTUNED_VSQX_DIR = os.path.join(
    _PROJECT_ROOT, "data", "use_vsqx_diffDYN", "non_DYN"
)  # パラメータ無しvsqx

# 前処理結果（.npz）の保存先
# PREPROCESSED_DIR = "./preprocessed_diffDYN"
PREPROCESSED_DIR = os.path.join(_PROJECT_ROOT, "processed_diffDYN_Active")

# データリーク防止のための楽曲リスト
SONGLIST_PATH = os.path.join(_PROJECT_ROOT, "songlist.txt")

# 推論・評価用
TARGET_WAV = os.path.join(_PROJECT_ROOT, "data", "wav_ALL", "Test", "Test1.wav")
TARGET_VSQX = os.path.join(_PROJECT_ROOT, "data", "vsqx_ALL", "Test", "Test1.vsqx")

EVAL_WAV_DIR = os.path.join(_PROJECT_ROOT, "data", "wav_ALL", "Test")
EVAL_VSQX_DIR = os.path.join(_PROJECT_ROOT, "data", "vsqx_ALL", "Test")
EVAL_OUTPUT_EXCEL = os.path.join(_SCRIPT_DIR, "output", "diffDYN_eval_results.xlsx")

OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output_2step")
MODEL_DIR = os.path.join(_SCRIPT_DIR, "models_2step")
MODEL_PATH = os.path.join(MODEL_DIR, "best_diffDYN_model.pth")

# Step 1 (Profile学習) の設定
STEP1_MODEL_PATH = os.path.join(MODEL_DIR, "step1_profile_model.pth")
STEP1_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output_step1")
STEP1_EPOCHS = 30
STEP1_LEARNING_RATE = 2e-4

# =============================================================================
# ===== アーキテクチャ・特徴量設定 =====
# =============================================================================
# 学習モードの選択: "disentangle_step1", "disentangle_step2", "baseline_single" など
TRAINING_MODE = "disentangle_step2"

# 使用する入力特徴量のリスト
INPUT_FEATURES = ["mel"]

# 各特徴量の次元数
FEATURE_DIMS = {
    "mel": 80,
    "rms": 1,
    "mfcc": 20
}

# =============================================================================
# ===== モデル設計設定 =====
# =============================================================================
# エンコーダの種類: "bilstm" または "transformer"
ENCODER_TYPE = "bilstm"

# BiLSTMの設定
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 4

# Transformerの設定
TF_D_MODEL = 128
TF_NHEAD = 4
TF_NUM_LAYERS = 3

# モデル共通設定
DROPOUT = 0.3
SMOOTHING_KERNEL_SIZE = 15
# スパイク防止用の学習可能な平滑化フィルタの範囲（奇数を推奨。5や7がおすすめ）

# =============================================================================
# ===== 学習・データセット設定 =====
# =============================================================================
SR = 44100
HOP_LENGTH = 256
N_MELS = 80
DYN_DEFAULT = 64.0

MEL_POWER_THRESHOLD_DB = -60.0  # これ以下のパワーは無音とみなしLoss計算から除外

# チャンク切り出し設定
CHUNK_SIZE = 600  # 学習時に切り出すフレーム数 (約3.5秒)
CHUNK_HOP = 300  # スライディングウィンドウのずらし幅

# 学習ハイパーパラメータ
BATCH_SIZE = 64
ACCUM_STEPS = 8  # 勾配累積のステップ数
EPOCHS = 50
LEARNING_RATE = 2e-4
NUM_WORKERS = 2

# Lossの重み (マルチタスク学習)
# Loss = (LOSS_UNTUNED_WEIGHT * Untuned推定誤差) + (LOSS_DYN_WEIGHT * DYN推定誤差) + (LOSS_DIFF_WEIGHT * DYN微分誤差)
LOSS_UNTUNED_WEIGHT = 0.1
LOSS_DYN_WEIGHT = 1.0
LOSS_DIFF_WEIGHT = 10.0
