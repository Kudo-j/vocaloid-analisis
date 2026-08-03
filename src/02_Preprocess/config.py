SR = 44100
HOP_LENGTH = 256


# =============================================================================
# ディレクトリ・実行設定 (ユーザー環境に合わせて修正)
# =============================================================================
# WAV_DIR = "./data/Step1/wav"
# VSQX_DIR = "./data/Step1/vsqx"
# OUTPUT_DIR = "./processed/Step1"

# --- テスト用設定 ---
WAV_DIR = "../../Testdata"
VSQX_DIR = "../../Testdata"
OUTPUT_DIR = "../../Testdata/build_dataset_test_output"


# 入力にする音響特徴量を辞書で指定（マルチブランチ対応）
INPUT_AUDIO_PARAMS = {
    "audio_rms": ["rms"],
    # "audio_mel": ["mel"] # 追加する場合はこのように書く
}

# 互換性のためフラットなリストも自動生成
FEATURE_TYPES = [f for features in INPUT_AUDIO_PARAMS.values() for f in features]
# 入力にするパラメータを辞書で指定（ブランチ分けに対応）
# 例1（単一結合）: {"all_inputs": ["NOTE_NUM", "PHONEME_P", "DYN", "PIT"]}
# 例2（ブランチ）: {"branch_basic": ["NOTE_NUM", "PHONEME_P"], "branch_voice": ["DYN", "PIT"]}
INPUT_VSQX_PARAMS = {}

# # 出力にするものを辞書で指定
# TARGET_PARAM_CONFIG = {
#     "dynamic": ["DYN", "PIT"], # ダイナミクスとピッチを予測
#     # "note": ["vibRate"]      # 必要ならビブラートなども追加可能
# }
# 生成ターゲット設定
TARGET_PARAM_CONFIG = {"dynamic": ["DYN"]}
