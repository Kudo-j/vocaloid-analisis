SR = 44100
HOP_LENGTH = 256


# =============================================================================
# ディレクトリ・実行設定 (ユーザー環境に合わせて修正)
# =============================================================================
WAV_DIR = "./data/Step1/wav"
VSQX_DIR = "./data/Step1/vsqx"
OUTPUT_DIR = "./processed/Step1"

# # --- テスト用設定 ---
# WAV_DIR = "../../Testdata"
# VSQX_DIR = "../../Testdata"
# OUTPUT_DIR = "../../Testdata/build_dataset_test_output"


# 特徴量抽出設定
# FEATURE_TYPES = ["mfcc","source_rms"]
FEATURE_TYPES = ["rms"]
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
