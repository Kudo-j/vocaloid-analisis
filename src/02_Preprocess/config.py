SR = 44100
HOP_LENGTH = 256


# =============================================================================
# ディレクトリ・実行設定 (ユーザー環境に合わせて修正)
# =============================================================================
# WAV_DIR = "./data/wav_ALL"
# VSQX_DIR = "./data/use_vsqx/got/aug"
# OUTPUT_DIR = "./processed/augDYN_DYN_mel"

# --- テスト用設定 ---
WAV_DIR = "../../Testdata"
VSQX_DIR = "../../Testdata"
OUTPUT_DIR = "../../Testdata/build_dataset_test_output"


# 特徴量抽出設定
# FEATURE_TYPES = ["mfcc","source_rms"]
FEATURE_TYPES = ["mel"]
# Target以外の動パラを抽出
# INPUT_DYNAMIC_IDS = [p for p in DYNAMIC_PARAM_IDS if p != "DYN"]
INPUT_VSQX_PARAMS = []


# # 出力にするものを辞書で指定
# TARGET_PARAM_CONFIG = {
#     "dynamic": ["DYN", "PIT"], # ダイナミクスとピッチを予測
#     # "note": ["vibRate"]      # 必要ならビブラートなども追加可能
# }
# 生成ターゲット設定
TARGET_PARAM_CONFIG = {"dynamic": ["DYN"]}
