SR = 44100
HOP_LENGTH = 256
# =============================================================================
# パラメータID定義
# =============================================================================
DYNAMIC_PARAM_IDS = [
    "DYN",
    "BRE",
    "BRI",
    "CLE",
    "GEN",
    "POR",
    "PIT",
    "PBS",
    "GWL",
    "XSY",
]

NOTE_PARAM_IDS = [
    "VEL",
    "accent",
    "bendDep",
    "bendLen",
    "decay",
    "fallPort",
    "opening",
    "risePort",
    "vibLen",
    "vibType",
    "vibDep",
    "vibRate",
]

MELODY_PARAM_IDS = ["NOTE_NUMBER", "DURATION", "PHONEME_P", "EVEC", "REPEAT"]
# MELODY_PARAM_IDS = ["NOTE_NUMBER", "DURATION", "PHONEME_P"]

STATIC_PARAM_IDS = [
    "BRE_1st_vocal",
    "BRI_1st_vocal",
    "CLE_1st_vocal",
    "GEN_1st_vocal",
    "OPE_1st_vocal",
    "BRE_2nd_vocal",
    "BRI_2nd_vocal",
    "CLE_2nd_vocal",
    "GEN_2nd_vocal",
    "OPE_2nd_vocal",
    "VOL_2nd_vocal",
]

ID_MAPPING = {
    "D": "DYN",
    "B": "BRE",
    "R": "BRI",
    "C": "CLE",
    "G": "GEN",
    "T": "POR",
    "P": "PIT",
    "S": "PBS",
    "W": "GWL",
    "X": "XSY",
}

# =============================================================================
# デフォルト値・上限値設定
# =============================================================================
DEFAULT_DYNAMIC_VALUES = {
    "DYN": 64,
    "BRE": 0,
    "BRI": 64,
    "CLE": 0,
    "GEN": 64,
    "POR": 64,
    "PIT": 0,
    "PBS": 1,
    "GWL": 0,
    "XSY": 0,
}

DEFAULT_NOTE_VALUES = {
    "VEL": 64,
    "accent": 50,
    "bendDep": 8,
    "bendLen": 0,
    "decay": 50,
    "fallPort": 0,
    "opening": 127,
    "risePort": 0,
    "vibLen": 66,
    "vibType": 1,
    "vibDep": 120,
    "vibRate": 30,
}

NOTE_PARAM_MAX_VALUES = {
    "VEL": 127.0,
    "accent": 100.0,
    "bendDep": 100.0,
    "bendLen": 100.0,
    "decay": 100.0,
    "fallPort": 1.0,
    "opening": 127.0,
    "risePort": 1.0,
    "vibLen": 100.0,
    "vibType": 16.0,
    "vibDep": 127,
    "vibRate": 127,
    "NOTE_NUMBER": 127.0,
    "DURATION": 7680.0,
}

# =============================================================================
# ボイスライブラリ設定
# =============================================================================
VOICE_LIB_ID_TO_NAME = {
    "BCNFCY43LB2LZCD4": "MIKU_V4X_Original_EVEC",
    "BFPL93T7GE3RWFC9": "MIKU_V4X_Soft_EVEC",
    "BKGHF6Y5PHTN4KD6": "MIKU_V4X_Solid_EVEC",
    "BHNME578NG2S3HD8": "MIKU_V4X_Dark",
    "BDGE8ZB2FCTKTDBA": "MIKU_V4X_Sweet",
}

VOICE_LIB_NAME_TO_ID = {
    "MIKU_V4X_Original_EVEC": 0,
    "MIKU_V4X_Soft_EVEC": 1,
    "MIKU_V4X_Solid_EVEC": 2,
    "MIKU_V4X_Dark": 3,
    "MIKU_V4X_Sweet": 4,
    "unknown": 5,
}

# EVEC ID定義
# モデルに「どんな強弱/表情か」を伝えるためのIDマップ
EVEC_ID_MAP = {
    "None": 0,
    # --- パターンA: ボイスリリース (Release) ---
    # 文末のブレスの長さなど
    "Release_Short": 1,
    "Release_Long": 2,
    # --- パターンB: ボイスカラー (Color / 母音用) ---
    # 母音(Vowel)に対する表情付け。子音用とはIDを分ける
    "Color_Soft": 3,
    "Color_Power": 4,
    # 必要に応じて Color_1 ~ Color_6 等を追加
    # --- パターンC: 子音拡張 (Expansion / 子音用) ---
    # 子音(Consonant)に対する強弱。母音用とはIDを分ける
    "Exp_Soft": 5,
    "Exp_Power": 6,
    # 必要に応じて Exp_1 ~ Exp_6 等を追加
}
# 母音リスト（パターンB判定用）
VOWELS = {"a", "i", "M", "e", "o"}

# =============================================================================
# ディレクトリ・実行設定 (ユーザー環境に合わせて修正)
# =============================================================================
WAV_DIR = "./data/wav_ALL"
VSQX_DIR = "./data/use_vsqx/got/aug"
OUTPUT_DIR = "./processed/augDYN_DYN_mel"
# WAV_DIR = "./predict/input_wav"
# VSQX_DIR = "./predict/input_vsqx"
# OUTPUT_DIR = "./processed/processed_predict"

# 特徴量抽出設定
# FEATURE_TYPES = ["mfcc","source_rms"]
FEATURE_TYPES = ["mel"]
# Target以外の動パラを抽出
#INPUT_DYNAMIC_IDS = [p for p in DYNAMIC_PARAM_IDS if p != "DYN"]
INPUT_VSQX_PARAMS = []


# # 出力にするものを辞書で指定
# TARGET_PARAM_CONFIG = {
#     "dynamic": ["DYN", "PIT"], # ダイナミクスとピッチを予測
#     # "note": ["vibRate"]      # 必要ならビブラートなども追加可能
# }
# 生成ターゲット設定
TARGET_PARAM_CONFIG = {"dynamic": ["DYN"]}
