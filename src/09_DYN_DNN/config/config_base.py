"""
config_base.py — 全実験共通の定数・パス設定
各実験configはこのファイルをimportし、必要に応じて上書きする。
"""
import os

# スクリプトの場所を基準にしたプロジェクトルート
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_09_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SRC_DIR = os.path.abspath(os.path.join(_09_ROOT, ".."))
_SRC_RESULT_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_RESULT_DIR, ".."))

# =============================================================================
# ===== データパス設定 =====
# =============================================================================
# パラメータ有り（チューニング済み）WAV・VSQXディレクトリ
TRAIN_TUNED_WAV_DIR = os.path.join(_PROJECT_ROOT, "data", "wav_ALL")
TRAIN_TUNED_VSQX_DIR = os.path.join(
    _PROJECT_ROOT, "data", "use_vsqx", "tuned",
)

# パラメータ無し（未チューニング）WAV・VSQXディレクトリ（実験Bで使用）
TRAIN_UNTUNED_WAV_DIR = os.path.join(_PROJECT_ROOT, "data", "wav_ALL")
TRAIN_UNTUNED_VSQX_DIR = os.path.join(
    _PROJECT_ROOT, "data", "use_vsqx", "untuned",
)

# データリーク防止のための楽曲リスト
SONGLIST_PATH = os.path.join(_PROJECT_ROOT, "songlist.txt")

# 推論・評価用
TARGET_WAV = os.path.join(_PROJECT_ROOT, "data", "wav_ALL", "Test", "Test1.wav")
TARGET_VSQX = os.path.join(_PROJECT_ROOT, "data", "vsqx_ALL", "Test", "Test1.vsqx")

EVAL_WAV_DIR = os.path.join(_PROJECT_ROOT, "data", "wav_ALL", "Test")
EVAL_VSQX_DIR = os.path.join(_PROJECT_ROOT, "data", "vsqx_ALL", "Test")

# util ディレクトリ（feature_extractor.py の場所）
UTIL_DIR = os.path.join(_SRC_DIR, "util")

# 外部モジュール参照パス
PATH_04_SRC = os.path.join(_SRC_DIR, "04_Parameter_Study", "src")
PATH_05 = os.path.join(_SRC_DIR, "05_DYN_Estimation")
PATH_07 = os.path.join(_SRC_DIR, "07_DYN_Direct_Estimation")

# =============================================================================
# ===== 音声処理共通定数 =====
# =============================================================================
SR = 44100
HOP_LENGTH = 256
N_MELS = 80
DYN_DEFAULT = 64.0

MEL_POWER_THRESHOLD_DB = -60.0  # これ以下のパワーは無音とみなしLoss計算から除外

# =============================================================================
# ===== VSQXパラメータID定義（Ref/config.py より完全コピー） =====
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

# VSQXのXMLタグID → パラメータ名のマッピング
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
# ===== デフォルト値・上限値設定（Ref/config.py より完全コピー） =====
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
# ===== ボイスライブラリ設定（Ref/config.py より完全コピー） =====
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
    "Release_Short": 1,
    "Release_Long": 2,
    # --- パターンB: ボイスカラー (Color / 母音用) ---
    "Color_Soft": 3,
    "Color_Power": 4,
    # --- パターンC: 子音拡張 (Expansion / 子音用) ---
    "Exp_Soft": 5,
    "Exp_Power": 6,
}
# 母音リスト（パターンB判定用）
VOWELS = {"a", "i", "M", "e", "o"}

# =============================================================================
# ===== VSQX入力・出力パラメータ設定 =====
# =============================================================================
# 前処理時にVSQXから抽出してモデル入力に含めるパラメータ
# 各実験configで上書き可能。空リスト = VSQXパラメータ入力なし
INPUT_VSQX_PARAMS = []

# 推定ターゲット設定（どのパラメータを出力として推定するか）
# 各実験configで上書き可能
TARGET_PARAM_CONFIG = {"dynamic": ["DYN"]}

# 音響特徴量設定（前処理時に抽出する音響特徴量の種類）
INPUT_FEATURES = ["rms"]

# 各音響特徴量ブランチの出力次元数
# 未指定の特徴量にはフォールバックとして BRANCH_HIDDEN_SIZE が使われる
# 例: FEATURE_HIDDEN_SIZES = {"mel": 64, "rms": 16}
FEATURE_HIDDEN_SIZES = {}

# ブランチの出力次元のデフォルト値
BRANCH_HIDDEN_SIZE = 32

# VSQXブランチ専用の出力次元数（0 or 未指定で BRANCH_HIDDEN_SIZE にフォールバック）
VSQX_HIDDEN_SIZE = 0

# 各特徴量ブランチの層タイプ
# "linear": 全結合層のみ（次元調整）, "cnn": 1次元CNN + 全結合層（周波数特徴抽出）
# 未指定のキーに対するデフォルトは "linear"
# 例: FEATURE_BRANCH_TYPES = {"mel": "cnn", "rms": "linear"}
FEATURE_BRANCH_TYPES = {}

# CNN ブランチの設定
CNN_NUM_FILTERS = 32      # 畳み込みフィルタ数
CNN_KERNEL_SIZE = 5       # カーネルサイズ（周波数方向）
CNN_NUM_LAYERS = 2        # CNN層の数

# メインモデル（ブランチ結合後の時系列学習モデル）
# "bilstm": BiLSTM, "transformer": Transformer Encoder
MAIN_MODEL_TYPE = "bilstm"

# =============================================================================
# ===== チャンク切り出し設定 =====
# =============================================================================
CHUNK_SIZE = 600   # 学習時に切り出すフレーム数 (約3.5秒)
CHUNK_HOP = 300    # スライディングウィンドウのずらし幅

# =============================================================================
# ===== 学習共通ハイパーパラメータ =====
# =============================================================================
BATCH_SIZE = 64
ACCUM_STEPS = 8
EPOCHS = 50
LEARNING_RATE = 2e-4
NUM_WORKERS = 2
DROPOUT = 0.3

# =============================================================================
# ===== Loss の重み =====
# =============================================================================
LOSS_DYN_WEIGHT = 1.0
LOSS_DIFF_WEIGHT = 10.0
# Disentangleモデル用（実験Bで使用）
LOSS_UNTUNED_WEIGHT = 0.1
