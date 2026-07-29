# =============================================================================
# 母音別パラメータ影響差異解析ツール 設定ファイル
# =============================================================================
# 母音ごとにVOCALOIDパラメータの効果がどう異なるかを横断的に比較する。
# このファイルを編集してから python compare_vowels.py を実行するだけでOK。

# ===== 入出力ディレクトリ =====
DATA_DIR = "./src_result/Para_Study_data"  # wav/ と vsqx/ の親ディレクトリ
OUTPUT_DIR = "./src_result/Para_Study_data//output/vowel_compare"  # 出力先ディレクトリ

# ===== 解析対象パラメータ =====
# リストで複数指定可。全パラメータを一括解析したい場合は "ALL" を含める。
# 例: ["DYN"]  or  ["DYN", "BRE", "BRI"]  or  ["ALL"]
TARGET_PARAMS = ["DYN"]

# ===== 母音条件（自動スキャンで検出するが、ここで明示的に定義も可） =====
# 7つの母音条件ラベル。ファイル名のプレフィクスと対応する。
# 各条件は {"label": 表示名, "prefix": ファイル名プレフィクス, "type": 分類} で定義。
VOWEL_CONDITIONS = [
    # --- 単母音 (isolated) ---
    {"label": "あ (単音)", "prefix": "あ", "type": "isolated", "vowel": "a"},
    {"label": "い (単音)", "prefix": "い", "type": "isolated", "vowel": "i"},
    {"label": "う (単音)", "prefix": "う", "type": "isolated", "vowel": "u"},
    {"label": "え (単音)", "prefix": "え", "type": "isolated", "vowel": "e"},
    {"label": "お (単音)", "prefix": "お", "type": "isolated", "vowel": "o"},
]

# ===== デフォルトファイル設定 =====
# 各母音条件に対応する「パラメータなし」のデフォルト音声/VSQX。
# デフォルトファイルは default/ サブディレクトリに {prefix}_default.{ext} の形式で格納。
# 探索ロジック:
#   1. DEFAULT_DIR_WAV/{prefix}_default.wav を探す
#   2. 見つからなければ DEFAULT_DIR_WAV/あ_default.wav (共通デフォルト) にフォールバック
DEFAULT_DIR_WAV = "./src_result/Para_Study_data/wav/default"  # デフォルトWAVディレクトリ
DEFAULT_DIR_VSQX = "./src_result/Para_Study_data/vsqx/default"  # デフォルトVSQXディレクトリ

# 共通フォールバックデフォルト（個別デフォルトが見つからない場合のファイル名）
FALLBACK_DEFAULT_PREFIX = "あ"  # フォールバック時に使用するプレフィクス

# ===== 音声設定 =====
SR = 44100
HOP_LENGTH = 256

# ===== 音響特徴量設定 =====
# 全特徴量を一括計算し、母音間の差異を網羅的にマトリクスで出力する。
ALL_FEATURE_TYPES = [
    "stft",
    "rms",
    "f0_world",
    "mfcc",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "hnr",
    "zero_crossing_rate",
    # "cqt",
    "mel",
    "aperiodicity",
    "source_mel",
    "source_rms",
    "source_stft",
    # フォルマント帯域振幅
    "formant_1_amp",
    "formant_2_amp",
    "formant_3_amp",
    "formant_4_amp",
    "formant_5_amp",
    "formant_6_amp",
    "formant_7_amp",
    "air_freq_amp",
]

# 算出しない特徴量
ALWAYS_EXCLUDE_FEATURES = [
    "f0_crepe",  # torch/torchcrepe 依存のため除外
]

# ===== フォーカス特徴量 =====
# 統合比較プロット（棒グラフ等）で軸を固定して7条件を並べる特徴量。
# ここで指定した特徴量は、母音間比較の「メイン可視化」に使用される。
# None = 自動選出（相関上位）
FOCUS_FEATURES = [
    "rms",
    "source_rms",
    "stft",
    "source_stft",
    "formant_1_amp",
    "formant_2_amp",
    "formant_3_amp",
    "formant_4_amp",
    "formant_5_amp",
]
# 例: FOCUS_FEATURES = ["rms", "spectral_centroid", "hnr", "mfcc"]

# ===== 特徴量グループ設定 (compare_config.py と同一) =====
FEATURE_GROUPS = {
    "spectrogram": ["stft", "mel", "cqt"],
    "pitch": ["f0_world", "f0_crepe"],
    "noise": ["aperiodicity", "hnr", "zero_crossing_rate"],
    "energy": ["rms", "source_rms"],
    "timbre": ["mfcc", "spectral_centroid", "spectral_bandwidth", "spectral_contrast"],
    "source": ["source_mel", "source_rms", "source_stft"],
    "formants": [
        "formant_1_amp",
        "formant_2_amp",
        "formant_3_amp",
        "formant_4_amp",
        "formant_5_amp",
        "formant_6_amp",
        "formant_7_amp",
        "air_freq_amp",
    ],
}
MAX_PER_GROUP = 1

# ===== 相関分析設定 =====
COMPUTE_CORRELATION = True
TOP_N_FEATURES = 5  # フォーカス特徴量が None の場合、相関上位何個を自動選出するか

# ===== 可視化設定 =====
DPI = 300
FIGSIZE_MATRIX = (20, 12)  # 相関マトリクスヒートマップ
FIGSIZE_BAR = (18, 10)  # 棒グラフ比較
FIGSIZE_DETAIL = (20, 24)  # 詳細時系列プロット


