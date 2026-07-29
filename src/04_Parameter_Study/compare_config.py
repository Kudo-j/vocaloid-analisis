# =============================================================================
# VOCALOIDパラメータ差分解析ツール 設定ファイル
# =============================================================================
# このファイルを編集してから python compare.py を実行するだけでOK

# ===== 入力ファイル設定 =====
DATA_NAME = "あ"
Para_id = "DYN"

WAV_A = f"./Para_Study_data/wav/default/{DATA_NAME}_default.wav"  # パラメータなしの歌唱音声
WAV_B = f"./Para_Study_data/wav/DYN/{DATA_NAME}_{Para_id}.wav"  # パラメータありの歌唱音声
VSQX_A = f"./Para_Study_data/vsqx/default/{DATA_NAME}_default.vsqx"  # パラメータなしのVSQX
VSQX_B = f"./Para_Study_data/vsqx/DYN/{DATA_NAME}_{Para_id}.vsqx"  # パラメータありのVSQX

# ===== 出力設定 =====
OUTPUT_DIR = f"./Para_Study_data/output/{Para_id}"
DPI = 300  # 出力画像の解像度
FIGSIZE = (18, 28)  # 出力画像サイズ (インチ)

# ===== 音声設定 =====
SR = 44100
HOP_LENGTH = 256

# ===== 注目パラメータ =====
# None → vsqx_A と vsqx_B のCC差分を自動検出して全て表示
# 文字列で指定 → そのパラメータにフォーカスして表示
# 例: "DYN", "BRE", "BRI", "CLE", "GEN", "POR", "PIT", "PBS", "GWL", "XSY"
TARGET_PARAM = None

# ===== 音響特徴量設定 =====
# デフォルトで全ての音響特徴量を算出し、相関係数が高い順に出力する。
# waveform は常に先頭に表示される（相関ランキングの対象外）。
#
# 利用可能な特徴量:
#   "stft"                - スペクトログラム (STFT, 多次元)
#   "mel"                 - メルスペクトログラム (80次元)
#   "rms"                 - RMSエネルギー (1次元)
#   "f0_world"            - 基本周波数 pyworld版 (1次元)
#   "f0_crepe"            - 基本周波数 CREPE版 (1次元)
#   "mfcc"                - MFCC (20次元)
#   "spectral_centroid"   - スペクトル重心 (1次元)
#   "spectral_bandwidth"  - スペクトル帯域幅 (1次元)
#   "spectral_contrast"   - スペクトルコントラスト (7次元)
#   "hnr"                 - 調波対雑音比 (1次元)
#   "zero_crossing_rate"  - ゼロ交差率 (1次元)
#   "cqt"                 - 定Q変換 (168次元)
#   "aperiodicity"        - 非周期成分 (多次元, pyworld D4C)

# 算出対象の全特徴量
ALL_FEATURE_TYPES = [
    "stft",
    "rms",
    "source_rms",
    "source_stft",
    "f0_world",
    "f0_crepe",
    "mfcc",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "hnr",
    "zero_crossing_rate",
    "cqt",
    # "mel",
    "aperiodicity",
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

# 相関係数の上位何個まで出力するか (None = 全て出力)
TOP_N_FEATURES = 1

# 相関の順位に関係なく必ず出力する特徴量
ALWAYS_INCLUDE_FEATURES = [
    "rms",
    "stft",
]

# 算出しない（スキップする）特徴量 (計算コスト削減用)
ALWAYS_EXCLUDE_FEATURES = [
    "f0_crepe",  # torch/torchcrepe が必要なためデフォルトで除外
]

# ===== 特徴量グループ設定 =====
# 同じグループに属する特徴量は、相関が最も高いものだけが出力される。
# これにより、似た特徴量（stft/mel/cqt等）が上位を独占するのを防ぐ。
# グループに属さない特徴量はグループ制限を受けない。
FEATURE_GROUPS = {
    "spectrogram": ["stft", "mel", "cqt"],
    "pitch": ["f0_world", "f0_crepe"],
    "noise": ["aperiodicity", "hnr", "zero_crossing_rate"],
    "energy": ["rms", "source_rms"],
    "timbre": ["mfcc", "spectral_centroid", "spectral_bandwidth", "spectral_contrast"],
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

# 各グループから最大何個まで選出するか
MAX_PER_GROUP = 1

# ===== パラメータ別の推奨特徴量プリセット =====
# TARGET_PARAM = None (自動検出) の場合にも参照される
PARAM_FEATURE_PRESETS = {
    "DYN": ["stft", "rms", "spectral_centroid"],
    "BRE": ["stft", "aperiodicity", "hnr", "spectral_contrast", "mfcc"],
    "BRI": ["stft", "spectral_centroid", "spectral_bandwidth"],
    "CLE": ["stft", "mfcc", "spectral_contrast"],
    "GEN": ["stft", "f0_world", "mfcc", "spectral_centroid"],
    "PIT": ["f0_world", "stft"],
    "POR": ["f0_world", "stft"],
    "OPE": ["stft", "mfcc", "spectral_bandwidth"],
    "GWL": ["stft", "aperiodicity", "hnr", "mfcc"],
    "XSY": ["stft", "mfcc", "spectral_centroid"],
}

# ===== 相関分析設定 =====
COMPUTE_CORRELATION = True  # 相関係数を算出するか（TOP_N選出には必須）
SORT_BY_CORRELATION = True  # 相関の強い順にプロットを並べるか


