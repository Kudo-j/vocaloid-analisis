# =============================================================================
# 複数音声の音響特徴量比較ビューア
# =============================================================================
# 複数のWAVファイルの指定音響特徴量を重ねてプロットし、
# 母音間やDYN値間の違いを直接比較する。
#
# 使い方:
#   1. 下の WAV_FILES に比較したいWAVファイルのパスとラベルを設定する
#   2. TARGET_FEATURES に見たい特徴量を設定する
#   3. python view_multi_features.py を実行する
# =============================================================================

import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# srcをパスに追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from audio_analyzer import (
    AudioAnalyzer,
    FEATURE_DISPLAY_NAMES,
    FEATURE_YLABELS,
    FEATURE_PLOT_TYPES,
)

# --- 日本語フォント設定 ---
try:
    import matplotlib.font_manager as fm

    jp_font_candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans"]
    for font_name in jp_font_candidates:
        if any(font_name in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = font_name
            break
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# ===== 設定 =====
# =============================================================================

# 比較したいWAVファイル（ラベル, パス）のリスト
WAV_FILES = [
    ("あ", "./src_result/Para_Study_data/wav/default/あ_default.wav"),
    ("い", "./src_result/Para_Study_data/wav/default/い_default.wav"),
    ("う", "./src_result/Para_Study_data/wav/default/う_default.wav"),
    ("え", "./src_result/Para_Study_data/wav/default/え_default.wav"),
    ("お", "./src_result/Para_Study_data/wav/default/お_default.wav"),
]

# 比較したい音響特徴量（1次元の line_1d 特徴量のみ対応、2Dの場合は重ね合わせが不可能）
# 利用可能: "rms", "source_rms", "f0_world", "hnr", "spectral_centroid",
#           "spectral_bandwidth", "zero_crossing_rate",
#           "formant_1_amp" ~ "formant_7_amp", "air_freq_amp"
TARGET_FEATURES = [
    "rms",
    "source_rms",
    "formant_1_amp",
    "formant_2_amp",
]

# 出力先
OUTPUT_DIR = "./src_result/Para_Study_data/output/multi_features"
OUTPUT_FILENAME = "multi_features_compare.png"

# 音声設定
SR = 44100
HOP_LENGTH = 256

# プロット設定
DPI = 300
FIGSIZE = (16, 12)  # 特徴量の数に応じて自動調整される

# 線の色（WAV_FILES の数だけ自動で割り当て）
# 固定色リスト。足りなければ繰り返す。
LINE_COLORS = [
    "#e74c3c",  # 赤
    "#3498db",  # 青
    "#2ecc71",  # 緑
    "#9b59b6",  # 紫
    "#f39c12",  # オレンジ
    "#1abc9c",  # ティール
    "#e67e22",  # ダークオレンジ
    "#34495e",  # ダークグレー
]

# =============================================================================
# ===== メイン処理 =====
# =============================================================================


def main():
    print("=" * 55)
    print("  Multi-File Feature Comparison Tool")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. 入力検証 ---
    # 1D 特徴量のみ許可
    for feat in TARGET_FEATURES:
        plot_type = FEATURE_PLOT_TYPES.get(feat, "line_1d")
        if plot_type != "line_1d":
            print(f"ERROR: '{feat}' は2次元特徴量のため、このツールでは非対応です。")
            print("       1次元 (line_1d) の特徴量のみ指定してください。")
            sys.exit(1)

    valid_files = []
    for label, path in WAV_FILES:
        if os.path.exists(path):
            valid_files.append((label, path))
        else:
            print(f"Warning: ファイルが見つかりません: {path}")

    if len(valid_files) == 0:
        print("ERROR: 比較可能なWAVファイルがありません。")
        sys.exit(1)

    print(f"\n[Config]")
    print(f"  Files: {len(valid_files)} / {len(WAV_FILES)}")
    for label, path in valid_files:
        print(f"    {label}: {path}")
    print(f"  Features: {TARGET_FEATURES}")

    # --- 2. 特徴量抽出 ---
    print("\n[Step 1] Extracting features...")
    analyzer = AudioAnalyzer(sr=SR, hop_length=HOP_LENGTH)

    # {label: {feature_name: np.ndarray, ...}, ...}
    all_data = {}
    for label, path in valid_files:
        print(f"  Processing: {label} ({os.path.basename(path)})")
        features = analyzer.extract_features(path, TARGET_FEATURES)
        all_data[label] = features
        print(f"    -> Extracted {len(features)} features")

    # --- 3. プロット ---
    print("\n[Step 2] Plotting...")

    n_features = len(TARGET_FEATURES)
    height_per_plot = 3.0
    fig_height = max(FIGSIZE[1], n_features * height_per_plot + 2)

    fig, axes = plt.subplots(
        n_features,
        1,
        figsize=(FIGSIZE[0], fig_height),
        sharex=False,
    )
    # 特徴量が1つだけの場合、axesを配列にする
    if n_features == 1:
        axes = [axes]

    fig.suptitle(
        f"Acoustic Feature Comparison ({len(valid_files)} files)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    colors = LINE_COLORS

    for feat_idx, feat_name in enumerate(TARGET_FEATURES):
        ax = axes[feat_idx]
        display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)
        ylabel = FEATURE_YLABELS.get(feat_name, "Value")

        ax.set_title(display_name, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

        for file_idx, (label, _path) in enumerate(valid_files):
            feat_data = all_data[label].get(feat_name)
            if feat_data is None:
                continue

            data = feat_data.flatten()
            import librosa

            time_axis = librosa.frames_to_time(
                np.arange(len(data)), sr=SR, hop_length=HOP_LENGTH
            )

            color = colors[file_idx % len(colors)]
            ax.plot(
                time_axis,
                data,
                color=color,
                linewidth=1.2,
                alpha=0.85,
                label=label,
            )

        ax.legend(loc="upper right", fontsize=10, framealpha=0.8)

        # 最後のグラフだけX軸ラベルを表示
        if feat_idx == n_features - 1:
            ax.set_xlabel("Time (sec)", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nSaved: {output_path}")
    print("\n" + "=" * 55)
    print("  Complete!")
    print(f"  Output: {os.path.abspath(output_path)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
