# =============================================================================
# スペクトル包絡の比較ビューア
# =============================================================================
# DYN=0, DYN=64(default), DYN=127 など、異なるDYN値の歌唱データの
# スペクトル包絡を重ねてプロットし、フォルマント構造の変化を直接観察する。
#
# 使い方:
#   1. 下の WAV_FILES に比較したいWAVファイルのパスとラベルを設定する
#   2. python view_spectral_envelope.py を実行する
#
# 入力データについて:
#   DYNが一定値(0, 64, 127)で持続する歌唱データが最適です。
#   「だんだん上がる」データだと、各フレームでDYN値が異なるため
#   スペクトル包絡の代表値を決めにくくなります。
# =============================================================================

import sys
import os
import numpy as np
import scipy.signal
import pyworld as pw
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# =============================================================================
# ===== 設定 =====
# =============================================================================

# 比較したいWAVファイル（ラベル, パス）のリスト
# DYN=0, 64, 127 のように一定値で持続するデータを用意してください
DATA_NAME = "お"
WAV_FILES = [
    ("DYN=1", f"./src_result/Para_Study_data/wav/DYN/Fix/{DATA_NAME}_DYNall1.wav"),
    ("DYN=32", f"./src_result/Para_Study_data/wav/DYN/Fix/{DATA_NAME}_DYNall32.wav"),
    ("DYN=64", f"./src_result/Para_Study_data/wav/default/{DATA_NAME}_default.wav"),
    ("DYN=127", f"./src_result/Para_Study_data/wav/DYN/Fix/{DATA_NAME}_DYNall127.wav"),
]

# 出力先
OUTPUT_DIR = "./src_result/Para_Study_data/output/spectral_envelope"
OUTPUT_FILENAME = f"spectral_envelope_log_{DATA_NAME}.png"

# 音声設定
SR = 44100
HOP_LENGTH = 256

# プロット設定
DPI = 300
FREQ_MAX = 22500  # 表示する最大周波数 (Hz)
FIGSIZE = (16, 10)

# 周波数軸を対数軸 (Log scale) にするかどうか
USE_LOG_FREQ_SCALE = True
FREQ_MIN_LOG = 200  # 対数軸の場合の最小周波数 (0Hzだと計算できないため)

# フォルマント帯域の区切り線を表示するか
SHOW_BAND_LINES = False
BAND_BOUNDARIES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000]

# ピーク（フォルマント位置）を自動検出して強調線を表示するか
SHOW_PEAK_LINES = True
# 分析手法の選択 ("CheapTrick" または "LPC")
ANALYSIS_METHOD = "CheapTrick"

# 差分プロットのペア指定（ラベル名で指定）
# 各タプルの (A, B) に対して「B - A」の差分を表示する
# 複数指定すると、差分グラフが複数段表示される
DIFF_PAIRS = [
    ("DYN=64", "DYN=127"),  # DYN=127 - DYN=64
    ("DYN=32", "DYN=64"),  # DYN=64 - DYN=32
    # ("DYN=1", "DYN=127"),   # DYN=127 - DYN=1
]

# 差分グラフの縦軸を全ペアで統一するか
# True: 全ペアで同じ縦軸スケール（比較しやすい）
# False: 各ペアごとに自動スケール（個別の形状が見やすい）
UNIFY_DIFF_YLIM = True

# =============================================================================
# ===== メイン処理 =====
# =============================================================================


def extract_spectral_envelope(wav_path, sr, hop_length, method="CheapTrick"):
    """
    指定された手法でスペクトル包絡を抽出する。

    Returns:
        sp: np.ndarray (n_frames, n_freq_bins) - スペクトル包絡
        freqs: np.ndarray (n_freq_bins,) - 周波数軸 (Hz)
        f0: np.ndarray (n_frames,) - 基本周波数
    """
    y, _ = librosa.load(wav_path, sr=sr)
    y = y.astype(np.float64)
    y = np.ascontiguousarray(y)

    frame_period = hop_length / sr * 1000

    # 粗い基本周波数の推定
    _f0, t = pw.dio(y, sr, frame_period=frame_period, f0_floor=65.0, f0_ceil=2093.0)
    # 精密な基本周波数の推定
    f0 = pw.stonemask(y, _f0, t, sr)

    if method == "CheapTrick":
        sp = pw.cheaptrick(y, f0, t, sr)
        freqs = np.linspace(0, sr / 2, sp.shape[1])
        return sp, freqs, f0
    elif method == "LPC":
        n_fft = 2048
        frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length).T
        window = scipy.signal.windows.hann(n_fft)
        frames_win = frames * window

        order = int(sr / 1000) + 2
        a_coeffs = librosa.lpc(frames_win, order=order)

        a_padded = np.pad(a_coeffs, ((0, 0), (0, n_fft - (order + 1))))
        A = np.fft.rfft(a_padded, axis=1)
        sp = 1.0 / (np.abs(A) ** 2 + 1e-8)

        freqs = np.linspace(0, sr / 2, sp.shape[1])

        # F0のフレーム数と揃える
        min_len = min(len(f0), sp.shape[0])
        return sp[:min_len], freqs, f0[:min_len]
    else:
        raise ValueError(f"Unknown method: {method}")


def get_voiced_mean_envelope(sp, f0):
    """
    有声フレーム（F0 > 0）のみを使ってスペクトル包絡の平均を取る。
    無声フレームや無音部分を除外し、安定した代表包絡を得る。
    """
    voiced_mask = f0 > 0
    if np.sum(voiced_mask) == 0:
        print("Warning: 有声フレームが見つかりません。全フレーム平均を使用します。")
        voiced_mask = np.ones(len(f0), dtype=bool)

    sp_voiced = sp[voiced_mask]
    sp_db = 10 * np.log10(sp_voiced + 1e-8)
    mean_envelope = np.mean(sp_db, axis=0)
    return mean_envelope


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 存在チェック ---
    valid_files = []
    for label, path in WAV_FILES:
        if os.path.exists(path):
            valid_files.append((label, path))
        else:
            print(f"Warning: ファイルが見つかりません: {path}")

    if len(valid_files) == 0:
        print("Error: 比較可能なWAVファイルがありません。")
        print("\n以下のいずれかのファイルを用意してください:")
        for label, path in WAV_FILES:
            print(f"  {label}: {path}")
        print("\nDYNが一定値で持続する歌唱データが必要です。")
        print("VOCALOIDエディタでDYN=0, DYN=127のデータをそれぞれ書き出してください。")
        sys.exit(1)

    # --- スペクトル包絡の抽出 ---
    envelopes = []
    freqs = None
    for label, path in valid_files:
        print(f"Processing: {label} ({path})")
        # スペクトル包絡抽出（上記extract_spectral_envelope関数で実行）
        sp, f, f0 = extract_spectral_envelope(
            path, SR, HOP_LENGTH, method=ANALYSIS_METHOD
        )
        # 有声区間の平均包絡を算出（上記get_voiced_mean_envelope関数で実行）
        mean_env = get_voiced_mean_envelope(sp, f0)
        envelopes.append((label, mean_env, path))
        if freqs is None:
            freqs = f

    # --- ラベル→包絡の辞書を作成 ---
    env_dict = {label: env for label, env, path in envelopes}

    # --- 有効な差分ペアをフィルタリング ---
    valid_diff_pairs = []
    for pair_a, pair_b in DIFF_PAIRS:
        if pair_a in env_dict and pair_b in env_dict:
            valid_diff_pairs.append((pair_a, pair_b))
        else:
            missing = []
            if pair_a not in env_dict:
                missing.append(pair_a)
            if pair_b not in env_dict:
                missing.append(pair_b)
            print(
                f"Warning: 差分ペア ({pair_a}, {pair_b}) のうち "
                f"{missing} のデータがありません。スキップします。"
            )

    n_diff = max(len(valid_diff_pairs), 1)  # 最低1行は差分領域を確保

    # --- プロット ---
    n_rows = 1 + n_diff  # 上段(包絡) + 差分ペア数
    height_ratios = [3] + [1] * n_diff
    fig_height = FIGSIZE[1] + (n_diff - 1) * 2.5  # ペアが増えたら高さ拡張

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(FIGSIZE[0], fig_height),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = [axes]

    # ===== 上段: スペクトル包絡の重ね合わせ =====
    ax1 = axes[0]

    # compare.py (diff_visualizer) の色使いに合わせる
    # デフォルト(64)=青, パラメータあり(127)=赤, 3色目(0)=紫（緑はDiff用なので避ける）
    color_map = {
        "DYN=0": "#9b59b6",  # 紫
        "DYN=64": "#3498db",  # 青
        "DYN=127": "#e74c3c",  # 赤
    }
    # マップにないラベルが来た時のための予備カラー
    fallback_colors = ["#f39c12", "#1abc9c", "#34495e"]

    for i, (label, env, path) in enumerate(envelopes):
        freq_mask = freqs <= FREQ_MAX
        color = color_map.get(label, fallback_colors[i % len(fallback_colors)])

        # デフォルト(DYN=64)の線を少し太くして目立たせる
        lw = 2.0 if label == "DYN=64" else 1.5

        ax1.plot(
            freqs[freq_mask],
            env[freq_mask],
            label=label,
            color=color,
            linewidth=lw,
            alpha=0.9,
        )

    # フォルマント帯域の区切り線
    if SHOW_BAND_LINES:
        for boundary in BAND_BOUNDARIES:
            if boundary <= FREQ_MAX:
                ax1.axvline(
                    x=boundary, color="gray", linestyle="--", alpha=0.3, linewidth=0.8
                )
                ax1.text(
                    boundary,
                    ax1.get_ylim()[1] - 2,
                    f"{boundary//1000}k",
                    fontsize=7,
                    color="gray",
                    ha="center",
                    va="top",
                )

    # ピーク（フォルマント位置）の自動検出と強調線
    peak_freqs = []
    if SHOW_PEAK_LINES and len(envelopes) > 0:
        # リファレンス包絡（DYN=64 または 先頭）を決定
        ref_env = None
        for label, env, path in envelopes:
            if "64" in label or "default" in label.lower():
                ref_env = env
                break
        if ref_env is None:
            ref_env = envelopes[0][1]

        # prominence: ピークの際立ち具合(dB), distance: ピーク間の最小距離(インデックス数)
        # fs=44100でn_fft=2048(bins=1025)の場合、1ビンは約21.5Hz。
        # distance=20 は約430Hz間隔。
        peaks, _ = scipy.signal.find_peaks(ref_env, prominence=2.0, distance=20)

        for p in peaks:
            pf = freqs[p]
            if pf <= FREQ_MAX:
                peak_freqs.append(pf)
                # 上段グラフにピーク線を引く
                ax1.axvline(
                    x=pf, color="darkorange", linestyle="--", alpha=0.9, linewidth=1.5
                )

    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("Amplitude (dB)", fontsize=12)
    ax1.set_title(f"Spectral Envelope Comparison - '{DATA_NAME}'", fontsize=14)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)

    if USE_LOG_FREQ_SCALE:
        ax1.set_xscale("log")
        ax1.set_xlim(FREQ_MIN_LOG, FREQ_MAX)
        import matplotlib.ticker as ticker

        ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        ax1.set_xlim(0, FREQ_MAX)

    # ===== 下段: 指定ペアごとの差分 =====
    if len(valid_diff_pairs) > 0:
        # 縦軸統一のため、先に全ペアの差分を計算
        freq_mask = freqs <= FREQ_MAX
        all_diffs = []
        for pair_a, pair_b in valid_diff_pairs:
            diff = env_dict[pair_b] - env_dict[pair_a]
            all_diffs.append(diff)

        # 統一スケールの計算
        if UNIFY_DIFF_YLIM:
            global_max = max(np.max(d[freq_mask]) for d in all_diffs)
            global_min = min(np.min(d[freq_mask]) for d in all_diffs)
            margin = (global_max - global_min) * 0.05
            unified_ylim = (global_min - margin, global_max + margin)

        for diff_idx, (pair_a, pair_b) in enumerate(valid_diff_pairs):
            ax = axes[1 + diff_idx]
            diff = all_diffs[diff_idx]
            diff_label = f"{pair_b} - {pair_a}"

            ax.fill_between(
                freqs[freq_mask],
                0,
                diff[freq_mask],
                where=diff[freq_mask] >= 0,
                color="coral",
                alpha=0.4,
                label="増加",
            )
            ax.fill_between(
                freqs[freq_mask],
                0,
                diff[freq_mask],
                where=diff[freq_mask] < 0,
                color="steelblue",
                alpha=0.4,
                label="減少",
            )
            ax.plot(
                freqs[freq_mask],
                diff[freq_mask],
                color="black",
                linewidth=0.8,
                alpha=0.7,
            )
            ax.axhline(y=0, color="black", linewidth=0.5)

            # フォルマント帯域の区切り線
            if SHOW_BAND_LINES:
                for boundary in BAND_BOUNDARIES:
                    if boundary <= FREQ_MAX:
                        ax.axvline(
                            x=boundary,
                            color="gray",
                            linestyle="--",
                            alpha=0.3,
                            linewidth=0.8,
                        )

            # ピーク（フォルマント位置）の強調線
            if SHOW_PEAK_LINES:
                for pf in peak_freqs:
                    ax.axvline(
                        x=pf,
                        color="darkorange",
                        linestyle="--",
                        alpha=0.9,
                        linewidth=1.5,
                    )

            ax.set_xlabel("Frequency (Hz)", fontsize=12)
            ax.set_ylabel("Δ Amplitude (dB)", fontsize=12)
            ax.set_title(f"Difference: {diff_label}", fontsize=12)
            ax.legend(fontsize=10, loc="upper right")
            ax.grid(True, alpha=0.3)

            if USE_LOG_FREQ_SCALE:
                ax.set_xscale("log")
                ax.set_xlim(FREQ_MIN_LOG, FREQ_MAX)
                import matplotlib.ticker as ticker

                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            else:
                ax.set_xlim(0, FREQ_MAX)

            # 縦軸スケールの統一
            if UNIFY_DIFF_YLIM:
                ax.set_ylim(unified_ylim)
    else:
        ax = axes[1]
        ax.text(
            0.5,
            0.5,
            "差分表示には DIFF_PAIRS にラベルを指定してください",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        if USE_LOG_FREQ_SCALE:
            ax.set_xscale("log")
            ax.set_xlim(FREQ_MIN_LOG, FREQ_MAX)
            import matplotlib.ticker as ticker

            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        else:
            ax.set_xlim(0, FREQ_MAX)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
