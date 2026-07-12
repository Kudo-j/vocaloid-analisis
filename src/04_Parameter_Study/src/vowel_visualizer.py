"""
母音別パラメータ比較の可視化モジュール。
横断比較用のヒートマップ、棒グラフ、時系列プロットを生成する。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import os
import librosa

from audio_analyzer import FEATURE_DISPLAY_NAMES, FEATURE_PLOT_TYPES


# --- 日本語フォント自動設定 ---
def _setup_japanese_font():
    import matplotlib.font_manager as fm
    candidates = [
        "IPAexGothic", "IPAGothic", "Noto Sans CJK JP",
        "Yu Gothic", "MS Gothic", "Meiryo",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            return

_setup_japanese_font()
plt.rcParams["axes.unicode_minus"] = False


# --- 母音タイプ別の色 ---
VOWEL_COLORS = {
    "a": "#e74c3c",   # 赤
    "i": "#3498db",   # 青
    "u": "#2ecc71",   # 緑
    "e": "#f39c12",   # 黄
    "o": "#9b59b6",   # 紫
    "aiueo": "#1abc9c",     # ティール
    "a-i-u-e-o": "#e67e22", # オレンジ
}

# 条件タイプ別のハッチングパターン
TYPE_HATCHES = {
    "isolated": "",       # ハッチなし（単母音）
    "continuous": "//",   # 連続歌唱
    "separated": "xx",    # 区切り発音
}


def _correlation_color_val(r_abs):
    """相関強度に応じた色(文字色用)"""
    if r_abs >= 0.7:
        return "#27ae60"
    elif r_abs >= 0.4:
        return "#f39c12"
    return "#95a5a6"


class VowelVisualizer:
    """母音別パラメータ比較の可視化"""

    def __init__(self, sr=44100, hop_length=256):
        self.sr = sr
        self.hop_length = hop_length

    # ==================================================================
    # 1. 相関マトリクスヒートマップ
    # ==================================================================
    def plot_correlation_heatmap(self, labels, features, matrix,
                                 param_name, output_path,
                                 figsize=(20, 12), dpi=300):
        """
        母音条件(行) × 音響特徴量(列) の |r| 相関係数ヒートマップ。
        """
        display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]
        n_rows, n_cols = matrix.shape

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=1.0)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("|r| (Correlation Coefficient)", fontsize=12)

        # セルにテキスト値を表示
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_title(
            f"Correlation Matrix: {param_name} × Vowel Conditions\n"
            f"|r| between parameter curve and acoustic feature difference",
            fontsize=14, fontweight="bold", pad=15
        )

        # 強度区分の凡例テキスト
        ax.text(1.0, -0.12,
                "Strong (|r|≥0.7)  /  Moderate (0.4≤|r|<0.7)  /  Weak (|r|<0.4)",
                transform=ax.transAxes, fontsize=9, ha="right",
                style="italic", color="#555")

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                   facecolor="white")
        plt.close(fig)
        print(f"  Saved: {output_path}")

    # ==================================================================
    # 2. 効果量ヒートマップ
    # ==================================================================
    def plot_effect_heatmap(self, labels, features, matrix,
                            param_name, output_path,
                            figsize=(20, 12), dpi=300):
        """母音条件 × 音響特徴量 の平均絶対差分ヒートマップ"""
        display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]
        n_rows, n_cols = matrix.shape

        # 列ごとに正規化（特徴量間でスケールが異なるため）
        col_max = np.nanmax(matrix, axis=0, keepdims=True)
        col_max[col_max < 1e-10] = 1.0
        normalized = matrix / col_max

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(normalized, cmap="PuBuGn", aspect="auto",
                       vmin=0, vmax=1.0)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Normalized Effect Size (per-feature max = 1.0)",
                       fontsize=11)

        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                norm_val = normalized[i, j]
                color = "white" if norm_val > 0.6 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, color=color)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_title(
            f"Effect Size Matrix: {param_name} × Vowel Conditions\n"
            f"Mean absolute difference (normalized per feature column)",
            fontsize=14, fontweight="bold", pad=15
        )

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                   facecolor="white")
        plt.close(fig)
        print(f"  Saved: {output_path}")

    # ==================================================================
    # 3. フォーカス特徴量の母音間棒グラフ比較
    # ==================================================================
    def plot_bar_comparison(self, all_results, focus_features, param_name,
                            output_path, figsize=(18, 10), dpi=300):
        """
        フォーカス特徴量ごとに、7母音条件の相関係数 |r| と効果量を
        並べた棒グラフを生成する。上段: |r|, 下段: 効果量。
        """
        n_features = len(focus_features)
        if n_features == 0:
            print("  WARNING: No focus features to plot bar comparison.")
            return

        fig, axes = plt.subplots(2, n_features, figsize=figsize, sharey="row")
        if n_features == 1:
            axes = axes.reshape(2, 1)

        cond_labels = list(all_results.keys())
        n_conds = len(cond_labels)
        x = np.arange(n_conds)
        bar_width = 0.7

        for j, fname in enumerate(focus_features):
            display = FEATURE_DISPLAY_NAMES.get(fname, fname)

            # --- 上段: |r| ---
            ax_r = axes[0, j]
            r_vals = []
            colors = []
            hatches = []
            for cond_label in cond_labels:
                result = all_results[cond_label]
                cond = result["condition"]
                r = abs(result["correlations"].get(fname, {}).get("r", 0.0))
                r_vals.append(r)
                colors.append(VOWEL_COLORS.get(cond["vowel"], "#7f8c8d"))
                hatches.append(TYPE_HATCHES.get(cond["type"], ""))

            bars = ax_r.bar(x, r_vals, bar_width, color=colors,
                            edgecolor="white", linewidth=0.5)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

            # 値ラベル
            for xi, val in zip(x, r_vals):
                ax_r.text(xi, val + 0.02, f"{val:.3f}", ha="center",
                         fontsize=8, fontweight="bold")

            ax_r.set_title(display, fontsize=11, fontweight="bold")
            ax_r.set_ylim(0, 1.15)
            ax_r.axhline(0.7, color="#27ae60", linestyle="--", alpha=0.5,
                         linewidth=1)
            ax_r.axhline(0.4, color="#f39c12", linestyle="--", alpha=0.5,
                         linewidth=1)
            ax_r.set_ylabel("|r|" if j == 0 else "")
            ax_r.set_xticks([])
            ax_r.grid(axis="y", alpha=0.3)

            # --- 下段: 効果量 ---
            ax_e = axes[1, j]
            e_vals = []
            for cond_label in cond_labels:
                result = all_results[cond_label]
                e = result["effect_sizes"].get(fname, 0.0)
                e_vals.append(e)

            bars_e = ax_e.bar(x, e_vals, bar_width, color=colors,
                              edgecolor="white", linewidth=0.5)
            for bar, hatch in zip(bars_e, hatches):
                bar.set_hatch(hatch)

            # 短いラベル
            short_labels = []
            for cond_label in cond_labels:
                result = all_results[cond_label]
                cond = result["condition"]
                short_labels.append(cond["prefix"])

            ax_e.set_xticks(x)
            ax_e.set_xticklabels(short_labels, rotation=45, ha="right",
                                 fontsize=9)
            ax_e.set_ylabel("Effect Size" if j == 0 else "")
            ax_e.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Vowel Comparison: {param_name}\n"
            f"Upper: Correlation |r|  /  Lower: Mean Absolute Difference",
            fontsize=14, fontweight="bold", y=1.02
        )

        # 凡例（条件タイプ）
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor="#ccc", edgecolor="black", label="単母音 (isolated)"),
            Patch(facecolor="#ccc", edgecolor="black", hatch="//",
                  label="連続歌唱 (continuous)"),
            Patch(facecolor="#ccc", edgecolor="black", hatch="xx",
                  label="区切り発音 (separated)"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=3,
                  fontsize=10, bbox_to_anchor=(0.5, -0.02))

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                   facecolor="white")
        plt.close(fig)
        print(f"  Saved: {output_path}")

    # ==================================================================
    # 4. 詳細時系列プロット
    # ==================================================================
    def plot_detail_timeseries(self, all_results, focus_features, param_name,
                               output_path, figsize=(20, 24), dpi=300):
        """
        フォーカス特徴量ごとに、全母音条件の差分時系列を重ねて表示する。
        上段にパラメータカーブ、以降にフォーカス特徴量の差分オーバーレイ。
        """
        n_features = len(focus_features)
        if n_features == 0:
            return

        n_rows = 1 + n_features  # パラメータカーブ + 特徴量
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
        if n_rows == 1:
            axes = [axes]

        cond_labels = list(all_results.keys())

        # --- 0行目: パラメータカーブ (代表1条件を表示) ---
        ax0 = axes[0]
        first_result = list(all_results.values())[0]
        pc = first_result["param_curve"]
        t_param = librosa.frames_to_time(
            np.arange(len(pc)), sr=self.sr, hop_length=self.hop_length
        )
        default_val = first_result["default_value"]

        ax0.axhline(default_val, color="#e74c3c", linestyle="--",
                    linewidth=1, alpha=0.7, label=f"Default ({default_val})")
        ax0.plot(t_param, pc, color="#2c3e50", linewidth=1.5,
                label=f"{param_name} Curve")
        ax0.fill_between(t_param, default_val, pc, alpha=0.15, color="#3498db")
        ax0.set_ylabel(f"{param_name} Value")
        ax0.set_title(f"VSQX Parameter: {param_name}", fontsize=12,
                     fontweight="bold")
        ax0.legend(loc="upper left", fontsize=9)
        ax0.grid(True, alpha=0.3)

        # --- 特徴量ごとの差分オーバーレイ ---
        for fi, fname in enumerate(focus_features):
            ax = axes[1 + fi]
            display = FEATURE_DISPLAY_NAMES.get(fname, fname)
            plot_type = FEATURE_PLOT_TYPES.get(fname, "line_1d")

            for cond_label in cond_labels:
                result = all_results[cond_label]
                cond = result["condition"]
                diff = result["diff_features"].get(fname)
                if diff is None:
                    continue

                color = VOWEL_COLORS.get(cond["vowel"], "#7f8c8d")
                r_val = abs(result["correlations"].get(
                    fname, {"r": 0.0}
                ).get("r", 0.0))

                if plot_type == "line_1d" or diff.ndim == 1 or diff.shape[1] == 1:
                    # 1次元: 差分の時系列を直接プロット
                    d = diff.flatten()
                    t = librosa.frames_to_time(
                        np.arange(len(d)), sr=self.sr,
                        hop_length=self.hop_length
                    )
                    ax.plot(t, d, color=color, linewidth=0.8, alpha=0.8,
                           label=f"{cond['prefix']} (|r|={r_val:.3f})")
                else:
                    # 多次元: 帯域平均の差分を表示
                    d_mean = np.mean(diff, axis=1)
                    t = librosa.frames_to_time(
                        np.arange(len(d_mean)), sr=self.sr,
                        hop_length=self.hop_length
                    )
                    ax.plot(t, d_mean, color=color, linewidth=0.8, alpha=0.8,
                           label=f"{cond['prefix']} (|r|={r_val:.3f})")

            ax.axhline(0, color="black", linestyle="--", linewidth=0.5,
                       alpha=0.5)
            ax.set_ylabel(f"Δ{display}")
            ax.set_title(f"{display} — Diff across vowel conditions",
                        fontsize=11, fontweight="bold")
            ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.8)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Time (sec)")

        fig.suptitle(
            f"Detail Time-Series Comparison: {param_name}\n"
            f"Diff = (with parameter) - (default) for each vowel condition",
            fontsize=14, fontweight="bold", y=1.01
        )

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                   facecolor="white")
        plt.close(fig)
        print(f"  Saved: {output_path}")
