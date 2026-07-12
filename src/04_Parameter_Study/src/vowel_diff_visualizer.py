"""
母音間パラメータ影響差異の可視化モジュール。
2つの母音条件のDiff（パラメータあり - なし）をさらに比較し、
「差分の差分 (Diff of Diffs)」や「比較オーバーレイ」を1枚の画像にまとめる。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI不要
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import os
import librosa

from audio_analyzer import FEATURE_DISPLAY_NAMES, FEATURE_PLOT_TYPES, FEATURE_YLABELS


# --- 日本語フォント自動設定 ---
def _setup_japanese_font():
    """利用可能な日本語フォントを自動検出して設定"""
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

    print("Warning: No Japanese font found. Japanese text may not display correctly.")


_setup_japanese_font()
plt.rcParams["axes.unicode_minus"] = False


class VowelDiffVisualizer:
    """母音間パラメータ影響差異の可視化"""

    def __init__(self, sr=44100, hop_length=256):
        self.sr = sr
        self.hop_length = hop_length

    # =================================================================
    # ヘルパー
    # =================================================================

    def _get_time_axis(self, total_frames):
        """フレーム数から時間軸を生成"""
        return librosa.frames_to_time(
            np.arange(total_frames), sr=self.sr, hop_length=self.hop_length
        )

    def _shade_non_singing_regions(self, ax, notes, t_start, t_end):
        """
        歌唱領域（ノートが存在する区間）以外の背景をグレーにシェーディングする。
        """
        if not notes:
            ax.axvspan(t_start, t_end, alpha=0.10, color="grey", zorder=0)
            return

        # ノート区間をマージ（重なりを統合）
        intervals = sorted([(n["start_sec"], n["end_sec"]) for n in notes])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # 非歌唱区間をシェーディング
        prev_end = t_start
        for s, e in merged:
            if s > prev_end:
                ax.axvspan(prev_end, s, alpha=0.10, color="grey", zorder=0)
            prev_end = e
        if prev_end < t_end:
            ax.axvspan(prev_end, t_end, alpha=0.10, color="grey", zorder=0)

    def _add_freq_labels(self, ax, fname, n_bands):
        """スペクトログラム系の特徴量に対してY軸に周波数ラベルを付与"""
        if fname in ("mel", "source_mel"):
            mel_f = librosa.mel_frequencies(n_mels=n_bands, fmin=0.0,
                                            fmax=self.sr / 2)
            tick_step = max(1, n_bands // 8)
            tick_indices = np.arange(0, n_bands, tick_step)
            ax.set_yticks(tick_indices + 0.5)
            ax.set_yticklabels(
                [f"{i}\n({mel_f[i]:.0f}Hz)" for i in tick_indices],
                fontsize=6
            )
        elif fname in ("stft", "source_stft", "aperiodicity"):
            freqs = np.linspace(0, self.sr / 2, n_bands)
            n_ticks = min(10, n_bands)
            tick_indices = np.linspace(0, n_bands - 1, n_ticks).astype(int)
            ax.set_yticks(tick_indices + 0.5)
            ax.set_yticklabels(
                [f"{freqs[i]:.0f}Hz" for i in tick_indices],
                fontsize=6
            )
            ax.set_ylabel("Frequency (Hz)")

    def _make_asymmetric_norm(self, data):
        """
        非対称 TwoSlopeNorm を生成する。
        正・負を独立にスケーリングし、5% のバッファ（Headroom）を上乗せして
        ピーク付近のベタ塗り（飽和）を防ぐ。
        """
        vmax_actual = np.max(data)
        vmin_actual = np.min(data)

        # 5%バッファで飽和を防止
        vmax_buf = vmax_actual * 1.05 if vmax_actual > 0 else 1.0
        vmin_buf = vmin_actual * 1.05 if vmin_actual < 0 else -1.0

        # セーフティ（全ゼロデータ等）
        if vmax_buf < 1e-10:
            vmax_buf = 1.0
        if vmin_buf > -1e-10:
            vmin_buf = -1.0

        return TwoSlopeNorm(vmin=vmin_buf, vcenter=0, vmax=vmax_buf)

    # =================================================================
    # Image B: Diff of Diffs
    # =================================================================

    def plot_diff_of_diffs(self, result_1, result_2, focus_features,
                           param_name, output_path, dpi=300):
        """
        2つの母音条件の差分（パラメータあり - なし）をさらに差し引きした
        「Diff of Diffs」を1枚の画像にまとめて出力する。

        (母音1_DYN - 母音1_default) - (母音2_DYN - 母音2_default)
        → 白なら違いなし、色ありなら母音固有の応答差異
        """
        label_1 = result_1["label"]
        label_2 = result_2["label"]

        diff_1 = result_1["diff_features"]
        diff_2 = result_2["diff_features"]

        notes_1 = result_1.get("notes", [])

        # 使用する特徴量
        feature_names = [f for f in focus_features
                         if f in diff_1 and f in diff_2]
        n_features = len(feature_names)

        # 段数: パラメータカーブ(1) + 各特徴量
        n_rows = 1 + n_features
        height_ratios = [2]
        for fname in feature_names:
            ptype = FEATURE_PLOT_TYPES.get(fname, "line_1d")
            if ptype == "spectrogram":
                height_ratios.append(4)
            elif ptype == "heatmap":
                height_ratios.append(3)
            else:
                height_ratios.append(2)

        total_h = sum(height_ratios)
        fig_height = max(18, total_h * 1.3)

        fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(
            n_rows, 2,
            height_ratios=height_ratios,
            width_ratios=[1, 0.03],
            hspace=0.35, wspace=0.03,
            top=0.93
        )

        # --- タイトル ---
        fig.suptitle(
            f"Image B: Diff of Diffs — {param_name}\n"
            f"（{label_1}）−（{label_2}）　※白=違いなし／色あり=違いあり",
            fontsize=13, fontweight="bold"
        )

        # --- パラメータカーブ (Sync Check) ---
        ax_param = fig.add_subplot(gs[0, 0])
        ax_param_cb = fig.add_subplot(gs[0, 1])
        ax_param_cb.axis("off")

        param_curve_1 = result_1["param_curve"]
        param_curve_2 = result_2["param_curve"]
        default_val = result_1["default_value"]

        min_len_p = min(len(param_curve_1), len(param_curve_2))
        t_param = self._get_time_axis(min_len_p)

        self._shade_non_singing_regions(ax_param, notes_1,
                                        t_param[0], t_param[-1])

        ax_param.axhline(y=default_val, color="#aaa", linestyle="--",
                         linewidth=1.0, alpha=0.7, label=f"Default ({default_val})")
        ax_param.plot(t_param, param_curve_1[:min_len_p],
                      color="#e74c3c", linewidth=1.2, alpha=0.8,
                      label=f"{label_1} Curve")
        ax_param.plot(t_param, param_curve_2[:min_len_p],
                      color="#3498db", linewidth=1.2, alpha=0.8,
                      label=f"{label_2} Curve")
        ax_param.set_title("VSQX Parameter Curve (Sync Check)",
                           fontsize=11, fontweight="bold")
        ax_param.set_ylabel("Value")
        ax_param.set_xlim(t_param[0], t_param[-1])
        ax_param.legend(loc="upper left", fontsize=9, framealpha=0.8)
        ax_param.grid(True, alpha=0.2)

        # --- 各特徴量の Diff of Diffs ---
        for fi, fname in enumerate(feature_names):
            ax = fig.add_subplot(gs[1 + fi, 0])
            ax_cb = fig.add_subplot(gs[1 + fi, 1])

            d1 = diff_1[fname]
            d2 = diff_2[fname]
            min_len = min(d1.shape[0], d2.shape[0])
            dod = d1[:min_len] - d2[:min_len]

            t = self._get_time_axis(min_len)
            display = FEATURE_DISPLAY_NAMES.get(fname, fname)
            ptype = FEATURE_PLOT_TYPES.get(fname, "line_1d")

            if ptype == "line_1d" or dod.ndim == 1 or dod.shape[1] == 1:
                # --- 1次元: 折れ線グラフ ---
                ax_cb.axis("off")
                self._shade_non_singing_regions(ax, notes_1, t[0], t[-1])

                dod_flat = dod.flatten()[:min_len]
                ax.fill_between(t, 0, dod_flat, where=(dod_flat > 0),
                                color="#e74c3c", alpha=0.3, interpolate=True)
                ax.fill_between(t, 0, dod_flat, where=(dod_flat < 0),
                                color="#3498db", alpha=0.3, interpolate=True)
                ax.plot(t, dod_flat, color="#2c3e50", linewidth=0.7)
                ax.axhline(0, color="black", linestyle="--",
                           linewidth=0.5, alpha=0.5)

                ax.set_xlim(t[0], t[-1])
                ylabel = FEATURE_YLABELS.get(fname, "Value")
                ax.set_ylabel(f"Δ {ylabel}")
                ax.set_title(f"{display}", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.2)

                # アノテーション
                mean_abs = np.mean(np.abs(dod_flat))
                max_abs = np.max(np.abs(dod_flat))
                ax.text(0.99, 0.97,
                        f"Mean|Δ|={mean_abs:.4f}, Max|Δ|={max_abs:.4f}",
                        transform=ax.transAxes, fontsize=8,
                        va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="lightyellow", alpha=0.85,
                                  edgecolor="#ccc"))

            else:
                # --- 多次元: ヒートマップ (非対称発散型カラーマップ) ---
                norm = self._make_asymmetric_norm(dod)

                img = ax.imshow(
                    dod.T, aspect="auto", origin="lower",
                    extent=[t[0], t[-1], 0, dod.shape[1]],
                    cmap="RdBu_r", norm=norm,
                    interpolation="nearest"
                )

                cbar = plt.colorbar(img, cax=ax_cb)
                cbar.set_label("Diff", fontsize=7)
                cbar.ax.tick_params(labelsize=6)

                ax.set_ylabel("Band")
                ax.set_title(f"{display}", fontsize=11, fontweight="bold")

                # Y軸の周波数ラベル
                self._add_freq_labels(ax, fname, dod.shape[1])

                # 差分の大きさを数値でアノテーション
                mean_abs = np.mean(np.abs(dod))
                max_abs = np.max(np.abs(dod))
                ax.text(0.99, 0.97,
                        f"Mean|Δ|={mean_abs:.4f}, Max|Δ|={max_abs:.4f}",
                        transform=ax.transAxes, fontsize=8,
                        va="top", ha="right", color="black",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="lightyellow", alpha=0.85,
                                  edgecolor="#ccc"))

            if fi == n_features - 1:
                ax.set_xlabel("Time (sec)")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"    Saved Image B: {output_path}")

    # =================================================================
    # Image A: Comparison Overlay
    # =================================================================

    def plot_comparison_overlay(self, result_1, result_2, focus_features,
                                param_name, output_path, dpi=300):
        """
        2つの母音条件のDiff（パラメータあり - なし）を並べて比較する。
        1次元特徴量はオーバーレイ折れ線、多次元は上下にスペクトログラムを並べる。
        """
        label_1 = result_1["label"]
        label_2 = result_2["label"]
        color_1 = "#e74c3c"
        color_2 = "#3498db"

        diff_1 = result_1["diff_features"]
        diff_2 = result_2["diff_features"]

        notes_1 = result_1.get("notes", [])

        feature_names = [f for f in focus_features
                         if f in diff_1 and f in diff_2]
        n_features = len(feature_names)

        # 段数を計算（多次元は2行使う）
        n_rows = 1  # パラメータカーブ
        height_ratios = [2]
        for fname in feature_names:
            ptype = FEATURE_PLOT_TYPES.get(fname, "line_1d")
            d_test = diff_1.get(fname)
            if d_test is not None and d_test.ndim > 1 and d_test.shape[1] > 1:
                if ptype in ("spectrogram", "heatmap"):
                    n_rows += 2
                    height_ratios.extend([3, 3])
                else:
                    n_rows += 1
                    height_ratios.append(2)
            else:
                n_rows += 1
                height_ratios.append(2)

        total_h = sum(height_ratios)
        fig_height = max(18, total_h * 1.3)

        fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(
            n_rows, 2,
            height_ratios=height_ratios,
            width_ratios=[1, 0.03],
            hspace=0.35, wspace=0.03,
            top=0.93
        )

        # --- タイトル ---
        fig.suptitle(
            f"Image A: Comparison Overlay — {param_name}\n"
            f"赤={label_1}　青={label_2}　各条件の (DYN−default) を比較",
            fontsize=13, fontweight="bold"
        )

        # --- パラメータカーブ ---
        ax_param = fig.add_subplot(gs[0, 0])
        ax_param_cb = fig.add_subplot(gs[0, 1])
        ax_param_cb.axis("off")

        param_curve_1 = result_1["param_curve"]
        param_curve_2 = result_2["param_curve"]
        default_val = result_1["default_value"]

        min_len_p = min(len(param_curve_1), len(param_curve_2))
        t_param = self._get_time_axis(min_len_p)

        self._shade_non_singing_regions(ax_param, notes_1,
                                        t_param[0], t_param[-1])

        ax_param.axhline(y=default_val, color="#aaa", linestyle="--",
                         linewidth=1.0, alpha=0.7, label=f"Default ({default_val})")
        ax_param.plot(t_param, param_curve_1[:min_len_p],
                      color=color_1, linewidth=1.2, alpha=0.8,
                      label=f"{label_1}")
        ax_param.plot(t_param, param_curve_2[:min_len_p],
                      color=color_2, linewidth=1.2, alpha=0.8,
                      label=f"{label_2}")
        ax_param.set_title("VSQX Parameter Curve (Sync Check)",
                           fontsize=11, fontweight="bold")
        ax_param.set_ylabel("Value")
        ax_param.set_xlim(t_param[0], t_param[-1])
        ax_param.legend(loc="upper left", fontsize=9, framealpha=0.8)
        ax_param.grid(True, alpha=0.2)

        # --- 特徴量プロット ---
        row_idx = 1
        for fi, fname in enumerate(feature_names):
            d1 = diff_1[fname]
            d2 = diff_2[fname]
            min_len = min(d1.shape[0], d2.shape[0])
            d1 = d1[:min_len]
            d2 = d2[:min_len]
            t = self._get_time_axis(min_len)

            display = FEATURE_DISPLAY_NAMES.get(fname, fname)
            ptype = FEATURE_PLOT_TYPES.get(fname, "line_1d")

            if ptype == "line_1d" or d1.ndim == 1 or d1.shape[1] == 1:
                # --- 1次元: オーバーレイ折れ線 ---
                ax = fig.add_subplot(gs[row_idx, 0])
                ax_cb = fig.add_subplot(gs[row_idx, 1])
                ax_cb.axis("off")
                row_idx += 1

                self._shade_non_singing_regions(ax, notes_1, t[0], t[-1])

                v1 = d1.flatten()[:min_len]
                v2 = d2.flatten()[:min_len]
                ax.plot(t, v1, color=color_1, linewidth=0.8,
                        alpha=0.8, label=label_1)
                ax.plot(t, v2, color=color_2, linewidth=0.8,
                        alpha=0.8, label=label_2)
                ax.axhline(0, color="black", linestyle="--",
                           linewidth=0.5, alpha=0.5)
                ax.set_xlim(t[0], t[-1])
                ylabel = FEATURE_YLABELS.get(fname, "Value")
                ax.set_ylabel(f"Δ {ylabel}")
                ax.set_title(f"{display}", fontsize=11, fontweight="bold")
                ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
                ax.grid(True, alpha=0.2)

                row_idx = row_idx  # already incremented

            else:
                # --- 多次元: 上下に2枚のスペクトログラムを並べる ---
                # 非対称のカラースケールを算出（両条件で共通）
                combined = np.concatenate([d1.ravel(), d2.ravel()])
                norm = self._make_asymmetric_norm(combined)

                # 条件1
                ax1 = fig.add_subplot(gs[row_idx, 0])
                ax1_cb = fig.add_subplot(gs[row_idx, 1])
                img1 = ax1.imshow(
                    d1.T, aspect="auto", origin="lower",
                    extent=[t[0], t[-1], 0, d1.shape[1]],
                    cmap="RdBu_r", norm=norm,
                    interpolation="nearest"
                )
                ax1.set_title(f"{display} — {label_1}",
                              fontsize=10, fontweight="bold",
                              color=color_1)
                ax1.set_ylabel("Band")
                self._add_freq_labels(ax1, fname, d1.shape[1])
                cbar1 = plt.colorbar(img1, cax=ax1_cb)
                cbar1.ax.tick_params(labelsize=6)
                row_idx += 1

                # 条件2
                ax2 = fig.add_subplot(gs[row_idx, 0])
                ax2_cb = fig.add_subplot(gs[row_idx, 1])
                img2 = ax2.imshow(
                    d2.T, aspect="auto", origin="lower",
                    extent=[t[0], t[-1], 0, d2.shape[1]],
                    cmap="RdBu_r", norm=norm,
                    interpolation="nearest"
                )
                ax2.set_title(f"{display} — {label_2}",
                              fontsize=10, fontweight="bold",
                              color=color_2)
                ax2.set_ylabel("Band")
                ax2.set_xlabel("Time (sec)")
                self._add_freq_labels(ax2, fname, d2.shape[1])
                cbar2 = plt.colorbar(img2, cax=ax2_cb)
                cbar2.set_label("Diff", fontsize=7)
                cbar2.ax.tick_params(labelsize=6)
                row_idx += 1

        # 最後の行のAxesにX軸ラベルを追加（1次元の最後の行用）
        # (多次元の場合は上で直接設定済み)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"    Saved Image A: {output_path}")
