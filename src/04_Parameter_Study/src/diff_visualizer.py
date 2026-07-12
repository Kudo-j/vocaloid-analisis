"""
差分解析結果の可視化モジュール。
matplotlib を使って相関順プロット、VSQXパラメータカーブ、
メルスペクトログラム差分などを1枚の画像にまとめる。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI不要
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import os

from audio_analyzer import FEATURE_DISPLAY_NAMES, FEATURE_PLOT_TYPES, FEATURE_YLABELS


# --- 日本語フォント自動設定 ---
def _setup_japanese_font():
    """利用可能な日本語フォントを自動検出して設定"""
    import matplotlib.font_manager as fm

    # 候補フォント (優先順)
    candidates = [
        "IPAexGothic", "IPAGothic", "Noto Sans CJK JP",
        "Yu Gothic", "MS Gothic", "Meiryo",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            print(f"Using Japanese font: {font_name}")
            return

    print("Warning: No Japanese font found. Japanese text may not display correctly.")


_setup_japanese_font()
plt.rcParams["axes.unicode_minus"] = False


# --- 相関強度の色分け ---
def _correlation_color(r_abs):
    """相関係数の絶対値に応じた色を返す"""
    if r_abs >= 0.7:
        return "#2ecc71"   # 緑 = 強い相関
    elif r_abs >= 0.4:
        return "#f39c12"   # 黄 = 中程度
    else:
        return "#95a5a6"   # 灰 = 弱い相関


def _correlation_label(r_abs):
    """相関強度のラベル"""
    if r_abs >= 0.7:
        return "Strong"
    elif r_abs >= 0.4:
        return "Moderate"
    else:
        return "Weak"


class DiffVisualizer:
    """差分解析結果の可視化"""

    def __init__(self, sr=44100, hop_length=256, figsize=(18, 28), dpi=300):
        self.sr = sr
        self.hop_length = hop_length
        self.figsize = figsize
        self.dpi = dpi

    def plot(self, param_name, vsqx_data_a, vsqx_data_b,
             features_a, features_b, diff_features,
             correlations, sorted_features,
             param_curve, default_value,
             static_diffs, file_a_name, file_b_name, output_path):
        """
        全情報を1枚の画像にまとめてプロットする。

        Args:
            param_name: 注目パラメータ名 (例: "DYN")
            vsqx_data_a: VSQXReader.read() の出力 (default)
            vsqx_data_b: VSQXReader.read() の出力 (with param)
            features_a: AudioAnalyzer.extract_features() の出力
            features_b: AudioAnalyzer.extract_features() の出力
            diff_features: AudioAnalyzer.compute_diff() の出力
            correlations: AudioAnalyzer.compute_correlations() の出力
            sorted_features: 相関順にソートされた [(name, info), ...]
            param_curve: np.ndarray (T,) — VSQXパラメータ時系列
            default_value: パラメータのデフォルト値
            static_diffs: VSQXReader.find_static_param_diffs() の出力
            output_path: 出力ファイルパス
        """
        n_feature_plots = len(sorted_features)
        # 段数: VSQXパラメータ(1) + 特徴量(n) + 相関テーブル(1)
        n_rows = 1 + n_feature_plots + 1

        # 各行の高さ比率を設定
        height_ratios = [2]  # VSQXパラメータカーブ
        for fname, _ in sorted_features:
            plot_type = FEATURE_PLOT_TYPES.get(fname, "line_1d")
            if plot_type == "spectrogram":
                height_ratios.append(4)
            elif plot_type == "heatmap":
                height_ratios.append(3)
            else:
                height_ratios.append(2)
        height_ratios.append(1.5)  # 相関テーブル

        # 合計高さに合わせてfigsize調整
        total_h = sum(height_ratios)
        fig_height = max(self.figsize[1], total_h * 1.2)

        fig = plt.figure(figsize=(self.figsize[0], fig_height))

        gs = gridspec.GridSpec(
            n_rows, 2,
            height_ratios=height_ratios,
            width_ratios=[1, 0.03],
            hspace=0.35,
            wspace=0.03,
            top=0.93  # タイトル用の余白
        )

        # タイムアクシスの共通設定
        time_axis = self._get_time_axis(param_curve)
        notes_b = vsqx_data_b.get("notes", [])

        # --- タイトル ---
        tempo = vsqx_data_b.get("tempo", 120.0)
        title = f"Parameter Diff Analysis: {param_name}"
        calculation_text = f"Diff = ({file_b_name}) - ({file_a_name})"
        subtitle_parts = [calculation_text, f"BPM={tempo:.0f}"]
        
        if vsqx_data_b.get("voice_info", {}).get("1st_vocal", "unknown") != "unknown":
            subtitle_parts.append(
                f"Voice: {vsqx_data_b['voice_info']['1st_vocal']}"
            )
        subtitle = " | ".join(subtitle_parts)

        # タイトルとサブタイトルを1つにまとめて suptitle に渡す（重なり防止）
        full_title = f"{title}\n"
        
        # 静的パラメータ差分テキスト（差分がある場合のみ）
        if static_diffs:
            static_text = "Static diffs: " + ", ".join(
                f"{k}: {v[0]}→{v[1]}" for k, v in static_diffs.items()
            )
            full_title += f"{subtitle}\n({static_text})"
        else:
            full_title += subtitle

        fig.suptitle(full_title, fontsize=14, fontweight="bold")

        # --- ① VSQXパラメータカーブ ---
        ax_param = fig.add_subplot(gs[0, 0])
        # 右列のカラーバーAxesを非表示（パラメータカーブにはカラーバーなし）
        ax_param_cb = fig.add_subplot(gs[0, 1])
        ax_param_cb.axis("off")
        self._plot_vsqx_param(ax_param, param_name, param_curve,
                               time_axis, default_value, notes_b,
                               file_a_name, file_b_name)

        # --- ②-n 音響特徴量プロット (相関順) ---
        for i, (fname, corr_info) in enumerate(sorted_features):
            ax = fig.add_subplot(gs[1 + i, 0])
            ax_cb = fig.add_subplot(gs[1 + i, 1])  # カラーバー用スペース
            plot_type = FEATURE_PLOT_TYPES.get(fname, "line_1d")
            r_val = corr_info["r"]
            r_abs = abs(r_val)
            corr_color = _correlation_color(r_abs)
            corr_label = _correlation_label(r_abs)

            display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
            ax.set_title(
                f"{display_name}  (r={r_val:.3f}, {corr_label})",
                fontsize=11, fontweight="bold",
                color=corr_color
            )

            if plot_type == "line_1d":
                self._plot_line_1d(
                    ax, fname, features_a, features_b,
                    diff_features, time_axis, notes_b,
                    file_a_name, file_b_name
                )
                ax_cb.axis("off")  # 1Dプロットにはカラーバー不要
            elif plot_type == "spectrogram":
                self._plot_spectrogram_diff(
                    ax, ax_cb, fname, features_a, features_b,
                    diff_features, time_axis, notes_b,
                    corr_info.get("r_per_band")
                )
            elif plot_type == "heatmap":
                self._plot_heatmap_diff(
                    ax, ax_cb, fname, features_a, features_b,
                    diff_features, time_axis, notes_b,
                    corr_info.get("r_per_band")
                )

            # 相関強度を示す左端の色帯
            ax.axvline(x=time_axis[0], color=corr_color, linewidth=4,
                       alpha=0.8)

        # --- ③ 相関係数サマリーテーブル ---
        ax_table = fig.add_subplot(gs[-1, :])
        self._plot_correlation_table(ax_table, sorted_features, param_name)

        # --- 保存 ---
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved: {output_path}")

    def _get_time_axis(self, data):
        """データ長に対応する時間軸"""
        import librosa
        return librosa.frames_to_time(
            np.arange(len(data)), sr=self.sr, hop_length=self.hop_length
        )

    def _add_note_regions(self, ax, notes, alpha=0.08):
        """ノート発声区間を背景色で表示"""
        for note in notes:
            ax.axvspan(note["start_sec"], note["end_sec"],
                      alpha=alpha, color="#3498db", zorder=0)

    def _plot_vsqx_param(self, ax, param_name, param_curve, time_axis,
                          default_value, notes, file_a_name, file_b_name):
        """VSQXパラメータカーブの描画"""
        self._add_note_regions(ax, notes)

        # デフォルト値の水平線
        ax.axhline(y=default_value, color="#e74c3c", linestyle="--",
                   linewidth=1.0, alpha=0.7, label=file_a_name)

        # パラメータカーブ
        ax.plot(time_axis, param_curve, color="#2c3e50", linewidth=1.5,
                label=file_b_name)

        # パラメータが変化している領域をハイライト
        diff_from_default = np.abs(param_curve - default_value)
        ax.fill_between(time_axis, default_value, param_curve,
                        alpha=0.15, color="#3498db")

        ax.set_title(f"VSQX Parameter: {param_name}", fontsize=12,
                     fontweight="bold")
        ax.set_ylabel(f"{param_name} Value")
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.legend(loc="upper left", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (sec)")

    def _plot_line_1d(self, ax, fname, features_a, features_b,
                       diff_features, time_axis, notes,
                       file_a_name, file_b_name):
        """1次元特徴量のオーバーレイ + 差分プロット"""
        self._add_note_regions(ax, notes)

        # fnameに "_diff" が含まれていても実データは元のキーから取得する
        real_fname = fname.replace("_diff", "")
        fa = features_a.get(real_fname)
        fb = features_b.get(real_fname)
        fd = diff_features.get(real_fname)

        if fa is None or fb is None:
            ax.text(0.5, 0.5, "Data not available", ha="center",
                   va="center", transform=ax.transAxes)
            return

        # --- Waveformの特別対応（2音声のオーバーレイのみ表示） ---
        if fname == "waveform":
            min_len = min(fa.shape[0], fb.shape[0])
            _sr = self.sr if hasattr(self, 'sr') else 44100
            t = np.arange(min_len) / _sr

            va = fa[:min_len].flatten()
            vb = fb[:min_len].flatten()

            ax.plot(t, va, color="#3498db", linewidth=1.2, alpha=1.0,
                    zorder=2, label=file_a_name)
            ax.plot(t, vb, color="#e74c3c", linewidth=0.35, alpha=0.5,
                    zorder=3, label=file_b_name)

        else:
            min_len = min(len(time_axis), fa.shape[0], fb.shape[0])
            t = time_axis[:min_len]
            lw = 0.8
            alpha_b = 0.8

            va = fa[:min_len].flatten()
            vb = fb[:min_len].flatten()

            # A と B のオーバーレイ
            ax.plot(t, va, color="#3498db", linewidth=lw, alpha=alpha_b,
                    label=file_a_name)
            ax.plot(t, vb, color="#e74c3c", linewidth=lw, alpha=alpha_b,
                    label=file_b_name)

            # 差分を同一軸にプロット
            if fd is not None:
                vd = fd[:min_len].flatten()
                ax.plot(t, vd, color="#2ecc71", linewidth=lw, alpha=0.9,
                        label="Diff")
                ax.axhline(0, color="black", linestyle="--", linewidth=0.5,
                           alpha=0.5)

        # グラフの横幅スケールを他のグラフ（フレーム単位の時間軸）に厳密に合わせる
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel("Time (sec)")
        
        # 縦軸のラベルを設定
        ylabel = FEATURE_YLABELS.get(real_fname, "Value")
        ax.set_ylabel(ylabel)
        
        leg = ax.legend(loc="upper left", fontsize=10, framealpha=0.8)
        leg.set_zorder(5)
        for line in leg.get_lines():
            line.set_linewidth(2.5)

        ax.grid(True, alpha=0.2, zorder=0)

    def _plot_spectrogram_diff(self, ax, ax_cb, fname, features_a, features_b,
                                diff_features, time_axis, notes,
                                r_per_band=None):
        """スペクトログラム差分の描画"""
        fd = diff_features.get(fname)
        if fd is None:
            ax.text(0.5, 0.5, "Data not available", ha="center",
                   va="center", transform=ax.transAxes)
            ax_cb.axis("off")
            return

        min_len = min(len(time_axis), fd.shape[0])
        fd = fd[:min_len]
        t = time_axis[:min_len]

        # 差分スペクトログラムを diverging カラーマップで表示
        vmax = np.percentile(np.abs(fd), 97)
        if vmax < 1e-10:
            vmax = 1.0

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        img = ax.imshow(
            fd.T, aspect="auto", origin="lower",
            extent=[t[0], t[-1], 0, fd.shape[1]],
            cmap="RdBu_r", norm=norm,
            interpolation="nearest"
        )

        # ノート区間の境界線
        for note in notes:
            ax.axvline(x=note["start_sec"], color="white", linewidth=0.3,
                       alpha=0.5)

        # カラーバーを専用の右列Axesに配置
        cbar = plt.colorbar(img, cax=ax_cb)
        cbar.set_label("Diff", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Band")

        # メルスペクトログラムの場合、Y軸に大まかな周波数(Hz)を併記する
        if fname == "mel":
            import librosa
            n_mels = fd.shape[1]
            mel_f = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=self.sr/2)

            tick_indices = np.arange(0, n_mels, 10)
            tick_locs = tick_indices + 0.5
            tick_labels = [f"{i}\n({mel_f[i]:.0f}Hz)" for i in tick_indices]

            ax.set_yticks(tick_locs)
            ax.set_yticklabels(tick_labels, fontsize=7)

        # STFT / Aperiodicity の場合、Y軸に周波数(Hz)を表示する
        if fname in ("stft", "aperiodicity"):
            n_bins = fd.shape[1]
            freqs = np.linspace(0, self.sr / 2, n_bins)

            # 約10個のティックを等間隔に配置
            n_ticks = min(10, n_bins)
            tick_indices = np.linspace(0, n_bins - 1, n_ticks).astype(int)
            tick_locs = tick_indices + 0.5
            tick_labels = [f"{freqs[i]:.0f}Hz" for i in tick_indices]

            ax.set_yticks(tick_locs)
            ax.set_yticklabels(tick_labels, fontsize=7)
            ax.set_ylabel("Frequency (Hz)")

        # 帯域ごとの相関をタイトル下に表示
        if r_per_band is not None:
            n_bands = len(r_per_band)
            max_r_band = np.argmax(np.abs(r_per_band))
            max_r_val = r_per_band[max_r_band]
            ax.text(0.99, 0.97,
                    f"Max |r|: band {max_r_band} (r={max_r_val:.3f})",
                    transform=ax.transAxes, fontsize=7,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="lightyellow", alpha=0.85,
                              edgecolor="#ccc"))

    def _plot_heatmap_diff(self, ax, ax_cb, fname, features_a, features_b,
                            diff_features, time_axis, notes,
                            r_per_band=None):
        """ヒートマップ差分の描画"""
        self._plot_spectrogram_diff(
            ax, ax_cb, fname, features_a, features_b,
            diff_features, time_axis, notes, r_per_band
        )



    def _plot_correlation_table(self, ax, sorted_features, param_name):
        """相関係数のサマリーテーブルを描画"""
        ax.axis("off")

        if not sorted_features:
            ax.text(0.5, 0.5, "No correlation data", ha="center",
                   va="center")
            return

        # テーブルデータ
        col_labels = ["Feature", "r value", "p value", "Strength"]
        table_data = []

        for fname, corr_info in sorted_features:
            display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
            r_val = corr_info["r"]
            p_val = corr_info["p"]
            r_abs = abs(r_val)

            p_str = f"< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            strength = _correlation_label(r_abs)

            table_data.append([display_name, f"{r_val:.4f}", p_str, strength])

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        # ヘッダーの色
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#2c3e50")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # 各行に相関強度の色をつける
        for i, (fname, corr_info) in enumerate(sorted_features):
            r_abs = abs(corr_info["r"])
            color = _correlation_color(r_abs)
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color + "20")  # 薄い背景色

        ax.set_title(f"Correlation Summary: {param_name} vs Audio Features",
                    fontsize=11, fontweight="bold", pad=10)
