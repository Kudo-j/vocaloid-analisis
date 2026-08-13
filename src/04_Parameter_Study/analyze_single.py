"""
単一音声の音響特徴量解析ツール
=================================
WAV_A と VSQX_A を入力し、設定された音響特徴量を抽出して
時間変化を1枚の画像にプロットします。差分や比較は行いません。

使い方:
  1. compare_config.py で WAV_A, VSQX_A, 抽出したい特徴量を設定
  2. python analyze_single.py を実行
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# srcをパスに追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "util"))

import compare_config as cfg
from vsqx_compare import VSQXCompare
from audio_analyzer import (
    AudioAnalyzer,
    FEATURE_DISPLAY_NAMES,
    FEATURE_PLOT_TYPES,
    FEATURE_YLABELS,
)


def _setup_japanese_font():
    import matplotlib.font_manager as fm

    candidates = [
        "IPAexGothic",
        "IPAGothic",
        "Noto Sans CJK JP",
        "Yu Gothic",
        "MS Gothic",
        "Meiryo",
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            return


_setup_japanese_font()
plt.rcParams["axes.unicode_minus"] = False


def _build_feature_list():
    """ALL_FEATURE_TYPES から ALWAYS_EXCLUDE を除いた特徴量リストを構築"""
    exclude = set(getattr(cfg, "ALWAYS_EXCLUDE_FEATURES", []))
    features = [f for f in cfg.ALL_FEATURE_TYPES if f not in exclude]
    return features


class SingleVisualizer:
    def __init__(self, sr=44100, shift_length=512, figsize=(16, 20), dpi=300):
        self.sr = sr
        self.shift_length = shift_length
        self.figsize = figsize
        self.dpi = dpi

    def _get_time_axis(self, data_len):
        import librosa

        return librosa.frames_to_time(
            np.arange(data_len), sr=self.sr, hop_length=self.shift_length
        )

    def _add_note_regions(self, ax, notes, alpha=0.1):
        for note in notes:
            ax.axvspan(
                note["start_sec"],
                note["end_sec"],
                alpha=alpha,
                color="#3498db",
                zorder=0,
            )

    def plot(self, features, vsqx_data, file_name, output_path):
        feature_names = list(features.keys())
        # waveformを先頭にするための処理
        if "waveform" in feature_names:
            feature_names.remove("waveform")
            feature_names.insert(0, "waveform")

        n_plots = len(feature_names)

        height_ratios = []
        for fname in feature_names:
            plot_type = FEATURE_PLOT_TYPES.get(fname, "line_1d")
            if plot_type == "spectrogram" or plot_type == "heatmap":
                height_ratios.append(4)
            else:
                height_ratios.append(2)

        total_h = sum(height_ratios)
        # 高さを特徴量の数に応じて可変にする
        fig_height = max(self.figsize[1], total_h * 1.5)

        fig = plt.figure(figsize=(self.figsize[0], fig_height))
        gs = gridspec.GridSpec(
            n_plots,
            2,
            height_ratios=height_ratios,
            width_ratios=[1, 0.03],
            hspace=0.4,
            wspace=0.03,
            top=0.96,
        )

        tempo = vsqx_data.get("tempo", 120.0)
        notes = vsqx_data.get("notes", [])

        voice_info = vsqx_data.get("voice_info", {}).get("1st_vocal", "unknown")
        subtitle = (
            f"BPM={tempo:.0f} | Voice: {voice_info}"
            if voice_info != "unknown"
            else f"BPM={tempo:.0f}"
        )
        fig.suptitle(
            f"Acoustic Features Analysis: {file_name}\n({subtitle})",
            fontsize=16,
            fontweight="bold",
        )

        # 共通の時間軸長を決定 (waveform以外)
        lengths = [
            feat.shape[0] for name, feat in features.items() if name != "waveform"
        ]
        max_len = max(lengths) if lengths else 0
        time_axis = self._get_time_axis(max_len)

        for i, fname in enumerate(feature_names):
            ax = fig.add_subplot(gs[i, 0])
            ax_cb = fig.add_subplot(gs[i, 1])

            feat_data = features[fname]
            plot_type = FEATURE_PLOT_TYPES.get(fname, "line_1d")
            display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)

            ax.set_title(display_name, fontsize=12, fontweight="bold")

            if fname == "waveform":
                _sr = self.sr
                t = np.arange(feat_data.shape[0]) / _sr
                ax.plot(
                    t, feat_data.flatten(), color="#3498db", linewidth=1.0, alpha=0.9
                )
                self._add_note_regions(ax, notes, alpha=0.2)
                ax.set_xlim(0, t[-1])
                ax.set_ylabel("Amplitude")
                ax_cb.axis("off")

            elif plot_type == "line_1d":
                min_len = min(len(time_axis), feat_data.shape[0])
                t = time_axis[:min_len]
                ax.plot(
                    t, feat_data[:min_len].flatten(), color="#2c3e50", linewidth=1.5
                )
                self._add_note_regions(ax, notes)
                ax.set_xlim(time_axis[0], time_axis[-1] if len(time_axis) > 0 else 1)
                ax.set_ylabel(FEATURE_YLABELS.get(fname, "Value"))
                ax.grid(True, alpha=0.3)
                ax_cb.axis("off")

            elif plot_type in ["spectrogram", "heatmap"]:
                min_len = min(len(time_axis), feat_data.shape[0])
                fd = feat_data[:min_len]
                t = time_axis[:min_len]

                # 正規化表示用
                if fname == "mel" or fname == "stft":
                    fd_plot = 10 * np.log10(np.maximum(fd, 1e-10))
                    cmap = "magma"
                else:
                    fd_plot = fd
                    cmap = "viridis"

                img = ax.imshow(
                    fd_plot.T,
                    aspect="auto",
                    origin="lower",
                    extent=[t[0], t[-1], 0, fd.shape[1]],
                    cmap=cmap,
                    interpolation="nearest",
                )

                for note in notes:
                    ax.axvline(
                        x=note["start_sec"], color="white", linewidth=0.5, alpha=0.5
                    )

                cbar = plt.colorbar(img, cax=ax_cb)
                cbar.ax.tick_params(labelsize=8)

                ax.set_xlim(time_axis[0], time_axis[-1] if len(time_axis) > 0 else 1)
                ax.set_ylabel("Band / Freq")

                if fname == "mel":
                    import librosa

                    n_mels = fd.shape[1]
                    mel_f = librosa.mel_frequencies(
                        n_mels=n_mels, fmin=0.0, fmax=self.sr / 2
                    )
                    tick_indices = np.arange(0, n_mels, 10)
                    ax.set_yticks(tick_indices + 0.5)
                    ax.set_yticklabels(
                        [f"{mel_f[i]:.0f}Hz" for i in tick_indices], fontsize=8
                    )
                elif fname in ("stft", "aperiodicity"):
                    n_bins = fd.shape[1]
                    freqs = np.linspace(0, self.sr / 2, n_bins)
                    n_ticks = min(10, n_bins)
                    tick_indices = np.linspace(0, n_bins - 1, n_ticks).astype(int)
                    ax.set_yticks(tick_indices + 0.5)
                    ax.set_yticklabels(
                        [f"{freqs[i]:.0f}Hz" for i in tick_indices], fontsize=8
                    )
                    ax.set_ylabel("Frequency (Hz)")

            if i == n_plots - 1:
                ax.set_xlabel("Time (sec)")
            else:
                ax.set_xticklabels([])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved visualization: {output_path}")


def main():
    print("=" * 50)
    print("  Single Audio Feature Analysis Tool")
    print("=" * 50)

    path_wav = cfg.WAV_A
    path_vsqx = cfg.VSQX_A
    output_dir = os.path.join(cfg.OUTPUT_DIR, "single_analysis")

    compute_features = _build_feature_list()
    if "waveform" not in compute_features:
        compute_features.append("waveform")

    print(f"\n[Config]")
    print(f"  WAV:  {path_wav}")
    print(f"  VSQX: {path_vsqx}")
    print(f"  Features: {len(compute_features)} items")

    for path in [path_wav, path_vsqx]:
        if not os.path.exists(path):
            print(f"\nERROR: File not found: {path}")
            sys.exit(1)

    print("\n[Step 1] Parsing VSQX...")
    reader = VSQXCompare(sr=cfg.SR, shift_length=cfg.SHIFT_LENGTH)
    vsqx_data = reader.read(path_vsqx)
    if vsqx_data is None:
        print("ERROR: Failed to parse VSQX file.")
        sys.exit(1)
    print(f"  tempo={vsqx_data['tempo']}BPM, notes={len(vsqx_data['notes'])}")

    print("\n[Step 2] Extracting audio features...")
    analyzer = AudioAnalyzer(sr=cfg.SR, shift_length=cfg.SHIFT_LENGTH)

    # AudioAnalyzer の extract_features_paired を使用するか、内部の機能で抽出するか
    # 単体抽出用の簡易ラッパー
    def extract_single(path, features_list):
        y, sr = analyzer._load_audio(path)
        y, f0, t_f0 = analyzer._extract_f0(y)
        res = {}
        for fname in features_list:
            if fname == "waveform":
                res[fname] = y
            else:
                method_name = f"_compute_{fname}"
                if hasattr(analyzer.extractor, method_name):
                    method = getattr(analyzer.extractor, method_name)
                    # _compute 系メソッドの引数要件に応じた処理 (通常は y, f0, t_f0 を渡せばよいように実装されているはず)
                    # extract_features_paired の中身を確認しなくても _compute を呼べば取得可能だが、
                    # extract_features_paired を A/B 同じファイルで呼んで A だけ取るのが一番安全で手っ取り早い。
                    pass
        return res

    # 既存の extract_features_paired を利用して単体抽出を行う（最も安全）
    features_a, _ = analyzer.extract_features_paired(
        path_wav, path_wav, compute_features
    )
    print(f"  Extracted {len(features_a)} features")

    print("\n[Step 3] Visualizing...")
    visualizer = SingleVisualizer(sr=cfg.SR, shift_length=cfg.SHIFT_LENGTH)
    base_name = os.path.splitext(os.path.basename(path_wav))[0]
    output_path = os.path.join(output_dir, f"{base_name}_features.png")

    visualizer.plot(features_a, vsqx_data, os.path.basename(path_wav), output_path)

    print("\n" + "=" * 50)
    print("  Analysis complete!")
    print(f"  Output: {os.path.abspath(output_path)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
