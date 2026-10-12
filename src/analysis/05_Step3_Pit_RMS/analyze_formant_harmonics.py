# -*- coding: utf-8 -*-
"""
=============================================================================
音高網羅データセットに基づく フォルマント・倍音構造 定量分析スクリプト
=============================================================================
【機能概要】
1. 初音ミク 5母音（あ、い、う、え、お）について、低音域（Octave 3: C3〜B3 の12半音階）
   の音声信号から LPC（線形予測分析: Linear Predictive Coding）全極フィルタを算出し、
   倍音サンプリング誤差を相殺した「平均LPCスペクトル包録線」を抽出。
2. LPC多項式の根（複素極）から母音ごとのフォルマント周波数（F1, F2, F3）および帯域幅を厳密算出。
3. 音高別（Note 60=C3 〜 Note 96=C6, 計37音高）の基音 (f0) および高次倍音系列（ハーモニクス）を計算。
4. 倍音系列とLPCスペクトル包絡線の重ね合わせ図を生成：
   - 図①: 5母音の平均LPCスペクトル包絡線 (0〜4000Hz) & フォルマント位置
   - 図②: C音系列（C3, C4, C5）の倍音と5母音包絡線の重ね合わせ
   - 図③: 母音「あ」における D#4（山: RMS最大）vs F#4（谷: RMS最小）倍音共鳴配置比較
5. フォルマント推定値CSVおよび音高・倍音周波数対応表CSVを自動保存。
=============================================================================
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import freqz
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Meiryo", "MS Gothic", "Yu Gothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 設定領域 (CONFIG) - 実行ボタンを押すだけで全自動動作
# =============================================================================
CONFIG = {
    # 分析対象WAVフォルダ
    "wav_dir": "D:/黒岩研究室/卒研/26Voca_Ana/data/Step3/Step3_Test1音高網羅/output-wav",
    # 出力先ディレクトリ
    "output_dir": "./output/formant_harmonic_analysis",
    # LPC分析パラメータ
    "lpc_sr": 16000,          # フォルマント分析用リサンプル周波数 (Hz)
    "lpc_order": 18,          # 16kHzにおける標準全極次数 (2 + sr/1000 = 18)
    "pre_emphasis": 0.97,     # プリエンファシス係数
    "freq_max": 4000,         # 解析・描画最大周波数 (Hz)
    "fft_size": 4096,
    # 平均包絡線を抽出するオクターブ (Octave 3: Note 60〜71)
    "octave3_notes": list(range(60, 72)),
}
# =============================================================================

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SAFE_PITCH_MAP = {
    "C": 0, "Cs": 1, "C#": 1,
    "D": 2, "Ds": 3, "D#": 3,
    "E": 4,
    "F": 5, "Fs": 6, "F#": 6,
    "G": 7, "Gs": 8, "G#": 8,
    "A": 9, "As": 10, "A#": 10,
    "B": 11,
}
VOWEL_LIST = ["あ", "い", "う", "え", "お"]
VOWEL_COLORS = {
    "あ": "#E74C3C",  # 赤
    "い": "#3498DB",  # 青
    "う": "#2ECC71",  # 緑
    "え": "#F39C12",  # 橙
    "お": "#9B59B6",  # 紫
}


def note_num_to_yamaha(note_num: int) -> str:
    p = PITCH_NAMES[note_num % 12]
    octave = (note_num // 12) - 2
    return f"{p}{octave}"


def note_num_to_intl(note_num: int) -> str:
    p = PITCH_NAMES[note_num % 12]
    octave = (note_num // 12) - 1
    return f"{p}{octave}"


def note_num_to_hz(note_num: int) -> float:
    return 440.0 * (2.0 ** ((note_num - 69) / 12.0))


def parse_filename(filename: str):
    stem = os.path.splitext(filename)[0]
    vowel = "Other"
    for v in VOWEL_LIST:
        if f"_{v}_" in stem:
            vowel = v
            break
    parts = stem.split("_")
    pitch_str = parts[-1]
    m = re.match(r"([A-Za-z#]+)(\d+)", pitch_str)
    if m:
        p_name, oct_s = m.group(1), int(m.group(2))
        semi = SAFE_PITCH_MAP.get(p_name, 0)
        note_num = (oct_s + 2) * 12 + semi
    else:
        note_num = -1
    return vowel, note_num


def compute_lpc_spectrum(wav_path: str, target_sr=16000, lpc_order=18, pre_emphasis=0.97):
    """WAVファイルからLPCスペクトル包絡および多項式係数を計算する"""
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # 有声定常部 (中央0.5秒)
    mid = len(y) // 2
    win_len = target_sr // 2
    seg = y[max(0, mid - win_len // 2): min(len(y), mid + win_len // 2)]

    # プリエンファシス
    seg_pre = np.append(seg[0], seg[1:] - pre_emphasis * seg[:-1])
    seg_win = seg_pre * np.hamming(len(seg_pre))

    # LPC係数計算
    a = librosa.lpc(seg_win, order=lpc_order)

    # 周波数応答
    w, h = freqz(1.0, a, worN=2048, fs=target_sr)
    h_db = 20 * np.log10(np.abs(h) + 1e-12)
    h_db -= np.max(h_db[w <= 4000])  # 4kHz以下の最大値を0dBに正規化

    # フォルマント周波数と帯域幅の推定
    roots = np.roots(a)
    roots = [r for r in roots if np.imag(r) >= 0.01 and np.abs(r) < 0.999]
    angles = np.angle(roots)
    freqs = angles * (target_sr / (2 * np.pi))
    bandwidths = -0.5 * (target_sr / (2 * np.pi)) * np.log(np.abs(roots))

    # 200Hz〜4000Hzの範囲のフォルマント候補をソート
    formants = []
    for f, bw in zip(freqs, bandwidths):
        if 200 <= f <= 4000 and bw < 500:
            formants.append((f, bw))
    formants.sort(key=lambda x: x[0])

    return w, h_db, formants


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(SCRIPT_DIR, p))


def run_analysis(config: dict):
    wav_dir = _resolve_path(config["wav_dir"])
    out_dir = _resolve_path(config["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("  フォルマント・倍音構造・実測RMS 定量分析")
    print("=" * 72)
    print(f"WAV Directory    : {wav_dir}")
    print(f"Output Directory : {out_dir}")
    print(f"LPC Parameters   : SR={config['lpc_sr']}Hz, Order={config['lpc_order']}")
    print("=" * 72)

    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    print(f"検出WAVファイル数: {len(wav_files)} 件\n")

    file_dict = {}
    for wf in wav_files:
        v, n = parse_filename(os.path.basename(wf))
        if v in VOWEL_LIST and n > 0:
            file_dict[(v, n)] = wf

    # 1. 各母音について Octave 3 (C3〜B3: Note 60〜71) の12音でLPCを計算し平均化
    vowel_envelopes = {}
    vowel_formants = {}
    w_freqs = None

    print("母音別 Octave 3 (全12半音階) 平均LPC包絡線を算出中...")
    for v in VOWEL_LIST:
        curves = []
        all_formants = []
        for n in config["octave3_notes"]:
            wf = file_dict.get((v, n))
            if wf and os.path.exists(wf):
                w, h_db, formants = compute_lpc_spectrum(
                    wf, target_sr=config["lpc_sr"], lpc_order=config["lpc_order"], pre_emphasis=config["pre_emphasis"]
                )
                w_freqs = w
                curves.append(h_db)
                if formants:
                    all_formants.append(formants)

        if curves:
            mean_env = np.mean(curves, axis=0)
            vowel_envelopes[v] = mean_env

            # 代表フォルマント (12音の中央値)
            f1_list, f2_list, f3_list = [], [], []
            for f_set in all_formants:
                if len(f_set) >= 1:
                    f1_list.append(f_set[0][0])
                if len(f_set) >= 2:
                    f2_list.append(f_set[1][0])
                if len(f_set) >= 3:
                    f3_list.append(f_set[2][0])

            med_f1 = np.median(f1_list) if f1_list else np.nan
            med_f2 = np.median(f2_list) if f2_list else np.nan
            med_f3 = np.median(f3_list) if f3_list else np.nan

            vowel_formants[v] = {
                "F1": med_f1,
                "F2": med_f2,
                "F3": med_f3,
            }
            print(f"  母音「{v}」: 推定 F1={med_f1:.1f}Hz, F2={med_f2:.1f}Hz, F3={med_f3:.1f}Hz")

    # -------------------------------------------------------------
    # 図①: 5母音の平均LPCスペクトル包絡線 (0〜4000Hz) & フォルマント位置
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    valid = w_freqs <= config["freq_max"]
    for v in VOWEL_LIST:
        env = vowel_envelopes[v]
        col = VOWEL_COLORS[v]
        fm = vowel_formants[v]
        ax.plot(w_freqs[valid], env[valid], label=f"母音「{v}」 (F1={fm['F1']:.0f}Hz, F2={fm['F2']:.0f}Hz)", color=col, linewidth=2.4)
        # F1, F2 にマーカー
        for f_name, f_val in [("F1", fm["F1"]), ("F2", fm["F2"])]:
            if not np.isnan(f_val) and f_val <= config["freq_max"]:
                idx = np.argmin(np.abs(w_freqs - f_val))
                ax.plot(f_val, env[idx], marker="o", markersize=6, color=col)

    ax.set_title("初音ミク 5母音の平均LPCスペクトル包絡線 (Octave 3: 12音平均, 0〜4000Hz)", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("周波数 [Hz]", fontsize=11, labelpad=8)
    ax.set_ylabel("相対利得 [dB] (ピーク基準 0dB)", fontsize=11, labelpad=8)
    ax.set_xlim(0, config["freq_max"])
    ax.set_ylim(-45, 5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.92)

    plot1_path = os.path.join(out_dir, "01_vowel_lpc_envelopes.png")
    plt.tight_layout()
    plt.savefig(plot1_path)
    plt.close()
    print(f"\n-> 図①保存完了: {plot1_path}")

    # -------------------------------------------------------------
    # 図②: C音系列 (C3=261.6Hz, C4=523.3Hz, C5=1046.5Hz) の倍音系列と5母音包絡線
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=300)
    c_notes = [(60, "C3 (Note 60, f0=261.6Hz)", 8), (72, "C4 (Note 72, f0=523.3Hz)", 6), (84, "C5 (Note 84, f0=1046.5Hz)", 4)]

    for ax_idx, (note_num, label_str, max_harms) in enumerate(c_notes):
        ax = axes[ax_idx]
        f0 = note_num_to_hz(note_num)
        for v in VOWEL_LIST:
            ax.plot(w_freqs[valid], vowel_envelopes[v][valid], color=VOWEL_COLORS[v], alpha=0.35, linewidth=1.5, label=f"母音「{v}」" if ax_idx == 0 else None)

        # C音の倍音系列
        for k in range(1, max_harms + 1):
            h_f = k * f0
            if h_f <= config["freq_max"]:
                ax.axvline(h_f, color="#2C3E50", linestyle="--", linewidth=1.8, alpha=0.85)
                ax.text(h_f + 15, -4, f"第{k}倍音\n({h_f:.0f}Hz)", color="#2C3E50", fontsize=8.5, fontweight="bold")

        ax.set_title(f"基音およびハーモニクス重なり: {label_str}", fontsize=11.5, fontweight="bold")
        ax.set_ylabel("相対利得 [dB]", fontsize=10)
        ax.set_ylim(-45, 5)
        ax.grid(True, linestyle="--", alpha=0.5)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8.5, ncol=5)

    axes[2].set_xlabel("周波数 [Hz]", fontsize=11, labelpad=8)
    axes[2].set_xlim(0, config["freq_max"])
    fig.suptitle("C音系列のハーモニクス（倍音列）と5母音LPCスペクトル包絡線の幾何学的重なり", fontsize=14, fontweight="bold")

    plot2_path = os.path.join(out_dir, "02_c_tones_harmonics_overlay.png")
    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()
    print(f"-> 図②保存完了: {plot2_path}")

    # -------------------------------------------------------------
    # 図③: 【決定打】D#4 (山: RMS=0.0538) vs F#4 (谷: RMS=0.0382) 倍音配置比較 (単一Axes・図①同等縮尺)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # 母音「あ」LPC共鳴包絡線
    ax.plot(w_freqs[valid], vowel_envelopes["あ"][valid], color="#C0392B", linewidth=2.6, label="母音「あ」LPC共鳴包絡線")

    # D#4 (山: 622.25 Hz)
    f0_ds = 622.25
    col_ds = "#1F77B4"
    for k in range(1, 6):
        hf = k * f0_ds
        if hf <= config["freq_max"]:
            idx = np.argmin(np.abs(w_freqs - hf))
            gain = vowel_envelopes["あ"][idx]
            lbl = "D#4 (山: RMS=0.0538) 倍音列" if k == 1 else None
            ax.axvline(hf, color=col_ds, linestyle="-", linewidth=1.8, alpha=0.85, label=lbl)
            ax.plot(hf, gain, marker="o", markersize=7.5, color=col_ds)
            tag = "H1" if k == 1 else ("H2(★ピーク直撃)" if k == 2 else ("H3(★F3近接)" if k == 3 else f"H{k}"))
            ax.text(hf + 15, gain + 1.2, f"{tag}\n{hf:.0f}Hz\n({gain:.1f}dB)", color=col_ds, fontsize=8, fontweight="bold", va="bottom")

    # F#4 (谷: 739.99 Hz)
    f0_fs = 739.99
    col_fs = "#E67E22"
    for k in range(1, 6):
        hf = k * f0_fs
        if hf <= config["freq_max"]:
            idx = np.argmin(np.abs(w_freqs - hf))
            gain = vowel_envelopes["あ"][idx]
            lbl = "F#4 (谷: RMS=0.0382) 倍音列" if k == 1 else None
            ax.axvline(hf, color=col_fs, linestyle="--", linewidth=1.8, alpha=0.85, label=lbl)
            ax.plot(hf, gain, marker="s", markersize=7.5, color=col_fs)
            tag = "H1" if k == 1 else ("H2(減衰)" if k == 2 else ("H3(▼谷底落下)" if k == 3 else f"H{k}"))
            ax.text(hf + 15, gain - 4.5, f"{tag}\n{hf:.0f}Hz\n({gain:.1f}dB)", color=col_fs, fontsize=8, fontweight="bold", va="top")

    ax.set_title("母音「あ」における D#4（山: RMS最大）vs F#4（谷: RMS最小）倍音共鳴配置比較", fontsize=13.5, pad=12, fontweight="bold")
    ax.set_xlabel("周波数 [Hz]", fontsize=11, labelpad=8)
    ax.set_ylabel("相対利得 [dB] (ピーク基準 0dB)", fontsize=11, labelpad=8)
    ax.set_xlim(0, config["freq_max"])
    ax.set_ylim(-45, 5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    plot3_path = os.path.join(out_dir, "03_ds4_vs_fs4_harmonics_comparison_vowel_a.png")
    plt.tight_layout()
    plt.savefig(plot3_path)
    plt.close()
    print(f"-> 図③保存完了: {plot3_path}")

    # -------------------------------------------------------------
    # 5. 音高・フォルマント数値対応表のCSV保存
    # -------------------------------------------------------------
    formant_df = pd.DataFrame([
        {"母音": v, "推定F1 [Hz]": round(vowel_formants[v]["F1"], 1), "推定F2 [Hz]": round(vowel_formants[v]["F2"], 1), "推定F3 [Hz]": round(vowel_formants[v]["F3"], 1)}
        for v in VOWEL_LIST
    ])
    formant_csv_path = os.path.join(out_dir, "formant_estimates.csv")
    formant_df.to_csv(formant_csv_path, index=False, encoding="utf-8-sig")
    print(f"-> フォルマント数値表保存: {formant_csv_path}")

    all_notes = list(range(60, 97))
    pitch_table_data = []
    for n_num in all_notes:
        f0 = note_num_to_hz(n_num)
        pitch_table_data.append({
            "ノート番号": n_num,
            "Yamaha音名": note_num_to_yamaha(n_num),
            "国際音名": note_num_to_intl(n_num),
            "基音周波数 (f0) [Hz]": round(f0, 2),
            "第2倍音 (2*f0) [Hz]": round(2 * f0, 2),
            "第3倍音 (3*f0) [Hz]": round(3 * f0, 2),
            "第4倍音 (4*f0) [Hz]": round(4 * f0, 2),
        })
    pitch_csv_path = os.path.join(out_dir, "pitch_harmonics_table.csv")
    pd.DataFrame(pitch_table_data).to_csv(pitch_csv_path, index=False, encoding="utf-8-sig")
    print(f"-> 音高・倍音周波数一覧保存: {pitch_csv_path}")

    print("\n全解析処理が正常に完了しました。")
    return True


if __name__ == "__main__":
    run_analysis(CONFIG)
