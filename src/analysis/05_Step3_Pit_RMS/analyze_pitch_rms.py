"""
analyze_pitch_rms.py
====================
音高網羅データセット (Step3_Test1音高網羅: 5母音 × 37音高 = 計185曲) を対象に、
音高（Note 60=C3 〜 Note 96=C6）と実効値（RMS）の物理的・音響的関係を
網羅的・定量的に解析・可視化するスクリプト。

主な機能:
  1. 全185曲のWAVから有声区間RMSおよび統計量（平均、最大、分散）を自動抽出
  2. シリーズ1: 音高（Note 60〜96） vs 平均RMS（全体表示および各母音別の計6枚）
  3. シリーズ2: オクターブ別（C3〜B3, C4〜B4, C5〜B5）RMS比較（全体表示および各母音別の計6枚）
  4. 詳細Excelレポート出力（全曲一覧、母音別サマリー、音高別サマリー、条件付き書式付き）
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# 日本語フォント設定（文字化け防止）
mpl.rcParams["font.family"] = ["Meiryo", "MS Gothic", "Yu Gothic", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False

# Excel スタイリング用
from openpyxl import Workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

# パス設定 (util フォルダのインポート)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "util")))

try:
    from feature_extractor import AudioFeatureExtractor
except ImportError:
    AudioFeatureExtractor = None


# =============================================================================
# 設定領域 (CONFIG) - 実行ボタンを押すだけで全自動動作
# =============================================================================
CONFIG = {
    # (必須) 分析対象WAVフォルダ
    "wav_dir": "./data/Step3/Step3_Test1音高網羅/output-wav",
    # (必須) 出力先ディレクトリ (グラフ画像、Excelレポート保存先)
    "output_dir": "./output/pitch_rms_analysis",
    # 音高範囲 (Yamaha規格: C3=60 〜 C6=96, 両端含む37段階)
    "min_note_num": 60,
    "max_note_num": 96,
    # サンプリング設定 (リポジトリ標準)
    "sr": 44100,
    "shift_length": 512,
    "frame_length": 1024,
    # 有声発声区間の自動検出閾値 (曲内最大RMSに対する比率)
    "silence_ratio": 0.05,
    # グラフ保存を行うか
    "save_plots": True,
    # Excelレポート保存を行うか
    "save_excel": True,
}
# =============================================================================


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SAFE_PITCH_MAP = {
    "C": 0,
    "Cs": 1,
    "C#": 1,
    "D": 2,
    "Ds": 3,
    "D#": 3,
    "E": 4,
    "F": 5,
    "Fs": 6,
    "F#": 6,
    "G": 7,
    "Gs": 8,
    "G#": 8,
    "A": 9,
    "As": 10,
    "A#": 10,
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
VOWEL_ROMAJI = {
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
}


def _resolve_dir(path_str: str) -> str:
    """相対パスまたは絶対パスを解決する。"""
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # スクリプトディレクトリ基準で解決
    cand = os.path.normpath(os.path.join(SCRIPT_DIR, path_str))
    if os.path.exists(cand) or os.path.exists(os.path.dirname(cand)):
        return os.path.abspath(cand)

    return os.path.abspath(path_str)


def note_num_to_name(note_num: int) -> str:
    """MIDIノート番号をYamaha音名表記 (例: 60 -> C3, 61 -> C#3) に変換する。"""
    pitch = PITCH_NAMES[note_num % 12]
    octave = (note_num // 12) - 2
    return f"{pitch}{octave}"


def note_num_to_hz(note_num: int) -> float:
    """MIDIノート番号を周波数 (Hz, A4=440Hz) に変換する。"""
    return 440.0 * (2.0 ** ((note_num - 69) / 12.0))


def parse_filename_metadata(filename: str) -> dict:
    """
    ファイル名 (例: Step3_Benchmark_Test_あ_1_Cs3.wav) から
    vowel, note_name, note_num, hz を高精度に抽出する。
    """
    stem = Path(filename).stem

    # 母音判定
    vowel = "Other"
    for v in VOWEL_LIST:
        if f"_{v}_" in stem or stem.startswith(f"{v}_") or stem.endswith(f"_{v}"):
            vowel = v
            break

    # 音名抽出 (末尾の _[A-G]s?[0-9] など)
    m = re.search(r"_([A-G][s#]?)(\d+)$", stem)
    if not m:
        # パターン2: アンダースコアなしで末尾にある場合
        m = re.search(r"([A-G][s#]?)(\d+)$", stem)

    if m:
        p_str = m.group(1)
        octave = int(m.group(2))
        semitone = SAFE_PITCH_MAP.get(p_str, 0)
        note_num = (octave + 2) * 12 + semitone
        standard_pitch = PITCH_NAMES[semitone]
        note_name = f"{standard_pitch}{octave}"
    else:
        note_num = -1
        note_name = "Unknown"

    hz = note_num_to_hz(note_num) if note_num > 0 else 0.0

    return {
        "filename": filename,
        "stem": stem,
        "vowel": vowel,
        "note_num": note_num,
        "note_name": note_name,
        "frequency_hz": round(hz, 2),
    }


def extract_rms_from_wav(
    wav_path: str, extractor, sr=44100, frame_length=1024, shift_length=512
) -> np.ndarray:
    """WAVファイルからRMS時系列配列を抽出する。"""
    import soundfile as sf
    import librosa

    # AudioFeatureExtractor を使用 (リポジトリ標準)
    if extractor is not None:
        feat = extractor.get_features(wav_path, ["rms"])
        if feat.ndim > 1:
            return feat.squeeze(-1)
        return feat

    # フォールバック: librosa
    try:
        y, file_sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    except Exception:
        y, _ = librosa.load(wav_path, sr=sr, mono=True)

    rms_series = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=shift_length
    )[0]
    return rms_series


def analyze_dataset(config: dict):
    """メイン解析処理"""
    wav_dir = _resolve_dir(config["wav_dir"])
    output_dir = _resolve_dir(config["output_dir"])
    min_note = config.get("min_note_num", 60)
    max_note = config.get("max_note_num", 96)
    sr = config.get("sr", 44100)
    shift_length = config.get("shift_length", 512)
    frame_length = config.get("frame_length", 1024)
    silence_ratio = config.get("silence_ratio", 0.05)
    save_plots = config.get("save_plots", True)
    save_excel = config.get("save_excel", True)

    print("=" * 72)
    print("  音高・母音別 RMS 物理特性解析 (Step 3 音高網羅データセット)")
    print("=" * 72)
    print(f"WAV Directory       : {wav_dir}")
    print(f"Output Directory    : {output_dir}")
    print(
        f"Pitch Range         : Note {min_note} ({note_num_to_name(min_note)}) 〜 Note {max_note} ({note_num_to_name(max_note)}) [37 steps]"
    )
    print(f"Sampling Parameters : SR={sr}Hz, Hop={shift_length}, Window={frame_length}")
    print(f"Silence Filter      : Max RMS比 {silence_ratio * 100:.1f}% 未満を除外")
    print("=" * 72)

    if not os.path.exists(wav_dir):
        print(f"[Error] 入力ディレクトリが存在しません: {wav_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
    if not wav_files:
        print(f"[Error] WAVファイルが見つかりません: {wav_dir}")
        return False

    print(f"検出WAVファイル数: {len(wav_files)} 件\n")

    # FeatureExtractor の準備
    extractor = None
    if AudioFeatureExtractor is not None:
        try:
            extractor = AudioFeatureExtractor(sr=sr, shift_length=shift_length)
        except Exception:
            extractor = None

    # 1. 各ファイルのRMS抽出と統計計算
    records = []

    print("WAVファイル解析中 (RMS抽出 & 有声区間集計)...")
    for idx, f in enumerate(wav_files):
        f_path = os.path.join(wav_dir, f)
        meta = parse_filename_metadata(f)
        vowel = meta["vowel"]
        note_num = meta["note_num"]
        note_name = meta["note_name"]
        hz = meta["frequency_hz"]

        if note_num < min_note or note_num > max_note or vowel not in VOWEL_LIST:
            continue

        try:
            rms_series = extract_rms_from_wav(
                f_path,
                extractor,
                sr=sr,
                frame_length=frame_length,
                shift_length=shift_length,
            )

            # 有声区間判定 (無音フレーム除外)
            max_val = np.max(rms_series) if len(rms_series) > 0 else 0.0
            voiced_mask = rms_series >= (max_val * silence_ratio)
            voiced_rms = rms_series[voiced_mask]

            if len(voiced_rms) > 0:
                mean_rms = float(np.mean(voiced_rms))
                std_rms = float(np.std(voiced_rms))
                peak_rms = float(max_val)
                min_voiced_rms = float(np.min(voiced_rms))
            else:
                mean_rms = 0.0
                std_rms = 0.0
                peak_rms = 0.0
                min_voiced_rms = 0.0

            row = {
                "filename": f,
                "vowel": vowel,
                "note_num": note_num,
                "note_name": note_name,
                "frequency_hz": hz,
                "mean_rms": mean_rms,
                "peak_rms": peak_rms,
                "std_rms": std_rms,
                "min_voiced_rms": min_voiced_rms,
                "total_frames": len(rms_series),
                "voiced_frames": len(voiced_rms),
            }
            records.append(row)

            if (idx + 1) % 40 == 0 or (idx + 1) == len(wav_files):
                print(f"  処理進捗: {idx + 1}/{len(wav_files)} 件完了")

        except Exception as e:
            print(f"  [Error] 読み込み失敗: {f} -> {e}")

    df_records = pd.DataFrame(records)
    print(f"\n集計完了: 有効データ {len(df_records)} 件\n")

    if df_records.empty:
        print("[Error] 有効な解析レコードがありませんでした。")
        return False

    # 2. 母音別の集計 & 回帰分析 (傾き, C3基準倍率)
    vowel_stats = {}
    for v in VOWEL_LIST:
        df_v = df_records[df_records["vowel"] == v].sort_values("note_num")
        if df_v.empty:
            continue
        c3_val = (
            df_v[df_v["note_num"] == min_note]["mean_rms"].values[0]
            if (min_note in df_v["note_num"].values)
            else df_v["mean_rms"].iloc[0]
        )
        c6_val = (
            df_v[df_v["note_num"] == max_note]["mean_rms"].values[0]
            if (max_note in df_v["note_num"].values)
            else df_v["mean_rms"].iloc[-1]
        )
        gain_ratio = c6_val / c3_val if c3_val > 0 else 0.0

        # 線形回帰 (音高番号 vs RMS)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_v["note_num"], df_v["mean_rms"]
        )

        vowel_stats[v] = {
            "c3_rms": c3_val,
            "c6_rms": c6_val,
            "gain_ratio": gain_ratio,
            "slope": slope,
            "intercept": intercept,
            "r2": r_value**2,
            "mean_rms_all": df_v["mean_rms"].mean(),
            "max_rms_all": df_v["mean_rms"].max(),
            "min_rms_all": df_v["mean_rms"].min(),
        }

    # 各レコードに C3基準ゲイン比 を付与
    df_records["c3_gain_ratio"] = df_records.apply(
        lambda r: (
            r["mean_rms"] / vowel_stats[r["vowel"]]["c3_rms"]
            if vowel_stats.get(r["vowel"], {}).get("c3_rms", 0) > 0
            else 1.0
        ),
        axis=1,
    )

    # 3. グラフ生成
    if save_plots:
        print("グラフ画像生成中...")

        # =============================================================
        # 【シリーズ1】音高通しグラフ (Note 60=C3 〜 Note 96=C6, 計6枚)
        # =============================================================
        octave_notes = [60, 72, 84, 96]
        all_notes = sorted(df_records["note_num"].unique())
        x_labels = [
            note_num_to_name(n) if n % 3 == 0 or n in octave_notes else ""
            for n in all_notes
        ]

        # 1-0: 全体表示 (5母音重ね描き & 回帰直線)
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        summary_texts = []
        for v in VOWEL_LIST:
            df_v = df_records[df_records["vowel"] == v].sort_values("note_num")
            color = VOWEL_COLORS[v]
            st = vowel_stats[v]

            ax.plot(
                df_v["note_num"],
                df_v["mean_rms"],
                marker="o",
                markersize=4,
                linewidth=2,
                color=color,
                label=f"母音「{v}」 (C6/C3倍率: {st['gain_ratio']:.2f}倍)",
            )
            trend_y = st["slope"] * df_v["note_num"] + st["intercept"]
            ax.plot(
                df_v["note_num"],
                trend_y,
                linestyle="--",
                linewidth=1.2,
                color=color,
                alpha=0.55,
            )
            summary_texts.append(
                f"「{v}」: C3={st['c3_rms']:.4f} → C6={st['c6_rms']:.4f} ({st['gain_ratio']:.2f}倍, R²={st['r2']:.3f})"
            )

        ax.set_title(
            "全体表示: 音高（Note 60=C3 〜 Note 96=C6）に伴う実測RMSの変化特性 (同一DYN)",
            fontsize=15,
            pad=12,
            fontweight="bold",
        )
        ax.set_xlabel("MIDI ノート番号 (音名)", fontsize=12, labelpad=8)
        ax.set_ylabel("有声区間 平均 RMS (実効値)", fontsize=12, labelpad=8)
        ax.set_xticks(all_notes)
        ax.set_xticklabels(x_labels, rotation=45, fontsize=9)
        for onote in octave_notes:
            ax.axvline(onote, color="#7F8C8D", linestyle=":", alpha=0.7, linewidth=1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        info_box = "\n".join(summary_texts)
        ax.text(
            0.98,
            0.03,
            info_box,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#F8F9F9",
                edgecolor="#BDC3C7",
                alpha=0.92,
            ),
        )
        p1_0_path = os.path.join(output_dir, "01_00_pitch_continuous_all.png")
        plt.tight_layout()
        plt.savefig(p1_0_path)
        plt.close()
        print(f"  -> シリーズ1 全体表示保存: {p1_0_path}")

        # 1-1 〜 1-5: 各母音の音高通しグラフ
        for v_idx, v in enumerate(VOWEL_LIST):
            df_v = df_records[df_records["vowel"] == v].sort_values("note_num")
            color = VOWEL_COLORS[v]
            st = vowel_stats[v]
            romaji = VOWEL_ROMAJI[v]

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.plot(
                df_v["note_num"],
                df_v["mean_rms"],
                marker="o",
                markersize=5,
                linewidth=2.2,
                color=color,
                label=f"実測RMS (C6/C3倍率: {st['gain_ratio']:.2f}倍)",
            )
            trend_y = st["slope"] * df_v["note_num"] + st["intercept"]
            ax.plot(
                df_v["note_num"],
                trend_y,
                linestyle="--",
                linewidth=1.5,
                color="#7F8C8D",
                label=f"線形回帰トレンド (R²={st['r2']:.3f})",
            )

            ax.set_title(
                f"母音「{v}」: 音高（Note 60=C3 〜 Note 96=C6）に伴う実測RMSの変化特性",
                fontsize=14,
                pad=12,
                fontweight="bold",
            )
            ax.set_xlabel("MIDI ノート番号 (音名)", fontsize=11, labelpad=8)
            ax.set_ylabel("有声区間 平均 RMS (実効値)", fontsize=11, labelpad=8)
            ax.set_xticks(all_notes)
            ax.set_xticklabels(x_labels, rotation=45, fontsize=9)
            for onote in octave_notes:
                ax.axvline(
                    onote, color="#BDC3C7", linestyle=":", alpha=0.7, linewidth=1
                )
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

            info_text = f"C3 RMS: {st['c3_rms']:.4f}\nC6 RMS: {st['c6_rms']:.4f}\n拡大倍率: {st['gain_ratio']:.2f}倍\n傾き: +{st['slope']*12:.4f} /octave\n決定係数 R²: {st['r2']:.3f}"
            ax.text(
                0.98,
                0.05,
                info_text,
                transform=ax.transAxes,
                fontsize=9.5,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#F8F9F9",
                    edgecolor="#BDC3C7",
                    alpha=0.92,
                ),
            )

            p_v_path = os.path.join(
                output_dir, f"01_0{v_idx+1}_pitch_continuous_vowel_{romaji}.png"
            )
            plt.tight_layout()
            plt.savefig(p_v_path)
            plt.close()
            print(f"  -> シリーズ1 母音「{v}」保存: {p_v_path}")

        # =============================================================
        # 【シリーズ2】オクターブ比較グラフ (案A: 横軸12音名 C〜B, 計6枚)
        # =============================================================
        df_records_oct = df_records.copy()
        df_records_oct["semitone"] = (df_records_oct["note_num"] - 60) % 12
        df_records_oct["octave"] = (df_records_oct["note_num"] - 60) // 12 + 3

        oct_colors = {3: "#2980B9", 4: "#27AE60", 5: "#E67E22"}
        octave_range = [3, 4, 5]

        # 2-0: 全体表示 (全母音平均の Octave 3, 4, 5 比較)
        mean_by_oct_semi = (
            df_records_oct.groupby(["octave", "semitone"])["mean_rms"]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for oct_val in octave_range:
            sub = mean_by_oct_semi[mean_by_oct_semi["octave"] == oct_val].sort_values(
                "semitone"
            )
            ax.plot(
                sub["semitone"],
                sub["mean_rms"],
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=oct_colors[oct_val],
                label=f"Octave {oct_val} 全母音平均 (C{oct_val}〜B{oct_val})",
            )

        ax.set_xticks(range(12))
        ax.set_xticklabels(PITCH_NAMES, fontsize=11, fontweight="bold")
        ax.set_title(
            "全体表示: オクターブ別 RMS比較 (全母音平均・12音名同一軸重ね描き)",
            fontsize=14,
            pad=12,
            fontweight="bold",
        )
        ax.set_xlabel("音名 (Pitch Class)", fontsize=11, labelpad=8)
        ax.set_ylabel("有声区間 平均 RMS (実効値)", fontsize=11, labelpad=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.92)

        p2_0_path = os.path.join(output_dir, "02_00_octave_overlay_all.png")
        plt.tight_layout()
        plt.savefig(p2_0_path)
        plt.close()
        print(f"  -> シリーズ2 全体表示保存: {p2_0_path}")

        # 2-1 〜 2-5: 各母音のオクターブ比較グラフ
        for v_idx, v in enumerate(VOWEL_LIST):
            df_v = df_records_oct[df_records_oct["vowel"] == v]
            romaji = VOWEL_ROMAJI[v]

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            for oct_val in octave_range:
                sub = df_v[df_v["octave"] == oct_val].sort_values("semitone")
                ax.plot(
                    sub["semitone"],
                    sub["mean_rms"],
                    marker="o",
                    markersize=6,
                    linewidth=2.2,
                    color=oct_colors[oct_val],
                    label=f"Octave {oct_val} (C{oct_val}〜B{oct_val})",
                )

            ax.set_xticks(range(12))
            ax.set_xticklabels(PITCH_NAMES, fontsize=11, fontweight="bold")
            ax.set_title(
                f"母音「{v}」: オクターブ別 RMS比較 (12音名同一軸重ね描き)",
                fontsize=14,
                pad=12,
                fontweight="bold",
            )
            ax.set_xlabel("音名 (Pitch Class)", fontsize=11, labelpad=8)
            ax.set_ylabel("有声区間 平均 RMS (実効値)", fontsize=11, labelpad=8)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper left", fontsize=10, framealpha=0.92)

            p2_v_path = os.path.join(
                output_dir, f"02_0{v_idx+1}_octave_overlay_vowel_{romaji}.png"
            )
            plt.tight_layout()
            plt.savefig(p2_v_path)
            plt.close()
            print(f"  -> シリーズ2 母音「{v}」保存: {p2_v_path}")

    # 4. Excel レポート作成
    if save_excel:
        excel_path = os.path.join(output_dir, "pitch_rms_analysis_report.xlsx")
        print(f"\nExcelレポート生成中: {excel_path}...")

        wb = Workbook()
        ws_all = wb.active
        ws_all.title = "全185曲詳細データ"

        # (1) 全曲詳細シート
        headers_all = [
            "ファイル名",
            "母音",
            "ノート番号",
            "音名",
            "周波数 [Hz]",
            "有声平均RMS",
            "ピークRMS",
            "標準偏差RMS",
            "有声区間最小RMS",
            "有声フレーム数",
            "総フレーム数",
            "C3基準倍率",
        ]
        ws_all.append(headers_all)

        for _, r in df_records.sort_values(["vowel", "note_num"]).iterrows():
            ws_all.append(
                [
                    r["filename"],
                    r["vowel"],
                    r["note_num"],
                    r["note_name"],
                    r["frequency_hz"],
                    round(r["mean_rms"], 5),
                    round(r["peak_rms"], 5),
                    round(r["std_rms"], 5),
                    round(r["min_voiced_rms"], 5),
                    r["voiced_frames"],
                    r["total_frames"],
                    round(r["c3_gain_ratio"], 3),
                ]
            )

        # (2) 母音別統計サマリーシート
        ws_vowel = wb.create_sheet(title="母音別サマリー")
        headers_vowel = [
            "母音",
            "C3 RMS (最低音)",
            "C6 RMS (最高音)",
            "C6/C3 ゲイン倍率",
            "線形回帰 傾き",
            "線形回帰 切片",
            "決定係数 R²",
            "全音域 平均RMS",
            "全音域 最大RMS",
            "全音域 最小RMS",
        ]
        ws_vowel.append(headers_vowel)

        for v in VOWEL_LIST:
            st = vowel_stats[v]
            ws_vowel.append(
                [
                    v,
                    round(st["c3_rms"], 5),
                    round(st["c6_rms"], 5),
                    round(st["gain_ratio"], 3),
                    round(st["slope"], 6),
                    round(st["intercept"], 5),
                    round(st["r2"], 4),
                    round(st["mean_rms_all"], 5),
                    round(st["max_rms_all"], 5),
                    round(st["min_rms_all"], 5),
                ]
            )

        # (3) 音高別母音マトリクスシート
        ws_pitch = wb.create_sheet(title="音高別母音マトリクス")
        headers_pitch = (
            ["ノート番号", "音名", "周波数 [Hz]"]
            + [f"「{v}」RMS" for v in VOWEL_LIST]
            + ["音高別平均", "音高別最大", "音高別最小", "母音間格差比 (Max/Min)"]
        )
        ws_pitch.append(headers_pitch)

        pivot_m = df_records.pivot(
            index="note_num", columns="vowel", values="mean_rms"
        )[VOWEL_LIST]
        for note in sorted(pivot_m.index):
            row_vals = [pivot_m.loc[note, v] for v in VOWEL_LIST]
            row_mean = np.mean(row_vals)
            row_max = np.max(row_vals)
            row_min = np.min(row_vals)
            row_spread = row_max / row_min if row_min > 0 else 0.0

            ws_pitch.append(
                [note, note_num_to_name(note), round(note_num_to_hz(note), 2)]
                + [round(val, 5) for val in row_vals]
                + [
                    round(row_mean, 5),
                    round(row_max, 5),
                    round(row_min, 5),
                    round(row_spread, 3),
                ]
            )

        # スタイリング (テーブルヘッダー、列幅、カラーバー)
        header_fill = PatternFill(
            start_color="2C3E50", end_color="2C3E50", fill_type="solid"
        )
        header_font = Font(name="Meiryo", size=10, bold=True, color="FFFFFF")
        regular_font = Font(name="Meiryo", size=9)
        center_align = Alignment(horizontal="center", vertical="center")
        thin_border = Border(
            left=Side(style="thin", color="D5D8DC"),
            right=Side(style="thin", color="D5D8DC"),
            top=Side(style="thin", color="D5D8DC"),
            bottom=Side(style="thin", color="D5D8DC"),
        )

        for ws in [ws_all, ws_vowel, ws_pitch]:
            for col_idx, col in enumerate(
                ws.iter_cols(min_row=1, max_row=ws.max_row), 1
            ):
                col_letter = get_column_letter(col_idx)
                max_len = 0
                for cell_idx, cell in enumerate(col, 1):
                    cell.border = thin_border
                    if cell_idx == 1:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = center_align
                    else:
                        cell.font = regular_font
                        if isinstance(cell.value, (int, float)):
                            pass
                        else:
                            cell.alignment = center_align
                    val_str = str(cell.value or "")
                    max_len = max(
                        max_len, len(val_str.encode("shift-jis", errors="ignore"))
                    )
                ws.column_dimensions[col_letter].width = max(max_len + 3, 11)

        # マトリクスシートに条件付き書式 (ヒートカラー)
        color_rule = ColorScaleRule(
            start_type="min",
            start_color="FFFFFF",
            mid_type="percentile",
            mid_value=50,
            mid_color="FAD7A0",
            end_type="max",
            end_color="E74C3C",
        )
        ws_pitch.conditional_formatting.add(f"D2:H{ws_pitch.max_row}", color_rule)

        wb.save(excel_path)
        print(f"Excelレポート保存完了: {excel_path}")

    # 5. コンソールサマリー表示
    print("\n" + "=" * 72)
    print("  解析結果サマリー (音高 C3〜C6 に伴うRMS変化)")
    print("=" * 72)
    for v in VOWEL_LIST:
        st = vowel_stats[v]
        print(
            f"  - 母音「{v}」: C3={st['c3_rms']:.4f} → C6={st['c6_rms']:.4f} "
            f"| ゲイン拡大: {st['gain_ratio']:.2f} 倍 (傾き: +{st['slope']*12:.4f}/octave, R²={st['r2']:.3f})"
        )

    # 母音間格差
    min_gain_v = min(vowel_stats.items(), key=lambda x: x[1]["gain_ratio"])
    max_gain_v = max(vowel_stats.items(), key=lambda x: x[1]["gain_ratio"])
    print(
        f"\n  ★ ゲイン上昇率の最大格差: 「{min_gain_v[0]}」({min_gain_v[1]['gain_ratio']:.2f}倍) 〜 「{max_gain_v[0]}」({max_gain_v[1]['gain_ratio']:.2f}倍)"
    )
    print(
        f"  ★ 結論: 音高上昇に伴うRMSの増加率は母音によって最大 {max_gain_v[1]['gain_ratio'] / min_gain_v[1]['gain_ratio']:.2f} 倍も乖離しています。"
    )
    print("=" * 72)
    return True


if __name__ == "__main__":
    analyze_dataset(CONFIG)
