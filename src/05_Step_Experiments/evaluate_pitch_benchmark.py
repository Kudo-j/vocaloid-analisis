"""
evaluate_pitch_benchmark.py
===========================
Step 3（音高多様化実験）専用: 音高網羅ベンチマーク評価スクリプト。

全音階（Note 60〜96, 37段階）× 5母音（あ・い・う・え・お）のベンチマークデータ
（計185曲）に対して学習済みモデルで推論を行い、
1. 各曲の正解 vs 予測DYN比較プロット（PNG）を出力
2. pitch_randomize_log.csv と突合して Seen（学習データに含まれる）/ Unseen（未学習）を自動判定
3. 音高別（Note 60〜96）、音素別、および Seen vs Unseen の詳細な誤差集計
4. train.py / evaluate.py 準拠のオープンオフィス対応Excelレポートを出力

使い方:
  1. 下記の CONFIG 領域を確認・編集する
  2. 実行ボタンを押す (python evaluate_pitch_benchmark.py)
"""

import os
import re
import sys
import json
import glob
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# 日本語フォントの設定（文字化け対策）
mpl.rcParams["font.family"] = ["Meiryo", "MS Gothic", "sans-serif"]

from openpyxl import Workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

# --- パス設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "util"))

from feature_extractor import AudioFeatureExtractor
from vsqx_dnn_formatter import VSQXDNNFormatter
from train_util.evaluate import Evaluator, denormalize_array
from train_util.inference import eval_song
from train_util.dataset import _load_npz_safe

# モデル
from models.cnn_bilstm import CNNBiLSTM
from models.bilstm_base import BiLSTMBase
from models.lstm_base import LSTMBase
from models.fusion_block import FeatureFusionBlock
from models.unified_model import UnifiedModel


# =============================================================================
# 設定領域 (CONFIG) - 実行ボタンを押すだけで動作可能
# =============================================================================
CONFIG = {
    # (必須) 評価対象の学習済みモデルフォルダ (best_model.pth, train_config.json が格納された場所)
    "model_dir": "./data/Step3/benchmark/model",

    # (必須) ベンチマークデータフォルダ (.npz 形式のデータセットフォルダ、または WAV/VSQX が格納されたフォルダ)
    "benchmark_dir": "./data/Step3/benchmark",

    # (必須) pitch_randomize_log.csv のパス (Seen/Unseen判定用)
    "log_csv_path": "./data/Step3/benchmark/pitch_randomize_log.csv",

    # (必須) 評価結果の保存先ディレクトリ
    "output_dir": "./data/Step3/Step3_benchmark_eval",

    # 各曲の比較波形プロット (PNG) を保存するかどうか
    "save_plots": True,

    # 使用デバイス ("cuda" または "cpu")
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # スライディングウィンドウ推論のバッチサイズ
    "batch_size": 32,
}
# =============================================================================


def _resolve_path(path_str: str, base_ref: str = None) -> str:
    """相対パスを安全に解決して絶対パスを返す。"""
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # 1. 26Voca_Ana プロジェクトルート基準（親フォルダまたはパス自体が存在する場合を最優先）
    proj_root = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    cand3 = os.path.normpath(os.path.join(proj_root, path_str.lstrip("./\\")))
    if os.path.exists(cand3) or os.path.exists(os.path.dirname(cand3)):
        return os.path.abspath(cand3)

    # 2. 親ディレクトリ（ワークスペースルート）基準
    root_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
    cand2 = os.path.normpath(os.path.join(root_dir, path_str.lstrip("./\\")))
    if os.path.exists(cand2):
        return os.path.abspath(cand2)

    # 3. スクリプトディレクトリ基準
    cand1 = os.path.normpath(os.path.join(SCRIPT_DIR, path_str))
    if os.path.exists(cand1):
        return os.path.abspath(cand1)

    return os.path.abspath(path_str)


def load_model_and_config(model_dir: str, device: torch.device):
    """学習済みモデルと設定情報をロードする。"""
    train_config_path = os.path.join(model_dir, "train_config.json")
    if not os.path.exists(train_config_path):
        train_config_path = os.path.join(model_dir, "..", "train_config.json")

    features_json_path = os.path.join(model_dir, "features.json")
    if not os.path.exists(features_json_path):
        features_json_path = os.path.join(model_dir, "..", "features.json")

    model_weights_path = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(model_dir):
        print(f"[Error] モデルディレクトリが存在しません: {model_dir}")
        print("  ※ CONFIG['model_dir'] に学習済みモデルフォルダ（best_model.pth がある場所）を指定してください。")
        return None, None, None

    if not os.path.exists(train_config_path):
        print(f"[Error] train_config.json が見つかりません: {model_dir}")
        return None, None, None
    if not os.path.exists(model_weights_path):
        print(f"[Error] best_model.pth が見つかりません: {model_dir}")
        return None, None, None

    with open(train_config_path, "r", encoding="utf-8") as f:
        train_config = json.load(f)

    features_meta = None
    if os.path.exists(features_json_path):
        with open(features_json_path, "r", encoding="utf-8") as f:
            raw_text = f.read().replace("NaN", "null")
            features_meta = json.loads(raw_text)

    model_type = train_config.get("MODEL_TYPE", "cnn_bilstm")
    fusion_strategy = train_config.get("FUSION_STRATEGY", "pass_through")
    branch_modules = train_config.get("BRANCH_MODULES", {})
    model_params = train_config.get("MODEL_PARAMS", {}).get(model_type, {})

    # モデルレジストリ
    model_registry = {
        "cnn_bilstm": CNNBiLSTM,
        "bilstm_base": BiLSTMBase,
        "lstm_base": LSTMBase,
    }
    base_model_type = model_type.split("_trial_")[0] if "_trial_" in model_type else model_type
    if base_model_type not in model_registry:
        raise ValueError(f"未知のモデル種別です: {base_model_type}")

    # 特徴量次元の決定
    x_dims = train_config.get("x_dims", {})
    y_dims = train_config.get("y_dims", {})
    if not x_dims and features_meta:
        x_dims = features_meta.get("x_dims", {})
    if not y_dims and features_meta:
        y_dims = features_meta.get("y_dims", {})

    output_dim = sum(y_dims.values()) if y_dims else 1

    # FeatureFusionBlock
    fusion_block = FeatureFusionBlock(
        x_dims_dict=x_dims,
        strategy=fusion_strategy,
        branch_modules=branch_modules,
    )
    core_input_dim = fusion_block.output_dim

    core_model = model_registry[base_model_type](
        input_dim=core_input_dim, output_dim=output_dim, **model_params
    )
    unified_model = UnifiedModel(fusion_block, core_model).to(device)

    state_dict = torch.load(model_weights_path, map_location=device)
    unified_model.load_state_dict(state_dict)
    unified_model.eval()

    print(f"モデルロード完了: {model_type} (Fusion: {fusion_strategy})")
    print(f"  重みファイル: {model_weights_path}")

    return unified_model, train_config, features_meta


def load_log_csv(csv_path: str) -> tuple[dict[str, dict], set[tuple[str, str]]]:
    """
    pitch_randomize_log.csv を読み込み、
    1. ファイル名をキーにしたメタデータ辞書
    2. CSVに登録されている (vowel, note_name) の既知ペア集合
    を返す。
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] ログCSVが見つかりません: {csv_path} (すべての曲がUnseenとして扱われます)")
        return {}, set()

    df_log = pd.read_csv(csv_path)
    metadata_map = {}
    seen_pairs = set()

    has_train = "split" in df_log.columns and (df_log["split"].astype(str).str.lower() == "train").any()

    for _, row in df_log.iterrows():
        out_f = str(row.get("output_file", "")).strip()
        orig_f = str(row.get("original_file", "")).strip()
        vowel = str(row.get("vowel", "")).strip()
        note_name = str(row.get("assigned_note_name", "")).strip().replace("s", "#")
        assigned_note = int(row.get("assigned_note", -1)) if pd.notnull(row.get("assigned_note")) else -1
        split_val = str(row.get("split", "")).strip().lower()

        # 学習データ (Train) に含まれる (vowel, note_name) のみを既知ペア集合に追加
        if has_train:
            if split_val == "train" and vowel and note_name:
                seen_pairs.add((vowel, note_name))
        else:
            raw_is_seen = row.get("is_seen")
            if (raw_is_seen is True or str(raw_is_seen).lower() == "true") and vowel and note_name:
                seen_pairs.add((vowel, note_name))

        stem_out = Path(out_f).stem if out_f else ""
        stem_orig = Path(orig_f).stem if orig_f else ""

        # is_seen カラムのパース
        raw_is_seen = row.get("is_seen")
        if pd.notnull(raw_is_seen):
            parsed_is_seen = (
                raw_is_seen
                if isinstance(raw_is_seen, bool)
                else str(raw_is_seen).strip().lower() in ["true", "1", "t", "yes"]
            )
        else:
            parsed_is_seen = None

        meta = {
            "split": str(row.get("split", "")),
            "vowel": vowel,
            "dyn_pattern_id": str(row.get("dyn_pattern_id", "")),
            "assigned_note": assigned_note,
            "assigned_note_name": note_name,
            "is_seen": parsed_is_seen,
            "is_benchmark": bool(row.get("is_benchmark", False)),
        }

        # 複数のキーで引けるように登録
        if out_f:
            metadata_map[out_f] = meta
        if stem_out:
            metadata_map[stem_out] = meta
        if stem_orig:
            metadata_map[stem_orig] = meta

    print(f"ログCSV読み込み完了: {len(df_log)} 件のメタデータを登録 (学習済み既知ペア: {len(seen_pairs)} 種)")
    return metadata_map, seen_pairs


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    song_name: str,
    note_name: str,
    note_num: int,
    vowel: str,
    is_seen: bool,
    mae: float,
    save_path: str,
):
    """
    1曲分の正解DYN vs 予測DYNの比較波形プロットを描画・保存する。
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    t_axis = np.arange(len(y_true))
    status_str = "Seen (既知)" if is_seen else "Unseen (未知)"

    ax.plot(t_axis, y_true, label="Ground Truth (正解DYN)", color="#1f77b4", linewidth=1.5, alpha=0.85)
    ax.plot(t_axis, y_pred, label="Predicted (予測DYN)", color="#ff7f0e", linewidth=1.5, linestyle="--", alpha=0.9)

    ax.set_title(
        f"{song_name} | 音素: '{vowel}' | 音高: {note_name} ({note_num}) | "
        f"状態: {status_str} | MAE: {mae:.2f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Time Frame (11.6ms / frame)", fontsize=10)
    ax.set_ylabel("DYN Value (0 - 127)", fontsize=10)
    ax.set_ylim(-5, 132)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    # 枠線に状態カラーを適用 (Unseenは赤枠で強調)
    if not is_seen:
        for spine in ax.spines.values():
            spine.set_edgecolor("#d62728")
            spine.set_linewidth(1.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_pitch_vs_mae(df: pd.DataFrame, save_path: str):
    """
    音高（縦軸） vs 精度 MAE（横軸）の相関プロットを描画・保存する。
    - Seen (緑) と Unseen (赤) を色分け
    - 母音ごとにマーカー形状を変更
    - 音高ごとの平均 MAE を折れ線で表示
    """
    if df.empty or "Note Num" not in df.columns or "MAE" not in df.columns:
        return

    unique_notes = sorted(df["Note Num"].unique())
    note_name_map = {}
    for _, r in df.iterrows():
        note_name_map[int(r["Note Num"])] = str(r["Note Name"])

    vowel_markers = {"あ": "o", "い": "^", "う": "s", "え": "D", "お": "p"}

    fig, ax = plt.subplots(figsize=(9, 12), dpi=130)

    # 散布図プロット (Seen: 緑, Unseen: 赤)
    for status, color, alpha in [("seen", "#2ca02c", 0.75), ("unseen", "#d62728", 0.85)]:
        sub_status = df[df["Status"] == status]
        for vowel, marker in vowel_markers.items():
            sub = sub_status[sub_status["Vowel"] == vowel]
            if not sub.empty:
                ax.scatter(
                    sub["MAE"],
                    sub["Note Num"],
                    color=color,
                    marker=marker,
                    s=55,
                    alpha=alpha,
                    edgecolors="black",
                    linewidths=0.6,
                )

    # 音高別平均 MAE のトレンドライン
    mean_by_note = df.groupby("Note Num")["MAE"].mean().sort_index()
    ax.plot(
        mean_by_note.values,
        mean_by_note.index,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.0,
        alpha=0.85,
        label="音高別平均 MAE",
    )

    ax.set_yticks(unique_notes)
    labels = [f"{note_name_map.get(n, '')} ({n})" for n in unique_notes]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("MAE (平均絶対誤差: DYN 0-127)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("音高 (Note Name & MIDI Number)", fontsize=10.5, fontweight="bold")
    ax.set_title("音高 (Pitch) vs 精度 (MAE) 相関プロット", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.55)

    # 状態と母音の凡例
    status_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor="k", markersize=8, label="Seen (既知)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markeredgecolor="k", markersize=8, label="Unseen (未知)"),
        Line2D([0], [0], color="#1f77b4", lw=2, label="音高別平均 MAE"),
    ]
    vowel_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label=f"母音: {v}")
        for v, m in vowel_markers.items()
    ]
    leg1 = ax.legend(handles=status_handles, loc="upper right", framealpha=0.9, title="状態")
    ax.add_artist(leg1)
    ax.legend(handles=vowel_handles, loc="lower right", framealpha=0.9, title="母音")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_seen_unseen_distribution(df: pd.DataFrame, save_path: str):
    """
    Seen vs Unseen の精度分布 (MAE 度数ヒストグラム) を描画・保存する。
    - 正規化なしの度数（曲数）表示
    - Seen (緑) と Unseen (赤) を重畳
    """
    if df.empty or "Status" not in df.columns or "MAE" not in df.columns:
        return

    seen_maes = df[df["Status"] == "seen"]["MAE"]
    unseen_maes = df[df["Status"] == "unseen"]["MAE"]
    bins = np.linspace(df["MAE"].min(), df["MAE"].max(), 25)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)

    if len(seen_maes) > 0:
        ax.hist(
            seen_maes,
            bins=bins,
            color="#2ca02c",
            alpha=0.6,
            edgecolor="black",
            linewidth=0.8,
            label=f"Seen (曲数: {len(seen_maes)}, 平均: {seen_maes.mean():.2f})",
        )
        ax.axvline(
            seen_maes.mean(),
            color="#1b631b",
            linestyle="--",
            linewidth=2,
            label=f"Seen 平均: {seen_maes.mean():.2f}",
        )

    if len(unseen_maes) > 0:
        ax.hist(
            unseen_maes,
            bins=bins,
            color="#d62728",
            alpha=0.6,
            edgecolor="black",
            linewidth=0.8,
            label=f"Unseen (曲数: {len(unseen_maes)}, 平均: {unseen_maes.mean():.2f})",
        )
        ax.axvline(
            unseen_maes.mean(),
            color="#8b1113",
            linestyle="--",
            linewidth=2,
            label=f"Unseen 平均: {unseen_maes.mean():.2f}",
        )

    ax.set_title("Seen vs Unseen 精度分布 (MAE 度数ヒストグラム)", fontsize=12, fontweight="bold")
    ax.set_xlabel("MAE (平均絶対誤差: DYN 0-127)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("曲数 (Frequency)", fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_pitch_vowel_heatmap(df: pd.DataFrame, save_path: str):
    """
    音高 × 母音 の 2次元 MAE 誤差ヒートマップを描画・保存する。
    - 縦軸: 低音から高音への昇順（C3が最下段、C6が最上段）
    - 横軸: 日本語5母音 (あ, い, う, え, お)
    - セル: 各組み合わせの平均 MAE 値を数値表示
    """
    if df.empty or "Note Num" not in df.columns or "Vowel" not in df.columns or "MAE" not in df.columns:
        return

    target_vowels = [v for v in ["あ", "い", "う", "え", "お"] if v in df["Vowel"].unique()]
    if not target_vowels:
        target_vowels = sorted(df["Vowel"].unique())

    pivot = df.pivot_table(
        index="Note Num", columns="Vowel", values="MAE", aggfunc="mean"
    )[target_vowels]
    pivot = pivot.sort_index(ascending=True)

    note_name_map = {}
    for _, r in df.iterrows():
        note_name_map[int(r["Note Num"])] = str(r["Note Name"])

    y_labels = [f"{note_name_map.get(n, '')} ({n})" for n in pivot.index]

    fig, ax = plt.subplots(figsize=(7, 12), dpi=130)
    mat = pivot.values
    # origin='lower' で低音(C3)を最下段、高音(C6)を最上段に配置
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", origin="lower")

    threshold = (np.nanmin(mat) + np.nanmax(mat)) / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if val > threshold else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.5,
                    fontweight="bold",
                )

    ax.set_xticks(range(len(target_vowels)))
    ax.set_xticklabels(target_vowels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.set_xlabel("母音 (Vowel)", fontsize=11, fontweight="bold")
    ax.set_ylabel("音高 (Note Name & MIDI Number)", fontsize=11, fontweight="bold")
    ax.set_title("音高 × 母音 MAE 誤差ヒートマップ", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE (平均絶対誤差)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def _write_evaluate_style_sheet(
    ws,
    df_subset: pd.DataFrame,
    table_display_name: str,
    final_cols: list[str],
    numeric_cols: list[str],
):
    """
    train_util/evaluate.py の記法・スタイル（サマリー行、ハイライト、TableStyleMedium9）
    に完全準拠して1つのワークシートを構築する。
    """
    if df_subset.empty:
        return

    # Total Loss が悪い順（降順）にソート
    df_sorted = df_subset.sort_values(by="Total Loss", ascending=False).reset_index(
        drop=True
    )

    # 各統計行の「データ名」を Total Loss 基準で決定
    max_song = df_sorted.loc[df_sorted["Total Loss"].idxmax(), "Song Name"]
    min_song = df_sorted.loc[df_sorted["Total Loss"].idxmin(), "Song Name"]
    median_val = df_sorted["Total Loss"].median()
    median_idx = (df_sorted["Total Loss"] - median_val).abs().idxmin()
    median_song = df_sorted.loc[median_idx, "Song Name"]

    # ==========================================================
    # サマリーテーブル (Rows 1-5)
    # ==========================================================
    summary_header = ["指標名", "データ名"] + numeric_cols
    summary_rows = [
        ["平均値 (Mean)", ""] + [df_sorted[c].mean() for c in numeric_cols],
        ["中央値 (Median)", median_song]
        + [df_sorted[c].median() for c in numeric_cols],
        ["最大値 (Max)", max_song] + [df_sorted[c].max() for c in numeric_cols],
        ["最小値 (Min)", min_song] + [df_sorted[c].min() for c in numeric_cols],
    ]

    # ヘッダー書き込み
    header_fill = PatternFill(
        start_color="4472C4", end_color="4472C4", fill_type="solid"
    )
    header_font = Font(name="Meiryo", bold=True, color="FFFFFF")
    for col_idx, val in enumerate(summary_header, 1):
        cell = ws.cell(row=1, column=col_idx, value=val)
        cell.fill = header_fill
        cell.font = header_font

    # 定義: 中央値・最大値・最小値用のソフトカラー背景
    fill_median = PatternFill(
        start_color="FFF2CC", end_color="FFF2CC", fill_type="solid"
    )  # 淡い黄
    fill_max = PatternFill(
        start_color="FCE4D6", end_color="FCE4D6", fill_type="solid"
    )  # 淡い赤
    fill_min = PatternFill(
        start_color="E2EFDA", end_color="E2EFDA", fill_type="solid"
    )  # 淡い緑

    # データ行書き込み＆各指標に応じた行の色付け
    summary_row_fills = [None, fill_median, fill_max, fill_min]
    bold_font = Font(name="Meiryo", bold=True)
    normal_font = Font(name="Meiryo")
    for row_offset, row_data in enumerate(summary_rows, 2):
        row_fill = summary_row_fills[row_offset - 2]
        for col_idx, val in enumerate(row_data, 1):
            cell = ws.cell(row=row_offset, column=col_idx, value=val)
            cell.font = normal_font
            if row_fill:
                cell.fill = row_fill
        # 指標名列を太字に
        ws.cell(row=row_offset, column=1).font = bold_font

    # サマリーブロックに罫線を適用
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    for row_idx in range(1, 6):
        for col_idx in range(1, len(summary_header) + 1):
            ws.cell(row=row_idx, column=col_idx).border = thin_border

    # ==========================================================
    # メインデータテーブル (Row 7+)
    # ==========================================================
    main_start_row = 7

    # ヘッダー書き込み
    for col_idx, col_name in enumerate(final_cols, 1):
        cell = ws.cell(row=main_start_row, column=col_idx, value=col_name)
        cell.font = bold_font

    # データ行書き込み（中央値・最大値・最小値の行のみハイライト）
    for row_offset, (_, row_data) in enumerate(df_sorted.iterrows(), 1):
        row_idx = main_start_row + row_offset
        song_name = row_data["Song Name"]

        # 対象の行に応じた色付け判定
        target_fill = None
        if song_name == max_song:
            target_fill = fill_max
        elif song_name == min_song:
            target_fill = fill_min
        elif song_name == median_song:
            target_fill = fill_median

        for col_idx, col_name in enumerate(final_cols, 1):
            val = row_data[col_name]
            cell = ws.cell(
                row=row_idx,
                column=col_idx,
                value=val,
            )
            cell.font = normal_font
            cell.border = thin_border
            if target_fill:
                cell.fill = target_fill

    # テーブル書式の適用
    num_data_rows = len(df_sorted)
    table_end_row = main_start_row + num_data_rows
    table_ref = (
        f"A{main_start_row}:{get_column_letter(len(final_cols))}{table_end_row}"
    )
    table = Table(displayName=table_display_name, ref=table_ref)
    style = TableStyleInfo(
        name="TableStyleMedium9",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    table.tableStyleInfo = style
    ws.add_table(table)

    # ==========================================================
    # 列幅の自動調整
    # ==========================================================
    total_cols = max(len(summary_header), len(final_cols))
    for col_idx in range(1, total_cols + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        for row_idx in range(1, table_end_row + 1):
            cell_val = ws.cell(row=row_idx, column=col_idx).value
            if cell_val is not None:
                if isinstance(cell_val, float):
                    cell_str = f"{cell_val:.6f}"
                else:
                    cell_str = str(cell_val)
                max_len = max(max_len, len(cell_str))
        ws.column_dimensions[col_letter].width = min(max_len + 3, 35)


def export_benchmark_excel(
    results: list[dict],
    save_excel_path: str,
):
    """
    train_util/evaluate.py の記法・スタイルに完全準拠したExcel評価レポートを出力する。
    - Sheet 1: Evaluation (全 185 曲の総合評価)
    - Sheet 2: Seen (既知 150 曲)
    - Sheet 3: Unseen (未知 35 曲)
    """
    df = pd.DataFrame(results)

    final_cols = [
        "Song Name",
        "Status",
        "Vowel",
        "Note Num",
        "Note Name",
        "Total Loss",
        "MSE",
        "MAE",
        "Correlation",
        "Valid Frames",
    ]
    numeric_cols = ["Total Loss", "MSE", "MAE", "Correlation", "Valid Frames"]

    os.makedirs(os.path.dirname(save_excel_path), exist_ok=True)
    wb = Workbook()

    # Sheet 1: Evaluation (全体)
    ws_eval = wb.active
    ws_eval.title = "Evaluation"
    _write_evaluate_style_sheet(
        ws_eval, df, "EvalResults", final_cols, numeric_cols
    )

    # Sheet 2: Seen (既知)
    df_seen = df[df["Status"] == "seen"].copy()
    if not df_seen.empty:
        ws_seen = wb.create_sheet(title="Seen")
        _write_evaluate_style_sheet(
            ws_seen, df_seen, "SeenResults", final_cols, numeric_cols
        )

    # Sheet 3: Unseen (未知)
    df_unseen = df[df["Status"] == "unseen"].copy()
    if not df_unseen.empty:
        ws_unseen = wb.create_sheet(title="Unseen")
        _write_evaluate_style_sheet(
            ws_unseen, df_unseen, "UnseenResults", final_cols, numeric_cols
        )

    wb.save(save_excel_path)
    print(f"\nExcel評価レポートを保存しました: {save_excel_path}")




AUDIO_FEATURE_NAMES = {
    "rms",
    "mel",
    "mfcc",
    "cqt",
    "stft",
    "f0_world",
    "f0_crepe",
    "source_rms",
    "source_mel",
    "source_stft",
    "hnr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "zero_crossing_rate",
    "cepstrum",
    "smoothed_cepstrum",
    "bark",
    "loudness",
    "mel_framenorm",
    "cqt_framenorm",
}


def setup_vsqx_and_audio_extractors(features_meta: dict):
    """
    predict.py に準拠し、AudioFeatureExtractor と VSQXDNNFormatter を初期化・設定する。
    """
    sr = features_meta.get("sr", 44100)
    shift_length = features_meta.get("shift_length", 512)
    x_feature_map = features_meta.get("x_feature_map", {})
    y_feature_map = features_meta.get("y_feature_map", {})

    audio_extractor = AudioFeatureExtractor(sr=sr, shift_length=shift_length)
    vsqx_extractor = VSQXDNNFormatter(sr=sr, shift_length=shift_length)

    input_vsqx_params_dict = {}
    for branch_name, feat_list in x_feature_map.items():
        feature_names = [f["name"] for f in feat_list]
        is_audio = any(ft in AUDIO_FEATURE_NAMES for ft in feature_names)
        if not is_audio:
            input_vsqx_params_dict[branch_name] = feature_names

    target_param_config = {}
    for branch_name, feat_list in y_feature_map.items():
        target_param_config[branch_name] = [f["name"] for f in feat_list]

    vsqx_extractor.vsqx_config["INPUT_VSQX_PARAMS"] = input_vsqx_params_dict
    vsqx_extractor.vsqx_config["TARGET_PARAM_CONFIG"] = target_param_config

    phoneme_map = features_meta.get("phoneme_map", {})
    if phoneme_map:
        vsqx_extractor.set_phoneme_map(phoneme_map)

    if "max_note_frames" in features_meta:
        vsqx_extractor.max_note_frames = int(features_meta["max_note_frames"])
    else:
        vsqx_extractor.max_note_frames = 100

    return audio_extractor, vsqx_extractor, input_vsqx_params_dict


def extract_song_data_from_wav_vsqx(
    wav_path: str,
    vsqx_path: str,
    features_meta: dict,
    audio_extractor: AudioFeatureExtractor,
    vsqx_extractor: VSQXDNNFormatter,
    input_vsqx_params_dict: dict,
) -> tuple[dict, dict, np.ndarray]:
    """
    predict.py (458-547行目) の処理フローに準拠し、WAVとVSQXから
    メモリ上で直接特徴量を抽出・正規化して (x_branches, y_dict, voicing_mask) を返す。
    """
    audio_global_mean = features_meta.get("audio_global_mean", {})
    audio_global_std = features_meta.get("audio_global_std", {})
    x_feature_map = features_meta.get("x_feature_map", {})
    y_feature_map = features_meta.get("y_feature_map", {})

    # A. 音響特徴量の抽出・正規化
    x_branches = {}
    total_frames = 0

    for branch_name, feat_list in x_feature_map.items():
        feature_names = [f["name"] for f in feat_list]
        is_audio = any(ft in AUDIO_FEATURE_NAMES for ft in feature_names)

        if is_audio:
            features = audio_extractor.get_features(wav_path, feature_names)
            total_frames = features.shape[0]

            # features.json の統計量で正規化 (build_dataset.py / predict.py と同一の処理)
            mean = np.array(audio_global_mean.get(branch_name, 0.0))
            std = np.array(audio_global_std.get(branch_name, 1.0))
            features = (features - mean) / (std + 1e-7)
            features = np.nan_to_num(features).astype(np.float32)

            x_branches[branch_name] = features

    # B. VSQXパラメータの抽出
    raw_vsqx = vsqx_extractor.parse_vsqx(vsqx_path, total_frames)
    if raw_vsqx is None:
        raise ValueError(f"VSQXのパースに失敗しました: {vsqx_path}")
    norm_vsqx = vsqx_extractor.normalize_features(raw_vsqx)

    # 発声マスク
    voicing_mask = norm_vsqx.get(
        "voicing_mask", np.ones((total_frames, 1), dtype=np.float32)
    )

    # VSQXの入力パラメータブランチを構築
    for branch_name, p_list in input_vsqx_params_dict.items():
        if not p_list:
            continue
        branch_data_list = []
        for pid in p_list:
            if pid in norm_vsqx:
                branch_data_list.append(norm_vsqx[pid])
            else:
                print(f"  [Warning] 入力パラメータ '{pid}' が見つかりません。")
        if branch_data_list:
            branch_arr = np.concatenate(branch_data_list, axis=-1).astype(np.float32)
            x_branches[branch_name] = branch_arr

    # C. 正解データの構築（NPZ互換ブランチ辞書形式）
    y_dict = {}
    for branch_name, feat_list in y_feature_map.items():
        branch_data_list = []
        for feat in feat_list:
            pid = feat["name"]
            dim = feat["dim"]
            gt = norm_vsqx.get(pid)
            if gt is not None:
                if gt.ndim == 1:
                    gt = gt[:, np.newaxis]
                branch_data_list.append(gt[:total_frames, :dim])
        if branch_data_list:
            y_dict[branch_name] = np.concatenate(
                branch_data_list, axis=-1
            ).astype(np.float32)

    return x_branches, y_dict, voicing_mask


def find_benchmark_pairs(benchmark_dir: str) -> list[tuple[str, str, str]]:
    """
    benchmark_dir 内を走査し、同名の .wav と .vsqx のペアを探索する。
    Returns:
        list of (song_name, wav_path, vsqx_path)
    """
    vsqx_files = sorted(glob.glob(os.path.join(benchmark_dir, "**", "*.vsqx"), recursive=True))
    wav_files = sorted(glob.glob(os.path.join(benchmark_dir, "**", "*.wav"), recursive=True))

    wav_map = {}
    for wp in wav_files:
        stem = Path(wp).stem
        wav_map[stem] = wp

    pairs = []
    for vp in vsqx_files:
        stem = Path(vp).stem
        # 1. 同じディレクトリの同名 .wav を優先探索
        same_dir_wav = os.path.splitext(vp)[0] + ".wav"
        if os.path.exists(same_dir_wav):
            pairs.append((stem, same_dir_wav, vp))
        # 2. benchmark_dir 全体から同名 stem の .wav を探索
        elif stem in wav_map:
            pairs.append((stem, wav_map[stem], vp))
        else:
            print(f"[Warning] 対応するWAVが見つかりません: {vp}")

    return pairs


def _evaluate_single_song(
    song_name: str,
    x_dict_np: dict,
    y_dict: dict,
    v_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    log_meta_map: dict,
    seen_pairs: set[tuple[str, str]],
    save_plots: bool,
    plots_dir: str,
    results: list,
    idx: int,
    total_count: int,
):
    """1曲分の推論、誤差集計、比較波形プロット描画を行う共通処理。"""
    # 入力辞書と正解テンソルの構築
    x_dict = {
        k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
        for k, v in x_dict_np.items()
    }

    # 推論
    with torch.no_grad():
        pred = model(x_dict)
        pred_np = pred.squeeze(0).cpu().numpy()

    # DYNの正解と予測を取り出し
    y_true_dyn = y_dict.get("dynamic", y_dict.get("all", np.zeros_like(pred_np)))
    if y_true_dyn.ndim == 1:
        y_true_dyn = y_true_dyn[:, None]
    if pred_np.ndim == 1:
        pred_np = pred_np[:, None]

    min_len = min(len(y_true_dyn), len(pred_np))
    y_true_dyn = y_true_dyn[:min_len]
    pred_np = pred_np[:min_len]
    if v_mask is not None:
        v_mask = v_mask[:min_len]

    # 逆正規化 (-1.0~1.0 -> 0~127)
    y_true_raw = denormalize_array(y_true_dyn[:, 0], "DYN")
    y_pred_raw = denormalize_array(pred_np[:, 0], "DYN")

    # 有声マスクの適用
    if v_mask is not None:
        voiced_idx = np.where(v_mask > 0.5)[0]
        if len(voiced_idx) > 0:
            y_true_eval = y_true_raw[voiced_idx]
            y_pred_eval = y_pred_raw[voiced_idx]
        else:
            y_true_eval, y_pred_eval = y_true_raw, y_pred_raw
    else:
        y_true_eval, y_pred_eval = y_true_raw, y_pred_raw

    diff = y_true_eval - y_pred_eval
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    if len(y_true_eval) > 1 and np.std(y_true_eval) > 1e-6 and np.std(y_pred_eval) > 1e-6:
        corr = float(np.corrcoef(y_true_eval, y_pred_eval)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    # メタ情報の取得
    meta = log_meta_map.get(song_name, log_meta_map.get(f"{song_name}.vsqx", {}))
    vowel = meta.get("vowel", "Other")
    if vowel == "Other":
        for v in ["あ", "い", "う", "え", "お"]:
            if v in song_name:
                vowel = v
                break

    note_name = meta.get("assigned_note_name", "")
    if not note_name:
        m_note = re.search(r"_([A-G](?:#|s)?\d+)$", song_name)
        if m_note:
            raw_n = m_note.group(1)
            note_name = raw_n.replace("s", "#")
        else:
            note_name = "Unknown"

    note_num = meta.get("assigned_note", -1)
    if note_num == -1 and note_name != "Unknown":
        SEMITONES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
        m_p = re.match(r"^([A-G]#?)(\d+)$", note_name)
        if m_p:
            pitch_cls, oct_str = m_p.group(1), m_p.group(2)
            if pitch_cls in SEMITONES:
                note_num = (int(oct_str) + 2) * 12 + SEMITONES[pitch_cls]

    # Seen / Unseen の判定:
    # 1. メタデータに is_seen が明示されている場合はそれを優先
    # 2. メタデータにない（CSV未登録ファイル）場合は、Trainの既知ペア (vowel, note_name) に含まれるかで判定
    if meta.get("is_seen") is not None:
        is_seen = bool(meta["is_seen"])
    else:
        is_seen = (vowel, note_name) in seen_pairs
    status = "seen" if is_seen else "unseen"

    results.append({
        "Song Name": song_name,
        "Status": status,
        "Vowel": vowel,
        "Note Num": note_num,
        "Note Name": note_name,
        "Total Loss": mse,
        "MSE": mse,
        "MAE": mae,
        "Correlation": corr,
        "Valid Frames": len(y_true_eval),
    })

    if save_plots:
        plot_path = os.path.join(plots_dir, f"{song_name}_compare.png")
        plot_prediction_comparison(
            y_true_raw, y_pred_raw, song_name, note_name, note_num, vowel, is_seen, mae, plot_path
        )

    if (idx + 1) % 10 == 0 or (idx + 1) == total_count:
        print(f"  [{idx+1:>3}/{total_count:>3}] {song_name:<32} ({status:<6}) MAE: {mae:5.2f}")


def process_benchmark_evaluation(config: dict):
    """メイン実行関数"""
    model_dir = _resolve_path(config["model_dir"])
    benchmark_dir = _resolve_path(config["benchmark_dir"])
    log_csv_path = _resolve_path(config["log_csv_path"])
    output_dir = _resolve_path(config["output_dir"])
    save_plots = config.get("save_plots", True)
    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.get("batch_size", 32)
    device = torch.device(device_str)

    print("=" * 68)
    print("  VOCALOID Step 3 音高網羅ベンチマーク評価システム")
    print("=" * 68)
    print(f"Model Directory     : {model_dir}")
    print(f"Benchmark Directory : {benchmark_dir}")
    print(f"Log CSV Path        : {log_csv_path}")
    print(f"Output Directory    : {output_dir}")
    print(f"Device              : {device}")
    print(f"Save Plots          : {save_plots}")
    print("=" * 68)

    # 1. モデルと設定のロード
    model, train_config, features_meta = load_model_and_config(model_dir, device)
    if model is None:
        return False

    # 2. ログCSVのロード
    log_meta_map, seen_pairs = load_log_csv(log_csv_path)

    # 3. ベンチマークデータの収集 (WAV+VSQXペア または .npz)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")

    pairs = find_benchmark_pairs(benchmark_dir)
    npz_files = sorted(glob.glob(os.path.join(benchmark_dir, "**", "*.npz"), recursive=True))

    results = []
    print("\nベンチマーク曲の推論と評価中...")

    if pairs:
        print(f"WAV & VSQX ペア形式で評価を実行します: {len(pairs)} 件")
        if features_meta is None:
            print("[Error] features.json が見つからないため、WAV+VSQXからの特徴量抽出ができません。")
            return False

        audio_extractor, vsqx_extractor, input_vsqx_params_dict = setup_vsqx_and_audio_extractors(features_meta)

        for idx, (song_name, wav_path, vsqx_path) in enumerate(pairs):
            try:
                x_dict_np, y_dict, v_mask = extract_song_data_from_wav_vsqx(
                    wav_path, vsqx_path, features_meta, audio_extractor, vsqx_extractor, input_vsqx_params_dict
                )
            except Exception as e:
                print(f"[Warning] 特徴量抽出失敗 ({song_name}): {e}")
                continue

            _evaluate_single_song(
                song_name=song_name,
                x_dict_np=x_dict_np,
                y_dict=y_dict,
                v_mask=v_mask,
                model=model,
                device=device,
                log_meta_map=log_meta_map,
                seen_pairs=seen_pairs,
                save_plots=save_plots,
                plots_dir=plots_dir,
                results=results,
                idx=idx,
                total_count=len(pairs),
            )

    elif npz_files:
        print(f"データセット形式 (.npz) で評価を実行します: {len(npz_files)} 件")
        for idx, fpath in enumerate(npz_files):
            song_name = Path(fpath).stem
            x_dict_np, y_dict, v_mask = _load_npz_safe(fpath)
            if x_dict_np is None:
                continue

            _evaluate_single_song(
                song_name=song_name,
                x_dict_np=x_dict_np,
                y_dict=y_dict,
                v_mask=v_mask,
                model=model,
                device=device,
                log_meta_map=log_meta_map,
                seen_pairs=seen_pairs,
                save_plots=save_plots,
                plots_dir=plots_dir,
                results=results,
                idx=idx,
                total_count=len(npz_files),
            )
    else:
        print(f"[Error] ベンチマークフォルダ内に評価可能なファイル (WAV+VSQXペア または .npz) が見つかりませんでした: {benchmark_dir}")
        return False

    # 4. Excelレポート出力
    excel_path = os.path.join(output_dir, "Step3_pitch_benchmark_eval.xlsx")
    export_benchmark_excel(results, excel_path)

    # 5. 分析グラフの出力
    df_res = pd.DataFrame(results)
    p_pitch_mae = os.path.join(output_dir, "benchmark_pitch_vs_mae.png")
    p_dist = os.path.join(output_dir, "benchmark_seen_unseen_mae_distribution.png")
    p_heatmap = os.path.join(output_dir, "benchmark_pitch_vowel_heatmap.png")

    if not df_res.empty:
        print("\n分析グラフを生成中...")
        plot_pitch_vs_mae(df_res, p_pitch_mae)
        plot_seen_unseen_distribution(df_res, p_dist)
        plot_pitch_vowel_heatmap(df_res, p_heatmap)
        print("  分析グラフ保存完了:")
        print(f"    1. 音高 vs 精度プロット   : {p_pitch_mae}")
        print(f"    2. 精度分布ヒストグラム   : {p_dist}")
        print(f"    3. 音高×母音ヒートマップ : {p_heatmap}")

    # 6. ターミナルサマリー表示
    seen_df = df_res[df_res["Status"] == "seen"]
    unseen_df = df_res[df_res["Status"] == "unseen"]

    seen_mae = seen_df["MAE"].mean() if len(seen_df) > 0 else 0.0
    unseen_mae = unseen_df["MAE"].mean() if len(unseen_df) > 0 else 0.0

    print("\n" + "=" * 68)
    print("  音高網羅ベンチマーク評価完了サマリー")
    print("=" * 68)
    print(f"評価曲数: 全 {len(df_res)} 曲")
    print(f"全体平均 MAE : {df_res['MAE'].mean():.4f}")
    print(f"Seen (既知) 平均 MAE   : {seen_mae:.4f} (計 {len(seen_df)} 曲)")
    print(f"Unseen (未知) 平均 MAE : {unseen_mae:.4f} (計 {len(unseen_df)} 曲)")
    if unseen_mae > 0 and seen_mae > 0:
        gap = unseen_mae - seen_mae
        pct = (gap / seen_mae) * 100
        print(f"未知パターン精度悪化率 : +{gap:.4f} (+{pct:.1f}%)")
    print("=" * 68)
    return True


if __name__ == "__main__":
    process_benchmark_evaluation(CONFIG)
