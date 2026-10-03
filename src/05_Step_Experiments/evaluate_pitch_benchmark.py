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
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    "model_dir": "./output/Step3_RMS_CNNBiLSTM/cnn_bilstm",

    # (必須) ベンチマークデータフォルダ (.npz 形式のデータセットフォルダ、または WAV/VSQX が格納されたフォルダ)
    "benchmark_dir": "./data/Step_org/Step3_60-96/benchmark",

    # (必須) pitch_randomize_log.csv のパス (Seen/Unseen判定用)
    "log_csv_path": "./data/Step_org/Step3_60-96/pitch_randomize_log.csv",

    # (必須) 評価結果の保存先ディレクトリ
    "output_dir": "./output/Step3_benchmark_eval",

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

    # スクリプトディレクトリ基準
    cand1 = os.path.normpath(os.path.join(SCRIPT_DIR, path_str))
    if os.path.exists(cand1):
        return os.path.abspath(cand1)

    # 親ディレクトリ（ワークスペースルート）基準
    root_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
    cand2 = os.path.normpath(os.path.join(root_dir, path_str.lstrip("./\\")))
    if os.path.exists(cand2):
        return os.path.abspath(cand2)

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
            features_meta = json.load(f)

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
    if model_type not in model_registry:
        raise ValueError(f"未知のモデル種別です: {model_type}")

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
        x_dims=x_dims,
        strategy=fusion_strategy,
        branch_modules=branch_modules,
    )
    core_input_dim = fusion_block.get_output_dim()

    core_model = model_registry[model_type](
        input_dim=core_input_dim, output_dim=output_dim, **model_params
    )
    unified_model = UnifiedModel(fusion_block, core_model).to(device)

    state_dict = torch.load(model_weights_path, map_location=device)
    unified_model.load_state_dict(state_dict)
    unified_model.eval()

    print(f"モデルロード完了: {model_type} (Fusion: {fusion_strategy})")
    print(f"  重みファイル: {model_weights_path}")

    return unified_model, train_config, features_meta


def load_log_csv(csv_path: str) -> dict[str, dict]:
    """
    pitch_randomize_log.csv を読み込み、ファイル名をキーにしたメタデータ辞書を構築する。
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] ログCSVが見つかりません: {csv_path} (Seen/Unseen判定は行われません)")
        return {}

    df_log = pd.read_csv(csv_path)
    metadata_map = {}

    for _, row in df_log.iterrows():
        out_f = str(row.get("output_file", ""))
        orig_f = str(row.get("original_file", ""))
        stem_out = Path(out_f).stem
        stem_orig = Path(orig_f).stem

        meta = {
            "split": str(row.get("split", "")),
            "vowel": str(row.get("vowel", "")),
            "dyn_pattern_id": str(row.get("dyn_pattern_id", "")),
            "assigned_note": int(row.get("assigned_note", -1)) if pd.notnull(row.get("assigned_note")) else -1,
            "assigned_note_name": str(row.get("assigned_note_name", "")),
            "is_seen": bool(row.get("is_seen", True)),
            "is_benchmark": bool(row.get("is_benchmark", False)),
        }

        # 複数のキーで引けるように登録
        if out_f:
            metadata_map[out_f] = meta
        if stem_out:
            metadata_map[stem_out] = meta
        if stem_orig:
            metadata_map[stem_orig] = meta

    print(f"ログCSV読み込み完了: {len(df_log)} 件のメタデータを登録")
    return metadata_map


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


def export_benchmark_excel(
    results: list[dict],
    save_excel_path: str,
):
    """
    ベンチマーク評価結果をオープンオフィス対応の高品質Excelレポートとして出力する。
    【Sheet 1: Summary】
      - 全体サマリー (Overall)
      - Seen vs Unseen 比較ブロック
      - 音素別 (あ〜お) 平均誤差テーブル
      - 音高別 (Note 60〜96) 平均誤差テーブル
    【Sheet 2: Details】
      - 全 185 曲の詳細データテーブル
    """
    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "Summary"
    ws_details = wb.create_sheet(title="Details")

    df = pd.DataFrame(results)

    # スタイル定義
    font_title = Font(name="Meiryo", size=14, bold=True, color="1F497D")
    font_section = Font(name="Meiryo", size=11, bold=True, color="1F497D")
    font_header = Font(name="Meiryo", size=10, bold=True, color="FFFFFF")
    font_bold = Font(name="Meiryo", size=10, bold=True)
    font_normal = Font(name="Meiryo", size=10)

    fill_header = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    fill_seen = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")  # 薄緑
    fill_unseen = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")  # 薄橙

    border_thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    align_center = Alignment(horizontal="center", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")

    # =============================================================
    # Sheet 1: Summary
    # =============================================================
    ws_summary.cell(row=1, column=1, value="Step 3 音高網羅ベンチマーク評価レポート").font = font_title

    # --- 1. 全体 & Seen vs Unseen 比較 ---
    ws_summary.cell(row=3, column=1, value="■ 汎化性能サマリー (Seen vs Unseen)").font = font_section

    headers_seen = ["区分 (Category)", "曲数 (Count)", "Mean Total Loss", "Mean MSE", "Mean MAE", "Mean Corr"]
    for col_idx, h in enumerate(headers_seen, 1):
        c = ws_summary.cell(row=4, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header
        c.alignment = align_center

    overall_row = [
        "全体 (Overall)",
        len(df),
        df["Total Loss"].mean(),
        df["MSE"].mean(),
        df["MAE"].mean(),
        df["Correlation"].mean(),
    ]

    seen_df = df[df["Is_Seen"] == True]
    unseen_df = df[df["Is_Seen"] == False]

    seen_row = [
        "Seen (既知音高×音素)",
        len(seen_df),
        seen_df["Total Loss"].mean() if len(seen_df) > 0 else 0,
        seen_df["MSE"].mean() if len(seen_df) > 0 else 0,
        seen_df["MAE"].mean() if len(seen_df) > 0 else 0,
        seen_df["Correlation"].mean() if len(seen_df) > 0 else 0,
    ]

    unseen_row = [
        "Unseen (未知音高×音素)",
        len(unseen_df),
        unseen_df["Total Loss"].mean() if len(unseen_df) > 0 else 0,
        unseen_df["MSE"].mean() if len(unseen_df) > 0 else 0,
        unseen_df["MAE"].mean() if len(unseen_df) > 0 else 0,
        unseen_df["Correlation"].mean() if len(unseen_df) > 0 else 0,
    ]

    summary_rows = [overall_row, seen_row, unseen_row]
    for r_idx, r_data in enumerate(summary_rows, 5):
        fill = None
        if "Seen (" in r_data[0]:
            fill = fill_seen
        elif "Unseen (" in r_data[0]:
            fill = fill_unseen

        for c_idx, val in enumerate(r_data, 1):
            cell = ws_summary.cell(row=r_idx, column=c_idx)
            if isinstance(val, float):
                cell.value = round(val, 4)
                cell.alignment = align_right
            else:
                cell.value = val
                cell.alignment = align_center if c_idx <= 2 else align_right

            cell.font = font_bold if r_idx == 5 else font_normal
            cell.border = border_thin
            if fill:
                cell.fill = fill

    # --- 2. 音素別サマリー ---
    ws_summary.cell(row=9, column=1, value="■ 音素別サマリー (Vowel Breakdown)").font = font_section
    headers_vowel = ["母音 (Vowel)", "曲数", "Mean Loss", "Mean MSE", "Mean MAE", "Mean Corr"]
    for col_idx, h in enumerate(headers_vowel, 1):
        c = ws_summary.cell(row=10, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header
        c.alignment = align_center

    vowel_group = df.groupby("Vowel")
    curr_row = 11
    for v_name, v_df in vowel_group:
        v_data = [
            v_name,
            len(v_df),
            v_df["Total Loss"].mean(),
            v_df["MSE"].mean(),
            v_df["MAE"].mean(),
            v_df["Correlation"].mean(),
        ]
        for c_idx, val in enumerate(v_data, 1):
            cell = ws_summary.cell(row=curr_row, column=c_idx)
            if isinstance(val, float):
                cell.value = round(val, 4)
                cell.alignment = align_right
            else:
                cell.value = val
                cell.alignment = align_center
            cell.font = font_normal
            cell.border = border_thin
        curr_row += 1

    # --- 3. 音高別サマリー (Note 60〜96) ---
    curr_row += 2
    ws_summary.cell(row=curr_row, column=1, value="■ 音高別サマリー (Pitch-by-Pitch Accuracy)").font = font_section
    curr_row += 1

    headers_pitch = [
        "Note Num",
        "Note Name",
        "曲数",
        "Seen数",
        "Unseen数",
        "Mean Loss",
        "Mean MSE",
        "Mean MAE",
        "Mean Corr",
    ]
    for col_idx, h in enumerate(headers_pitch, 1):
        c = ws_summary.cell(row=curr_row, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header
        c.alignment = align_center
    curr_row += 1

    pitch_group = df.groupby(["Note Num", "Note Name"])
    for (n_num, n_name), p_df in pitch_group:
        p_seen = len(p_df[p_df["Is_Seen"] == True])
        p_unseen = len(p_df[p_df["Is_Seen"] == False])
        p_data = [
            n_num,
            n_name,
            len(p_df),
            p_seen,
            p_unseen,
            p_df["Total Loss"].mean(),
            p_df["MSE"].mean(),
            p_df["MAE"].mean(),
            p_df["Correlation"].mean(),
        ]
        for c_idx, val in enumerate(p_data, 1):
            cell = ws_summary.cell(row=curr_row, column=c_idx)
            if isinstance(val, float):
                cell.value = round(val, 4)
                cell.alignment = align_right
            else:
                cell.value = val
                cell.alignment = align_center
            cell.font = font_normal
            cell.border = border_thin
        curr_row += 1

    # =============================================================
    # Sheet 2: Details
    # =============================================================
    detail_cols = [
        "Song Name",
        "Vowel",
        "Note Num",
        "Note Name",
        "Is_Seen",
        "Status",
        "Total Loss",
        "MSE",
        "MAE",
        "Correlation",
        "Valid Frames",
    ]

    for col_idx, h in enumerate(detail_cols, 1):
        c = ws_details.cell(row=1, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header
        c.alignment = align_center

    df_sorted = df.sort_values(by=["Note Num", "Vowel"]).reset_index(drop=True)
    for r_idx, (_, row) in enumerate(df_sorted.iterrows(), 2):
        is_s = row["Is_Seen"]
        status_text = "Seen" if is_s else "Unseen"
        row_vals = [
            row["Song Name"],
            row["Vowel"],
            row["Note Num"],
            row["Note Name"],
            is_s,
            status_text,
            round(row["Total Loss"], 4),
            round(row["MSE"], 4),
            round(row["MAE"], 4),
            round(row["Correlation"], 4),
            int(row["Valid Frames"]),
        ]
        fill = fill_seen if is_s else fill_unseen

        for c_idx, val in enumerate(row_vals, 1):
            cell = ws_details.cell(row=r_idx, column=c_idx, value=val)
            cell.font = font_normal
            cell.border = border_thin
            if c_idx in [5, 6]:
                cell.fill = fill
                cell.alignment = align_center
            elif isinstance(val, (int, np.integer)):
                cell.alignment = align_center
            elif isinstance(val, (float, np.floating)):
                cell.alignment = align_right
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    # 列幅の自動調整
    for ws in [ws_summary, ws_details]:
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = min(max_len + 4, 35)

    os.makedirs(os.path.dirname(save_excel_path), exist_ok=True)
    wb.save(save_excel_path)
    print(f"\nExcel評価レポートを保存しました: {save_excel_path}")


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
    log_meta_map = load_log_csv(log_csv_path)

    # 3. ベンチマークデータの収集 (.npz または .vsqx)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")

    npz_files = sorted(glob.glob(os.path.join(benchmark_dir, "**", "*.npz"), recursive=True))
    vsqx_files = sorted(glob.glob(os.path.join(benchmark_dir, "**", "*.vsqx"), recursive=True))

    results = []
    print("\nベンチマーク曲の推論と評価中...")

    # (A) .npz が存在する場合はデータセットから高速評価
    if npz_files:
        print(f"データセット形式 (.npz) で評価を実行します: {len(npz_files)} 件")
        for fpath in npz_files:
            song_name = Path(fpath).stem
            data = _load_npz_safe(fpath)
            if data is None:
                continue

            # 入力辞書と正解テンソルの構築
            x_dict = {
                k[2:]: torch.tensor(data[k], dtype=torch.float32).unsqueeze(0).to(device)
                for k in data.files
                if k.startswith("x_")
            }
            y_dict = {
                k[2:]: data[k]
                for k in data.files
                if k.startswith("y_")
            }

            v_mask = data["voicing_mask"] if "voicing_mask" in data.files else None

            # 推論
            with torch.no_grad():
                pred = model(x_dict)
                pred_np = pred.squeeze(0).cpu().numpy()

            # DYNの正解と予測を取り出し
            y_true_dyn = y_dict.get("dynamic", np.zeros_like(pred_np))
            if y_true_dyn.ndim == 1:
                y_true_dyn = y_true_dyn[:, None]
            if pred_np.ndim == 1:
                pred_np = pred_np[:, None]

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
            corr = float(np.corrcoef(y_true_eval, y_pred_eval)[0, 1]) if len(y_true_eval) > 1 else 0.0
            if np.isnan(corr):
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
            note_num = meta.get("assigned_note", -1)
            if not note_name:
                m_note = re.search(r"_([A-G]#?\d+)$", song_name)
                note_name = m_note.group(1) if m_note else "Unknown"

            is_seen = meta.get("is_seen", True)

            results.append({
                "Song Name": song_name,
                "Vowel": vowel,
                "Note Num": note_num,
                "Note Name": note_name,
                "Is_Seen": is_seen,
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

    # (B) .npz がなく .vsqx のみの場合は推論環境の案内
    elif vsqx_files:
        print(f"[Notice] ベンチマークフォルダ内に {len(vsqx_files)} 件の .vsqx ファイルが検出されました。")
        print("DNNモデルの推論には対応する歌声WAVが必要です。")
        print("VOCALOIDソフトでWAVレンダリングを行った後、build_dataset.py で .npz を生成してください。")
        return False
    else:
        print(f"[Error] ベンチマークフォルダ内にファイルが見つかりませんでした: {benchmark_dir}")
        return False

    # 4. Excelレポート出力
    excel_path = os.path.join(output_dir, "Step3_pitch_benchmark_eval.xlsx")
    export_benchmark_excel(results, excel_path)

    # 5. ターミナルサマリー表示
    df_res = pd.DataFrame(results)
    seen_mae = df_res[df_res["Is_Seen"] == True]["MAE"].mean()
    unseen_mae = df_res[df_res["Is_Seen"] == False]["MAE"].mean()

    print("\n" + "=" * 68)
    print("  音高網羅ベンチマーク評価完了サマリー")
    print("=" * 68)
    print(f"評価曲数: 全 {len(df_res)} 曲")
    print(f"全体平均 MAE : {df_res['MAE'].mean():.4f}")
    print(f"Seen (既知) 平均 MAE   : {seen_mae:.4f} (計 {len(df_res[df_res['Is_Seen'] == True])} 曲)")
    print(f"Unseen (未知) 平均 MAE : {unseen_mae:.4f} (計 {len(df_res[df_res['Is_Seen'] == False])} 曲)")
    if unseen_mae > 0 and seen_mae > 0:
        gap = unseen_mae - seen_mae
        pct = (gap / seen_mae) * 100
        print(f"未知パターン精度悪化率 : +{gap:.4f} (+{pct:.1f}%)")
    print("=" * 68)
    return True


if __name__ == "__main__":
    process_benchmark_evaluation(CONFIG)
