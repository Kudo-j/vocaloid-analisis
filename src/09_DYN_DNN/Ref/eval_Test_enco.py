import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
from torch.utils.data import DataLoader
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from tqdm import tqdm

# --- 定義モジュールのインポート ---
from dataset import VocalDataset, collate_fn
from model_encoder import TransformerModel

# ==========================================
# 設定セクション (train_encoder.pyと合わせる)
# ==========================================
DATA_NAME = "DYN"
DATA_DIR = f"./processed/processed_{DATA_NAME}"  # features.jsonがある場所
EVAL_DATA_DIR = f"./processed/processed_{DATA_NAME}/Test"
MODEL_DIR = "./models"  # train_encoder.pyの保存先に合わせる
MODEL_PATH = os.path.join(MODEL_DIR, f"{DATA_NAME}_model.pth")
OUTPUT_EXCEL = f"./eval_check/{DATA_NAME}_encoder_eval.xlsx"

# モデルハイパーパラメータ (train_encoder.py準拠)
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_features_info(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_param_names(feature_info):
    """Excelの列名用に、出力パラメータの名前リストを作成する"""
    names = []
    for feat in feature_info["y_feature_map"]:
        dim = feat["dim"]
        name = feat["name"]
        if dim == 1:
            names.append(name)
        else:
            for i in range(dim):
                names.append(f"{name}_{i}")
    return names


def calculate_masked_loss(pred, target, mask):
    """
    Masked MSEを計算する
    pred, target: (B, T, D)
    mask: (B, T) True=有効, False=無効(パディング/無音)
    """
    mask_expanded = mask.unsqueeze(-1).float()

    squared_diff = (pred - target) ** 2
    masked_squared_diff = squared_diff * mask_expanded

    valid_elements = mask_expanded.sum()

    if valid_elements == 0:
        return 0.0, torch.zeros(pred.shape[-1], device=pred.device)

    # 全体MSE
    total_mse = masked_squared_diff.sum() / valid_elements

    # パラメータ別MSE (次元ごとの平均)
    dim_wise_mse = masked_squared_diff.sum(dim=(0, 1)) / mask_expanded.sum(
        dim=(0, 1)
    ).clamp(min=1)

    return total_mse, dim_wise_mse


def main():
    print(f"--- Starting Evaluation using {DEVICE} ---")

    # 1. features.json から次元数を読み込む
    features_json_path = os.path.join(DATA_DIR, "features.json")
    # DATA_DIRが見つからない場合のフォールバック (train_encoderとパスが違う可能性があるため)
    if not os.path.exists(features_json_path):
        alt_features_path = f"./processed_{DATA_NAME}/features.json"
        if os.path.exists(alt_features_path):
            features_json_path = alt_features_path
            print(f"Using alternate features.json path: {features_json_path}")
        else:
            print(f"Error: {features_json_path} が見つかりません。")
            return

    feature_info = load_features_info(features_json_path)

    input_dim = feature_info["num_x_features"]
    output_dim = feature_info["num_y_features"]

    if output_dim == 0:
        print("Error: Yの特徴量数が0です。設定を確認してください。")
        return

    print(f"X features: {input_dim}, Y features: {output_dim}")

    # 2. モデル準備 (エンコーダ専用)
    model = TransformerModel(
        num_x_features=input_dim,
        num_y_features=output_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # 3. 評価対象ファイルのリスト取得
    eval_data_dir = EVAL_DATA_DIR
    if not os.path.exists(eval_data_dir):
        # フォールバック
        alt_eval_dir = f"./processed_{DATA_NAME}/Test"
        if os.path.exists(alt_eval_dir):
            eval_data_dir = alt_eval_dir
            print(f"Using alternate test data path: {eval_data_dir}")

    target_files = sorted(glob.glob(os.path.join(eval_data_dir, "*.npz")))
    if not target_files:
        print(f"No .npz files found in {eval_data_dir}.")
        return

    print(f"Found {len(target_files)} files to evaluate.")

    param_names = get_param_names(feature_info)
    results = []

    # 出力先ディレクトリの作成
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)

    # 4. 楽曲ごとのループ処理
    for file_path in tqdm(target_files, desc="Evaluating Songs"):
        file_name = os.path.basename(file_path)
        song_name = os.path.splitext(file_name)[0]

        # Dataset作成 (mode="eval")
        ds = VocalDataset(
            data_dir=eval_data_dir,
            file_list=[song_name],
            feature_info=feature_info,
            mode="eval",
        )

        # データセットが空（ファイルが見つからない等）の場合はスキップ
        if len(ds) == 0:
            continue

        dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        total_mse_accum = 0.0
        total_diff_accum = 0.0
        param_mse_accum = torch.zeros(output_dim, device=DEVICE)

        total_frames = 0
        total_diff_frames = 0

        with torch.no_grad():
            for batch in dl:
                x, y, voicing_mask, lengths = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                voicing_mask = voicing_mask.to(DEVICE)

                # Padding Mask (True: パディング部分)
                max_len = x.size(1)
                padding_mask = torch.arange(max_len, device=DEVICE).expand(
                    len(lengths), max_len
                ) >= lengths.unsqueeze(1).to(DEVICE)

                # 有効フレームのベース（パディングでない部分）
                valid_mask = ~padding_mask

                # Silence Mask を voicing_mask に置き換え
                is_voicing = voicing_mask.squeeze(-1) > 0.5
                valid_mask = valid_mask & is_voicing

                # エンコーダモデルでの一括推論
                # padding_maskをsrc_key_padding_maskとして渡す
                pred = model(x, src_key_padding_mask=padding_mask)

                # MSE
                batch_mse, batch_param_mse = calculate_masked_loss(pred, y, valid_mask)
                valid_count = valid_mask.sum().item()

                if valid_count > 0:
                    total_mse_accum += batch_mse.item() * valid_count
                    param_mse_accum += batch_param_mse * valid_count
                    total_frames += valid_count

                # Diff Loss
                if max_len > 1:
                    target_diff = y[:, 1:, :] - y[:, :-1, :]
                    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
                    diff_mask = valid_mask[:, 1:] & valid_mask[:, :-1]

                    batch_diff_loss, _ = calculate_masked_loss(
                        pred_diff, target_diff, diff_mask
                    )
                    diff_valid_count = diff_mask.sum().item()

                    if diff_valid_count > 0:
                        total_diff_accum += batch_diff_loss.item() * diff_valid_count
                        total_diff_frames += diff_valid_count

        # --- 集計 ---
        if total_frames > 0:
            avg_mse = total_mse_accum / total_frames
            avg_param_mse = (param_mse_accum / total_frames).cpu().numpy()
        else:
            avg_mse = 0.0
            avg_param_mse = np.zeros(output_dim)

        if total_diff_frames > 0:
            avg_diff = total_diff_accum / total_diff_frames
        else:
            avg_diff = 0.0

        # Total Loss (学習時のLoss相当)
        total_loss = avg_mse + avg_diff

        res_dict = {
            "Song Name": song_name,
            "Total Loss": total_loss,
            "MSE Loss": avg_mse,
            "Diff Loss": avg_diff,
            "Valid Frames": total_frames,
        }

        for i, pname in enumerate(param_names):
            res_dict[f"MSE_{pname}"] = avg_param_mse[i]

        results.append(res_dict)

    # 5. Excel保存
    if not results:
        print("\n[Warning] 結果が得られませんでした。")
        return

    df = pd.DataFrame(results)

    # 列の並び順を整理
    base_cols = ["Song Name", "Total Loss", "MSE Loss", "Diff Loss", "Valid Frames"]
    param_cols = [c for c in df.columns if c.startswith("MSE_")]

    final_cols = [c for c in base_cols + param_cols if c in df.columns]
    df = df[final_cols]

    # Total Loss が悪い順にソート (初期表示用)
    df_sorted = df.sort_values(by="Total Loss", ascending=False).reset_index(drop=True)

    # --- 平均行の作成 ---
    numeric_cols = [c for c in final_cols if c != "Song Name"]
    avg_row = {"Song Name": "AVERAGE"}
    for col in numeric_cols:
        avg_row[col] = df_sorted[col].mean()
    df_avg = pd.concat([df_sorted, pd.DataFrame([avg_row])], ignore_index=True)

    try:
        # まず pandas でExcelに書き出す
        df_avg.to_excel(OUTPUT_EXCEL, index=False)

        # --- openpyxl でテーブル機能と太字装飾を適用 ---
        wb = load_workbook(OUTPUT_EXCEL)
        ws = wb.active

        num_cols = len(final_cols)
        num_data_rows = len(df_sorted)  # 平均行を含まないデータ行数

        # テーブル範囲: ヘッダー(1行目) + データ行 (平均行は含めない)
        table_ref = f"A1:{get_column_letter(num_cols)}{num_data_rows + 1}"
        table = Table(displayName="EvalResults", ref=table_ref)
        style = TableStyleInfo(
            name="TableStyleMedium9",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        table.tableStyleInfo = style
        ws.add_table(table)

        # 平均行 (テーブル外の最下行) を太字にする
        avg_row_idx = num_data_rows + 2  # ヘッダー(1) + データ行 + 1
        bold_font = Font(bold=True)
        for col_idx in range(1, num_cols + 1):
            cell = ws.cell(row=avg_row_idx, column=col_idx)
            cell.font = bold_font

        # 列幅を自動調整 (簡易)
        for col_idx in range(1, num_cols + 1):
            col_letter = get_column_letter(col_idx)
            max_len = len(str(ws.cell(row=1, column=col_idx).value or ""))
            for row_idx in range(2, avg_row_idx + 1):
                cell_val = ws.cell(row=row_idx, column=col_idx).value
                if cell_val is not None:
                    max_len = max(max_len, len(str(cell_val)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 30)

        wb.save(OUTPUT_EXCEL)

        print(f"\nEvaluation completed! Results saved to: {OUTPUT_EXCEL}")
        print("Top 5 worst songs by Total Loss:")
        print(df_sorted[["Song Name", "Total Loss", "MSE Loss", "Diff Loss"]].head(5))
    except Exception as e:
        print(f"Error saving Excel: {e}")
        df_avg.to_csv(OUTPUT_EXCEL.replace(".xlsx", ".csv"), index=False)
        print("Saved as CSV instead.")


if __name__ == "__main__":
    main()

