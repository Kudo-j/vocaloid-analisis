import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 定義モジュールのインポート ---
from dataset import VocalDataset, collate_fn
from model_decoder import TransformerModel

# ==========================================
# 設定セクション (train_diff.pyと合わせる)
# ==========================================
DATA_NAME = "DYN"
DATA_DIR = f"./processed/processed_{DATA_NAME}"  # features.jsonがある場所
EVAL_DATA_DIR = f"./processed/processed_{DATA_NAME}/Test"
MODEL_DIR = "./predict/models"
MODEL_PATH = os.path.join(MODEL_DIR, f"{DATA_NAME}_model.pth")
OUTPUT_EXCEL = f"./eval_check/{DATA_NAME}_lin .xlsx"
# モデルハイパーパラメータ
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.25
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
    if not os.path.exists(features_json_path):
        print(f"Error: {features_json_path} が見つかりません。")
        return

    feature_info = load_features_info(features_json_path)

    # train_diff.py のキー名に合わせる
    input_dim = feature_info["num_x_features"]
    output_dim = feature_info["num_y_features"]

    if output_dim == 0:
        print("Error: Yの特徴量数が0です。設定を確認してください。")
        return

    print(f"X features: {input_dim}, Y features: {output_dim}")

    # 2. モデル準備 (引数名を修正済み)
    model = TransformerModel(
        num_x_features=input_dim,
        num_y_features=output_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
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
    target_files = sorted(glob.glob(os.path.join(EVAL_DATA_DIR, "*.npz")))
    if not target_files:
        print("No .npz files found.")
        return

    print(f"Found {len(target_files)} files to evaluate.")

    param_names = get_param_names(feature_info)
    results = []

    # 4. 楽曲ごとのループ処理
    for file_path in tqdm(target_files, desc="Evaluating Songs"):
        file_name = os.path.basename(file_path)
        song_name = os.path.splitext(file_name)[0]

        # Dataset作成 (mode="eval")
        ds = VocalDataset(
            data_dir=EVAL_DATA_DIR,
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
                # 【修正】voicing_mask を受け取る
                x, y, voicing_mask, lengths = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                voicing_mask = voicing_mask.to(DEVICE)

                # Padding Mask
                max_len = x.size(1)
                padding_mask = torch.arange(max_len, device=DEVICE).expand(
                    len(lengths), max_len
                ) >= lengths.unsqueeze(1).to(DEVICE)

                # 有効フレームのベース（パディングでない部分）
                valid_mask = ~padding_mask

                # 【修正】Silence Mask を直接 voicing_mask に置き換え
                # voicing_maskは (Batch, Seq, 1) なので、squeeze(-1) で (Batch, Seq) にし、bool型にしてANDをとる
                is_voicing = voicing_mask.squeeze(-1) > 0.5
                valid_mask = valid_mask & is_voicing

                # Predict（自己回帰推論）
                seq_len = x.size(1)
                pred = model.inference(
                    x, max_len=seq_len, src_key_padding_mask=padding_mask
                )

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

        # ★ここを修正: 学習時と同じ定義で Total Loss を算出
        total_loss = avg_mse + avg_diff

        res_dict = {
            "Song Name": song_name,
            "Total Loss": total_loss,  # Value + Diff (学習時のLoss相当)
            "MSE Loss": avg_mse,  # 値のズレ (旧 Total MSE)
            "Diff Loss": avg_diff,  # 変化量のズレ
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

    # 列の並び順を整理 (Total Loss を先頭に)
    base_cols = ["Song Name", "Total Loss", "MSE Loss", "Diff Loss", "Valid Frames"]
    param_cols = [c for c in df.columns if c.startswith("MSE_")]

    # 実際に存在する列だけでフィルタリング
    final_cols = [c for c in base_cols + param_cols if c in df.columns]

    df = df[final_cols]

    # Total Loss (学習時の指標) が悪い順にソート
    df_sorted = df.sort_values(by="Total Loss", ascending=False)

    try:
        df_sorted.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\nEvaluation completed! Results saved to: {OUTPUT_EXCEL}")
        print("Top 5 worst songs by Total Loss:")
        print(df_sorted[["Song Name", "Total Loss", "MSE Loss", "Diff Loss"]].head(5))
    except Exception as e:
        print(f"Error saving Excel: {e}")
        df_sorted.to_csv(OUTPUT_EXCEL.replace(".xlsx", ".csv"), index=False)
        print("Saved as CSV instead.")


if __name__ == "__main__":
    main()
