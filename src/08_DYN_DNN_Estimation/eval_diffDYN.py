import os
import glob
import torch
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- 日本語フォント対応 (文字化け防止) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Meiryo', 'MS Gothic', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
# ------------------------------------
from torch.utils.data import DataLoader
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from tqdm import tqdm

import config_diffDYN as cfg
from dataset import DisentangleDataset, collate_fn
from model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_masked_mse(pred, target, mask):
    """
    Masked MSEを計算する
    pred, target: (B, T, D)
    mask: (B, T, 1) or (B, T) True=有効, False=無効(無音)
    """
    squared_diff = ((pred - target) ** 2) * mask
    valid_elements = mask.sum() * target.shape[-1]
    if valid_elements == 0:
        return 0.0
    return (squared_diff.sum() / valid_elements).item()

def calculate_masked_diff_loss(pred, target, mask):
    """
    微分Lossを計算する
    """
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]
    mask_diff = mask[:, 1:, :] * mask[:, :-1, :]
    
    movement_weight = torch.abs(target_diff) * 25.0
    squared_diff = ((pred_diff - target_diff) ** 2) * mask_diff * movement_weight
    
    valid_elements = mask_diff.sum()
    if valid_elements == 0:
        return 0.0
    return (squared_diff.sum() / valid_elements).item()

def _shade_unvoiced(ax, time_axis, voicing_mask_1d, label_added=False):
    in_unvoiced = False
    start_t = 0.0
    added = label_added
    for i, v in enumerate(voicing_mask_1d):
        if v < 0.5 and not in_unvoiced:
            in_unvoiced = True
            start_t = time_axis[i]
        elif v >= 0.5 and in_unvoiced:
            in_unvoiced = False
            lbl = "Unvoiced" if not added else None
            ax.axvspan(start_t, time_axis[i], alpha=0.18, color="#9E9E9E", label=lbl)
            added = True
    if in_unvoiced:
        lbl = "Unvoiced" if not added else None
        ax.axvspan(start_t, time_axis[-1], alpha=0.18, color="#9E9E9E", label=lbl)

def denormalize_dyn(val_array):
    """ DYN (-1.0 ~ 1.0) を 0~127 の整数値に逆正規化・丸め """
    if np.max(val_array) > 2.0 or np.min(val_array) < -2.0:
        # すでに 0~127 のスケールになっている場合はそのまま返す
        return np.clip(np.round(val_array), 0, 127)
    val = (val_array + 1.0) / 2.0 * 127.0
    return np.clip(np.round(val), 0, 127)

def predict_full_song(model, song_name, data_dir):
    npz_path = os.path.join(data_dir, f"{song_name}.npz")
    if not os.path.exists(npz_path):
        return None, None, None, None, None
        
    with np.load(npz_path) as data:
        # configで指定されたすべての入力特徴量を抽出
        features_t = []
        features_ut = []
        for feat in cfg.INPUT_FEATURES:
            feat_key_t = f"{feat}_tuned" if f"{feat}_tuned" in data else feat
            feat_key_ut = f"{feat}_untuned" if f"{feat}_untuned" in data else feat
            if feat_key_t in data:
                features_t.append(data[feat_key_t])
            if feat_key_ut in data:
                features_ut.append(data[feat_key_ut])
                
        # 結合
        x_t = np.concatenate(features_t, axis=-1)
        x_ut = np.concatenate(features_ut, axis=-1) if features_ut else None
        
        dyns = data["dyn"]
        masks = data["voicing_mask"]
        
    total_frames = len(x_t)
    X_tensor = torch.from_numpy(x_t).unsqueeze(0).to(DEVICE)
    
    pred_ut_chunks = []
    pred_dyn_chunks = []
    
    is_disentangle = cfg.TRAINING_MODE in ["disentangle_step1", "disentangle_step2"]
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
            for start in range(0, total_frames, cfg.CHUNK_SIZE):
                end = min(start + cfg.CHUNK_SIZE, total_frames)
                x_chunk = X_tensor[:, start:end, :]
                
                if is_disentangle:
                    p_ut, p_dyn = model(x_chunk)
                    pred_ut_chunks.append(p_ut.cpu().numpy())
                else:
                    p_dyn = model(x_chunk)
                    # BaselineモデルはUTを予測しないため、ダミー（0埋め）を返す
                    pred_ut_chunks.append(np.zeros_like(x_chunk.cpu().numpy()))
                    
                pred_dyn_chunks.append(p_dyn.cpu().numpy())
                
    pred_ut = np.concatenate(pred_ut_chunks, axis=1).squeeze(0)
    pred_dyn = np.concatenate(pred_dyn_chunks, axis=1).squeeze(0)
    
    return pred_dyn, dyns, pred_ut, x_ut, masks

def plot_eval_results(song_name, pred_dyn, gt_dyn, pred_mel, gt_mel, mask, rank_label, total_loss, save_path, sr=44100, hop_length=256):
    pred_dyn = denormalize_dyn(pred_dyn.flatten())
    gt_dyn = denormalize_dyn(gt_dyn.flatten())
    mask = mask.flatten()
    
    total_frames = len(gt_dyn)
    time_axis = librosa.frames_to_time(np.arange(total_frames), sr=sr, hop_length=hop_length)
    
    voiced_idx = mask >= 0.5
    mae = np.mean(np.abs(pred_dyn[voiced_idx] - gt_dyn[voiced_idx])) if np.any(voiced_idx) else 0.0
    
    window_frames = int(5.0 * sr / hop_length)
    w = min(window_frames, total_frames)
    
    # Max Movement (GT)
    gt_diff = np.abs(np.diff(gt_dyn, prepend=gt_dyn[0]))
    cum_diff = np.cumsum(gt_diff)
    window_sums = cum_diff[w:] - cum_diff[:-w]
    max_move_start = int(np.argmax(window_sums)) if len(window_sums) > 0 else 0
    max_move_end = max_move_start + w
    
    # Max Error
    abs_error = np.abs(pred_dyn - gt_dyn) * mask
    cum_err = np.cumsum(abs_error)
    err_sums = cum_err[w:] - cum_err[:-w]
    max_err_start = int(np.argmax(err_sums)) if len(err_sums) > 0 else 0
    max_err_end = max_err_start + w
    
    fig, axes = plt.subplots(5, 1, figsize=(18, 22), gridspec_kw={'height_ratios': [1.5, 1, 1, 1.2, 1.2]})
    fig.suptitle(f"[{rank_label}] {song_name}  |  DYN MAE = {mae:.2f}  |  Total Loss = {total_loss:.4f}", fontsize=18, fontweight="bold", y=0.98)
    
    # --- 1. DYN Full ---
    ax1 = axes[0]
    _shade_unvoiced(ax1, time_axis, mask)
    ax1.plot(time_axis, gt_dyn, label="Ground Truth", color="#2196F3", alpha=0.7, linewidth=1.0)
    ax1.plot(time_axis, pred_dyn, label="Predicted", color="#FF5722", alpha=0.7, linewidth=1.0)
    ax1.axvspan(time_axis[max_move_start], time_axis[min(max_move_end, total_frames - 1)], alpha=0.12, color="green", label="Max Movement (5s)")
    ax1.axvspan(time_axis[max_err_start], time_axis[min(max_err_end, total_frames - 1)], alpha=0.12, color="red", label="Max Error (5s)")
    ax1.set_title("DYN: Full Comparison", fontsize=13)
    ax1.set_ylabel("DYN")
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # --- 2. Max Movement ---
    ax2 = axes[1]
    t_slice2 = time_axis[max_move_start:max_move_end]
    gt_slice2 = gt_dyn[max_move_start:max_move_end]
    pred_slice2 = pred_dyn[max_move_start:max_move_end]
    mask_slice2 = mask[max_move_start:max_move_end]
    v2 = mask_slice2 >= 0.5
    mae2 = np.mean(np.abs(pred_slice2[v2] - gt_slice2[v2])) if np.any(v2) else 0.0
    
    _shade_unvoiced(ax2, t_slice2, mask_slice2)
    ax2.plot(t_slice2, gt_slice2, label="Ground Truth", color="#2196F3", alpha=0.8, linewidth=1.5)
    ax2.plot(t_slice2, pred_slice2, label="Predicted", color="#FF5722", alpha=0.8, linewidth=1.5)
    ax2.fill_between(t_slice2, gt_slice2, pred_slice2, alpha=0.15, color="orange")
    ax2.set_title(f"DYN: Max Ground Truth Movement (5s)  |  Local MAE = {mae2:.2f}", fontsize=13)
    ax2.set_ylabel("DYN")
    ax2.set_xlim(t_slice2[0], t_slice2[-1])
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    # --- 3. Max Error ---
    ax3 = axes[2]
    t_slice3 = time_axis[max_err_start:max_err_end]
    gt_slice3 = gt_dyn[max_err_start:max_err_end]
    pred_slice3 = pred_dyn[max_err_start:max_err_end]
    mask_slice3 = mask[max_err_start:max_err_end]
    v3 = mask_slice3 >= 0.5
    mae3 = np.mean(np.abs(pred_slice3[v3] - gt_slice3[v3])) if np.any(v3) else 0.0
    
    _shade_unvoiced(ax3, t_slice3, mask_slice3)
    ax3.plot(t_slice3, gt_slice3, label="Ground Truth", color="#2196F3", alpha=0.8, linewidth=1.5)
    ax3.plot(t_slice3, pred_slice3, label="Predicted", color="#FF5722", alpha=0.8, linewidth=1.5)
    ax3.fill_between(t_slice3, gt_slice3, pred_slice3, alpha=0.15, color="red")
    ax3.set_title(f"DYN: Max Prediction Error (5s)  |  Local MAE = {mae3:.2f}", fontsize=13)
    ax3.set_ylabel("DYN")
    ax3.set_xlim(t_slice3[0], t_slice3[-1])
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.4)
    
    # --- 4. Untuned Feature Ground Truth ---
    ax4 = axes[3]
    if gt_mel is not None and gt_mel.shape[-1] > 1: # 複数次元特徴量 (Mel等)
        vmin = min(gt_mel.min(), pred_mel.min())
        vmax = max(gt_mel.max(), pred_mel.max())
        librosa.display.specshow(gt_mel.T, sr=sr, hop_length=hop_length, x_axis='time', ax=ax4, cmap='magma', vmin=vmin, vmax=vmax)
        ax4.set_ylabel("Bins")
    else: # 1次元特徴量 (RMS等)
        if gt_mel is not None:
            ax4.plot(time_axis, gt_mel, label="GT Feature", color="#2196F3")
            ax4.legend(loc="upper right", fontsize=9)
        ax4.set_ylabel("Feature")
    ax4.set_title("Untuned Feature: Ground Truth", fontsize=13)
    ax4.set_xlabel("")
    ax4.set_xlim(0, time_axis[-1])
    
    # --- 5. Untuned Feature Predicted ---
    ax5 = axes[4]
    is_disentangle = cfg.TRAINING_MODE in ["disentangle_step1", "disentangle_step2"]
    if is_disentangle and pred_mel is not None and pred_mel.shape[-1] > 1:
        librosa.display.specshow(pred_mel.T, sr=sr, hop_length=hop_length, x_axis='time', ax=ax5, cmap='magma', vmin=vmin, vmax=vmax)
        ax5.set_ylabel("Bins")
    elif is_disentangle:
        if pred_mel is not None:
            ax5.plot(time_axis, pred_mel, label="Predicted Feature", color="#FF5722")
            ax5.legend(loc="upper right", fontsize=9)
        ax5.set_ylabel("Feature")
    else:
        # Baselineの場合UT予測はないので空にする
        ax5.text(0.5, 0.5, "Baseline Model (No UT Prediction)", ha='center', va='center', transform=ax5.transAxes, color='gray')
        ax5.set_ylabel("")
        
    ax5.set_title("Untuned Feature: Predicted", fontsize=13)
    ax5.set_xlabel("Time (s)")
    ax5.set_xlim(0, time_axis[-1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def main():
    print(f"--- Starting Evaluation using {DEVICE} ---")

    # 1. モデル準備
    model = get_model(cfg).to(DEVICE)
    if not os.path.exists(cfg.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {cfg.MODEL_PATH}")

    state_dict = torch.load(cfg.MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # 2. 評価対象ファイルのリスト取得
    eval_data_dir = os.path.join(cfg.PREPROCESSED_DIR, "Test")
    target_files = sorted(glob.glob(os.path.join(eval_data_dir, "*.npz")))
    
    if not target_files:
        print(f"No .npz files found in {eval_data_dir}.")
        return

    print(f"Found {len(target_files)} files to evaluate.")
    results = []

    # 出力先ディレクトリの作成
    os.makedirs(os.path.dirname(cfg.EVAL_OUTPUT_EXCEL), exist_ok=True)

    # 3. 楽曲ごとのループ処理
    for file_path in tqdm(target_files, desc="Evaluating Songs"):
        file_name = os.path.basename(file_path)
        song_name = os.path.splitext(file_name)[0]

        # Dataset作成 (mode="eval")
        # evalモードの場合、ファイル名のリストを渡すだけでその曲の全チャンクを取得
        ds = DisentangleDataset(
            data_dir=eval_data_dir,
            file_list=[song_name],
            mode="eval",
        )

        if len(ds) == 0:
            continue

        dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        total_ut_accum = 0.0
        total_dyn_accum = 0.0
        total_diff_accum = 0.0
        total_dyn_mae_raw_accum = 0.0

        total_frames = 0
        total_diff_frames = 0

        with torch.no_grad():
            for batch in dl:
                features_t_dict, features_ut_dict, dyns, masks, _ = batch
                
                # 特徴量を結合してTensor化
                x_t = torch.cat([features_t_dict[f] for f in cfg.INPUT_FEATURES], dim=-1).to(DEVICE)
                if len(features_ut_dict) > 0:
                    x_ut = torch.cat([features_ut_dict[f] for f in cfg.INPUT_FEATURES], dim=-1).to(DEVICE)
                else:
                    x_ut = None
                    
                dyns = dyns.to(DEVICE)
                masks = masks.to(DEVICE)

                # エンコーダモデルでの一括推論
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                    is_disentangle = cfg.TRAINING_MODE in ["disentangle_step1", "disentangle_step2"]
                    if is_disentangle:
                        pred_ut, pred_dyn = model(x_t)
                    else:
                        pred_dyn = model(x_t)
                        pred_ut = None

                valid_count = masks.sum().item()

                if valid_count > 0:
                    if is_disentangle and x_ut is not None:
                        batch_ut_mse = calculate_masked_mse(pred_ut, x_ut, masks)
                    else:
                        batch_ut_mse = 0.0
                        
                    batch_dyn_mse = calculate_masked_mse(pred_dyn, dyns, masks)
                    
                    # --- Raw MAE for DYN (逆正規化後の整数スケールでの絶対誤差) ---
                    # もしDYNが既に 0〜127 のスケールだった場合は変換式を通さない
                    if dyns.max() > 2.0 or dyns.min() < -2.0:
                        pred_dyn_raw = torch.round(pred_dyn).clamp(0, 127)
                        dyns_raw = torch.round(dyns).clamp(0, 127)
                    else:
                        pred_dyn_raw = torch.round((pred_dyn + 1.0) / 2.0 * 127.0).clamp(0, 127)
                        dyns_raw = torch.round((dyns + 1.0) / 2.0 * 127.0).clamp(0, 127)
                        
                    # ブロードキャストの暴走を防ぐため、1次元に揃えてから計算する
                    abs_err = torch.abs(pred_dyn_raw.squeeze(-1) - dyns_raw.squeeze(-1)) * masks.squeeze(-1)
                    batch_dyn_mae_raw = (abs_err.sum() / valid_count).item()
                    
                    total_ut_accum += batch_ut_mse * valid_count
                    total_dyn_accum += batch_dyn_mse * valid_count
                    total_dyn_mae_raw_accum += batch_dyn_mae_raw * valid_count
                    total_frames += valid_count

                # Diff Loss
                if x_t.size(1) > 1:
                    mask_diff = masks[:, 1:, :] * masks[:, :-1, :]
                    diff_valid_count = mask_diff.sum().item()

                    if diff_valid_count > 0:
                        batch_diff_loss = calculate_masked_diff_loss(pred_dyn, dyns, masks)
                        total_diff_accum += batch_diff_loss * diff_valid_count
                        total_diff_frames += diff_valid_count

        # --- 集計 ---
        if total_frames > 0:
            avg_ut = total_ut_accum / total_frames
            avg_dyn = total_dyn_accum / total_frames
            avg_dyn_mae_raw = total_dyn_mae_raw_accum / total_frames
        else:
            avg_ut = 0.0
            avg_dyn = 0.0
            avg_dyn_mae_raw = 0.0

        if total_diff_frames > 0:
            avg_diff = total_diff_accum / total_diff_frames
        else:
            avg_diff = 0.0

        total_loss = (cfg.LOSS_UNTUNED_WEIGHT * avg_ut) + \
                     (cfg.LOSS_DYN_WEIGHT * avg_dyn) + \
                     (cfg.LOSS_DIFF_WEIGHT * avg_diff)

        res_dict = {
            "Song Name": song_name,
            "Total Loss": total_loss,
            "MSE_UT": avg_ut,
            "MSE_DYN": avg_dyn,
            "Diff_DYN": avg_diff,
            "MAE_DYN (Raw)": avg_dyn_mae_raw,
            "Valid Frames": total_frames,
        }

        results.append(res_dict)

    # 4. Excel保存
    if not results:
        print("\n[Warning] 結果が得られませんでした。")
        return

    df = pd.DataFrame(results)

    final_cols = ["Song Name", "Total Loss", "MSE_UT", "MSE_DYN", "Diff_DYN", "MAE_DYN (Raw)", "Valid Frames"]
    df = df[final_cols]

    # Total Loss が良い順(昇順)にソート
    df_sorted = df.sort_values(by="Total Loss", ascending=True).reset_index(drop=True)

    # --- 5曲のプロット生成 ---
    n_songs = len(df_sorted)
    if n_songs > 0:
        target_indices = {
            "Best": 0,
            "Q1": n_songs // 4,
            "Median": n_songs // 2,
            "Q3": (n_songs * 3) // 4,
            "Worst": n_songs - 1
        }
        
        # 重複するインデックスを削除 (曲数が少ない場合)
        unique_indices = {}
        for rank, idx in target_indices.items():
            if idx < n_songs and idx not in unique_indices.values():
                unique_indices[rank] = idx
                
        print(f"\n--- Generating Plots for {len(unique_indices)} Representative Songs ---")
        for rank_label, idx in unique_indices.items():
            row = df_sorted.iloc[idx]
            song_name = row["Song Name"]
            total_loss = row["Total Loss"]
            
            pred_dyn, gt_dyn, pred_mel, gt_mel, mask = predict_full_song(model, song_name, eval_data_dir)
            if pred_dyn is not None:
                save_path = os.path.join(cfg.OUTPUT_DIR, "plots", f"{rank_label}_loss{total_loss:.2f}_{song_name}.png")
                plot_eval_results(song_name, pred_dyn, gt_dyn, pred_mel, gt_mel, mask, rank_label, total_loss, save_path)
                print(f"  Saved plot: {save_path}")

    # --- 平均行の作成 ---
    numeric_cols = [c for c in final_cols if c != "Song Name"]
    avg_row = {"Song Name": "AVERAGE"}
    for col in numeric_cols:
        avg_row[col] = df_sorted[col].mean()
    df_avg = pd.concat([df_sorted, pd.DataFrame([avg_row])], ignore_index=True)

    try:
        # まず pandas でExcelに書き出す
        df_avg.to_excel(cfg.EVAL_OUTPUT_EXCEL, index=False)

        # --- openpyxl でテーブル機能と太字装飾を適用 ---
        wb = load_workbook(cfg.EVAL_OUTPUT_EXCEL)
        ws = wb.active

        num_cols = len(final_cols)
        num_data_rows = len(df_sorted)

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
        avg_row_idx = num_data_rows + 2
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

        wb.save(cfg.EVAL_OUTPUT_EXCEL)

        print(f"\nEvaluation completed! Results saved to: {cfg.EVAL_OUTPUT_EXCEL}")
        print("Top 5 best songs by Total Loss:")
        print(df_sorted[["Song Name", "Total Loss", "MSE_UT", "MSE_DYN", "Diff_DYN", "MAE_DYN (Raw)"]].head(5))
        print("\nTop 5 worst songs by Total Loss:")
        print(df_sorted[["Song Name", "Total Loss", "MSE_UT", "MSE_DYN", "Diff_DYN", "MAE_DYN (Raw)"]].tail(5))
    except Exception as e:
        print(f"Error saving Excel: {e}")
        df_avg.to_csv(cfg.EVAL_OUTPUT_EXCEL.replace(".xlsx", ".csv"), index=False)
        print("Saved as CSV instead.")

if __name__ == "__main__":
    main()
