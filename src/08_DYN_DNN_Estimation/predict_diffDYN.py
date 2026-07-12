import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'sans-serif']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "05_DYN_Estimation"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "07_DYN_Direct_Estimation"))

import config_diffDYN as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader
from model import DisentangleNet
from predict_direct import write_dyn_to_vsqx, interpolate_note_gaps

def plot_results(time_axis, mel, dyn_curve, gt_dyn_curve, notes, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    mel_mean = np.mean(mel, axis=1)
    axes[0].plot(time_axis, mel_mean, label="Mel Power", alpha=0.7)
    axes[0].set_title("Input: Mel Spectrogram Power")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 無声区間のグレー背景
    unvoiced_regions = []
    last_sec = 0.0
    for note in notes:
        if note["start_sec"] > last_sec:
            unvoiced_regions.append((last_sec, note["start_sec"]))
        last_sec = note["end_sec"]
    if len(time_axis) > 0 and time_axis[-1] > last_sec:
        unvoiced_regions.append((last_sec, time_axis[-1]))

    for start_sec, end_sec in unvoiced_regions:
        for ax in axes:
            ax.axvspan(start_sec, end_sec, color="gray", alpha=0.15)

    for note in notes:
        mid = (note["start_sec"] + note["end_sec"]) / 2
        axes[1].text(mid, 130, note.get("lyric", ""), ha='center', va='top', fontsize=8, color='darkgreen')

    axes[1].plot(time_axis, dyn_curve, label="推定DYN (DNN)", color="tab:blue")
    if gt_dyn_curve is not None:
        min_len = min(len(time_axis), len(gt_dyn_curve))
        axes[1].plot(time_axis[:min_len], gt_dyn_curve[:min_len], label="正解DYN", color="tab:red", alpha=0.7, linestyle="--")
    axes[1].axhline(y=64, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_ylim(-5, 132)
    axes[1].set_title("Estimated DYN Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if gt_dyn_curve is not None:
        error = dyn_curve[:min_len] - gt_dyn_curve[:min_len]
        axes[2].plot(time_axis[:min_len], error, color="tab:orange", alpha=0.7)
        
        voiced_errors = []
        for note in notes:
            s_idx = librosa.time_to_frames(note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
            e_idx = librosa.time_to_frames(note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
            s_idx = max(0, min(s_idx, min_len))
            e_idx = max(0, min(e_idx, min_len))
            if s_idx < e_idx:
                voiced_errors.extend(error[s_idx:e_idx])
                
        valid_mae = np.mean(np.abs(voiced_errors)) if voiced_errors else 0.0
        axes[2].set_title(f"Estimation Error (Voiced MAE: {valid_mae:.2f})")
    axes[2].axhline(y=0, color="gray", linestyle=":")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def smooth_dyn(dyn_curve):
    if len(dyn_curve) < 31: return dyn_curve
    return np.clip(savgol_filter(dyn_curve, 31, 3), 0, 127)

def main():
    print("=" * 60)
    print(f"  DYN DNN Estimation ({cfg.ENCODER_TYPE.upper()}) - Prediction")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(cfg.MODEL_PATH):
        print("[エラー] モデルが見つかりません。")
        return

    model = DisentangleNet(cfg).to(device)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    wav_path = cfg.TARGET_WAV
    vsqx_path = cfg.TARGET_VSQX
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH, use_part_offset=True)
    
    # scaler.json の読み込み
    scaler_path = os.path.join(cfg.PREPROCESSED_DIR, "scaler.json")
    if not os.path.exists(scaler_path):
        print("[エラー] scaler.json が見つかりません。preprocess.py を先に実行してください。")
        return
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    global_mean = np.array(scaler["mean"], dtype=np.float32)
    global_std = np.array(scaler["std"], dtype=np.float32)
    
    mel = extractor.get_features(wav_path, ["mel"])
    total_frames = len(mel)
    
    # Zスコア標準化（学習時と同じスケール）
    mel = ((mel - global_mean) / (global_std + 1e-8)).astype(np.float32)
    
    vsqx_data = reader.read(vsqx_path)
    notes = vsqx_data.get("notes", []) if vsqx_data else []
    
    gt_dyn_points = vsqx_data["dynamic_params"].get("DYN", []) if vsqx_data else []
    gt_dyn_curve = reader.interpolate_param_to_frames(gt_dyn_points, total_frames, cfg.DYN_DEFAULT) if gt_dyn_points else None

    print("\n[Step 1] DNNによる推論...")
    with torch.no_grad():
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(device) # (1, Seq, 80)
        _, dyn_pred_tensor = model(mel_tensor)
        dyn_raw = dyn_pred_tensor.squeeze().cpu().numpy()
    
    # DYN逆正規化: [-1, 1] -> [0, 127]
    dyn_raw = (dyn_raw + 1.0) / 2.0 * 127.0

    # ノートガード（ノート外の無声部分は64にリセット）
    voicing_mask = np.zeros(total_frames, dtype=bool)
    for note in notes:
        s_idx = librosa.time_to_frames(note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
        e_idx = librosa.time_to_frames(note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
        s_idx = max(0, min(s_idx, total_frames))
        e_idx = max(0, min(e_idx, total_frames))
        if s_idx < e_idx:
            voicing_mask[s_idx:e_idx] = True
            
    dyn_raw[~voicing_mask] = cfg.DYN_DEFAULT
    dyn_raw = np.clip(dyn_raw, 0, 127)

    print("[Step 2] 補間と平滑化...")
    dyn_interp = interpolate_note_gaps(dyn_raw, notes, total_frames)
    dyn_smooth = smooth_dyn(dyn_interp)

    print("\n[Step 3] 出力...")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    time_axis = librosa.frames_to_time(np.arange(total_frames), sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    
    img_path = os.path.join(cfg.OUTPUT_DIR, f"{base_name}_{cfg.ENCODER_TYPE}_dyn.png")
    plot_results(time_axis, mel, dyn_smooth, gt_dyn_curve, notes, img_path)
    
    out_vsqx_path = os.path.join(cfg.OUTPUT_DIR, f"{base_name}_{cfg.ENCODER_TYPE}_dyn.vsqx")
    write_dyn_to_vsqx(vsqx_path, out_vsqx_path, dyn_smooth, cfg.SR, cfg.HOP_LENGTH)
    
    print(f"  完了！ 画像: {img_path}")

if __name__ == "__main__":
    main()
