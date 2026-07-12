import os
import sys
import numpy as np
import pandas as pd
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "05_DYN_Estimation"))

import config_direct as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader
from pair_finder import find_pairs
from predict_direct import estimate_dyn_direct, interpolate_note_gaps, smooth_dyn

def main():
    print("=" * 60)
    print("  DYN Direct Estimation (Ridge) - Evaluation")
    print("=" * 60)

    if not os.path.exists(cfg.MODEL_PATH):
        print("[エラー] モデルが見つかりません。")
        return

    model_data = joblib.load(cfg.MODEL_PATH)
    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    pairs = find_pairs(cfg.EVAL_WAV_DIR, cfg.EVAL_VSQX_DIR)
    if not pairs:
        print("[エラー] 評価用データが見つかりません。")
        return

    results = []
    
    for i, (base_name, wav_path, vsqx_path) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {base_name} 評価中...")
        mel = extractor.get_features(wav_path, ["mel"])
        total_frames = len(mel)
        vsqx_data = reader.read(vsqx_path)
        notes = vsqx_data.get("notes", [])
        
        gt_points = vsqx_data["dynamic_params"].get("DYN", [])
        gt_curve = reader.interpolate_param_to_frames(gt_points, total_frames, cfg.DYN_DEFAULT)

        dyn_raw = estimate_dyn_direct(mel, model_data)
        dyn_interp = interpolate_note_gaps(dyn_raw, notes, total_frames)
        dyn_smooth = smooth_dyn(dyn_interp)

        min_len = min(len(gt_curve), len(dyn_smooth))
        gt = gt_curve[:min_len]
        pred = dyn_smooth[:min_len]

        # 有声区間（ノート内）のみのインデックスを取得
        voiced_indices = []
        for note in notes:
            import librosa
            s_idx = librosa.time_to_frames(note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
            e_idx = librosa.time_to_frames(note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
            s_idx = max(0, min(s_idx, min_len))
            e_idx = max(0, min(e_idx, min_len))
            if s_idx < e_idx:
                voiced_indices.extend(range(s_idx, e_idx))
                
        if voiced_indices:
            gt_voiced = gt[voiced_indices]
            pred_voiced = pred[voiced_indices]
            
            mae = np.mean(np.abs(gt_voiced - pred_voiced))
            rmse = np.sqrt(np.mean((gt_voiced - pred_voiced)**2))
            corr = np.corrcoef(gt_voiced, pred_voiced)[0, 1] if np.std(gt_voiced)>0 and np.std(pred_voiced)>0 else 0.0
        else:
            mae, rmse, corr = 0.0, 0.0, 0.0

        results.append({
            "File": base_name,
            "Voiced MAE": mae,
            "Voiced RMSE": rmse,
            "Voiced Correlation": corr
        })
        print(f"  → Voiced MAE: {mae:.2f}, RMSE: {rmse:.2f}, Corr: {corr:.3f}")

    df = pd.DataFrame(results)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    df.to_excel(cfg.EVAL_OUTPUT_EXCEL, index=False)
    print(f"\n[完了] 評価結果を保存しました: {cfg.EVAL_OUTPUT_EXCEL}")
    
    print("\n--- 全体サマリー ---")
    print(f"平均 MAE (有声区間): {df['Voiced MAE'].mean():.2f}")
    print(f"平均 RMSE (有声区間): {df['Voiced RMSE'].mean():.2f}")
    print(f"平均 相関係数 (有声区間): {df['Voiced Correlation'].mean():.3f}")

if __name__ == "__main__":
    main()
