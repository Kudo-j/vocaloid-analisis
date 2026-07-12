import os
import sys
import json
import numpy as np
import librosa
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "05_DYN_Estimation"))

import config_diffDYN as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader

def get_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]

import re

def find_4way_pairs():
    # Tuned(有り), Untuned(無し) のWAVとVSQXの4点セットを探す
    tuned_wavs = glob.glob(os.path.join(cfg.TRAIN_TUNED_WAV_DIR, "**", "*.wav"), recursive=True)
    tuned_vsqxs = glob.glob(os.path.join(cfg.TRAIN_TUNED_VSQX_DIR, "**", "*.vsqx"), recursive=True)
    untuned_wavs = glob.glob(os.path.join(cfg.TRAIN_UNTUNED_WAV_DIR, "**", "*.wav"), recursive=True)
    untuned_vsqxs = glob.glob(os.path.join(cfg.TRAIN_UNTUNED_VSQX_DIR, "**", "*.vsqx"), recursive=True)

    tuned_w_dict = {get_base_name(f): f for f in tuned_wavs}
    tuned_v_dict = {get_base_name(f): f for f in tuned_vsqxs}
    untuned_w_dict = {get_base_name(f): f for f in untuned_wavs}
    untuned_v_dict = {get_base_name(f): f for f in untuned_vsqxs}

    pairs = []
    for tuned_base, tuned_wav in tuned_w_dict.items():
        # 例: "8番出口_Original_Soft_aug3_1" -> "8番出口_Original_Soft_1"
        untuned_base = re.sub(r'_aug\d*', '', tuned_base)
        
        if (tuned_base in tuned_v_dict and 
            untuned_base in untuned_w_dict and 
            untuned_base in untuned_v_dict):
            pairs.append((
                tuned_base, 
                tuned_wav, 
                tuned_v_dict[tuned_base], 
                untuned_w_dict[untuned_base],
                untuned_v_dict[untuned_base]
            ))
    
    return pairs

def main():
    print("=" * 60)
    print("  DYN DNN Estimation - Preprocess")
    print("=" * 60)

    pairs = find_4way_pairs()
    if not pairs:
        print("[エラー] 4点セット（有WAV/VSQX、無WAV/VSQX）が揃うペアが見つかりません。")
        return
    print(f"見つかったペア数: {len(pairs)}")

    os.makedirs(cfg.PREPROCESSED_DIR, exist_ok=True)
    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH, use_part_offset=True)

    # =================================================================
    # 1パス目: 全ファイルのMelを走査し、全体の平均・標準偏差を計算
    # =================================================================
    print("\n[Pass 1] Melスペクトログラムの統計量を計算中...")
    sum_x = None
    sum_sq_x = None
    total_frames_all = 0
    valid_pairs = []  # DYN混入チェックを通過したペアを記録

    for i, (base, tuned_wav, tuned_vsqx, untuned_wav, untuned_vsqx) in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  Scanning: {i+1}/{len(pairs)}...")
        try:
            # 無しVSQX の DYN 混入チェック
            untuned_vsqx_data = reader.read(untuned_vsqx)
            if untuned_vsqx_data and "DYN" in untuned_vsqx_data["dynamic_params"]:
                untuned_dyn_points = untuned_vsqx_data["dynamic_params"]["DYN"]
                has_tuned_dyn = any(val != cfg.DYN_DEFAULT for _, val in untuned_dyn_points)
                if has_tuned_dyn:
                    print(f"  [{i+1}] {base} -> [警告] 無しVSQXにDYNパラメータが存在。除外します。")
                    continue

            # 特徴量抽出（統計量計算用）
            mel_tuned = extractor.get_features(tuned_wav, ["mel"])
            mel_untuned = extractor.get_features(untuned_wav, ["mel"])

            if sum_x is None:
                dim = mel_tuned.shape[1]
                sum_x = np.zeros(dim, dtype=np.float64)
                sum_sq_x = np.zeros(dim, dtype=np.float64)

            # Tuned と Untuned 両方の統計を合算（同じスケーラーで標準化するため）
            for mel_data in [mel_tuned, mel_untuned]:
                sum_x += np.sum(mel_data, axis=0)
                sum_sq_x += np.sum(mel_data ** 2, axis=0)
                total_frames_all += mel_data.shape[0]

            valid_pairs.append((base, tuned_wav, tuned_vsqx, untuned_wav, untuned_vsqx))

        except Exception as e:
            print(f"  [{i+1}] {base} -> エラー (Pass 1): {e}")

    if total_frames_all == 0:
        print("[エラー] 有効なデータがありません。")
        return

    # 平均・標準偏差の計算
    global_mean = (sum_x / total_frames_all).astype(np.float32)
    global_std = np.sqrt((sum_sq_x / total_frames_all) - (global_mean.astype(np.float64) ** 2)).astype(np.float32)
    global_std = np.maximum(global_std, 1e-8)  # ゼロ除算防止

    # scaler.json 保存
    scaler_path = os.path.join(cfg.PREPROCESSED_DIR, "scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({
            "mean": global_mean.tolist(),
            "std": global_std.tolist(),
            "total_frames": total_frames_all
        }, f)
    print(f"  統計量を保存しました: {scaler_path}")
    print(f"  合計フレーム数: {total_frames_all}")

    # =================================================================
    # 2パス目: 標準化済みデータを npz として保存
    # =================================================================
    print(f"\n[Pass 2] 標準化済みデータの保存中... ({len(valid_pairs)} ペア)")

    for i, (base, tuned_wav, tuned_vsqx, untuned_wav, untuned_vsqx) in enumerate(valid_pairs):
        out_path = os.path.join(cfg.PREPROCESSED_DIR, f"{base}.npz")
        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(valid_pairs)}] {base} (Skip: Already processed)")
            continue
            
        print(f"  [{i+1}/{len(valid_pairs)}] Processing {base}...")
        try:
            # 特徴量抽出
            mel_tuned = extractor.get_features(tuned_wav, ["mel"])
            mel_untuned = extractor.get_features(untuned_wav, ["mel"])
            
            # 長さを揃える
            min_len = min(len(mel_tuned), len(mel_untuned))
            mel_tuned = mel_tuned[:min_len]
            mel_untuned = mel_untuned[:min_len]

            # Zスコア標準化
            mel_tuned = ((mel_tuned - global_mean) / (global_std + 1e-8)).astype(np.float32)
            mel_untuned = ((mel_untuned - global_mean) / (global_std + 1e-8)).astype(np.float32)

            # DYN抽出 (パラメータ有りVSQXから取得) + 正規化 [-1.0, 1.0]
            vsqx_data = reader.read(tuned_vsqx)
            dyn_points = vsqx_data["dynamic_params"].get("DYN", [])
            dyn_curve = reader.interpolate_param_to_frames(dyn_points, min_len, cfg.DYN_DEFAULT)
            dyn_curve = dyn_curve[:min_len].astype(np.float32)
            dyn_normalized = (dyn_curve / 127.0) * 2.0 - 1.0  # [0,127] -> [-1,1]

            # Voicing Mask計算 (ノートベースでの有音判定)
            notes = vsqx_data.get("notes", [])
            voicing_mask = np.zeros((min_len, 1), dtype=np.float32)
            for note in notes:
                s_idx = librosa.time_to_frames(note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
                e_idx = librosa.time_to_frames(note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
                s_idx = max(0, min(s_idx, min_len))
                e_idx = max(0, min(e_idx, min_len))
                if s_idx < e_idx:
                    voicing_mask[s_idx:e_idx] = 1.0

            np.savez(
                out_path,
                mel_tuned=mel_tuned,
                mel_untuned=mel_untuned,
                dyn=dyn_normalized.reshape(-1, 1),
                voicing_mask=voicing_mask
            )

        except Exception as e:
            print(f"    -> エラー: {e}")

    # features.json保存
    with open(os.path.join(cfg.PREPROCESSED_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump({"sr": cfg.SR, "hop_length": cfg.HOP_LENGTH, "n_mels": cfg.N_MELS}, f)
    
    print("\n[完了] すべての前処理が完了しました。")

if __name__ == "__main__":
    main()
