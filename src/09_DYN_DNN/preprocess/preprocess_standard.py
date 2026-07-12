"""
preprocess_standard.py — 標準前処理スクリプト（実験A/C向け）
08_DYN_DNN_Estimation/preprocess.py を完全コピーし、以下を変更:
  - configのインポートをスクリプト先頭で切り替え可能にした
  - INPUT_FEATURESで指定した特徴量のみを抽出・保存
  - untuned データは不要（tuned WAV + VSQX のペアのみ）
  - feature_extractor.py を util/ から参照
"""

import os
import sys
import json
import numpy as np
import librosa
import glob

# =============================================================================
# ===== config 切り替え（ここを書き換えて実験を切り替え） =====
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_09_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, _09_ROOT)

import config.exp_rms_only as cfg

# import config.exp_diff_learning as cfg
# import config.exp_multibranch as cfg

# 外部モジュールのパス追加
sys.path.insert(0, cfg.PATH_05)
sys.path.insert(0, cfg.PATH_04_SRC)
sys.path.insert(0, cfg.UTIL_DIR)

from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader


def get_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def find_4way_pairs():
    """Tuned(有り), Untuned(無し) のWAVとVSQXの4点セットを探す"""
    import re
    
    # 全てのWAVとVSQXを収集 (ディレクトリ関係なく検索)
    all_wavs = []
    all_wavs.extend(glob.glob(os.path.join(cfg.TRAIN_TUNED_WAV_DIR, "**", "*.wav"), recursive=True))
    if hasattr(cfg, "TRAIN_UNTUNED_WAV_DIR"):
        all_wavs.extend(glob.glob(os.path.join(cfg.TRAIN_UNTUNED_WAV_DIR, "**", "*.wav"), recursive=True))
    
    all_vsqxs = []
    all_vsqxs.extend(glob.glob(os.path.join(cfg.TRAIN_TUNED_VSQX_DIR, "**", "*.vsqx"), recursive=True))
    if hasattr(cfg, "TRAIN_UNTUNED_VSQX_DIR"):
        all_vsqxs.extend(glob.glob(os.path.join(cfg.TRAIN_UNTUNED_VSQX_DIR, "**", "*.vsqx"), recursive=True))

    all_wavs = list(set(all_wavs))
    all_vsqxs = list(set(all_vsqxs))

    songs = {}

    def get_song_info(filename):
        base = get_base_name(filename)
        is_untuned = "untuned" in base.lower()
        is_aug = "aug" in base.lower()
        
        # ベース名の抽出 (純粋な曲名)
        clean_base = re.sub(r'_?untuned', '', base, flags=re.IGNORECASE)
        clean_base = re.sub(r'_?aug\d*', '', clean_base, flags=re.IGNORECASE)
        
        # augの識別子 (例: aug1)
        aug_match = re.search(r'(aug\d*)', base, flags=re.IGNORECASE)
        aug_id = aug_match.group(1) if aug_match else None
        
        category = "untuned" if is_untuned else ("aug" if is_aug else "original")
        return clean_base, category, aug_id

    # WAVの分類
    for w in all_wavs:
        clean_base, cat, aug_id = get_song_info(w)
        if clean_base not in songs:
            songs[clean_base] = {"original": {}, "untuned": {}, "aug": {}}
            
        if cat == "aug":
            if aug_id not in songs[clean_base]["aug"]:
                songs[clean_base]["aug"][aug_id] = {}
            songs[clean_base]["aug"][aug_id]["wav"] = w
        else:
            songs[clean_base][cat]["wav"] = w

    # VSQXの分類
    for v in all_vsqxs:
        clean_base, cat, aug_id = get_song_info(v)
        if clean_base not in songs:
            songs[clean_base] = {"original": {}, "untuned": {}, "aug": {}}
            
        if cat == "aug":
            if aug_id not in songs[clean_base]["aug"]:
                songs[clean_base]["aug"][aug_id] = {}
            songs[clean_base]["aug"][aug_id]["vsqx"] = v
        else:
            songs[clean_base][cat]["vsqx"] = v

    pairs = []
    
    # ペアのマッチング
    for clean_base, data in songs.items():
        untuned_wav = data["untuned"].get("wav")
        untuned_vsqx = data["untuned"].get("vsqx")
        
        # untunedデータが揃っていなければ比較ペアは作れない
        if not untuned_wav or not untuned_vsqx:
            continue
            
        # 1. オリジナルペアの構築
        orig_wav = data["original"].get("wav")
        orig_vsqx = data["original"].get("vsqx")
        if orig_wav and orig_vsqx:
            pairs.append((clean_base, orig_wav, orig_vsqx, untuned_wav, untuned_vsqx))
            
        # 2. 拡張ペアの構築
        for aug_id, aug_data in data["aug"].items():
            a_wav = aug_data.get("wav")
            a_vsqx = aug_data.get("vsqx")
            if a_wav and a_vsqx:
                pair_name = f"{clean_base}_{aug_id}"
                pairs.append((pair_name, a_wav, a_vsqx, untuned_wav, untuned_vsqx))

    return sorted(pairs, key=lambda x: x[0])


def main():
    print("=" * 60)
    print(f"  09_DYN_DNN Preprocess (Standard)")
    print(f"  Config: {cfg.EXPERIMENT_NAME}")
    print(f"  Features: {cfg.INPUT_FEATURES}")
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
    # 1パス目: 全ファイルの特徴量を走査し、統計量（平均・標準偏差）を計算
    # =================================================================
    print(f"\n[Pass 1] 特徴量の統計量を計算中...")
    sum_x = None
    sum_sq_x = None
    total_frames_all = 0
    valid_pairs = []

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
            features_t = extractor.get_features(tuned_wav, cfg.INPUT_FEATURES)
            features_ut = extractor.get_features(untuned_wav, cfg.INPUT_FEATURES)

            if sum_x is None:
                dim = features_t.shape[1]
                sum_x = np.zeros(dim, dtype=np.float64)
                sum_sq_x = np.zeros(dim, dtype=np.float64)

            # Tuned と Untuned 両方の統計を合算（同じスケーラーで標準化するため）
            for feats in [features_t, features_ut]:
                sum_x += np.sum(feats, axis=0)
                sum_sq_x += np.sum(feats ** 2, axis=0)
                total_frames_all += feats.shape[0]

            valid_pairs.append((base, tuned_wav, tuned_vsqx, untuned_wav, untuned_vsqx))

        except Exception as e:
            print(f"  [{i+1}] {base} -> エラー (Pass 1): {e}")

    if total_frames_all == 0:
        print("[エラー] 有効なデータがありません。")
        return

    # 平均・標準偏差の計算
    global_mean = (sum_x / total_frames_all).astype(np.float32)
    global_std = np.sqrt(
        (sum_sq_x / total_frames_all) - (global_mean.astype(np.float64) ** 2)
    ).astype(np.float32)
    global_std = np.maximum(global_std, 1e-8)  # ゼロ除算防止

    # scaler.json 保存
    scaler_path = os.path.join(cfg.PREPROCESSED_DIR, "scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": global_mean.tolist(),
                "std": global_std.tolist(),
                "total_frames": total_frames_all,
                "features": cfg.INPUT_FEATURES,
            },
            f,
        )
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
            features_t = extractor.get_features(tuned_wav, cfg.INPUT_FEATURES)
            features_ut = extractor.get_features(untuned_wav, cfg.INPUT_FEATURES)
            
            # 長さを揃える
            min_len = min(len(features_t), len(features_ut))
            features_t = features_t[:min_len]
            features_ut = features_ut[:min_len]
            total_frames = min_len

            # Zスコア標準化
            features_t = ((features_t - global_mean) / (global_std + 1e-8)).astype(
                np.float32
            )
            features_ut = ((features_ut - global_mean) / (global_std + 1e-8)).astype(
                np.float32
            )

            # DYN抽出（パラメータ有りVSQXから取得）+ 正規化 [-1.0, 1.0]
            vsqx_data = reader.read(tuned_vsqx)
            dyn_points = vsqx_data["dynamic_params"].get("DYN", [])
            dyn_curve = reader.interpolate_param_to_frames(
                dyn_points, total_frames, cfg.DYN_DEFAULT
            )
            dyn_curve = dyn_curve[:total_frames].astype(np.float32)
            dyn_normalized = (dyn_curve / 127.0) * 2.0 - 1.0  # [0,127] -> [-1,1]

            # Voicing Mask計算（ノートベースでの有音判定）
            notes = vsqx_data.get("notes", [])
            voicing_mask = np.zeros((total_frames, 1), dtype=np.float32)
            for note in notes:
                s_idx = librosa.time_to_frames(
                    note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
                )
                e_idx = librosa.time_to_frames(
                    note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
                )
                s_idx = max(0, min(s_idx, total_frames))
                e_idx = max(0, min(e_idx, total_frames))
                if s_idx < e_idx:
                    voicing_mask[s_idx:e_idx] = 1.0

            # 特徴量を個別キーで保存（INPUT_FEATURES が複数の場合に対応）
            save_dict = {
                "dyn": dyn_normalized.reshape(-1, 1),
                "voicing_mask": voicing_mask,
            }

            # 特徴量を各キーで保存（Tuned / Untuned 両方）
            col_start = 0
            for feat_name in cfg.INPUT_FEATURES:
                feat_dim = extractor.get_feature_dim(feat_name)
                save_dict[f"{feat_name}_tuned"] = features_t[:, col_start : col_start + feat_dim]
                save_dict[f"{feat_name}_untuned"] = features_ut[:, col_start : col_start + feat_dim]
                col_start += feat_dim

            np.savez(out_path, **save_dict)

        except Exception as e:
            print(f"    -> エラー: {e}")

    # features.json保存
    features_info = {
        "sr": cfg.SR,
        "hop_length": cfg.HOP_LENGTH,
        "input_features": cfg.INPUT_FEATURES,
        "feature_dims": {f: extractor.get_feature_dim(f) for f in cfg.INPUT_FEATURES},
    }
    with open(
        os.path.join(cfg.PREPROCESSED_DIR, "features.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(features_info, f, indent=2)

    print("\n[完了] すべての前処理が完了しました。")


if __name__ == "__main__":
    main()
