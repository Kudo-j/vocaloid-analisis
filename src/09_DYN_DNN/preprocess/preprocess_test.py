"""
preprocess_test.py — テスト用前処理スクリプト
Untuned（パラ無し）データ作成待ちの間に、Tunedデータのみで学習テストを行うためのプログラム。
WAVとVSQXの2点ペアのみを探し、単一の特徴量としてNPZに保存します。
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

# 外部モジュールのパス追加
sys.path.insert(0, cfg.PATH_05)
sys.path.insert(0, cfg.PATH_04_SRC)
sys.path.insert(0, cfg.UTIL_DIR)

from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader


def get_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def find_wav_vsqx_pairs():
    """チューニング済みWAVとVSQXのペアを探す"""
    wav_files = glob.glob(
        os.path.join(cfg.TRAIN_TUNED_WAV_DIR, "**", "*.wav"), recursive=True
    )
    vsqx_files = glob.glob(
        os.path.join(cfg.TRAIN_TUNED_VSQX_DIR, "**", "*.vsqx"), recursive=True
    )

    wav_dict = {get_base_name(f): f for f in wav_files}
    vsqx_dict = {get_base_name(f): f for f in vsqx_files}

    pairs = []
    for base, wav_path in wav_dict.items():
        if base in vsqx_dict:
            pairs.append((base, wav_path, vsqx_dict[base]))

    return sorted(pairs, key=lambda x: x[0])


def main():
    print("=" * 60)
    print(f"  09_DYN_DNN Preprocess (Test: Tuned Only)")
    print(f"  Config: {cfg.EXPERIMENT_NAME}")
    print(f"  Features: {cfg.INPUT_FEATURES}")
    print("=" * 60)

    pairs = find_wav_vsqx_pairs()
    if not pairs:
        print("[エラー] WAV-VSQXペアが見つかりません。")
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

    for i, (base, wav_path, vsqx_path) in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  Scanning: {i+1}/{len(pairs)}...")
        try:
            # 特徴量抽出（統計量計算用）
            features = extractor.get_features(wav_path, cfg.INPUT_FEATURES)

            if sum_x is None:
                dim = features.shape[1]
                sum_x = np.zeros(dim, dtype=np.float64)
                sum_sq_x = np.zeros(dim, dtype=np.float64)

            sum_x += np.sum(features, axis=0)
            sum_sq_x += np.sum(features**2, axis=0)
            total_frames_all += features.shape[0]

            valid_pairs.append((base, wav_path, vsqx_path))

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

    for i, (base, wav_path, vsqx_path) in enumerate(valid_pairs):
        out_path = os.path.join(cfg.PREPROCESSED_DIR, f"{base}.npz")
        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(valid_pairs)}] {base} (Skip: Already processed)")
            continue

        print(f"  [{i+1}/{len(valid_pairs)}] Processing {base}...")
        try:
            # 特徴量抽出
            features = extractor.get_features(wav_path, cfg.INPUT_FEATURES)
            total_frames = features.shape[0]

            # Zスコア標準化
            features = ((features - global_mean) / (global_std + 1e-8)).astype(
                np.float32
            )

            # DYN抽出（VSQXから取得）+ 正規化 [-1.0, 1.0]
            vsqx_data = reader.read(vsqx_path)
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

            # 特徴量を各キーで保存
            col_start = 0
            for feat_name in cfg.INPUT_FEATURES:
                feat_dim = extractor.get_feature_dim(feat_name)
                save_dict[feat_name] = features[:, col_start : col_start + feat_dim]
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
