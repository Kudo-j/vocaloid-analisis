import os
import sys
import numpy as np
import joblib
import librosa
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "05_DYN_Estimation"))

import config_direct as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader
from pair_finder import find_pairs

def extract_direct_data(pairs, extractor, reader):
    all_mel = []
    all_dyn = []
    skip = cfg.NOTE_BOUNDARY_SKIP_FRAMES
    threshold = cfg.MEL_POWER_THRESHOLD_DB

    for i, (base_name, wav_path, vsqx_path) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {base_name}")
        try:
            mel = extractor.get_features(wav_path, ["mel"])
            total_frames = len(mel)
            mel_power = np.mean(mel, axis=1)

            vsqx_data = reader.read(vsqx_path)
            if not vsqx_data or not vsqx_data.get("notes"):
                continue

            dyn_points = vsqx_data["dynamic_params"].get("DYN", [])
            dyn_curve = reader.interpolate_param_to_frames(
                dyn_points, total_frames, default_value=cfg.DYN_DEFAULT
            )

            frame_count = 0
            for note in vsqx_data["notes"]:
                s_idx = librosa.time_to_frames(note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
                e_idx = librosa.time_to_frames(note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
                
                # ノート境界のスキップ
                inner_start = max(0, s_idx + skip)
                inner_end = min(total_frames, e_idx - skip)

                for f in range(inner_start, inner_end):
                    # 音量ガード：無声子音等の不安定なフレームは学習から除外
                    if mel_power[f] >= threshold:
                        all_mel.append(mel[f])
                        all_dyn.append(dyn_curve[f])
                        frame_count += 1

            print(f"    → {frame_count} フレーム")
        except Exception as e:
            print(f"    → エラー: {e}")

    return np.array(all_mel, dtype=np.float32), np.array(all_dyn, dtype=np.float32)

def main():
    print("=" * 60)
    print("  DYN Direct Estimation (Ridge) - Training")
    print("=" * 60)

    pairs = find_pairs(cfg.TRAIN_WAV_DIR, cfg.TRAIN_VSQX_DIR)
    if not pairs:
        print("[エラー] ペアが見つかりません。")
        return

    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    print("\n[Step 1] 学習データの抽出...")
    X, y = extract_direct_data(pairs, extractor, reader)

    if len(X) == 0:
        print("[エラー] 学習データが0件です。")
        return

    print(f"\n  合計学習フレーム数: {len(X)}")
    if len(X) > cfg.MAX_TRAIN_FRAMES:
        print(f"  サンプリング: {len(X)} → {cfg.MAX_TRAIN_FRAMES}")
        np.random.seed(42)
        indices = np.random.choice(len(X), cfg.MAX_TRAIN_FRAMES, replace=False)
        X, y = X[indices], y[indices]

    print("\n[Step 2] Ridge回帰モデルの学習...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=cfg.RIDGE_ALPHA)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    r_squared = model.score(X_scaled, y)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))

    print(f"  R² スコア: {r_squared:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    std_y = np.std(y)
    std_pred = np.std(y_pred)
    recommended_scale = std_y / std_pred if std_pred > 1e-8 else 1.0

    print(f"\n  --- 推奨振幅補正倍率 ---")
    print(f"  正解DYNの標準偏差: {std_y:.4f}")
    print(f"  予測DYNの標準偏差: {std_pred:.4f}")
    print(f"  ★ 推奨 DIRECT_OUTPUT_SCALE = {recommended_scale:.2f}")
    print(f"  → config_direct.py に設定してください")

    coef_abs = np.abs(model.coef_)
    top_indices = np.argsort(coef_abs)[::-1][:10]
    print(f"\n  回帰係数 Top 10 (Melバンド):")
    for r, idx in enumerate(top_indices):
        print(f"    #{r+1}: Band {idx:3d} → coef={model.coef_[idx]:+.6f}")

    print("\n[Step 3] モデル保存...")
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "mean_dyn": float(np.mean(y)), # スケール補正の基準点として使用
        "sr": cfg.SR,
        "hop_length": cfg.HOP_LENGTH,
        "n_mels": cfg.N_MELS,
        "recommended_scale": recommended_scale
    }, cfg.MODEL_PATH)
    print(f"  保存完了: {cfg.MODEL_PATH}")

if __name__ == "__main__":
    main()
