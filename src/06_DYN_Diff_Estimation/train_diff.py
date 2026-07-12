"""
train_diff.py
=============
時間差分モデルによるDYN推定 — 学習スクリプト

アプローチ:
  フレーム間のMelスペクトログラムの変化量 (ΔMel/Δt) から、
  DYNの変化量 (ΔDYN/Δt) を予測するRidge回帰モデルを学習する。

  母音ごとの基準値の差は、隣接フレーム間で差分を取ることにより
  自動的に相殺（キャンセル）されるため、ベースラインモデル (Phase 1) が不要。

学習データ:
  DYNが連続的に変動している実際の歌唱データ (WAV + VSQX)。
  VSQXに書き込まれたDYNの制御点から正解値を生成する。

使い方:
  1. config_diff.py の TRAIN_WAV_DIR, TRAIN_VSQX_DIR を設定
  2. このファイルを実行
  3. models/ に diff_dyn_model.pkl が生成される
"""

import os
import sys
import numpy as np
import joblib
import librosa

# 既存モジュールへのパス追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "05_DYN_Estimation"))

import config_diff as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader
from pair_finder import find_pairs


def extract_diff_data(pairs, extractor, reader):
    """
    ペアリストから「Melの時間差分 → DYNの時間差分」の学習データを抽出する。

    各ペアについて:
      1. WAVからMelスペクトログラムを抽出
      2. VSQXからDYN制御点を読み取り、フレーム時系列に補間
      3. ノートの発声区間内で、DIFF_STEPフレーム間隔の差分を計算
      4. ノート境界付近のフレームはスキップ

    Args:
        pairs: [(base_name, wav_path, vsqx_path), ...]
        extractor: AudioFeatureExtractor
        reader: VSQXReader

    Returns:
        X: np.ndarray (N, n_mels) - Melの時間差分
        y: np.ndarray (N,) - DYNの時間差分
    """
    all_delta_mel = []
    all_delta_dyn = []
    all_anchor_X = []
    all_anchor_y = []
    step = cfg.DIFF_STEP
    skip = cfg.NOTE_BOUNDARY_SKIP_FRAMES

    for i, (base_name, wav_path, vsqx_path) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {base_name}")

        try:
            # Melスペクトログラムの抽出
            mel = extractor.get_features(wav_path, ["mel"])
            total_frames = len(mel)

            # VSQXの読み込み
            vsqx_data = reader.read(vsqx_path)
            if vsqx_data is None:
                print(f"    → VSQX読み込み失敗。スキップ")
                continue

            notes = vsqx_data.get("notes", [])
            if not notes:
                print(f"    → ノートなし。スキップ")
                continue

            # DYN制御点をフレーム時系列に補間
            dyn_points = vsqx_data["dynamic_params"].get("DYN", [])
            dyn_curve = reader.interpolate_param_to_frames(
                dyn_points, total_frames, default_value=cfg.DYN_DEFAULT
            )

            # ノート区間ごとに差分データを計算
            note_frames_count = 0
            for note in notes:
                s_idx = librosa.time_to_frames(
                    note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
                )
                e_idx = librosa.time_to_frames(
                    note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
                )
                s_idx = max(0, s_idx)
                e_idx = min(total_frames, e_idx)

                # ノート境界を除外した有効範囲
                inner_start = s_idx + skip
                inner_end = e_idx - skip

                if inner_end - inner_start < step + 1:
                    continue  # ノートが短すぎる

                # アンカーデータの抽出 (ノート開始時点の安定区間のパワーとDYN)
                anchor_f_start = min(s_idx + skip, e_idx - 1)
                anchor_f_end = min(anchor_f_start + step, e_idx)
                if anchor_f_end > anchor_f_start:
                    note_power = np.mean(np.mean(mel[anchor_f_start:anchor_f_end], axis=1))
                    note_dyn = np.mean(dyn_curve[anchor_f_start:anchor_f_end])
                else:
                    note_power = np.mean(mel[anchor_f_start])
                    note_dyn = dyn_curve[anchor_f_start]
                
                all_anchor_X.append(note_power)
                all_anchor_y.append(note_dyn)

                # DIFF_STEPフレーム間隔で差分を計算
                for f in range(inner_start + step, inner_end):
                    delta_mel = mel[f] - mel[f - step]
                    delta_dyn = dyn_curve[f] - dyn_curve[f - step]
                    all_delta_mel.append(delta_mel)
                    all_delta_dyn.append(delta_dyn)
                    note_frames_count += 1

            print(f"    → {note_frames_count} フレーム (差分データ)")

        except Exception as e:
            print(f"    → エラー: {e}")
            continue

    if not all_delta_mel:
        return np.array([]), np.array([]), np.array([]), np.array([])

    X = np.array(all_delta_mel, dtype=np.float32)
    y = np.array(all_delta_dyn, dtype=np.float32)
    anchor_X = np.array(all_anchor_X, dtype=np.float32)
    anchor_y = np.array(all_anchor_y, dtype=np.float32)
    return X, y, anchor_X, anchor_y


def main():
    print("=" * 60)
    print("  DYN Estimation Training (Time-Diff Model)")
    print("=" * 60)

    # パスの存在チェック
    print(f"\n[Config]")
    print(f"  学習用WAVフォルダ: {cfg.TRAIN_WAV_DIR}")
    print(f"  学習用VSQXフォルダ: {cfg.TRAIN_VSQX_DIR}")
    print(f"  モデル保存先: {cfg.MODEL_PATH}")
    print(f"  SR: {cfg.SR}, HOP: {cfg.HOP_LENGTH}")
    print(f"  Mel次元数: {cfg.N_MELS}")
    print(f"  差分ステップ: {cfg.DIFF_STEP} フレーム")
    print(f"  Ridge正則化: α={cfg.RIDGE_ALPHA}")

    if not os.path.isdir(cfg.TRAIN_WAV_DIR):
        print(f"\n[エラー] WAVフォルダが存在しません: {cfg.TRAIN_WAV_DIR}")
        return
    if not os.path.isdir(cfg.TRAIN_VSQX_DIR):
        print(f"\n[エラー] VSQXフォルダが存在しません: {cfg.TRAIN_VSQX_DIR}")
        return

    # ペアの検索
    print(f"\n[Step 1] WAV-VSQXペアの検索...")
    pairs = find_pairs(cfg.TRAIN_WAV_DIR, cfg.TRAIN_VSQX_DIR)
    if not pairs:
        print("\n[エラー] ペアが1つも見つかりませんでした。")
        return

    # 特徴量抽出器とリーダーの準備
    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    # 学習データの抽出
    print(f"\n[Step 2] 差分学習データとアンカーデータの抽出...")
    X, y, anchor_X, anchor_y = extract_diff_data(pairs, extractor, reader)

    if len(X) == 0:
        print("\n[エラー] 学習データが0件です。データを確認してください。")
        return

    print(f"\n  合計学習フレーム数 (抽出直後): {len(X)}")

    # メモリ枯渇防止のためのサンプリング
    if len(X) > cfg.MAX_TRAIN_FRAMES:
        print(f"  [情報] メモリ制限のため、{len(X)} → {cfg.MAX_TRAIN_FRAMES} フレームにサンプリング")
        np.random.seed(42)
        indices = np.random.choice(len(X), cfg.MAX_TRAIN_FRAMES, replace=False)
        X = X[indices]
        y = y[indices]

    print(f"  ΔMel形状: {X.shape}")
    print(f"  ΔDYN 統計: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.1f}, max={y.max():.1f}")

    # アンカーモデルの学習 (Melパワー -> 初期DYN)
    print(f"\n[Step 3] アンカーモデル (開始DYN推定) の学習...")
    from sklearn.linear_model import LinearRegression
    anchor_X_2d = anchor_X.reshape(-1, 1)
    anchor_model = LinearRegression()
    anchor_model.fit(anchor_X_2d, anchor_y)
    anchor_r2 = anchor_model.score(anchor_X_2d, anchor_y)
    print(f"  R² スコア: {anchor_r2:.4f}")
    print(f"  学習された式: Start_DYN = {anchor_model.coef_[0]:.4f} * Mel_Power + {anchor_model.intercept_:.4f}")

    # Ridge回帰モデルの学習
    print(f"\n[Step 4] Ridge回帰モデル (差分) の学習...")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=cfg.RIDGE_ALPHA)
    model.fit(X_scaled, y)

    # 学習精度の確認
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    r_squared = model.score(X_scaled, y)
    mae = np.mean(np.abs(residuals))

    print(f"  R² スコア: {r_squared:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # 推奨出力倍率の計算
    # モデルの予測値（y_pred）の標準偏差が、正解値（y）の標準偏差より
    # どの程度小さいかを計算し、振幅を復元するための倍率を算出する
    std_gt = np.std(y)
    std_pred = np.std(y_pred)
    if std_pred > 1e-8:
        recommended_scale = std_gt / std_pred
    else:
        recommended_scale = 1.0

    print(f"\n  --- 推奨出力倍率 ---")
    print(f"  正解ΔDYNの標準偏差: {std_gt:.4f}")
    print(f"  予測ΔDYNの標準偏差: {std_pred:.4f}")
    print(f"  ★ 推奨 DIFF_OUTPUT_SCALE = {recommended_scale:.2f}")
    print(f"  → config_diff.py の DIFF_OUTPUT_SCALE にこの値を設定してください")

    # 係数の確認 (上位10個)
    coef_abs = np.abs(model.coef_)
    top_indices = np.argsort(coef_abs)[::-1][:10]
    print(f"\n  回帰係数の絶対値 Top 10 (Melバンドのインデックス):")
    for rank, idx in enumerate(top_indices):
        print(f"    #{rank+1}: Band {idx:3d} → coef={model.coef_[idx]:+.6f} (|coef|={coef_abs[idx]:.6f})")

    # モデルの保存
    print(f"\n[Step 5] モデルの保存...")
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    model_data = {
        "model": model,
        "anchor_model": anchor_model,
        "scaler": scaler,
        "sr": cfg.SR,
        "hop_length": cfg.HOP_LENGTH,
        "n_mels": cfg.N_MELS,
        "diff_step": cfg.DIFF_STEP,
        "ridge_alpha": cfg.RIDGE_ALPHA,
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "recommended_scale": recommended_scale,
        "n_training_files": len(pairs),
        "n_training_frames": len(X),
    }

    joblib.dump(model_data, cfg.MODEL_PATH)
    print(f"  保存完了: {cfg.MODEL_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  学習完了!")
    print(f"  次は predict_diff.py または eval_diff.py を実行してください。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
