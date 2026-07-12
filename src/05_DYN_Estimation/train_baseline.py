"""
train_baseline.py
=================
2段階学習によるDYN推定モデルの構築。

Phase 1: ベースラインモデル (MFCC → Mel)
    DYN=64（デフォルト）の音声データを用い、MFCCから
    「その母音・その音高で本来あるべきMelスペクトログラム」を予測する
    Ridge回帰モデルを学習する。

Phase 2: DYNスケールモデル (ΔMel → DYN値)
    DYN値が既知のデータ（DYN=1, 32, 64, 127の母音データ）を用い、
    「ベースラインMelとの差分（ΔMel）」からDYN値を推定する
    Ridge回帰モデルを学習する。（多項式特徴量によりDYN下限値の非線形性を吸収）

使い方:
  1. config_dyn.py の各フォルダパスを設定
  2. このファイルをVSCode上で実行 (F5 or ▶)
  3. models/ に baseline_mel_model.pkl, dyn_scale_model.pkl が生成される
"""

import os
import sys
import re
import numpy as np
import joblib
import librosa

# 既存モジュールへのパス追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))

import config_dyn as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader
from pair_finder import find_pairs


# =============================================================================
# ===== Phase 1: ベースラインモデルの学習 =====
# =============================================================================

def extract_baseline_data(pairs, extractor, reader):
    """
    DYN=64のペアリストから MFCC と Melスペクトログラム を抽出し、
    発声区間のフレームのみを結合して返す。

    Args:
        pairs: [(base_name, wav_path, vsqx_path), ...]
        extractor: AudioFeatureExtractor
        reader: VSQXReader

    Returns:
        X: np.ndarray (N, n_mfcc) - MFCC
        Y: np.ndarray (N, n_mels) - Melスペクトログラム (dB)
    """
    all_mfcc = []
    all_mel = []

    for i, (base_name, wav_path, vsqx_path) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {base_name}")

        # VSQXの読み込み (ノート情報の取得)
        vsqx_data = reader.read(vsqx_path)
        if vsqx_data is None:
            print(f"    [警告] VSQXの読み込みに失敗。スキップ")
            continue

        if not vsqx_data["notes"]:
            print(f"    [警告] ノートが存在しません。スキップ")
            continue

        # MFCC抽出 (N, n_mfcc)
        mfcc = extractor.get_features(wav_path, ["mfcc"])

        # Melスペクトログラム抽出 (N, n_mels)
        mel = extractor.get_features(wav_path, ["mel"])

        # 時間軸の長さを合わせる
        min_len = min(len(mfcc), len(mel))
        mfcc = mfcc[:min_len]
        mel = mel[:min_len]

        # 発声マスク (voicing_mask) の作成
        voicing_mask = np.zeros(min_len, dtype=bool)
        for note in vsqx_data["notes"]:
            s_idx = librosa.time_to_frames(
                note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
            )
            e_idx = librosa.time_to_frames(
                note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
            )
            voicing_mask[max(0, s_idx):min(min_len, e_idx)] = True

        # ノートが存在する（発声している）フレームのみ抽出
        mfcc = mfcc[voicing_mask]
        mel = mel[voicing_mask]

        print(f"    → {len(mfcc)} フレーム (ノート区間のみ)")

        if len(mfcc) > 0:
            all_mfcc.append(mfcc)
            all_mel.append(mel)

    if not all_mfcc:
        raise ValueError("有効な学習データが見つかりませんでした。")

    X = np.vstack(all_mfcc)
    Y = np.vstack(all_mel)

    return X, Y


# =============================================================================
# ===== Phase 2: DYNスケールモデルの学習 =====
# =============================================================================

def parse_dyn_value_from_filename(filename):
    """
    ファイル名からDYN値を抽出する。
    例: "あ_DYN127" → 127, "い_DYN1" → 1, "お_DYN32" → 32

    Args:
        filename: 拡張子なしのファイル名

    Returns:
        int or None: DYN値。抽出できない場合はNone。
    """
    match = re.search(r"DYN(?:all)?(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def extract_dyn_scale_data(pairs, extractor, reader, baseline_model_data,
                          force_dyn_value=None):
    """
    DYN値が既知のデータから ΔMel と DYN値 を抽出する。

    処理の流れ:
      1. 各WAVから MFCC と Mel を抽出
      2. ベースラインモデルを使って「本来のMel（DYN=64相当）」を予測
      3. ΔMel = 実際のMel - ベースラインMel を算出
      4. ファイル名からDYN値を取得し、(ΔMel, DYN値) のペアを蓄積

    Args:
        pairs: [(base_name, wav_path, vsqx_path), ...]
        extractor: AudioFeatureExtractor
        reader: VSQXReader
        baseline_model_data: Phase 1で学習済みのモデルデータ
        force_dyn_value: int or None - Noneの場合はファイル名からDYN値を取得。
                         整数を指定した場合は全ペアをそのDYN値として扱う。

    Returns:
        X: np.ndarray (N, n_mels) - ΔMelの値
        y: np.ndarray (N,) - DYN値 (0-127)
    """
    baseline_model = baseline_model_data["model"]
    baseline_scaler = baseline_model_data["scaler"]

    all_delta_mel = []
    all_dyn_values = []

    for i, (base_name, wav_path, vsqx_path) in enumerate(pairs):
        # DYN値の決定
        if force_dyn_value is not None:
            dyn_value = force_dyn_value
        else:
            dyn_value = parse_dyn_value_from_filename(base_name)
            if dyn_value is None:
                print(f"  [{i+1}/{len(pairs)}] {base_name} → DYN値不明。スキップ")
                continue

        if dyn_value == 0:
            print(f"  [{i+1}/{len(pairs)}] {base_name} → DYN=0は無音のためスキップ")
            continue

        print(f"  [{i+1}/{len(pairs)}] {base_name} (DYN={dyn_value})")

        # VSQXの読み込み
        vsqx_data = reader.read(vsqx_path)
        if vsqx_data is None or not vsqx_data["notes"]:
            print(f"    [警告] スキップ")
            continue

        # 特徴量抽出
        mfcc = extractor.get_features(wav_path, ["mfcc"])
        mel = extractor.get_features(wav_path, ["mel"])

        min_len = min(len(mfcc), len(mel))
        mfcc = mfcc[:min_len]
        mel = mel[:min_len]

        # 発声マスクの作成
        voicing_mask = np.zeros(min_len, dtype=bool)
        for note in vsqx_data["notes"]:
            s_idx = librosa.time_to_frames(
                note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
            )
            e_idx = librosa.time_to_frames(
                note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
            )
            voicing_mask[max(0, s_idx):min(min_len, e_idx)] = True

        # 発声区間のみ抽出
        mfcc_voiced = mfcc[voicing_mask]
        mel_voiced = mel[voicing_mask]

        if len(mfcc_voiced) == 0:
            print(f"    [警告] 有声フレームなし。スキップ")
            continue

        # ベースラインMelの予測
        mfcc_scaled = baseline_scaler.transform(mfcc_voiced)
        baseline_mel = baseline_model.predict(mfcc_scaled)

        # ΔMelの算出
        delta_mel = mel_voiced - baseline_mel

        # DYN値の配列を作成（全フレームで同一値）
        dyn_array = np.full(len(delta_mel), float(dyn_value))

        all_delta_mel.append(delta_mel)
        all_dyn_values.append(dyn_array)

        print(f"    → {len(delta_mel)} フレーム")

    if not all_delta_mel:
        raise ValueError("有効なDYNスケール学習データが見つかりませんでした。")

    X = np.vstack(all_delta_mel)
    y = np.concatenate(all_dyn_values)

    return X, y


# =============================================================================
# ===== メイン処理 =====
# =============================================================================

def main():
    print("=" * 60)
    print("  DYN Estimation Model Training (2-Phase)")
    print("=" * 60)

    # =============================================
    # Phase 1: ベースラインモデルの学習
    # =============================================
    print(f"\n{'='*60}")
    print(f"  Phase 1: ベースラインモデル (MFCC → Mel)")
    print(f"{'='*60}")

    print(f"\n[Config]")
    print(f"  学習用WAVフォルダ: {cfg.BASELINE_WAV_DIR}")
    print(f"  学習用VSQXフォルダ: {cfg.BASELINE_VSQX_DIR}")
    print(f"  モデル保存先: {cfg.BASELINE_MODEL_PATH}")
    print(f"  SR: {cfg.SR}, HOP: {cfg.HOP_LENGTH}")
    print(f"  MFCC次元数: {cfg.N_MFCC}, Mel次元数: {cfg.N_MELS}")

    if not os.path.isdir(cfg.BASELINE_WAV_DIR):
        print(f"\n[エラー] WAVフォルダが存在しません: {cfg.BASELINE_WAV_DIR}")
        return
    if not os.path.isdir(cfg.BASELINE_VSQX_DIR):
        print(f"\n[エラー] VSQXフォルダが存在しません: {cfg.BASELINE_VSQX_DIR}")
        return

    # ペアの検索
    print(f"\n[Step 1-1] WAV-VSQXペアの検索...")
    pairs = find_pairs(cfg.BASELINE_WAV_DIR, cfg.BASELINE_VSQX_DIR)
    if not pairs:
        print("\n[エラー] ペアが1つも見つかりませんでした。")
        return

    # 特徴量抽出器とリーダーの準備
    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    # 学習データの抽出
    print(f"\n[Step 1-2] ベースライン学習データの抽出...")
    X_baseline, Y_baseline = extract_baseline_data(pairs, extractor, reader)
    print(f"\n  合計学習フレーム数 (抽出直後): {len(X_baseline)}")

    # メモリ枯渇防止のためのサンプリング
    if len(X_baseline) > cfg.MAX_TRAIN_FRAMES_PHASE1:
        print(f"  [情報] メモリ制限のため、{len(X_baseline)} フレームから {cfg.MAX_TRAIN_FRAMES_PHASE1} フレームにサンプリングします。")
        np.random.seed(42)
        indices = np.random.choice(len(X_baseline), cfg.MAX_TRAIN_FRAMES_PHASE1, replace=False)
        X_baseline = X_baseline[indices]
        Y_baseline = Y_baseline[indices]

    print(f"  MFCC形状: {X_baseline.shape}")
    print(f"  Mel形状: {Y_baseline.shape}")

    # Ridge回帰モデルの学習
    print(f"\n[Step 1-3] Ridge回帰モデルの学習...")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler_baseline = StandardScaler()
    X_baseline_scaled = scaler_baseline.fit_transform(X_baseline)

    model_baseline = Ridge(alpha=1.0)
    model_baseline.fit(X_baseline_scaled, Y_baseline)

    # 学習精度の確認
    Y_pred = model_baseline.predict(X_baseline_scaled)
    residuals = Y_baseline - Y_pred
    rmse_per_band = np.sqrt(np.mean(residuals ** 2, axis=0))
    rmse_overall = np.sqrt(np.mean(residuals ** 2))
    r_squared = model_baseline.score(X_baseline_scaled, Y_baseline)

    print(f"  R² スコア (全Melバンド平均): {r_squared:.4f}")
    print(f"  RMSE (全体): {rmse_overall:.4f} dB")
    print(f"  RMSE (バンドごと min/max): [{rmse_per_band.min():.4f}, {rmse_per_band.max():.4f}] dB")

    # モデルの保存
    print(f"\n[Step 1-4] ベースラインモデルの保存...")
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    baseline_model_data = {
        "model": model_baseline,
        "scaler": scaler_baseline,
        "sr": cfg.SR,
        "hop_length": cfg.HOP_LENGTH,
        "n_mfcc": cfg.N_MFCC,
        "n_mels": cfg.N_MELS,
        "r_squared": r_squared,
        "rmse": rmse_overall,
        "n_training_files": len(pairs),
        "n_training_frames": len(X_baseline),
    }
    joblib.dump(baseline_model_data, cfg.BASELINE_MODEL_PATH)
    print(f"  保存完了: {cfg.BASELINE_MODEL_PATH}")

    # =============================================
    # Phase 2: DYNスケールモデルの学習
    # =============================================
    print(f"\n{'='*60}")
    print(f"  Phase 2: DYNスケールモデル (ΔMel → DYN値)")
    print(f"{'='*60}")

    print(f"\n[Config]")
    print(f"  DYN既知WAVフォルダ: {cfg.DYN_KNOWN_WAV_DIR}")
    print(f"  DYN既知VSQXフォルダ: {cfg.DYN_KNOWN_VSQX_DIR}")
    print(f"  モデル保存先: {cfg.DYN_MODEL_PATH}")
    print(f"  多項式特徴量: {cfg.USE_POLYNOMIAL} (degree={cfg.POLYNOMIAL_DEGREE})")

    if not os.path.isdir(cfg.DYN_KNOWN_WAV_DIR):
        print(f"\n[エラー] DYN既知WAVフォルダが存在しません: {cfg.DYN_KNOWN_WAV_DIR}")
        print("  Phase 2 をスキップします。predict_dyn.py は動作しません。")
        return
    if not os.path.isdir(cfg.DYN_KNOWN_VSQX_DIR):
        print(f"\n[エラー] DYN既知VSQXフォルダが存在しません: {cfg.DYN_KNOWN_VSQX_DIR}")
        print("  Phase 2 をスキップします。predict_dyn.py は動作しません。")
        return

    # ペアの検索
    print(f"\n[Step 2-1] DYN既知ペアの検索...")
    dyn_pairs = find_pairs(cfg.DYN_KNOWN_WAV_DIR, cfg.DYN_KNOWN_VSQX_DIR)
    if not dyn_pairs:
        print("\n[エラー] DYN既知ペアが1つも見つかりませんでした。")
        return

    # ΔMelとDYN値の抽出 (DYN=1, 32, 127 等)
    print(f"\n[Step 2-2] ΔMelとDYN値の抽出...")
    print(f"  --- DYN既知データ (Fixフォルダ) ---")
    X_dyn, y_dyn = extract_dyn_scale_data(
        dyn_pairs, extractor, reader, baseline_model_data
    )

    # DYN=64（デフォルト）のデータも追加
    if os.path.isdir(cfg.DYN_DEFAULT_WAV_DIR) and os.path.isdir(cfg.DYN_DEFAULT_VSQX_DIR):
        print(f"\n  --- DYN=64 デフォルトデータ ---")
        default_pairs = find_pairs(cfg.DYN_DEFAULT_WAV_DIR, cfg.DYN_DEFAULT_VSQX_DIR)
        if default_pairs:
            X_default, y_default = extract_dyn_scale_data(
                default_pairs, extractor, reader, baseline_model_data,
                force_dyn_value=64
            )
            X_dyn = np.vstack([X_dyn, X_default])
            y_dyn = np.concatenate([y_dyn, y_default])
            print(f"  デフォルトデータ追加完了: {len(X_default)} フレーム")
    else:
        print(f"\n  [情報] デフォルトデータフォルダが見つかりません。DYN=64なしで学習します。")

    print(f"\n  合計学習フレーム数 (抽出直後): {len(X_dyn)}")

    # メモリ枯渇防止のためのサンプリング
    if len(X_dyn) > cfg.MAX_TRAIN_FRAMES_PHASE2:
        print(f"  [情報] メモリ制限のため、{len(X_dyn)} フレームから {cfg.MAX_TRAIN_FRAMES_PHASE2} フレームにサンプリングします。")
        np.random.seed(42)
        indices = np.random.choice(len(X_dyn), cfg.MAX_TRAIN_FRAMES_PHASE2, replace=False)
        X_dyn = X_dyn[indices]
        y_dyn = y_dyn[indices]

    print(f"  ΔMel形状: {X_dyn.shape}")
    print(f"  DYN値の分布: {np.unique(y_dyn)}")

    # Ridge回帰モデルの学習（多項式特徴量あり/なし）
    print(f"\n[Step 2-3] DYNスケールモデルの学習...")

    scaler_dyn = StandardScaler()
    X_dyn_scaled = scaler_dyn.fit_transform(X_dyn)

    poly_transformer = None
    if cfg.USE_POLYNOMIAL:
        from sklearn.preprocessing import PolynomialFeatures
        poly_transformer = PolynomialFeatures(
            degree=cfg.POLYNOMIAL_DEGREE,
            interaction_only=False,
            include_bias=False
        )
        X_dyn_poly = poly_transformer.fit_transform(X_dyn_scaled)
        print(f"  多項式変換後の次元数: {X_dyn_poly.shape[1]}")
    else:
        X_dyn_poly = X_dyn_scaled

    model_dyn = Ridge(alpha=10.0)
    model_dyn.fit(X_dyn_poly, y_dyn)

    # 学習精度の確認
    y_pred_dyn = model_dyn.predict(X_dyn_poly)
    dyn_residuals = y_dyn - y_pred_dyn
    dyn_rmse = np.sqrt(np.mean(dyn_residuals ** 2))
    dyn_mae = np.mean(np.abs(dyn_residuals))
    dyn_r_squared = model_dyn.score(X_dyn_poly, y_dyn)

    print(f"  R² スコア: {dyn_r_squared:.4f}")
    print(f"  RMSE: {dyn_rmse:.4f}")
    print(f"  MAE: {dyn_mae:.4f}")

    # DYN値ごとの精度
    for dv in sorted(np.unique(y_dyn)):
        mask = y_dyn == dv
        pred_dv = y_pred_dyn[mask]
        mae_dv = np.mean(np.abs(pred_dv - dv))
        print(f"    DYN={int(dv):>3d}: 予測平均={pred_dv.mean():.1f}, MAE={mae_dv:.2f}")

    # モデルの保存
    print(f"\n[Step 2-4] DYNスケールモデルの保存...")
    dyn_model_data = {
        "model": model_dyn,
        "scaler": scaler_dyn,
        "poly_transformer": poly_transformer,
        "r_squared": dyn_r_squared,
        "rmse": dyn_rmse,
        "mae": dyn_mae,
        "n_training_frames": len(X_dyn),
        "dyn_values_used": sorted(np.unique(y_dyn).tolist()),
    }
    joblib.dump(dyn_model_data, cfg.DYN_MODEL_PATH)
    print(f"  保存完了: {cfg.DYN_MODEL_PATH}")

    # =============================================
    # 完了
    # =============================================
    print(f"\n{'='*60}")
    print(f"  全ての学習が完了しました！")
    print(f"  ベースラインモデル: {cfg.BASELINE_MODEL_PATH}")
    print(f"  DYNスケールモデル: {cfg.DYN_MODEL_PATH}")
    print(f"  次のステップ: predict_dyn.py を実行してください。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
