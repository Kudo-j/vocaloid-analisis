import os
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import binary_opening, binary_closing
import pandas as pd
import shutil

import config_diffDYN as cfg

plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]

# --- 分析設定 ---
CHECK_NUM_FILES = 3        # 詳細チェック(画像作成)する数
ANALYSIS_NUM_FILES = 5000  # 統計チェックする数
SEARCH_WINDOW_SEC = 1.0    # 探索範囲 ±n秒
CORR_THRESHOLD = -1.0      # 信頼度足切りライン
OUTLIER_THRESHOLD = 5      # このフレーム数以上ずれていたら異常とみなす
LAG_OFFSET = -4            # 許容されるベースのズレ（VOCALOIDのエンジン起因の遅れ）

MOVE_OUTLIERS = True       # 実際にファイルを移動させるかどうか

def binarize_audio_envelope(audio_env, percentile=20, min_gap_frames=5, min_dur_frames=3):
    """
    音響エンベロープをバイナリゲート(発声=1/無音=0)に変換する。
    """
    if np.max(audio_env) - np.min(audio_env) < 1e-8:
        return np.zeros_like(audio_env)

    noise_floor = np.percentile(audio_env, percentile)
    peak_level = np.percentile(audio_env, 95)
    threshold = noise_floor + (peak_level - noise_floor) * 0.3

    gate = (audio_env > threshold).astype(bool)

    if min_gap_frames > 0:
        gate = binary_closing(gate, structure=np.ones(min_gap_frames * 2 + 1, dtype=bool))
    if min_dur_frames > 0:
        gate = binary_opening(gate, structure=np.ones(min_dur_frames * 2 + 1, dtype=bool))

    return gate.astype(np.float64)

def calc_lag_and_corr(signal_a, signal_b, sec_per_frame, max_lag_sec):
    if np.std(signal_a) > 1e-6:
        signal_a = (signal_a - np.mean(signal_a)) / np.std(signal_a)
    else:
        return 0, 0.0
    if np.std(signal_b) > 1e-6:
        signal_b = (signal_b - np.mean(signal_b)) / np.std(signal_b)
    else:
        return 0, 0.0

    corr = signal.correlate(signal_a, signal_b, mode="full")
    lags = signal.correlation_lags(len(signal_a), len(signal_b), mode="full")
    valid_mask = (lags * sec_per_frame >= -max_lag_sec) & (lags * sec_per_frame <= max_lag_sec)

    if not np.any(valid_mask):
        return 0, 0.0

    masked_corr = corr.copy()
    masked_corr[~valid_mask] = -np.inf
    best_idx = np.argmax(masked_corr)
    max_corr_val = masked_corr[best_idx] / len(signal_a)
    return lags[best_idx], max_corr_val

def visualize_detail(name, env_t, gate_t, env_ut, gate_ut, note_gate, lag_info, ms_per_frame, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 1. Tuned Audio
    ax = axes[0]
    env_t_norm = (env_t - np.min(env_t)) / (np.ptp(env_t) + 1e-8)
    ax.plot(env_t_norm, color="gray", alpha=0.5, label="Audio Env (Tuned)")
    ax.plot(gate_t, color="green", alpha=0.8, label="Audio Gate (Tuned)")
    ax.set_title(f"1. Tuned WAV vs VSQX (Lag: {lag_info['lag_t']} frames) - {name}")
    ax.legend(loc="upper right")

    # 2. Untuned Audio
    ax = axes[1]
    env_ut_norm = (env_ut - np.min(env_ut)) / (np.ptp(env_ut) + 1e-8)
    ax.plot(env_ut_norm, color="gray", alpha=0.5, label="Audio Env (Untuned)")
    ax.plot(gate_ut, color="orange", alpha=0.8, label="Audio Gate (Untuned)")
    ax.set_title(f"2. Untuned WAV vs VSQX (Lag: {lag_info['lag_ut']} frames)")
    ax.legend(loc="upper right")

    # 3. All Gates Sync
    ax = axes[2]
    ax.plot(gate_t, color="green", alpha=0.6, label="Gate Tuned")
    ax.plot(gate_ut, color="orange", alpha=0.6, label="Gate Untuned")
    ax.plot(note_gate, color="blue", alpha=0.8, linestyle="--", label="Note Gate (VSQX Mask)")
    ax.set_title(f"3. Gate Sync Comparison")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_timelag_{name}.png"))
    plt.close()

def move_outliers(outliers_df, npz_dir):
    """異常値ファイルを指定フォルダに移動する"""
    if outliers_df.empty:
        return

    outlier_dir = os.path.join(npz_dir, "outlier")
    if not os.path.exists(outlier_dir):
        os.makedirs(outlier_dir)
        print(f"Created directory: {outlier_dir}")

    print(f"\nMoving {len(outliers_df)} files to {outlier_dir} ...")

    count = 0
    for _, row in outliers_df.iterrows():
        filename = row["name"]
        src_npz = os.path.join(npz_dir, f"{filename}.npz")
        dst_npz = os.path.join(outlier_dir, f"{filename}.npz")

        try:
            if os.path.exists(src_npz):
                shutil.move(src_npz, dst_npz)
                count += 1
            else:
                print(f"  Warning: File not found {src_npz}")
        except Exception as e:
            print(f"  Error moving {filename}: {e}")

    print(f"Moved {count} files successfully.")

def main():
    print("=" * 60)
    print("  DYN DNN Estimation - Timelag Check")
    print("=" * 60)

    npz_dir = cfg.PREPROCESSED_DIR
    out_dir = cfg.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"対象フォルダ: {npz_dir}")

    features_json = os.path.join(npz_dir, "features.json")
    if not os.path.exists(features_json):
        print("[エラー] features.jsonが見つかりません。")
        return
    with open(features_json, "r", encoding="utf-8") as f:
        info = json.load(f)

    sr = info.get("sr", cfg.SR)
    hop = info.get("hop_length", cfg.HOP_LENGTH)
    sec_per_frame = hop / sr
    ms_per_frame = sec_per_frame * 1000

    print(f"SR: {sr}, Hop: {hop} ({ms_per_frame:.2f} ms/frame)")

    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    if not npz_files:
        print("[エラー] NPZファイルが見つかりません。")
        return
    
    # 常に同じ順番でチェックするためソートしてからシャッフル（シード固定）
    npz_files.sort()
    random.seed(42)
    random.shuffle(npz_files)

    print(f"\n=== Phase 1: 詳細チェック ({CHECK_NUM_FILES} files) ===")
    for fpath in npz_files[:CHECK_NUM_FILES]:
        try:
            filename = os.path.basename(fpath).replace(".npz", "")
            with np.load(fpath) as data:
                mel_t = data["mel_tuned"]
                mel_ut = data["mel_untuned"]
                note_gate = data["voicing_mask"].flatten()

            env_t = np.mean(mel_t, axis=1)
            gate_t = binarize_audio_envelope(env_t)
            lag_t, _ = calc_lag_and_corr(gate_t, note_gate, sec_per_frame, SEARCH_WINDOW_SEC)

            env_ut = np.mean(mel_ut, axis=1)
            gate_ut = binarize_audio_envelope(env_ut)
            lag_ut, _ = calc_lag_and_corr(gate_ut, note_gate, sec_per_frame, SEARCH_WINDOW_SEC)
            
            lag_info = {"lag_t": lag_t, "lag_ut": lag_ut}
            visualize_detail(filename, env_t, gate_t, env_ut, gate_ut, note_gate, lag_info, ms_per_frame, out_dir)
            print(f"  Processed {filename} -> Lag Tuned: {lag_t} frames, Lag Untuned: {lag_ut} frames")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n=== Phase 2: 全体統計分析 ({min(ANALYSIS_NUM_FILES, len(npz_files))} files) ===")
    results = []
    for i, fpath in enumerate(npz_files[:ANALYSIS_NUM_FILES]):
        if i > 0 and i % 100 == 0:
            print(f"  Scanning... {i}/{len(npz_files)}")
        try:
            filename = os.path.basename(fpath).replace(".npz", "")
            with np.load(fpath) as data:
                mel_t = data["mel_tuned"]
                mel_ut = data["mel_untuned"]
                note_gate = data["voicing_mask"].flatten()

            env_t = np.mean(mel_t, axis=1)
            gate_t = binarize_audio_envelope(env_t)
            lag_t, corr_t = calc_lag_and_corr(gate_t, note_gate, sec_per_frame, SEARCH_WINDOW_SEC)

            env_ut = np.mean(mel_ut, axis=1)
            gate_ut = binarize_audio_envelope(env_ut)
            lag_ut, corr_ut = calc_lag_and_corr(gate_ut, note_gate, sec_per_frame, SEARCH_WINDOW_SEC)
            
            # オフセットを引いて、VOCALOID特有の遅れを補正した「真のズレ」を計算
            corrected_lag_t = lag_t - LAG_OFFSET
            corrected_lag_ut = lag_ut - LAG_OFFSET
            
            results.append({
                "name": filename,
                "lag_t": lag_t,
                "corr_t": corr_t,
                "lag_ut": lag_ut,
                "corr_ut": corr_ut,
                # 補正後のズレの絶対値のうち、大きい方を「代表ズレ」として記録
                "max_abs_lag": max(abs(corrected_lag_t), abs(corrected_lag_ut))
            })
        except:
            continue

    df = pd.DataFrame(results)
    if len(df) == 0:
        return
    
    # 信頼度の低いデータ（どちらかが相関ゼロ等）を除外
    df_valid = df[(df["corr_t"] >= CORR_THRESHOLD) & (df["corr_ut"] >= CORR_THRESHOLD)].copy()

    print("\n" + "=" * 50)
    print("【同期ズレ集計結果 (Tuned / Untuned 総合)】")
    for i in range(4):
        if i < 3:
            cnt = len(df_valid[df_valid["max_abs_lag"] == i])
            label = f"{i}フレーム"
        else:
            cnt = len(df_valid[df_valid["max_abs_lag"] >= i])
            label = f"{i}フレーム以上"
        ratio = 0 if len(df_valid) == 0 else (cnt / len(df_valid) * 100)
        print(f"  {label:<10} : {cnt:4d} 件 ({ratio:5.1f}%)")

    # グラフ保存 (Tuned と Untuned の2つの分布を重ねて表示)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(df_valid["lag_t"], bins=50, color="green", alpha=0.5, label="Tuned Lag")
    ax1.hist(df_valid["lag_ut"], bins=50, color="orange", alpha=0.5, label="Untuned Lag")
    ax1.set_title("ズレ分布")
    ax1.set_xlabel("Lag (frames)")
    ax1.legend()
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(df_valid["lag_t"], df_valid["corr_t"], color="green", alpha=0.5, s=15, label="Tuned")
    ax2.scatter(df_valid["lag_ut"], df_valid["corr_ut"], color="orange", alpha=0.5, s=15, label="Untuned")
    ax2.set_title("信頼度 vs ズレ")
    ax2.set_xlabel("Lag (frames)")
    ax2.set_ylabel("Correlation")
    
    plt.tight_layout()
    stats_img = os.path.join(out_dir, "alignment_stats.png")
    plt.savefig(stats_img)
    print(f"Graphs saved: {stats_img}")

    outliers = df_valid[df_valid["max_abs_lag"] >= OUTLIER_THRESHOLD]
    if not outliers.empty:
        print(f"\n【検出】 {OUTLIER_THRESHOLD}フレーム以上ずれているファイル: {len(outliers)}件")
        csv_path = os.path.join(out_dir, "alignment_outliers.csv")
        outliers.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path}")
        
        if MOVE_OUTLIERS:
            move_outliers(outliers, npz_dir)
        else:
            print("  ※移動機能はOFFになっています (MOVE_OUTLIERS = False)")
    else:
        print("\n  異常ファイルはありませんでした。同期は完璧です！")

if __name__ == "__main__":
    main()
