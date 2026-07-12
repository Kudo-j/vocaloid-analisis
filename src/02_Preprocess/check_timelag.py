import os
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from scipy.ndimage import binary_opening, binary_closing
import pandas as pd
import shutil  # ★追加: ファイル移動用
import config

plt.rcParams["font.family"] = "Meiryo"

# --- 設定パラメータ ---
# DATA_DIR = "./processed/processed_only_mel/l"  # .npzがあるフォルダ
DATA_DIR = config.OUTPUT_DIR
WAV_DIR = "./data/wav_ALL"  # .wavがあるフォルダ
FEATURES_JSON = os.path.join(DATA_DIR, "features.json")

# ★追加: 異常値を移動させる先のフォルダ
OUTLIER_DIR = os.path.join(DATA_DIR, "outlier")
# ★追加: 実際にファイルを移動させるかどうか (Falseなら移動せずリスト出力のみ)
MOVE_OUTLIERS = True

# 分析設定
CHECK_NUM_FILES = 0  # 詳細チェック(画像作成)する数
ANALYSIS_NUM_FILES = 5000  # 統計チェックする数
SEARCH_WINDOW_SEC = 1.0  # 探索範囲 ±n秒
CORR_THRESHOLD = -1.0  # 信頼度足切りライン
OUTLIER_THRESHOLD = 5  # このフレーム数以上ずれていたら移動対象
# ★追加: 平均的なズレの補正値（マイナスのズレが多いので、マイナス値を設定）
# これにより、判定の中心が「-4フレーム」になります。
LAG_OFFSET = -4


def load_feature_info():
    if not os.path.exists(FEATURES_JSON):
        print(f"Error: {FEATURES_JSON} not found.")
        return None
    with open(FEATURES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_indices(feature_info):
    x_map = feature_info["x_feature_map"]
    indices = {
        "audio": {"start": -1, "dim": 0},
        "note": {"start": -1, "dim": 0},
        "phoneme": {"start": -1, "dim": 0},
    }
    current_idx = 0
    for feat in x_map:
        name = feat["name"]
        dim = feat["dim"]
        if "Audio" in name:
            indices["audio"] = {"start": current_idx, "dim": dim}
        elif "NOTE_NUMBER" in name:
            indices["note"] = {"start": current_idx, "dim": dim}
        elif "phoneme" in name.lower() or "PHONEME" in name:
            indices["phoneme"] = {"start": current_idx, "dim": dim}
        current_idx += dim
    return indices


def binarize_audio_envelope(
    audio_env, percentile=20, min_gap_frames=5, min_dur_frames=3
):
    """
    音響エンベロープをバイナリゲート(発声=1/無音=0)に変換する。
    特徴量の種類(mel/stft/cqt等)に依存しない汎用的な判定を実現。

    手順:
    1. パーセンタイルベースの適応閾値でバイナリ化
    2. モルフォロジー処理で微小な穴・スパイクを除去し安定化

    Args:
        audio_env: 音響エンベロープ (1D array)
        percentile: ノイズフロア推定用パーセンタイル (低い値=保守的)
        min_gap_frames: この長さ以下の無音区間(穴)を埋める
        min_dur_frames: この長さ以下の発声区間(スパイク)を除去
    Returns:
        binary gate (1D array, 0.0 or 1.0)
    """
    if np.max(audio_env) - np.min(audio_env) < 1e-8:
        # 全く変動がない場合 (全無音や全一定値)
        return np.zeros_like(audio_env)

    # --- 1. 適応閾値の算出 ---
    # ノイズフロア: 下位パーセンタイルの値
    noise_floor = np.percentile(audio_env, percentile)
    # ピークレベル: 上位の値
    peak_level = np.percentile(audio_env, 95)

    # 閾値 = ノイズフロアとピークの中間点（やや低め）
    # ノイズフロアから30%上の地点を閾値にする
    threshold = noise_floor + (peak_level - noise_floor) * 0.3

    # --- 2. バイナリ化 ---
    gate = (audio_env > threshold).astype(bool)

    # --- 3. モルフォロジー処理 ---
    # closing: 短い無音区間(穴)を埋める
    if min_gap_frames > 0:
        closing_struct = np.ones(min_gap_frames * 2 + 1, dtype=bool)
        gate = binary_closing(gate, structure=closing_struct)

    # opening: 短い発声スパイク(ノイズ)を除去
    if min_dur_frames > 0:
        opening_struct = np.ones(min_dur_frames * 2 + 1, dtype=bool)
        gate = binary_opening(gate, structure=opening_struct)

    return gate.astype(np.float64)


def extract_f0_from_wav(wav_path, sr, hop_length, length_frames):
    try:
        y, _ = librosa.load(wav_path, sr=sr)
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length,
        )
        f0 = np.nan_to_num(f0)
        if len(f0) < length_frames:
            f0 = np.pad(f0, (0, length_frames - len(f0)))
        else:
            f0 = f0[:length_frames]
        return f0
    except Exception as e:
        print(f"Warning: Failed to extract f0 from {wav_path}: {e}")
        return np.zeros(length_frames)


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
    valid_mask = (lags * sec_per_frame >= -max_lag_sec) & (
        lags * sec_per_frame <= max_lag_sec
    )

    if not np.any(valid_mask):
        return 0, 0.0

    masked_corr = corr.copy()
    masked_corr[~valid_mask] = -np.inf
    best_idx = np.argmax(masked_corr)
    max_corr_val = masked_corr[best_idx] / len(signal_a)
    return lags[best_idx], max_corr_val


def visualize_detail(
    name, audio_env, audio_gate, note_gate, wav_f0, lag_info, ms_per_frame
):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 1. Audio Envelope + Binary Gate の比較
    ax = axes[0]
    env_norm = audio_env / (np.max(np.abs(audio_env)) + 1e-8)
    ax.plot(env_norm, color="gray", alpha=0.5, label="Audio Env (raw)")
    ax.plot(audio_gate, color="green", alpha=0.8, label="Audio Gate (binary)")
    ax.set_title(f"1. Audio Envelope → Binary Gate ({name})")
    ax.legend(loc="upper right")

    # 2. Audio Gate vs Note Gate の同期
    ax = axes[1]
    ax.plot(audio_gate, color="green", alpha=0.6, label="Audio Gate")
    ax.plot(note_gate, color="blue", alpha=0.8, linestyle="--", label="Voicing Mask")
    ax.set_title(f"2. Gate Sync (Lag: {lag_info['note_lag']*ms_per_frame:.1f} ms)")
    ax.legend(loc="upper right")

    # 3. Pitch curve (参考)
    ax = axes[2]
    if np.sum(wav_f0) > 0:
        f0_norm = wav_f0 / (np.max(wav_f0) + 1e-6)
        ax.plot(f0_norm, color="black", alpha=0.7, label="Wav F0")
        ax.plot(note_gate, color="red", alpha=0.6, linestyle="--", label="Voicing Mask")
        ax.set_title(
            f"3. Pitch Curve Check (Lag: {lag_info['pitch_lag']*ms_per_frame:.1f} ms)"
        )
    else:
        ax.text(0.5, 0.5, "No F0 Data", ha="center")
        ax.set_title("3. Pitch Curve Check")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"debug_detail_{name}.png")
    plt.close()


def move_outliers(outliers_df):
    """異常値ファイルを指定フォルダに移動する"""
    if outliers_df.empty:
        return

    # 移動先フォルダ作成
    if not os.path.exists(OUTLIER_DIR):
        os.makedirs(OUTLIER_DIR)
        print(f"Created directory: {OUTLIER_DIR}")

    print(f"\nMoving {len(outliers_df)} files to {OUTLIER_DIR} ...")

    count = 0
    for _, row in outliers_df.iterrows():
        filename = row["name"]

        # 1. NPZの移動
        src_npz = os.path.join(DATA_DIR, f"{filename}.npz")
        dst_npz = os.path.join(OUTLIER_DIR, f"{filename}.npz")

        try:
            if os.path.exists(src_npz):
                shutil.move(src_npz, dst_npz)

                # 2. WAVもあれば一緒に移動 (セットで管理したいため)
                src_wav = os.path.join(WAV_DIR, f"{filename}.wav")
                dst_wav = os.path.join(OUTLIER_DIR, f"{filename}.wav")
                if os.path.exists(src_wav):
                    shutil.move(src_wav, dst_wav)

                count += 1
            else:
                print(f"  Warning: File not found {src_npz}")
        except Exception as e:
            print(f"  Error moving {filename}: {e}")

    print(f"Moved {count} file sets successfully.")


def analyze_alignment():
    info = load_feature_info()
    if info is None:
        return

    indices = get_feature_indices(info)
    sr = info.get("sr", 24000)
    hop = info.get("hop_length", 128)
    sec_per_frame = hop / sr
    ms_per_frame = sec_per_frame * 1000

    print(f"SR: {sr}, Hop: {hop} ({ms_per_frame:.2f} ms/frame)")

    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    if not npz_files:
        return
    random.shuffle(npz_files)

    # --- Phase 1: 詳細チェック ---
    print(f"\n=== Phase 1: 詳細ピッチ確認 ({CHECK_NUM_FILES} files) ===")
    for fpath in npz_files[:CHECK_NUM_FILES]:
        try:
            filename = os.path.basename(fpath).replace(".npz", "")
            with np.load(fpath) as data:
                x = data["x"]
                # 必要なデータを取り出したらインデントの中で処理
                # ★修正: preprocess.pyで新たに保存するようにした 'audio_env' があればそれを使う
                if "audio_env" in data:
                    audio_env = data["audio_env"].flatten()
                else:
                    # 過去のデータとの互換性用（特徴量から直接平均を取る）
                    idx_a, dim_a = indices["audio"]["start"], indices["audio"]["dim"]
                    audio_env = np.mean(x[:, idx_a : idx_a + dim_a], axis=1)

                # 【修正】xから探すのではなく、voicing_maskを直接ゲートとして使う
                if "voicing_mask" in data:
                    note_gate = data["voicing_mask"].flatten()
                else:
                    note_gate = np.zeros_like(audio_env)

            # ★改善: audio_envをバイナリゲートに変換
            audio_gate = binarize_audio_envelope(audio_env)

            wav_path = os.path.join(WAV_DIR, f"{filename}.wav")
            wav_f0 = np.zeros_like(audio_env)
            pitch_lag = 0
            if os.path.exists(wav_path):
                wav_f0 = extract_f0_from_wav(wav_path, sr, hop, len(audio_env))
                if np.sum(wav_f0) > 0 and np.sum(note_gate) != 0:
                    pitch_lag, _ = calc_lag_and_corr(
                        wav_f0, note_gate, sec_per_frame, SEARCH_WINDOW_SEC
                    )

            note_lag, _ = calc_lag_and_corr(
                audio_gate, note_gate, sec_per_frame, SEARCH_WINDOW_SEC
            )
            visualize_detail(
                filename,
                audio_env,
                audio_gate,
                note_gate,
                wav_f0,
                {"note_lag": note_lag, "pitch_lag": pitch_lag},
                ms_per_frame,
            )
            print(f"  Processed {filename}")
        except Exception as e:
            print(f"  Error in detail check: {e}")

    # --- Phase 2: 全体統計分析 ---
    print(
        f"\n=== Phase 2: 全体統計分析 ({min(ANALYSIS_NUM_FILES, len(npz_files))} files) ==="
    )
    results = []
    target_files = npz_files[:ANALYSIS_NUM_FILES]

    for i, fpath in enumerate(target_files):
        if i % 100 == 0:
            print(f"  Scanning... {i}")
        try:
            filename = os.path.basename(fpath).replace(".npz", "")
            with np.load(fpath) as data:
                x = data["x"]
                # 必要なデータを取り出したらインデントの中で処理
                # ★修正: preprocess.pyで新たに保存するようにした 'audio_env' があればそれを使う
                if "audio_env" in data:
                    audio_env = data["audio_env"].flatten()
                else:
                    # 過去のデータとの互換性用
                    idx_a, dim_a = indices["audio"]["start"], indices["audio"]["dim"]
                    audio_env = np.mean(x[:, idx_a : idx_a + dim_a], axis=1)

                # ▼ Phase 1 と同じように voicing_mask を使う
                if "voicing_mask" in data:
                    note_gate = data["voicing_mask"].flatten()
                else:
                    note_gate = np.zeros_like(audio_env)
            # ★改善: audio_envをバイナリゲートに変換
            audio_gate = binarize_audio_envelope(audio_env)
            lag, corr = calc_lag_and_corr(
                audio_gate, note_gate, sec_per_frame, SEARCH_WINDOW_SEC
            )
            results.append(
                {
                    "name": filename,
                    "lag_frame": lag,
                    "lag_ms": lag * ms_per_frame,
                    "corr": corr,
                }
            )
        except:
            continue

    df = pd.DataFrame(results)
    if len(df) == 0:
        return
    df_valid = df[df["corr"] >= CORR_THRESHOLD].copy()
    # ★修正: オフセットを引いて「補正後のズレ」を計算
    # 例: 実際が-4, オフセットが-4 → 補正後 0 (ど真ん中)
    # 例: 実際が-16, オフセットが-4 → 補正後 -12 (アウト)
    df_valid["corrected_lag"] = df_valid["lag_frame"] - LAG_OFFSET
    df_valid["abs_lag"] = df_valid["corrected_lag"].abs()  # ← corrected_lag に変更！

    # 集計表示
    # テキスト統計
    print("\n" + "=" * 50)
    print("【同期ズレ集計結果】")

    # ずれ件数表示
    for i in range(4):
        if i < 3:
            cnt = len(df_valid[df_valid["abs_lag"] == i])
            label = f"{i}フレーム"
        else:
            cnt = len(df_valid[df_valid["abs_lag"] >= i])
            label = f"{i}フレーム以上"

        ratio = 0 if len(df_valid) == 0 else (cnt / len(df_valid) * 100)
        print(f"  {label:<10} : {cnt:4d} 件 ({ratio:5.1f}%)")

    # グラフ保存
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(df_valid["lag_frame"], bins=50, color="#4F81BD", edgecolor="black")
    ax1.set_title("ズレ分布")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(df_valid["lag_frame"], df_valid["corr"], alpha=0.5, s=15)
    ax2.set_title("信頼度 vs ズレ")
    plt.tight_layout()
    plt.savefig("alignment_stats.png")
    print("Graphs saved.")

    # ★追加: 異常ファイルの移動
    outliers = df_valid[df_valid["abs_lag"] >= OUTLIER_THRESHOLD]
    if not outliers.empty:
        print(
            f"\n【検出】 {OUTLIER_THRESHOLD}フレーム以上ずれているファイル: {len(outliers)}件"
        )
        outliers.to_csv("alignment_outliers.csv", index=False)
        print("  CSV saved.")

        if MOVE_OUTLIERS:
            move_outliers(outliers)
        else:
            print("  ※移動機能はOFFになっています (MOVE_OUTLIERS = False)")
    else:
        print("\n  異常ファイルはありませんでした。")


if __name__ == "__main__":
    analyze_alignment()
