import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# プロジェクトルートのパスを追加してfeature_extractorとvsqx_parserをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "util"))
from feature_extractor import AudioFeatureExtractor
from vsqx_parser import VSQXParser

# ==========================================
# CONFIG領域：WAV/VSQXファイル群が格納されたディレクトリを指定してください
# ==========================================
CONFIG = {
    "WAV_DIR": "./data/ノート開始点終了点テスト",
    "WAV_FILES": [
        "あ_32分音符.wav",
        "あ_16分音符.wav",
        "あ_8分音符.wav",
        "あ_4分音符.wav",
        "あ_2分音符.wav",
        "あ_全音符.wav",
    ],
    "VSQX_FILES": [
        "あ_32分音符.vsqx",
        "あ_16分音符.vsqx",
        "あ_8分音符.vsqx",
        "あ_4分音符.vsqx",
        "あ_2分音符.vsqx",
        "あ_全音符.vsqx",
    ],
    "LABELS": ["32", "16", "8", "4", "2", "1"],
    "SR": 44100,
    "HOP_LENGTH": 512,
    "COMPARE_FRAMES": 20,  # 比較抽出フレーム（約348ms）
}
# ==========================================


def verify_envelope():
    wav_dir = CONFIG["WAV_DIR"]
    wav_files = CONFIG["WAV_FILES"]
    vsqx_files = CONFIG["VSQX_FILES"]
    labels = CONFIG["LABELS"]
    sr = CONFIG["SR"]
    hop_length = CONFIG["HOP_LENGTH"]
    compare_frames = CONFIG["COMPARE_FRAMES"]

    # インスタンス化
    extractor = AudioFeatureExtractor(sr=sr, shift_length=hop_length)
    vsqx_parser = VSQXParser(sr=sr, shift_length=hop_length)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, file_name in enumerate(wav_files):
        wav_path = os.path.join(wav_dir, file_name)
        vsqx_path = os.path.join(wav_dir, vsqx_files[i])
        label = labels[i]

        if not os.path.exists(wav_path) or not os.path.exists(vsqx_path):
            print(f"警告: ファイルが見つかりません: {file_name} または {vsqx_files[i]}")
            continue

        print(f"読み込み中: {file_name} & {vsqx_files[i]}")

        # AudioFeatureExtractorを使用してRMS（dB）を取得
        rms = extractor.get_features(wav_path, "rms").flatten()

        # VSQXパースによる絶対時間の取得
        vsqx_data = vsqx_parser.read_sparse(vsqx_path)
        if not vsqx_data or not vsqx_data["notes"]:
            print(f"警告: VSQXからノート情報が取得できませんでした: {vsqx_files[i]}")
            continue

        note = vsqx_data["notes"][0]
        tempo_map = vsqx_data["tempo_map"]
        resolution = vsqx_data["resolution"]

        # Tick から 秒数への変換
        start_sec = vsqx_parser.tick_to_second(
            note["abs_start_tick"], tempo_map, resolution
        )
        end_sec = vsqx_parser.tick_to_second(
            note["abs_end_tick"], tempo_map, resolution
        )

        # 秒数 から フレームへの変換
        start_frame = int(start_sec * sr / hop_length)
        end_frame = int(end_sec * sr / hop_length)

        # ==========================================
        # アタック部分（VSQXノートオン位置合わせ）
        # ==========================================
        attack_end = min(len(rms), start_frame + compare_frames)
        attack_rms = rms[start_frame:attack_end]

        # x軸を絶対時間(ms)に変換
        x_attack_ms = np.arange(len(attack_rms)) * hop_length / sr * 1000
        axes[0].plot(x_attack_ms, attack_rms, label=label, marker=".")

        # ==========================================
        # リリース部分（VSQXノートオフ地点合わせ）
        # ==========================================
        # ノートオフの前後を切り出す（前30フレーム≒348ms 〜 後20フレーム≒232ms）
        pre_frames = compare_frames
        post_frames = 20

        rel_start = max(0, end_frame - pre_frames)
        rel_end = min(len(rms), end_frame + post_frames)

        release_rms = rms[rel_start:rel_end]

        # X軸の計算：end_frame（ノートオフの瞬間）がピッタリ X=0ms になるようにする
        offset_frames = rel_start - end_frame
        x_release_ms = (
            (np.arange(len(release_rms)) + offset_frames) * hop_length / sr * 1000
        )

        axes[1].plot(x_release_ms, release_rms, label=label, marker=".")

    # グラフの装飾
    axes[0].set_title("Attack Phase (Aligned at VSQX Note On)")
    axes[0].set_xlabel("Time from VSQX Note On (ms)")
    axes[0].set_ylabel("RMS Energy (dB)")
    axes[0].grid(True)
    axes[0].legend()

    # 縦線でノートオフ位置を明示
    axes[1].axvline(
        x=0, color="r", linestyle="--", alpha=0.5, label="VSQX Note Off (0 ms)"
    )
    axes[1].set_title("Release Phase (Aligned at VSQX Note Off)")
    axes[1].set_xlabel("Time to/from VSQX Note Off (ms)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verify_envelope()
