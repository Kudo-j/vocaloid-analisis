"""
test_sync.py
============
WAVファイルとVSQXファイルの同期（タイムラグ）を確認するためのテストスクリプト。
WAVのMelパワー（RMS）と、VSQXから取得したノート区間を重ねてプロットします。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa

# 日本語フォントの設定（Windows環境向け）
plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'sans-serif']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))

from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader

def main():
    print("=" * 50)
    print("  Synchronization Test")
    print("=" * 50)

    # テスト対象
    wav_path = os.path.join(_PROJECT_ROOT, "src_result", "Para_Study_data", "wav", "default", "あ_default.wav")
    vsqx_path = os.path.join(_PROJECT_ROOT, "src_result", "Para_Study_data", "vsqx", "default", "あ_default.vsqx")
    
    output_path = os.path.join(SCRIPT_DIR, "output", "test_sync_result.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"WAV: {wav_path}")
    print(f"VSQX: {vsqx_path}")

    # 特徴量抽出
    extractor = AudioFeatureExtractor(sr=44100, hop_length=256)
    try:
        mel = extractor.get_features(wav_path, ["mel"])
        # Melの全帯域平均をRMSの代わりに用いる
        rms = np.mean(mel, axis=1)
    except Exception as e:
        print(f"WAVの読み込みに失敗しました: {e}")
        return

    # VSQX読み込み
    reader = VSQXReader(sr=44100, hop_length=256)
    vsqx_data = reader.read(vsqx_path)
    if vsqx_data is None:
        print("VSQXの読み込みに失敗しました")
        return

    notes = vsqx_data.get("notes", [])
    
    # 時間軸の作成
    time_axis = librosa.frames_to_time(np.arange(len(rms)), sr=44100, hop_length=256)

    # 描画
    plt.figure(figsize=(14, 6))
    
    # 音声パワーの描画
    plt.plot(time_axis, rms, label="Audio Power (Mel Mean)", color="tab:blue")
    
    # ノート区間の描画
    for i, note in enumerate(notes):
        s_sec = note["start_sec"]
        e_sec = note["end_sec"]
        lyric = note["lyric"]
        
        # 背景を塗りつぶす
        plt.axvspan(s_sec, e_sec, color="green", alpha=0.2, label="Note Segment" if i == 0 else "")
        plt.axvline(x=s_sec, color="green", linestyle=":", alpha=0.8)
        plt.axvline(x=e_sec, color="red", linestyle=":", alpha=0.8)
        
        # 歌詞テキスト
        ymin, ymax = plt.ylim()
        plt.text(s_sec + (e_sec - s_sec) / 2, ymax - (ymax - ymin) * 0.1, lyric,
                 ha='center', va='top', color='darkgreen', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    plt.xlabel("Time (sec)")
    plt.ylabel("Power (dB)")
    plt.title(f"Time Synchronization Check: {os.path.basename(wav_path)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"\nテスト完了！結果のグラフを保存しました:\n{output_path}")

if __name__ == "__main__":
    main()
