"""
plot_rendered_eval.py
====================
Vocaloidソフトでレンダリングされた予測音声(WAV)および予測VSQXと、
正解音声(WAV)および正解VSQXの比較プロットを生成する評価スクリプト。

比較プロットのレイアウト (evaluate.py 規定に準拠):
    上段: DYN パラメータ比較 (正解 VSQX DYN vs 予測 VSQX DYN)
    下段: RMS 音響特徴量比較 (正解 WAV RMS vs 予測 WAV RMS)

使い方:
    1. predict.py で出力された予測VSQXをVocaloidソフトで読み込み、WAVとして書き出す
    2. 予測WAVを predict_config.py の PRED_WAV_SEARCH_DIR に配置する
    3. python plot_rendered_eval.py を実行する
"""

import os
import sys
import json
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib as mpl

# 日本語フォント
mpl.rcParams["font.family"] = ["Meiryo", "MS Gothic", "sans-serif"]

# --- パス設定 ---
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "util"))

from feature_extractor import AudioFeatureExtractor
from vsqx_dnn_formatter import VSQXDNNFormatter
from train_util.evaluate import Evaluator

# 設定の読み込み
import predict_config as cfg


def find_file_recursive(search_dir, filename_stem, extension):
    """search_dir 以下を再帰的に検索し、指定名のファイルを探す。"""
    pattern = os.path.join(search_dir, "**", f"{filename_stem}{extension}")
    found = glob.glob(pattern, recursive=True)
    if len(found) == 0:
        raise FileNotFoundError(
            f"'{filename_stem}{extension}' が見つかりません。\n"
            f"  検索フォルダ: {os.path.abspath(search_dir)}"
        )
    elif len(found) > 1:
        paths_str = "\n".join(f"    - {os.path.abspath(p)}" for p in found)
        raise ValueError(
            f"'{filename_stem}{extension}' が複数見つかりました。\n"
            f"  検索フォルダ: {os.path.abspath(search_dir)}\n"
            f"  該当ファイル:\n{paths_str}"
        )
    resolved = os.path.abspath(found[0])
    print(f"  Found: {resolved}")
    return resolved


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("plot_rendered_eval.py - レンダリング後の正解・予測比較プロット")
    print("=" * 60)

    # --- 1. features.json から sr, shift_length 等の取得 ---
    if not os.path.exists(cfg.FEATURES_JSON_PATH):
        print(f"[Error] features.json not found: {os.path.abspath(cfg.FEATURES_JSON_PATH)}")
        return

    with open(cfg.FEATURES_JSON_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read().replace("NaN", "null")
        feature_info = json.loads(raw_text)

    sr = feature_info.get("sr", 44100)
    shift_length = feature_info.get("shift_length", 512)
    x_feature_map = feature_info.get("x_feature_map", {})
    y_feature_map = feature_info.get("y_feature_map", {})
    audio_global_mean = feature_info.get("audio_global_mean", {})
    audio_global_std = feature_info.get("audio_global_std", {})

    target_param_names = []
    for branch_name, feat_list in y_feature_map.items():
        for feat in feat_list:
            target_param_names.append(feat["name"])

    # --- 2. ファイル検索 ---
    errors = []
    gt_wav_path = gt_vsqx_path = pred_wav_path = pred_vsqx_path = None

    print(f"\n対象曲名: {cfg.INPUT_FILE_NAME}")

    # A. 正解WAV
    print(f"正解WAV検索中: {os.path.abspath(cfg.WAV_SEARCH_DIR)}")
    try:
        gt_wav_path = find_file_recursive(cfg.WAV_SEARCH_DIR, cfg.INPUT_FILE_NAME, ".wav")
    except (FileNotFoundError, ValueError) as e:
        errors.append(f"Ground Truth WAV: {e}")

    # B. 正解VSQX
    print(f"正解VSQX検索中: {os.path.abspath(cfg.VSQX_SEARCH_DIR)}")
    try:
        gt_vsqx_path = find_file_recursive(cfg.VSQX_SEARCH_DIR, cfg.INPUT_FILE_NAME, ".vsqx")
    except (FileNotFoundError, ValueError) as e:
        errors.append(f"Ground Truth VSQX: {e}")

    # C. 予測VSQX (predict.pyの出力ファイル)
    pred_vsqx_name = f"{cfg.INPUT_FILE_NAME}_predict"
    print(f"予測VSQX検索中: {os.path.abspath(cfg.OUTPUT_DIR)}")
    try:
        pred_vsqx_path = find_file_recursive(cfg.OUTPUT_DIR, pred_vsqx_name, ".vsqx")
    except (FileNotFoundError, ValueError) as e:
        errors.append(f"Predicted VSQX: {e}")

    # D. レンダリング済み予測WAV
    # "{INPUT_FILE_NAME}_predict.wav" または "{INPUT_FILE_NAME}.wav" を検索
    print(f"予測WAV検索中: {os.path.abspath(cfg.PRED_WAV_SEARCH_DIR)}")
    try:
        try:
            pred_wav_path = find_file_recursive(cfg.PRED_WAV_SEARCH_DIR, pred_vsqx_name, ".wav")
        except FileNotFoundError:
            pred_wav_path = find_file_recursive(cfg.PRED_WAV_SEARCH_DIR, cfg.INPUT_FILE_NAME, ".wav")
    except (FileNotFoundError, ValueError) as e:
        errors.append(f"Rendered Predicted WAV: {e}")

    if errors:
        print("\n" + "=" * 60)
        for e in errors:
            print(f"[Error] {e}")
        print("=" * 60)
        print("処理を中止します。上記ファイルを用意した上で再実行してください。")
        return

    # --- 3. RMS特徴量の抽出 (正解WAV vs 予測WAV) ---
    print("\nWAVデータからRMS特徴量を抽出中...")
    audio_extractor = AudioFeatureExtractor(sr=sr, shift_length=shift_length)

    gt_rms = audio_extractor.get_features(gt_wav_path, ["rms"])
    pred_rms = audio_extractor.get_features(pred_wav_path, ["rms"])

    min_frames = min(len(gt_rms), len(pred_rms))
    gt_rms = gt_rms[:min_frames]
    pred_rms = pred_rms[:min_frames]

    # x_feature_map から "rms" を含む branch_name を検索
    rms_branch_name = None
    for b_name, feat_list in x_feature_map.items():
        feat_names = [f["name"] if isinstance(f, dict) else f for f in feat_list]
        if "rms" in feat_names:
            rms_branch_name = b_name
            break

    # 指定コードと同一のキー参照および標準化計算
    if rms_branch_name and rms_branch_name in audio_global_mean:
        mean = np.array(audio_global_mean[rms_branch_name])
        std = np.array(audio_global_std[rms_branch_name])
    else:
        mean = np.array(audio_global_mean.get("rms", 0.0))
        std = np.array(audio_global_std.get("rms", 1.0))

    gt_rms_norm = (gt_rms - mean) / (std + 1e-7)
    gt_rms_norm = np.nan_to_num(gt_rms_norm).astype(np.float32)

    pred_rms_norm = (pred_rms - mean) / (std + 1e-7)
    pred_rms_norm = np.nan_to_num(pred_rms_norm).astype(np.float32)

    if gt_rms_norm.ndim > 1:
        gt_rms_norm = gt_rms_norm[:, 0]
    if pred_rms_norm.ndim > 1:
        pred_rms_norm = pred_rms_norm[:, 0]

    # --- 4. DYN等のパラメータ抽出 (正解VSQX vs 予測VSQX) ---
    print("VSQXデータからパラメータを抽出中...")
    vsqx_extractor = VSQXDNNFormatter(sr=sr, shift_length=shift_length)

    # y_feature_map からパラメータ名（文字列）のみを抽出した辞書を作成して渡す
    target_param_config = {
        branch: [feat["name"] if isinstance(feat, dict) else feat for feat in feat_list]
        for branch, feat_list in y_feature_map.items()
    }
    vsqx_extractor.vsqx_config["TARGET_PARAM_CONFIG"] = target_param_config

    # max_note_frames の復元 (features.json から設定。存在しない場合は学習スケール不整合防止のためエラー)
    if "max_note_frames" in feature_info:
        vsqx_extractor.max_note_frames = int(feature_info["max_note_frames"])
    else:
        raise ValueError(
            "features.json に 'max_note_frames' が含まれていません。\n"
            "学習時と評価時で正規化スケールの齟齬を防ぐため、build_dataset.py を再実行して最新の features.json を使用してください。"
        )

    gt_vsqx_raw = vsqx_extractor.parse_vsqx(gt_vsqx_path, min_frames)
    pred_vsqx_raw = vsqx_extractor.parse_vsqx(pred_vsqx_path, min_frames)

    gt_vsqx_norm = vsqx_extractor.normalize_features(gt_vsqx_raw)
    pred_vsqx_norm = vsqx_extractor.normalize_features(pred_vsqx_raw)

    voicing_mask = gt_vsqx_norm.get("voicing_mask", np.ones((min_frames, 1), dtype=np.float32))
    v_mask = voicing_mask.squeeze(-1) if voicing_mask.ndim > 1 else voicing_mask

    # --- 5. 比較グラフの作成 ---
    print("\n比較グラフを描画中...")

    # evaluate.py と同一のレイアウト構築
    # 上段: パラメータ比較 (DYN等), 下段: 音響特徴量比較 (RMS)
    num_params = len(target_param_names)
    num_extra = 1  # RMS
    total_plots = num_params + num_extra

    time_axis = np.arange(min_frames) * shift_length / sr

    fig, axes = plt.subplots(
        total_plots, 1, figsize=(12, 3.5 * total_plots), sharex=True
    )
    if total_plots == 1:
        axes = [axes]

    # 上段: パラメータ比較
    for i, pname in enumerate(target_param_names):
        ax = axes[i]
        gt_param = gt_vsqx_raw["dynamic"].get(pname, np.full(min_frames, 64.0))[:min_frames]
        pred_param = pred_vsqx_raw["dynamic"].get(pname, np.full(min_frames, 64.0))[:min_frames]

        # 無音区間のシャドウ描画
        voiced_idx = v_mask > 0.5
        mae = np.mean(np.abs(gt_param[voiced_idx] - pred_param[voiced_idx])) if np.any(voiced_idx) else 0.0

        # 無音区間シェード
        in_unvoiced = False
        start_t = 0.0
        for idx_m, v in enumerate(v_mask):
            if v < 0.5 and not in_unvoiced:
                in_unvoiced = True
                start_t = time_axis[idx_m]
            elif v >= 0.5 and in_unvoiced:
                in_unvoiced = False
                ax.axvspan(start_t, time_axis[idx_m], alpha=0.18, color="#9E9E9E")
        if in_unvoiced:
            ax.axvspan(start_t, time_axis[-1], alpha=0.18, color="#9E9E9E")

        ax.plot(time_axis, gt_param, label="正解 VSQX DYN", color="steelblue", linewidth=1.5)
        ax.plot(time_axis, pred_param, label="予測 VSQX DYN", color="coral", linewidth=1.5, linestyle="--")

        ax.set_title(f"DYN パラメータ比較: {cfg.INPUT_FILE_NAME}  |  MAE = {mae:.2f}")
        ax.set_ylabel("DYN Value (0~127)")
        ax.set_ylim(0, 127)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    # 下段: RMS 音響特徴量比較 (正解WAV vs 予測WAV)
    ax_rms = axes[num_params]

    voiced_idx = v_mask > 0.5
    rms_mae = np.mean(np.abs(gt_rms_norm[voiced_idx] - pred_rms_norm[voiced_idx])) if np.any(voiced_idx) else 0.0

    in_unvoiced = False
    start_t = 0.0
    for idx_m, v in enumerate(v_mask):
        if v < 0.5 and not in_unvoiced:
            in_unvoiced = True
            start_t = time_axis[idx_m]
        elif v >= 0.5 and in_unvoiced:
            in_unvoiced = False
            ax_rms.axvspan(start_t, time_axis[idx_m], alpha=0.18, color="#9E9E9E")
    if in_unvoiced:
        ax_rms.axvspan(start_t, time_axis[-1], alpha=0.18, color="#9E9E9E")

    ax_rms.plot(time_axis, gt_rms_norm, label="正解 WAV RMS (標準化)", color="steelblue", linewidth=1.5)
    ax_rms.plot(time_axis, pred_rms_norm, label="予測 WAV RMS (標準化)", color="coral", linewidth=1.5, linestyle="--")

    ax_rms.set_title(f"RMS 音響特徴量比較 (標準化): {cfg.INPUT_FILE_NAME}  |  MAE = {rms_mae:.4f}")
    ax_rms.set_xlabel("時間 (秒)")
    ax_rms.set_ylabel("RMS (Z-score)")
    ax_rms.legend(loc="upper right")
    ax_rms.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.OUTPUT_DIR, f"rendered_eval_{cfg.INPUT_FILE_NAME}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("=== 比較プロット作成完了 ===")
    print(f"  正解WAV:   {gt_wav_path}")
    print(f"  予測WAV:   {pred_wav_path}")
    print(f"  正解VSQX:  {gt_vsqx_path}")
    print(f"  予測VSQX:  {pred_vsqx_path}")
    print(f"  出力グラフ: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
