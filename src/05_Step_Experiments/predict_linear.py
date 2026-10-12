"""
predict_linear.py
=================
学習済み線形モデル (.pkl) を使い、WAV + VSQX から直接推論を行うスクリプト。

出力:
    1. パラメータごとの3枚組グラフ (全体 / 動き最大5秒 / 誤差最大5秒)
    2. 予測値を書き込んだ新しい VSQX ファイル

CONTEXT_SIZE 等の学習設定は .pkl から自動取得するため、
config で指定する必要があるのはパスと曲名のみ。
"""

import os
import re
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import joblib

# 日本語フォント
mpl.rcParams["font.family"] = ["Meiryo", "MS Gothic", "sans-serif"]

# --- パス設定 ---
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "util"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_Preprocess"))

from feature_extractor import AudioFeatureExtractor
from vsqx_dnn_formatter import VSQXDNNFormatter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
#  config 領域
# =============================================================================

# --- 推論対象 ---
INPUT_FILE_NAME = "あ_DYN_Train_1"  # 拡張子なしのファイル名

# --- ファイル検索ディレクトリ ---
WAV_SEARCH_DIR = "./data/Step1/wav"
VSQX_SEARCH_DIR = "./data/Step1/vsqx"

# --- 学習済みモデル ---
MODEL_PATH = os.path.join(
    BASE_DIR, "output", "output_linear", "Context_0", "Step1_RMS_Linear_model.pkl"
)

# --- features.json (正規化パラメータ) ---
FEATURES_JSON_PATH = "./processed/Step1/features.json"

# --- 出力先 ---
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "predict_linear")

# --- VSQX書き込み時の音声開始Tick ---
AUDIO_START_TICK = 1920 * 4

# =============================================================================
# config 領域ここまで
# =============================================================================

# --- 正規表現パターン (VSQX書き込み用) ---
RE_RESOLUTION = re.compile(r"<resolution>(\d+)</resolution>")
RE_TEMPO_V = re.compile(r"<tempo>\s*<t>0</t>\s*<v>(\d+)</v>\s*</tempo>", re.DOTALL)
RE_VS_PART = re.compile(r"<vsPart>.*?</vsPart>", re.DOTALL)
RE_PART_T = re.compile(r"<vsPart>\s*<t>(\d+)</t>")
RE_SINGER = re.compile(r"(<singer>.*?</singer>)", re.DOTALL)
RE_CC_TAG = re.compile(r'<cc>.*?<v id="([A-Z]+)">.*?</cc>', re.DOTALL)
RE_NOTE = re.compile(r"<note>.*?</note>", re.DOTALL)


# =============================================================================
# ユーティリティ関数
# =============================================================================


def denormalize_dyn_to_int(norm_val):
    """DYN の正規化値 (-1~1) を VSQX 用の整数 (0~127) に変換する。"""
    raw = (norm_val + 1.0) / 2.0 * 127.0
    return int(np.clip(np.round(raw), 0, 127))


def denormalize_array(arr, pname):
    """正規化された配列を元のスケールに戻す (グラフ表示用、浮動小数)。"""
    if pname == "PIT":
        return arr * 8192.0
    elif pname == "PBS":
        return (arr + 1.0) / 2.0 * 24.0
    elif pname in ["fallPort", "risePort"]:
        return np.clip(arr * 127.0, 0, 127)
    elif pname == "vibType":
        return arr
    else:
        # DYN, BRI, CLE 等
        return np.clip((arr + 1.0) / 2.0 * 127.0, 0, 127)


def _apply_context_window(X, context_size):
    """文脈窓を適用する (analyze_linear.py と同一ロジック)。"""
    if context_size == 0:
        return X
    N, D = X.shape
    X_padded = np.pad(X, ((context_size, context_size), (0, 0)), mode="edge")
    window_size = 2 * context_size + 1
    X_ctx = np.zeros((N, D * window_size), dtype=X.dtype)
    for i in range(window_size):
        X_ctx[:, i * D : (i + 1) * D] = X_padded[i : i + N]
    return X_ctx


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


# =============================================================================
# グラフ描画
# =============================================================================


def _shade_unvoiced(ax, time_axis, vm_1d):
    """無声区間をグレー背景で塗る。"""
    in_unvoiced = False
    start_t = 0.0
    added = False
    for i, v in enumerate(vm_1d):
        if v < 0.5 and not in_unvoiced:
            in_unvoiced = True
            start_t = time_axis[i]
        elif v >= 0.5 and in_unvoiced:
            in_unvoiced = False
            lbl = "Unvoiced" if not added else None
            ax.axvspan(start_t, time_axis[i], alpha=0.18, color="#9E9E9E", label=lbl)
            added = True
    if in_unvoiced:
        lbl = "Unvoiced" if not added else None
        ax.axvspan(start_t, time_axis[-1], alpha=0.18, color="#9E9E9E", label=lbl)


def plot_3panel(
    denorm_gt, denorm_pred, vm, time_axis, pid, mae, save_path, sr, shift_length
):
    """3枚組グラフ (全体 / 動き最大5秒 / 誤差最大5秒) を描画する。"""
    min_len = len(denorm_gt)
    w = min(int(5.0 * sr / shift_length), min_len)

    # (A) 正解値の動き最大
    gt_diff = np.abs(np.diff(denorm_gt, prepend=denorm_gt[0]))
    cum_diff = np.cumsum(gt_diff)
    window_sums = cum_diff[w:] - cum_diff[:-w]
    max_move_start = int(np.argmax(window_sums)) if len(window_sums) > 0 else 0
    max_move_end = max_move_start + w

    # (B) 予測誤差最大 (有声のみ)
    abs_error = np.abs(denorm_pred - denorm_gt) * vm
    cum_err = np.cumsum(abs_error)
    err_sums = cum_err[w:] - cum_err[:-w]
    max_err_start = int(np.argmax(err_sums)) if len(err_sums) > 0 else 0
    max_err_end = max_err_start + w

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle(f"{pid}  |  MAE = {mae:.2f}", fontsize=16, fontweight="bold", y=0.98)

    # --- グラフ1: 全体 ---
    ax1 = axes[0]
    _shade_unvoiced(ax1, time_axis, vm)
    ax1.plot(
        time_axis,
        denorm_gt,
        label="Ground Truth",
        color="#2196F3",
        alpha=0.7,
        linewidth=1.0,
    )
    ax1.plot(
        time_axis,
        denorm_pred,
        label="Predicted",
        color="#FF5722",
        alpha=0.7,
        linewidth=1.0,
    )
    ax1.axvspan(
        time_axis[max_move_start],
        time_axis[min(max_move_end, min_len - 1)],
        alpha=0.12,
        color="green",
        label="Max Movement (5s)",
    )
    ax1.axvspan(
        time_axis[max_err_start],
        time_axis[min(max_err_end, min_len - 1)],
        alpha=0.12,
        color="red",
        label="Max Error (5s)",
    )
    ax1.set_title("Full Comparison", fontsize=13)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(pid)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --- グラフ2: 動き最大5秒 ---
    ax2 = axes[1]
    sl2 = slice(max_move_start, max_move_end)
    vm2, t2, gt2, pred2 = vm[sl2], time_axis[sl2], denorm_gt[sl2], denorm_pred[sl2]
    voiced2 = vm2 >= 0.5
    local_mae2 = (
        np.mean(np.abs(pred2[voiced2] - gt2[voiced2])) if np.any(voiced2) else 0.0
    )
    _shade_unvoiced(ax2, t2, vm2)
    ax2.plot(t2, gt2, label="Ground Truth", color="#2196F3", alpha=0.8, linewidth=1.5)
    ax2.plot(t2, pred2, label="Predicted", color="#FF5722", alpha=0.8, linewidth=1.5)
    ax2.fill_between(t2, gt2, pred2, alpha=0.15, color="orange")
    ax2.set_title(
        f"Max Ground Truth Movement (5s)  |  Local MAE = {local_mae2:.2f}", fontsize=13
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(pid)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # --- グラフ3: 誤差最大5秒 ---
    ax3 = axes[2]
    sl3 = slice(max_err_start, max_err_end)
    vm3, t3, gt3, pred3 = vm[sl3], time_axis[sl3], denorm_gt[sl3], denorm_pred[sl3]
    voiced3 = vm3 >= 0.5
    local_mae3 = (
        np.mean(np.abs(pred3[voiced3] - gt3[voiced3])) if np.any(voiced3) else 0.0
    )
    _shade_unvoiced(ax3, t3, vm3)
    ax3.plot(t3, gt3, label="Ground Truth", color="#2196F3", alpha=0.8, linewidth=1.5)
    ax3.plot(t3, pred3, label="Predicted", color="#FF5722", alpha=0.8, linewidth=1.5)
    ax3.fill_between(t3, gt3, pred3, alpha=0.15, color="red")
    ax3.set_title(
        f"Max Prediction Error (5s)  |  Local MAE = {local_mae3:.2f}", fontsize=13
    )
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel(pid)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


# =============================================================================
# VSQX 書き込み
# =============================================================================


def write_vsqx(
    vsqx_path, output_path, predictions_int, sr, shift_length, audio_start_tick
):
    """
    予測値(整数)をVSQXファイルに書き込む。
    predict_en.py の update_vsqx_xml を線形モデル用に簡略化したもの。
    """
    # VSQXパーサーからID_MAPPINGを取得
    _parser = VSQXDNNFormatter(sr=sr, shift_length=shift_length)
    inv_map = {v: k for k, v in _parser.vsqx_config["ID_MAPPING"].items()}

    try:
        with open(vsqx_path, "r", encoding="utf-8") as f:
            content = f.read()

        res_match = RE_RESOLUTION.search(content)
        resolution = int(res_match.group(1)) if res_match else 480
        tempo_match = RE_TEMPO_V.search(content)
        bpm = int(tempo_match.group(1)) / 100.0 if tempo_match else 120.0

        part_match = RE_VS_PART.search(content)
        if not part_match:
            print("Error: vsPart not found.")
            return
        part_content = part_match.group(0)

        part_t_match = RE_PART_T.search(part_content)
        if not part_t_match:
            print("Error: Part start tick not found.")
            return
        part_start_tick = int(part_t_match.group(1))

        # 対象パラメータの既存CCタグを削除
        target_tag_ids = set()
        for param_id in predictions_int.keys():
            tag_id = inv_map.get(param_id)
            if tag_id:
                target_tag_ids.add(tag_id)

        placeholders = []

        def _mask_note(match):
            placeholders.append(match.group(0))
            return f"__NOTE_PLACEHOLDER_{len(placeholders)-1}__"

        masked_part = RE_NOTE.sub(_mask_note, part_content)

        def _remove_target_cc(match):
            if match.group(1) in target_tag_ids:
                return ""
            return match.group(0)

        cleaned_part = RE_CC_TAG.sub(_remove_target_cc, masked_part)

        for i, note_str in enumerate(placeholders):
            cleaned_part = cleaned_part.replace(f"__NOTE_PLACEHOLDER_{i}__", note_str)

        # 予測値のCC文字列を生成
        cc_strings = []
        for param_id, int_values in predictions_int.items():
            tag_id_str = inv_map.get(param_id)
            if not tag_id_str:
                continue
            last_val = -9999
            for frame_idx, int_val in enumerate(int_values):
                if int_val == last_val:
                    continue
                time_sec = librosa.frames_to_time(
                    frame_idx, sr=sr, hop_length=shift_length
                )
                audio_tick = time_sec * bpm * resolution / 60.0
                rel_tick = int(audio_tick + audio_start_tick - part_start_tick)
                if rel_tick < 0:
                    continue
                cc_strings.append(
                    f"\t\t\t<cc>\n"
                    f"\t\t\t\t<t>{rel_tick}</t>\n"
                    f'\t\t\t\t<v id="{tag_id_str}">{int_val}</v>\n'
                    f"\t\t\t</cc>"
                )
                last_val = int_val

        if cc_strings:
            final_cc_block = "\n".join(cc_strings)
            augmented_part = RE_SINGER.sub(
                rf"\1\n{final_cc_block}", cleaned_part, count=1
            )
        else:
            augmented_part = cleaned_part

        final_content = content.replace(part_content, augmented_part, 1)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"VSQX saved: {output_path}")

    except Exception as e:
        print(f"Error writing VSQX: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# メイン処理
# =============================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("predict_linear.py - 線形モデルによる推論")
    print("=" * 60)

    # --- 1. モデルのロード ---
    print(f"\nモデル読み込み中: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: モデルファイルが見つかりません: {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)

    # モデルに埋め込まれたメタデータを取得
    context_size = getattr(model, "context_size", 0)
    method = getattr(model, "method", "unknown")
    data_dir = getattr(model, "data_dir", "")
    model_feature_names = getattr(model, "feature_names", None)

    print(f"  手法: {method}")
    print(f"  文脈窓: {context_size}")
    if model_feature_names:
        print(f"  特徴量名 (先頭5): {model_feature_names[:5]}")

    # --- 2. features.json のロード ---
    print(f"\nfeatures.json 読み込み中: {FEATURES_JSON_PATH}")
    if not os.path.exists(FEATURES_JSON_PATH):
        print(f"Error: features.json が見つかりません: {FEATURES_JSON_PATH}")
        return
    with open(FEATURES_JSON_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read().replace("NaN", "null")
        feature_info = json.loads(raw_text)

    sr = feature_info.get("sr", 44100)
    shift_length = feature_info.get("shift_length", 512)
    audio_global_mean = feature_info.get("audio_global_mean", {})
    audio_global_std = feature_info.get("audio_global_std", {})

    # 入力音響特徴量の種類を特定
    # features.json の x_feature_map から音響ブランチを特定
    x_feature_map = feature_info.get("x_feature_map", {})
    input_vsqx_params = feature_info.get("input_vsqx_params", [])

    # 出力パラメータ名
    y_feature_map = feature_info.get("y_feature_map", {})
    target_param_names = []
    for branch_name, features in y_feature_map.items():
        for feat in features:
            target_param_names.append(feat["name"])

    print(f"  SR: {sr}, HOP: {shift_length}")
    print(f"  出力パラメータ: {target_param_names}")

    # --- 3. ファイル検索 ---
    errors = []
    wav_path = vsqx_path = None

    print(f"\n入力ファイル名: {INPUT_FILE_NAME}")
    print(f"WAV検索中: {os.path.abspath(WAV_SEARCH_DIR)}")
    try:
        wav_path = find_file_recursive(WAV_SEARCH_DIR, INPUT_FILE_NAME, ".wav")
    except (FileNotFoundError, ValueError) as e:
        errors.append(str(e))

    print(f"VSQX検索中: {os.path.abspath(VSQX_SEARCH_DIR)}")
    try:
        vsqx_path = find_file_recursive(VSQX_SEARCH_DIR, INPUT_FILE_NAME, ".vsqx")
    except (FileNotFoundError, ValueError) as e:
        errors.append(str(e))

    if errors:
        for e in errors:
            print(f"[Error] {e}")
        return

    # --- 4. 特徴量の抽出と正規化 ---
    print("\n音響特徴量を抽出中...")
    audio_extractor = AudioFeatureExtractor(sr=sr, shift_length=shift_length)

    # ブランチごとに特徴量を抽出・正規化・結合
    x_branches = {}
    total_frames = 0

    for branch_name, feat_list in x_feature_map.items():
        # 音響ブランチかVSQXブランチかを判定
        feature_types = [f["name"] for f in feat_list]

        # 音響特徴量の場合 (rms, mel, mfcc 等)
        is_audio_branch = any(
            ft
            in [
                "rms",
                "mel",
                "mfcc",
                "cqt",
                "stft",
                "f0_world",
                "f0_crepe",
                "source_rms",
                "source_mel",
                "source_stft",
                "hnr",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_contrast",
                "zero_crossing_rate",
                "cepstrum",
                "smoothed_cepstrum",
            ]
            for ft in feature_types
        )

        if is_audio_branch:
            features = audio_extractor.get_features(wav_path, feature_types)
            total_frames = features.shape[0]

            # features.json の統計量で正規化
            mean = np.array(audio_global_mean.get(branch_name, 0.0))
            std = np.array(audio_global_std.get(branch_name, 1.0))
            features = (features - mean) / (std + 1e-7)
            features = np.nan_to_num(features).astype(np.float64)

            x_branches[branch_name] = features
            print(f"  {branch_name}: {feature_types} -> shape {features.shape}")

    # --- 5. VSQXからの正解データと発声マスクの取得 ---
    print("VSQXから正解データを抽出中...")
    vsqx_extractor = VSQXDNNFormatter(sr=sr, shift_length=shift_length)

    # config の注入 (build_dataset.py と同じ)
    import config as preprocess_config

    vsqx_extractor.vsqx_config["INPUT_VSQX_PARAMS"] = (
        preprocess_config.INPUT_VSQX_PARAMS
    )
    vsqx_extractor.vsqx_config["TARGET_PARAM_CONFIG"] = (
        preprocess_config.TARGET_PARAM_CONFIG
    )

    # max_note_frames の復元 (features.json から設定。存在しない場合は学習スケール不整合防止のためエラー)
    if "max_note_frames" in feature_info:
        vsqx_extractor.max_note_frames = int(feature_info["max_note_frames"])
    else:
        raise ValueError(
            "features.json に 'max_note_frames' が含まれていません。\n"
            "学習時と推論時で正規化スケールの齟齬を防ぐため、build_dataset.py を再実行して最新の features.json を使用してください。"
        )

    raw_vsqx = vsqx_extractor.parse_vsqx(vsqx_path, total_frames)
    if raw_vsqx is None:
        print("Error: VSQXのパースに失敗しました。")
        return
    norm_vsqx = vsqx_extractor.normalize_features(raw_vsqx)

    # 発声マスク
    voicing_mask = norm_vsqx.get(
        "voicing_mask", np.ones((total_frames, 1), dtype=np.float32)
    )
    vm = voicing_mask.squeeze(-1)

    # VSQXの入力パラメータがある場合はそれも結合
    for branch_name, p_list_config in preprocess_config.INPUT_VSQX_PARAMS.items():
        if not p_list_config:
            continue
        branch_data_list = []
        for pid in p_list_config:
            if pid in norm_vsqx:
                branch_data_list.append(norm_vsqx[pid])
        if branch_data_list:
            branch_arr = np.concatenate(branch_data_list, axis=-1).astype(np.float64)
            x_branches[branch_name] = branch_arr

    # 入力を結合 (x_で始まるブランチ名のソート順で結合)
    sorted_branch_names = sorted(x_branches.keys())
    X_input = np.concatenate([x_branches[k] for k in sorted_branch_names], axis=-1)
    print(f"入力配列: {X_input.shape}")

    # --- 6. 文脈窓の適用 ---
    if context_size > 0:
        X_input = _apply_context_window(X_input, context_size)
        print(f"文脈窓適用後: {X_input.shape}")

    # --- 7. 推論 ---
    print("\n推論実行中...")
    Y_pred = model.predict(X_input)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred[:, np.newaxis]
    print(f"予測出力: {Y_pred.shape}")

    # --- 8. 正解値の取得 ---
    Y_true_dict = {}
    for pid in target_param_names:
        gt = norm_vsqx.get(pid)
        if gt is not None:
            if gt.ndim == 2 and gt.shape[1] == 1:
                gt = gt.flatten()
            Y_true_dict[pid] = gt

    # --- 9. グラフ描画 ---
    print("\n3枚組グラフを描画中...")
    time_axis = librosa.frames_to_time(
        np.arange(total_frames), sr=sr, hop_length=shift_length
    )
    voiced_idx = vm >= 0.5

    col_idx = 0
    for pid in target_param_names:
        if pid not in Y_true_dict:
            col_idx += 1
            continue

        gt_norm = Y_true_dict[pid]
        pred_norm = Y_pred[:, col_idx].flatten()
        min_len = min(len(gt_norm), len(pred_norm), len(vm))
        gt_norm = gt_norm[:min_len]
        pred_norm = pred_norm[:min_len]
        vm_clip = vm[:min_len]
        ta_clip = time_axis[:min_len]

        denorm_gt = denormalize_array(gt_norm, pid)
        denorm_pred = denormalize_array(pred_norm, pid)
        vi = vm_clip >= 0.5

        mae = np.mean(np.abs(denorm_pred[vi] - denorm_gt[vi])) if np.any(vi) else 0.0

        save_path = os.path.join(OUTPUT_DIR, f"{INPUT_FILE_NAME}_{pid}.png")
        plot_3panel(
            denorm_gt,
            denorm_pred,
            vm_clip,
            ta_clip,
            pid,
            mae,
            save_path,
            sr,
            shift_length,
        )

        print(f"  {pid}: MAE = {mae:.2f}")
        col_idx += 1

    # --- 10. VSQX への書き込み ---
    print("\nVSQX書き込み中...")
    predictions_int = {}
    col_idx = 0
    for pid in target_param_names:
        pred_norm = Y_pred[:, col_idx].flatten()
        # 完全に整数化 (小数混入によるVSQX形式不備を防止)
        int_values = [denormalize_dyn_to_int(v) for v in pred_norm]
        predictions_int[pid] = int_values
        col_idx += 1

    vsqx_output_path = os.path.join(OUTPUT_DIR, f"{INPUT_FILE_NAME}_predict.vsqx")
    write_vsqx(
        vsqx_path, vsqx_output_path, predictions_int, sr, shift_length, AUDIO_START_TICK
    )

    print("\n" + "=" * 60)
    print("=== 処理完了 ===")
    print(f"  WAV入力:   {wav_path}")
    print(f"  VSQX入力:  {vsqx_path}")
    print(f"  VSQX出力:  {vsqx_output_path}")
    print(f"  グラフ出力: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
