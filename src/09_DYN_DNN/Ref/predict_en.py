import os
import re
import json
import glob
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace  # 引数オブジェクト用
from scipy import stats
from typing import Dict, List, Any, Optional

# ユーザー定義モジュール
import sys
import os

# 02_Preprocess フォルダへのパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "02_Preprocess")))

import config
from preprocess import VSQXPreprocessor
from model_encoder import TransformerModel
from train_encoder import (
    D_MODEL,
    NHEAD,
    NUM_ENCODER_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
)
from dataset import CHUNK_SIZE  # 学習時と同じチャンクサイズを使用

INPUT_FILE_NAME = "フロムナインス_Fromninth_noedit_Original_Soft_aug_1"  # ★ 拡張子なしのファイル名
DATA_NAME = "augDYN_DYN_mfccsrcrms"

# スクリプトの1つ上のディレクトリ(predict/)を基準にする
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 入力ファイル検索フォルダ
WAV_SEARCH_DIR = "./data/wav_ALL"
VSQX_SEARCH_DIR = "./data/vsqx_ALL"

# 出力ファイル
OUTPUT_DIR = "./predict/output"
VSQX_OUTPUT = os.path.join(OUTPUT_DIR, f"{DATA_NAME}_{INPUT_FILE_NAME}_predict.vsqx")
PLOT_OUTPUT = os.path.join(OUTPUT_DIR, f"{DATA_NAME}_{INPUT_FILE_NAME}.png")

# 学習済みモデルとメタデータ
MODEL_PATH = f"./predict/models/{DATA_NAME}_model.pth"
FEATURES_JSON_PATH = f"./processed/{DATA_NAME}/features.json"

# --- 正規表現パターン (テキストベース書き込み用) ---
RE_RESOLUTION = re.compile(r"<resolution>(\d+)</resolution>")
RE_TEMPO_V = re.compile(r"<tempo>\s*<t>0</t>\s*<v>(\d+)</v>\s*</tempo>", re.DOTALL)
RE_VS_PART = re.compile(r"<vsPart>.*?</vsPart>", re.DOTALL)
RE_PART_T = re.compile(r"<vsPart>\s*<t>(\d+)</t>")
RE_SINGER = re.compile(r"(<singer>.*?</singer>)", re.DOTALL)
RE_CC_TAG = re.compile(r'<cc>.*?<v id="([A-Z]+)">.*?</cc>', re.DOTALL)
RE_NOTE = re.compile(r"<note>.*?</note>", re.DOTALL)

AUDIO_START_TICK = 1920 * 4

# =============================================================================
# ユーティリティ関数
# =============================================================================


def denormalize_value(value: float, param_id: str) -> int:
    """
    正規化された値(-1.0 ~ 1.0など)を、VSQX用の整数値(0~127など)に戻す。
    configの設定値を使用する。
    """
    # 1. Dynamic Parameters
    if param_id in config.DYNAMIC_PARAM_IDS:
        if param_id == "PIT":
            # PITは -8192 ~ 8191 (中心0)
            return int(value * 8192.0)
        elif param_id == "PBS":
            # PBSは 0 ~ 24 (正規化は 0~24 を -1~1 にしている想定)
            return int((value + 1.0) / 2.0 * 24.0)
        else:
            # 一般的なCC (0~127)
            val = (value + 1.0) / 2.0 * 127.0
            return int(np.clip(val, 0, 127))

    # 2. Note Parameters
    elif param_id in config.NOTE_PARAM_IDS:
        max_val = config.NOTE_PARAM_MAX_VALUES.get(param_id, 127.0)
        if param_id in ["fallPort", "risePort"]:
            # 0.0 ~ 1.0
            return int(np.clip(value * max_val, 0, 127))
        elif param_id == "vibType":
            return int(value)
        else:
            # -1 ~ 1 -> 0 ~ max
            val = (value + 1.0) / 2.0 * max_val
            return int(np.clip(val, 0, int(max_val)))

    # デフォルト
    return int(value)


def update_vsqx_xml(
    vsqx_path: str,
    output_path: str,
    predictions: Dict[str, np.ndarray],
    sr: int,
    hop_length: int,
    audio_start_tick: int = 1920 * 4,
):
    """
    テキストベース（正規表現）でVSQXファイルを書き換える関数。
    lxmlを使わず、プレーンテキストとして処理することで、CDATA・XML宣言・
    インデントなど元ファイルのフォーマットを完全に保持する。
    (DPara_add_AllTrack.py の書き込み方式を踏襲)

    処理内容:
    1. ファイルをテキストとして読み込む
    2. 予測対象のタグIDに該当する既存<cc>タグを正規表現で削除する
    3. 予測値を新しい<cc>タグ文字列として生成し、<singer>直後に挿入する
    4. テキストとしてそのまま書き出す
    """
    try:
        with open(vsqx_path, "r", encoding="utf-8") as f:
            content = f.read()

        # --- メタ情報の取得 ---
        res_match = RE_RESOLUTION.search(content)
        resolution = int(res_match.group(1)) if res_match else 480

        tempo_match = RE_TEMPO_V.search(content)
        bpm = int(tempo_match.group(1)) / 100.0 if tempo_match else 120.0

        # --- vsPart の処理 ---
        part_match = RE_VS_PART.search(content)
        if not part_match:
            print("Error: vsPart not found in VSQX.")
            return
        part_content = part_match.group(0)

        # パート開始Tickの取得
        part_t_match = RE_PART_T.search(part_content)
        if not part_t_match:
            print("Error: Part start tick not found.")
            return
        part_start_tick = int(part_t_match.group(1))

        # --- 1. 既存パラメータの削除 ---
        # 予測対象のタグID一覧 (DYN -> D)
        inv_map = {v: k for k, v in config.ID_MAPPING.items()}
        target_tag_ids = set()
        for param_id in predictions.keys():
            tag_id = inv_map.get(param_id)
            if tag_id:
                target_tag_ids.add(tag_id)

        # ノートをプレースホルダで保護してからCCを削除する
        # (DPara_add_AllTrack.py と同じ手法)
        placeholders = []

        def _mask_note(match):
            placeholders.append(match.group(0))
            return f"__NOTE_PLACEHOLDER_{len(placeholders)-1}__"

        masked_part = RE_NOTE.sub(_mask_note, part_content)

        removed_count = 0

        def _remove_target_cc(match):
            nonlocal removed_count
            cc_id = match.group(1)
            if cc_id in target_tag_ids:
                removed_count += 1
                return ""
            return match.group(0)

        cleaned_part = RE_CC_TAG.sub(_remove_target_cc, masked_part)

        # プレースホルダを復元
        for i, note_str in enumerate(placeholders):
            cleaned_part = cleaned_part.replace(f"__NOTE_PLACEHOLDER_{i}__", note_str)

        print(
            f"Removed {removed_count} existing control events for targets: "
            f"{list(predictions.keys())}"
        )

        # --- 2. 予測値のCC文字列を生成 ---
        # 音声の Frame 0 に対応するVSQXの絶対Tick
        pre_measure_ticks = audio_start_tick
        cc_strings = []

        for param_id, values in predictions.items():
            tag_id_str = inv_map.get(param_id)
            if not tag_id_str:
                print(f"Warning: No VSQX tag ID found for {param_id}. Skipping.")
                continue

            last_val = -9999
            for frame_idx, val in enumerate(values):
                int_val = denormalize_value(val, param_id)

                # 間引き処理: 値が変わった時だけ書き込む
                if int_val == last_val:
                    continue

                # 時刻変換
                time_sec = librosa.frames_to_time(
                    frame_idx, sr=sr, hop_length=hop_length
                )
                audio_tick = time_sec * bpm * resolution / 60.0

                # 絶対Tick = 音声上のTick + プレメジャーのTick
                target_abs_tick = audio_tick + pre_measure_ticks

                # 相対Tick
                rel_tick = int(target_abs_tick - part_start_tick)

                # Part範囲外のガード
                if rel_tick < 0:
                    continue

                cc_strings.append(
                    f"\t\t\t<cc>\n"
                    f"\t\t\t\t<t>{rel_tick}</t>\n"
                    f'\t\t\t\t<v id="{tag_id_str}">{int_val}</v>\n'
                    f"\t\t\t</cc>"
                )
                last_val = int_val

        # --- 3. <singer>直後にCC文字列を挿入 ---
        if cc_strings:
            final_cc_block = "\n".join(cc_strings)
            augmented_part = RE_SINGER.sub(
                rf"\1\n{final_cc_block}", cleaned_part, count=1
            )
        else:
            augmented_part = cleaned_part

        # --- 4. 元のvsPartを置換して書き出し ---
        final_content = content.replace(part_content, augmented_part, 1)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"Successfully saved VSQX to: {output_path}")

    except Exception as e:
        print(f"Error writing VSQX: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# グラフ描画関数 (正解値との比較プロット3枚)
# =============================================================================


def _shade_unvoiced(ax, time_axis, voicing_mask_1d, label_added=False):
    """
    無声区間（voicing_mask == 0）をグラフ背景に灰色で塗る。
    連続する無声フレームをまとめて1つの axvspan にすることで描画を軽量化する。
    """
    in_unvoiced = False
    start_t = 0.0
    added = label_added
    for i, v in enumerate(voicing_mask_1d):
        if v < 0.5 and not in_unvoiced:
            in_unvoiced = True
            start_t = time_axis[i]
        elif v >= 0.5 and in_unvoiced:
            in_unvoiced = False
            lbl = "Unvoiced" if not added else None
            ax.axvspan(start_t, time_axis[i], alpha=0.18, color="#9E9E9E", label=lbl)
            added = True
    # 末尾まで無声の場合
    if in_unvoiced:
        lbl = "Unvoiced" if not added else None
        ax.axvspan(start_t, time_axis[-1], alpha=0.18, color="#9E9E9E", label=lbl)


def plot_predictions_with_gt(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    voicing_mask: Optional[np.ndarray],
    plot_output_path: str,
    sr: int,
    hop_length: int,
):
    """
    予測値と正解値を比較する3枚のグラフを生成する。
    1. 全体プロット (正解値 vs 予測値)
    2. 正解値の動きが最も大きい5秒間の拡大プロット
    3. 正解値と予測値の差が最も大きい5秒間の拡大プロット
    各グラフにはMAE (Mean Absolute Error) をタイトル付近に表示する。
    無声区間（voicing_mask == 0）は灰色背景で表示し、MAE計算から除外する。
    """
    # 5秒間に対応するフレーム数
    window_frames = int(5.0 * sr / hop_length)

    for pid, pred_vals in predictions.items():
        if pid not in ground_truth:
            print(f"  Skipping plot for '{pid}': no ground truth available.")
            continue

        gt_vals = ground_truth[pid]

        # フレーム数を合わせる (短い方に合わせる)
        min_len = min(len(pred_vals), len(gt_vals))
        pred_vals = pred_vals[:min_len]
        gt_vals = gt_vals[:min_len]

        # voicing_mask の準備 (1D化 & 長さ合わせ)
        if voicing_mask is not None:
            vm = voicing_mask[:min_len].flatten()
        else:
            vm = np.ones(min_len, dtype=np.float32)  # マスクなし=全て有声扱い

        voiced_idx = vm >= 0.5  # 有声フレームのブールマスク

        # 逆正規化
        denorm_pred = np.array([denormalize_value(v, pid) for v in pred_vals])
        denorm_gt = np.array([denormalize_value(v, pid) for v in gt_vals])

        # MAE計算 (有声区間のみ)
        if np.any(voiced_idx):
            mae = np.mean(np.abs(denorm_pred[voiced_idx] - denorm_gt[voiced_idx]))
        else:
            mae = 0.0

        # 時間軸 (秒)
        time_axis = librosa.frames_to_time(
            np.arange(min_len), sr=sr, hop_length=hop_length
        )

        # --- ウィンドウの特定 ---
        # ウィンドウサイズ制限
        w = min(window_frames, min_len)

        # (A) 正解値の動きが最も大きい5秒間
        gt_diff = np.abs(np.diff(denorm_gt, prepend=denorm_gt[0]))
        # 移動和で各ウィンドウの変動量を計算
        cum_diff = np.cumsum(gt_diff)
        window_sums = cum_diff[w:] - cum_diff[:-w]
        if len(window_sums) > 0:
            max_move_start = int(np.argmax(window_sums))
        else:
            max_move_start = 0
        max_move_end = max_move_start + w

        # (B) 予測値と正解値の差が最も大きい5秒間 (有声区間のみ考慮)
        abs_error = np.abs(denorm_pred - denorm_gt) * vm  # 無声区間は0
        cum_err = np.cumsum(abs_error)
        err_sums = cum_err[w:] - cum_err[:-w]
        if len(err_sums) > 0:
            max_err_start = int(np.argmax(err_sums))
        else:
            max_err_start = 0
        max_err_end = max_err_start + w

        # --- 3枚のグラフ描画 ---
        fig, axes = plt.subplots(3, 1, figsize=(18, 14))
        fig.suptitle(
            f"{pid}  |  MAE = {mae:.2f}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # --- グラフ1: 全体プロット ---
        ax1 = axes[0]
        _shade_unvoiced(ax1, time_axis, vm)  # 無声区間を灰色に
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
        # 最大変動区間をハイライト
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

        # --- グラフ2: 正解値の動き最大 5秒 ---
        ax2 = axes[1]
        vm_slice2 = vm[max_move_start:max_move_end]
        t_slice2 = time_axis[max_move_start:max_move_end]
        gt_slice2 = denorm_gt[max_move_start:max_move_end]
        pred_slice2 = denorm_pred[max_move_start:max_move_end]
        voiced2 = vm_slice2 >= 0.5
        local_mae2 = (
            np.mean(np.abs(pred_slice2[voiced2] - gt_slice2[voiced2]))
            if np.any(voiced2)
            else 0.0
        )
        _shade_unvoiced(ax2, t_slice2, vm_slice2)  # 無声区間を灰色に
        ax2.plot(
            t_slice2,
            gt_slice2,
            label="Ground Truth",
            color="#2196F3",
            alpha=0.8,
            linewidth=1.5,
        )
        ax2.plot(
            t_slice2,
            pred_slice2,
            label="Predicted",
            color="#FF5722",
            alpha=0.8,
            linewidth=1.5,
        )
        ax2.fill_between(t_slice2, gt_slice2, pred_slice2, alpha=0.15, color="orange")
        ax2.set_title(
            f"Max Ground Truth Movement (5s)  |  Local MAE = {local_mae2:.2f}",
            fontsize=13,
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel(pid)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.4)

        # --- グラフ3: 予測誤差最大 5秒 ---
        ax3 = axes[2]
        vm_slice3 = vm[max_err_start:max_err_end]
        t_slice3 = time_axis[max_err_start:max_err_end]
        gt_slice3 = denorm_gt[max_err_start:max_err_end]
        pred_slice3 = denorm_pred[max_err_start:max_err_end]
        voiced3 = vm_slice3 >= 0.5
        local_mae3 = (
            np.mean(np.abs(pred_slice3[voiced3] - gt_slice3[voiced3]))
            if np.any(voiced3)
            else 0.0
        )
        _shade_unvoiced(ax3, t_slice3, vm_slice3)  # 無声区間を灰色に
        ax3.plot(
            t_slice3,
            gt_slice3,
            label="Ground Truth",
            color="#2196F3",
            alpha=0.8,
            linewidth=1.5,
        )
        ax3.plot(
            t_slice3,
            pred_slice3,
            label="Predicted",
            color="#FF5722",
            alpha=0.8,
            linewidth=1.5,
        )
        ax3.fill_between(t_slice3, gt_slice3, pred_slice3, alpha=0.15, color="red")
        ax3.set_title(
            f"Max Prediction Error (5s)  |  Local MAE = {local_mae3:.2f}",
            fontsize=13,
        )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel(pid)
        ax3.legend(loc="upper right", fontsize=9)
        ax3.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # ファイル名にパラメータ名を付加 (複数パラメータ対応)
        base, ext = os.path.splitext(plot_output_path)
        save_path = f"{base}_{pid}{ext}"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {save_path}")


# =============================================================================
# メイン予測処理
# =============================================================================


def predict_vsqx(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 特徴量定義のロード
    if not os.path.exists(args.features_json):
        print(f"Error: {args.features_json} not found.")
        return

    with open(args.features_json, "r", encoding="utf-8") as f:
        raw_text = f.read()
        # features.json に NaN が含まれる場合への対策
        raw_text = raw_text.replace("NaN", "null")
        feature_info = json.loads(raw_text)

    num_x_features = feature_info["num_x_features"]
    num_y_features = feature_info["num_y_features"]
    y_feature_map = feature_info["y_feature_map"]

    print(f"Model input dim: {num_x_features}, output dim: {num_y_features}")

    # 2. Preprocessorの初期化
    preprocessor = VSQXPreprocessor()
    preprocessor.set_phoneme_map(feature_info["phoneme_map"])

    # 3. 入力データの作成 (X)
    print("Processing input files...")

    # A. 音声 (features.jsonに記録された特徴量タイプを使用)
    feature_types = feature_info.get("feature_types", config.FEATURE_TYPES)
    print(f"  Audio feature types: {feature_types}")
    audio_features = preprocessor.extractor.get_features(args.wav_path, feature_types)
    audio_features = np.nan_to_num(stats.zscore(audio_features, axis=0))
    total_frames = audio_features.shape[0]

    # ===== 音声の書き出し開始位置 (時間同期の要) =====
    # DAWからWAVを書き出した際、VSQX上のどのTickから書き出したかを指定します。
    # 例: Test1.wav は vsPart の開始位置 (Tick 9600) から書き出されているため 9600。
    # プレメジャー(4小節)の終了時点から書き出している場合は 1920 * 4 = 7680。

    # B. VSQXパラメータ
    raw_vsqx = preprocessor.parse_vsqx(
        args.vsqx_path, total_frames, pre_measure_ticks=AUDIO_START_TICK
    )
    if raw_vsqx is None:
        print("Error parsing VSQX.")
        return
    norm_vsqx = preprocessor.normalize_features(raw_vsqx)

    # C. 特徴量の結合
    input_param_ids = preprocessor._flatten_input_config(config.INPUT_VSQX_PARAMS)

    X_list = [audio_features]

    for pid in input_param_ids:
        data, _ = preprocessor._get_feature_data(pid, norm_vsqx)
        if data is not None:
            X_list.append(data)
        else:
            print(f"Warning: Input parameter '{pid}' not found in VSQX.")

    X_input = np.concatenate(X_list, axis=-1).astype(np.float32)

    # Tensor化
    X_tensor = torch.from_numpy(X_input).unsqueeze(0).to(device)

    # 4. モデルのロード
    model = TransformerModel(
        num_x_features=num_x_features,
        num_y_features=num_y_features,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 5. 推論 (学習時と同じCHUNK_SIZEで自己回帰推論を行う)
    print(f"Running inference... (frames={total_frames}, chunk_size={CHUNK_SIZE})")
    print("This will use encoder-only generation.")
    output_chunks = []

    with torch.no_grad():
        for start in range(0, total_frames, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total_frames)
            x_chunk = X_tensor[:, start:end, :]
            chunk_len = end - start

            # Encoder-onlyの場合はそのままforwardパスを通すだけでOKです
            out_chunk = model(x_chunk)
            output_chunks.append(out_chunk.cpu())

            if (start // CHUNK_SIZE) % 5 == 0:
                print(
                    f"  Processed Chunk {start//CHUNK_SIZE + 1}/{(total_frames + CHUNK_SIZE - 1)//CHUNK_SIZE}"
                )

        output = torch.cat(output_chunks, dim=1)
        output_np = output.squeeze(0).numpy()

    # 6. 出力データの分解
    predictions = {}
    current_idx = 0
    target_ids_flat = []
    for val in config.TARGET_PARAM_CONFIG.values():
        if isinstance(val, list):
            target_ids_flat.extend(val)
        else:
            target_ids_flat.append(val)

    for meta in y_feature_map:
        name = meta["name"]
        dim = meta["dim"]

        val_slice = output_np[:, current_idx : current_idx + dim]
        current_idx += dim

        if dim == 1:
            val_slice = val_slice.flatten()

        if name in target_ids_flat:
            predictions[name] = val_slice

    # 7. VSQXへの書き込み
    print(f"Writing predictions to {args.output_path}...")
    update_vsqx_xml(
        vsqx_path=args.vsqx_path,
        output_path=args.output_path,
        predictions=predictions,
        sr=config.SR,
        hop_length=config.HOP_LENGTH,
        audio_start_tick=AUDIO_START_TICK,
    )

    # 8. 正解値の取得と比較グラフ描画
    if args.plot_output_path:
        # --- 正解値の取得 (preprocess.py の手法を流用) ---
        ground_truth = {}
        try:
            print("Loading ground truth from VSQX...")
            # parse_vsqx + normalize_features は既に実行済み (norm_vsqx)
            # ターゲットパラメータの正解値を抽出
            for pid in predictions.keys():
                gt_data, _ = preprocessor._get_feature_data(pid, norm_vsqx)
                if gt_data is not None:
                    if gt_data.ndim == 2 and gt_data.shape[1] == 1:
                        gt_data = gt_data.flatten()
                    ground_truth[pid] = gt_data
                    print(f"  Ground truth loaded for '{pid}' (frames={len(gt_data)})")
                else:
                    print(f"  Warning: Ground truth for '{pid}' not found in VSQX.")
        except Exception as e:
            print(f"Warning: Could not load ground truth: {e}")

        # --- 発声マスクの取得 (preprocess.py の voicing_mask を流用) ---
        voicing_mask = norm_vsqx.get("voicing_mask", None)
        if voicing_mask is not None:
            voiced_ratio = np.mean(voicing_mask >= 0.5) * 100
            print(f"  Voicing mask loaded (voiced: {voiced_ratio:.1f}%)")

        # --- 3枚のグラフを描画 ---
        plot_predictions_with_gt(
            predictions=predictions,
            ground_truth=ground_truth,
            voicing_mask=voicing_mask,
            plot_output_path=args.plot_output_path,
            sr=config.SR,
            hop_length=config.HOP_LENGTH,
        )

    print("Prediction process finished.")


# =============================================================================
# ファイル検索ヘルパー
# =============================================================================
def find_file_recursive(search_dir, filename_stem, extension):
    """
    search_dir 以下を再帰的に検索し、指定名のファイルを探す。

    Args:
        search_dir: 検索ルートディレクトリ
        filename_stem: 拡張子なしのファイル名
        extension: 拡張子 (例: ".wav", ".vsqx")
    Returns:
        見つかったファイルの絶対パス (str)
    Raises:
        FileNotFoundError: ファイルが見つからない場合
        ValueError: 同名ファイルが複数見つかった場合
    """
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
            f"  該当ファイル:\n{paths_str}\n"
            f"  INPUT_FILE_NAME をより具体的な名前に変更するか、フォルダ構成を見直してください。"
        )

    resolved = os.path.abspath(found[0])
    print(f"  Found: {resolved}")
    return resolved


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("predict.py - WAV + VSQX Prediction")
    print("=" * 60)

    # --- 1. ファイル検索 ---
    errors = []
    wav_path = None
    vsqx_path = None

    print(f"\n入力ファイル名: {INPUT_FILE_NAME}")

    # WAV検索
    print(f"WAV検索中: {os.path.abspath(WAV_SEARCH_DIR)}")
    try:
        wav_path = find_file_recursive(WAV_SEARCH_DIR, INPUT_FILE_NAME, ".wav")
    except (FileNotFoundError, ValueError) as e:
        errors.append(str(e))

    # VSQX検索
    print(f"VSQX検索中: {os.path.abspath(VSQX_SEARCH_DIR)}")
    try:
        vsqx_path = find_file_recursive(VSQX_SEARCH_DIR, INPUT_FILE_NAME, ".vsqx")
    except (FileNotFoundError, ValueError) as e:
        errors.append(str(e))

    # モデル・features.json の存在チェック
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found: {os.path.abspath(MODEL_PATH)}")
    if not os.path.exists(FEATURES_JSON_PATH):
        errors.append(f"features.json not found: {os.path.abspath(FEATURES_JSON_PATH)}")

    # --- 2. エラーチェック ---
    if errors:
        print("\n" + "=" * 60)
        for e in errors:
            print(f"[Error] {e}")
        print("=" * 60)
        print("処理を中止します。上記を確認してください。")
    else:
        # --- 3. 実行 ---
        args = Namespace(
            wav_path=wav_path,
            vsqx_path=vsqx_path,
            output_path=VSQX_OUTPUT,
            plot_output_path=PLOT_OUTPUT,
            model_path=MODEL_PATH,
            features_json=FEATURES_JSON_PATH,
        )

        predict_vsqx(args)

        print("\n" + "=" * 60)
        print("=== 処理完了 ===")
        print(f"  WAV入力:   {wav_path}")
        print(f"  VSQX入力:  {vsqx_path}")
        print(f"  VSQX出力:  {VSQX_OUTPUT}")
        print(f"  プロット:  {PLOT_OUTPUT}")
        print("=" * 60)
