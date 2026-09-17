"""
predict.py
==========
学習済みDNNモデルを用いて、WAV + VSQX から推論を行い、
予測パラメータを書き込んだ新しい VSQX ファイルを出力するスクリプト。

処理フロー:
    1. predict_config.py から設定を読み込む
    2. features.json から特徴量仕様・正規化パラメータを自動取得
    3. AudioFeatureExtractor / VSQXDNNFormatter で特徴量を抽出・正規化
    4. スライディングウィンドウ (Overlap-Add) 推論を実行
    5. 予測値を VSQX に書き込んで保存
    6. 正解VSQXとの比較プロットを出力 (evaluate.py の形式に準拠)

使い方:
    1. predict_config.py の config 領域を編集する
    2. python predict.py を実行する
"""

import os
import re
import sys
import json
import glob
import torch
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

# モデル
from models.cnn_bilstm import CNNBiLSTM
from models.bilstm_base import BiLSTMBase
from models.fusion_block import FeatureFusionBlock
from models.unified_model import UnifiedModel
from train_util.inference import run_sliding_window_inference

# 設定の読み込み
import predict_config as cfg


# =============================================================================
# ユーティリティ関数
# =============================================================================


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


def _build_model(model_type, input_dim, output_dim, model_params):
    """config の MODEL_TYPE に基づいてモデルを動的にインスタンス化する。"""
    model_registry = {
        "cnn_bilstm": CNNBiLSTM,
        "bilstm_base": BiLSTMBase,
    }
    base_type = (
        model_type.split("_trial_")[0] if "_trial_" in model_type else model_type
    )
    if base_type not in model_registry:
        raise ValueError(f"未知のモデル種別: {base_type}")
    params = model_params.get(base_type, {})
    return model_registry[base_type](
        input_dim=input_dim, output_dim=output_dim, **params
    )


def denormalize_to_int(norm_val, param_id, vsqx_config):
    """
    正規化された値を VSQX 用の整数値に戻す。
    """
    if param_id == "PIT":
        return int(norm_val * 8192.0)
    elif param_id == "PBS":
        return int(np.clip((norm_val + 1.0) / 2.0 * 24.0, 0, 24))
    elif param_id in vsqx_config.get("DYNAMIC_PARAM_IDS", []):
        val = (norm_val + 1.0) / 2.0 * 127.0
        return int(np.clip(np.round(val), 0, 127))
    elif param_id in [p for p in vsqx_config.get("NOTE_PARAM_IDS", [])]:
        max_val = vsqx_config.get("NOTE_PARAM_MAX_VALUES", {}).get(param_id, 127.0)
        if param_id in ["fallPort", "risePort"]:
            return int(np.clip(norm_val * max_val, 0, 127))
        elif param_id == "vibType":
            return int(norm_val)
        else:
            val = (norm_val + 1.0) / 2.0 * max_val
            return int(np.clip(val, 0, int(max_val)))
    else:
        val = (norm_val + 1.0) / 2.0 * 127.0
        return int(np.clip(np.round(val), 0, 127))


# --- 正規表現パターン (VSQX テキストベース書き込み用) ---
RE_RESOLUTION = re.compile(r"<resolution>(\d+)</resolution>")
RE_TEMPO_V = re.compile(r"<tempo>\s*<t>0</t>\s*<v>(\d+)</v>\s*</tempo>", re.DOTALL)
RE_VS_PART = re.compile(r"<vsPart>.*?</vsPart>", re.DOTALL)
RE_PART_T = re.compile(r"<vsPart>\s*<t>(\d+)</t>")
RE_SINGER = re.compile(r"(<singer>.*?</singer>)", re.DOTALL)
RE_CC_TAG = re.compile(r'<cc>.*?<v id="([A-Z]+)">.*?</cc>', re.DOTALL)
RE_NOTE = re.compile(r"<note>.*?</note>", re.DOTALL)


def update_vsqx_xml(
    vsqx_path, output_path, predictions_int, sr, shift_length, vsqx_config
):
    """
    テキストベース（正規表現）で VSQX ファイルを書き換える関数。
    predict_en.py の update_vsqx_xml を移植したもの。
    """
    inv_map = {v: k for k, v in vsqx_config["ID_MAPPING"].items()}

    try:
        with open(vsqx_path, "r", encoding="utf-8") as f:
            content = f.read()

        # メタ情報の取得
        res_match = RE_RESOLUTION.search(content)
        resolution = int(res_match.group(1)) if res_match else 480
        tempo_match = RE_TEMPO_V.search(content)
        bpm = int(tempo_match.group(1)) / 100.0 if tempo_match else 120.0

        # preMeasure の取得 (VSQX内から自動取得)
        pre_measure_match = re.search(r"<preMeasure>(\d+)</preMeasure>", content)
        pre_measure = int(pre_measure_match.group(1)) if pre_measure_match else 4
        audio_start_tick = pre_measure * resolution * 4  # 4拍子前提

        # vsPart の処理
        part_match = RE_VS_PART.search(content)
        if not part_match:
            print("Error: vsPart not found in VSQX.")
            return
        part_content = part_match.group(0)

        part_t_match = RE_PART_T.search(part_content)
        if not part_t_match:
            print("Error: Part start tick not found.")
            return
        part_start_tick = int(part_t_match.group(1))

        # 1. 既存パラメータの削除
        target_tag_ids = set()
        for param_id in predictions_int.keys():
            tag_id = inv_map.get(param_id)
            if tag_id:
                target_tag_ids.add(tag_id)

        # ノートをプレースホルダで保護してからCCを削除する
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
            f"  Removed {removed_count} existing control events for targets: "
            f"{list(predictions_int.keys())}"
        )

        # 2. 予測値のCC文字列を生成
        cc_strings = []
        for param_id, int_values in predictions_int.items():
            tag_id_str = inv_map.get(param_id)
            if not tag_id_str:
                print(f"  Warning: No VSQX tag ID found for {param_id}. Skipping.")
                continue

            last_val = -9999
            for frame_idx, int_val in enumerate(int_values):
                # 間引き処理: 値が変わった時だけ書き込む
                if int_val == last_val:
                    continue

                time_sec = librosa.frames_to_time(
                    frame_idx, sr=sr, hop_length=shift_length
                )
                audio_tick = time_sec * bpm * resolution / 60.0
                target_abs_tick = audio_tick + audio_start_tick
                rel_tick = int(target_abs_tick - part_start_tick)

                if rel_tick < 0:
                    continue

                cc_strings.append(
                    f"\t\t\t<cc>\n"
                    f"\t\t\t\t<t>{rel_tick}</t>\n"
                    f'\t\t\t\t<v id="{tag_id_str}">{int_val}</v>\n'
                    f"\t\t\t</cc>"
                )
                last_val = int_val

        # 3. <singer>直後にCC文字列を挿入
        if cc_strings:
            final_cc_block = "\n".join(cc_strings)
            augmented_part = RE_SINGER.sub(
                rf"\1\n{final_cc_block}", cleaned_part, count=1
            )
        else:
            augmented_part = cleaned_part

        # 4. 元のvsPartを置換して書き出し
        final_content = content.replace(part_content, augmented_part, 1)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"  VSQX saved: {output_path}")

    except Exception as e:
        print(f"  Error writing VSQX: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# メイン処理
# =============================================================================


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("predict.py - DNN モデル推論 & 予測VSQX生成")
    print("=" * 60)

    # --- 1. ファイル検索 ---
    errors = []
    wav_path = vsqx_path = None

    print(f"\n入力ファイル名: {cfg.INPUT_FILE_NAME}")
    print(f"WAV検索中: {os.path.abspath(cfg.WAV_SEARCH_DIR)}")
    try:
        wav_path = find_file_recursive(cfg.WAV_SEARCH_DIR, cfg.INPUT_FILE_NAME, ".wav")
    except (FileNotFoundError, ValueError) as e:
        errors.append(str(e))

    print(f"VSQX検索中: {os.path.abspath(cfg.VSQX_SEARCH_DIR)}")
    try:
        vsqx_path = find_file_recursive(
            cfg.VSQX_SEARCH_DIR, cfg.INPUT_FILE_NAME, ".vsqx"
        )
    except (FileNotFoundError, ValueError) as e:
        errors.append(str(e))

    if not os.path.exists(cfg.MODEL_PATH):
        errors.append(f"Model file not found: {os.path.abspath(cfg.MODEL_PATH)}")
    if not os.path.exists(cfg.FEATURES_JSON_PATH):
        errors.append(
            f"features.json not found: {os.path.abspath(cfg.FEATURES_JSON_PATH)}"
        )
    if not os.path.exists(cfg.TRAIN_SETTINGS_PATH):
        errors.append(
            f"train_settings.json not found: {os.path.abspath(cfg.TRAIN_SETTINGS_PATH)}"
        )

    if errors:
        print("\n" + "=" * 60)
        for e in errors:
            print(f"[Error] {e}")
        print("=" * 60)
        print("処理を中止します。上記を確認してください。")
        return

    # --- 2. features.json のロード ---
    print(f"\nfeatures.json 読み込み中: {cfg.FEATURES_JSON_PATH}")
    with open(cfg.FEATURES_JSON_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read().replace("NaN", "null")
        feature_info = json.loads(raw_text)

    sr = feature_info.get("sr", 44100)
    shift_length = feature_info.get("shift_length", 512)
    audio_global_mean = feature_info.get("audio_global_mean", {})
    audio_global_std = feature_info.get("audio_global_std", {})
    x_feature_map = feature_info.get("x_feature_map", {})
    y_feature_map = feature_info.get("y_feature_map", {})
    x_dims = feature_info.get("x_dims", {})
    y_dims = feature_info.get("y_dims", {})

    # 出力パラメータ名の取得
    y_param_names = []
    total_y_dim = 0
    for branch_name, features in y_feature_map.items():
        for feat in features:
            dim = feat["dim"]
            name = feat["name"]
            total_y_dim += dim
            if dim == 1:
                y_param_names.append(name)
            else:
                for i in range(dim):
                    y_param_names.append(f"{name}_{i}")

    print(f"  SR: {sr}, SHIFT: {shift_length}")
    print(f"  出力パラメータ: {y_param_names}")

    # --- 3. train_settings.json のロード (モデル構築パラメータ) ---
    with open(cfg.TRAIN_SETTINGS_PATH, "r", encoding="utf-8") as f:
        train_settings = json.load(f)

    FUSION_STRATEGY = train_settings["FUSION_STRATEGY"]
    BRANCH_MODULES = train_settings["BRANCH_MODULES"]
    MODEL_PARAMS = train_settings.get("MODEL_PARAMS", {})

    # --- 4. モデルの構築とロード ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    fusion_block = FeatureFusionBlock(
        x_dims_dict=x_dims, strategy=FUSION_STRATEGY, branch_modules=BRANCH_MODULES
    )
    total_x_dim = fusion_block.output_dim
    core_model = _build_model(cfg.MODEL_TYPE, total_x_dim, total_y_dim, MODEL_PARAMS)
    model = UnifiedModel(fusion_block, core_model).to(device)

    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    model.eval()
    print(f"  Model loaded: {cfg.MODEL_PATH}")
    print(f"  総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # --- 5. 特徴量の抽出と正規化 ---
    print("\n音響特徴量を抽出中...")
    audio_extractor = AudioFeatureExtractor(sr=sr, shift_length=shift_length)
    vsqx_extractor = VSQXDNNFormatter(sr=sr, shift_length=shift_length)

    # config の注入 (build_dataset.py と同じ方式)
    # features.json から入力/ターゲットパラメータを復元
    input_vsqx_params_flat = feature_info.get("input_vsqx_params", [])
    target_vsqx_params_flat = feature_info.get("target_vsqx_params", [])

    # x_feature_map からブランチ構造を復元して INPUT_VSQX_PARAMS を再構築
    input_vsqx_params_dict = {}
    for branch_name, feat_list in x_feature_map.items():
        feature_names = [f["name"] for f in feat_list]
        # 音響特徴量ではないブランチ = VSQXパラメータブランチ
        is_audio = any(
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
            for ft in feature_names
        )
        if not is_audio:
            input_vsqx_params_dict[branch_name] = feature_names

    # ターゲットもブランチ構造を復元
    target_param_config = {}
    for branch_name, feat_list in y_feature_map.items():
        target_param_config[branch_name] = [f["name"] for f in feat_list]

    vsqx_extractor.vsqx_config["INPUT_VSQX_PARAMS"] = input_vsqx_params_dict
    vsqx_extractor.vsqx_config["TARGET_PARAM_CONFIG"] = target_param_config

    # 音素マップの復元 (features.json に保存されている場合)
    phoneme_map = feature_info.get("phoneme_map", {})
    if phoneme_map:
        vsqx_extractor.set_phoneme_map(phoneme_map)

    # max_note_frames の復元 (features.json から設定、無ければスキャン)
    if "max_note_frames" in feature_info:
        vsqx_extractor.max_note_frames = int(feature_info["max_note_frames"])
    else:
        raise ValueError(
            "features.json に 'max_note_frames' が含まれていません。\n"
            "学習時と推論時で正規化スケールの齟齬を防ぐため、build_dataset.py を再実行して最新の features.json を使用してください。"
        )

    # A. 音響特徴量の抽出・正規化
    x_branches = {}
    total_frames = 0

    for branch_name, feat_list in x_feature_map.items():
        feature_names = [f["name"] for f in feat_list]

        # 音響特徴量かどうかの判定
        is_audio = any(
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
            for ft in feature_names
        )

        if is_audio:
            features = audio_extractor.get_features(wav_path, feature_names)
            total_frames = features.shape[0]

            # features.json の統計量で正規化 (build_dataset.py と同一の処理)
            mean = np.array(audio_global_mean.get(branch_name, 0.0))
            std = np.array(audio_global_std.get(branch_name, 1.0))
            features = (features - mean) / (std + 1e-7)
            features = np.nan_to_num(features).astype(np.float32)

            x_branches[branch_name] = features
            print(f"  {branch_name}: {feature_names} -> shape {features.shape}")

    # B. VSQXパラメータの抽出
    print("VSQXからパラメータを抽出中...")
    raw_vsqx = vsqx_extractor.parse_vsqx(vsqx_path, total_frames)
    if raw_vsqx is None:
        print("Error: VSQXのパースに失敗しました。")
        return
    norm_vsqx = vsqx_extractor.normalize_features(raw_vsqx)

    # 発声マスク
    voicing_mask = norm_vsqx.get(
        "voicing_mask", np.ones((total_frames, 1), dtype=np.float32)
    )
    v_mask = voicing_mask.squeeze(-1) if voicing_mask.ndim > 1 else voicing_mask

    # VSQXの入力パラメータブランチを構築
    for branch_name, p_list in input_vsqx_params_dict.items():
        if not p_list:
            continue
        branch_data_list = []
        for pid in p_list:
            if pid in norm_vsqx:
                branch_data_list.append(norm_vsqx[pid])
            else:
                print(f"  Warning: 入力パラメータ '{pid}' が見つかりません。")
        if branch_data_list:
            branch_arr = np.concatenate(branch_data_list, axis=-1).astype(np.float32)
            x_branches[branch_name] = branch_arr
            print(f"  {branch_name}: {p_list} -> shape {branch_arr.shape}")

    # --- 6. スライディングウィンドウ (Overlap-Add) 推論 ---
    print(
        f"\n推論実行中... (frames={total_frames}, chunk={cfg.CHUNK_SIZE}, hop={cfg.CHUNK_HOP})"
    )

    # 推論用にテンソルを準備
    x_tensors = {k: torch.from_numpy(v).unsqueeze(0) for k, v in x_branches.items()}

    # 共有ユーティリティで加重オーバーラップ加算推論を実行
    y_pred_full = run_sliding_window_inference(
        model=model,
        x_tensors=x_tensors,
        total_frames=total_frames,
        chunk_size=cfg.CHUNK_SIZE,
        chunk_hop=cfg.CHUNK_HOP,
        device=device,
        device_type=device_type,
        offset=0.1,
    )

    # downstream が pred_sum を期待しているので代入
    pred_sum = y_pred_full

    print("  推論完了")
    print("  推論完了: 完了")

    # --- 7. 正解値の取得 ---
    y_true_full = np.zeros((total_frames, total_y_dim), dtype=np.float32)
    col_idx = 0
    for branch_name, feat_list in y_feature_map.items():
        for feat in feat_list:
            pid = feat["name"]
            dim = feat["dim"]
            gt = norm_vsqx.get(pid)
            if gt is not None:
                if gt.ndim == 1:
                    gt = gt[:, np.newaxis]
                y_true_full[:, col_idx : col_idx + dim] = gt[:total_frames, :dim]
            col_idx += dim

    # --- 8. Evaluator を使ったプロット ---
    print("\n比較グラフを描画中...")
    evaluator = Evaluator(device=device)
    evaluator.add_song_result(
        cfg.INPUT_FILE_NAME,
        y_true_full,
        y_pred_full,
        y_param_names,
        voicing_mask=v_mask,
    )

    # RMS等の追加特徴量 (x_branches の音響ブランチの先頭次元)
    extra_features = {}
    for branch_name, feat_list in x_feature_map.items():
        feature_names = [f["name"] for f in feat_list]
        is_audio = any(
            ft in ["rms", "mel", "mfcc", "cqt", "stft"] for ft in feature_names
        )
        if is_audio and branch_name in x_branches:
            feat_data = x_branches[branch_name]
            if feat_data.ndim > 1:
                feat_data = feat_data[:, 0]
            # ブランチ名から表示名を生成 (例: "audio_rms" -> "RMS")
            display_name = feature_names[0].upper()
            extra_features[display_name] = feat_data

    wave_path = os.path.join(cfg.OUTPUT_DIR, f"waveform_{cfg.INPUT_FILE_NAME}.png")
    evaluator.plot_waveform(
        cfg.INPUT_FILE_NAME,
        y_true_full,
        y_pred_full,
        wave_path,
        y_param_names,
        voicing_mask=v_mask,
        extra_features=extra_features,
        sr=sr,
        shift_length=shift_length,
        remove_intro_silence=getattr(cfg, "REMOVE_INTRO_SILENCE", True),
    )

    # --- 9. VSQX への書き込み ---
    print("\nVSQX書き込み中...")

    # 予測値を整数化
    predictions_int = {}
    col_idx = 0
    for branch_name, feat_list in y_feature_map.items():
        for feat in feat_list:
            pid = feat["name"]
            dim = feat["dim"]
            if dim == 1:
                pred_norm = y_pred_full[:, col_idx].flatten()
                int_values = [
                    denormalize_to_int(v, pid, vsqx_extractor.vsqx_config)
                    for v in pred_norm
                ]
                predictions_int[pid] = int_values
            col_idx += dim

    vsqx_output_path = os.path.join(
        cfg.OUTPUT_DIR, f"{cfg.INPUT_FILE_NAME}_predict.vsqx"
    )
    update_vsqx_xml(
        vsqx_path,
        vsqx_output_path,
        predictions_int,
        sr,
        shift_length,
        vsqx_extractor.vsqx_config,
    )

    # --- 完了メッセージ ---
    print("\n" + "=" * 60)
    print("=== 処理完了 ===")
    print(f"  WAV入力:   {wav_path}")
    print(f"  VSQX入力:  {vsqx_path}")
    print(f"  VSQX出力:  {vsqx_output_path}")
    print(f"  グラフ出力: {wave_path}")
    print("=" * 60)
    print("\n次のステップ:")
    print("  1. 出力された予測VSQXをVocaloidソフトで開く")
    print("  2. 予測WAVとして書き出す")
    print(f"  3. 予測WAVを {cfg.PRED_WAV_SEARCH_DIR} に配置する")
    print("  4. plot_rendered_eval.py を実行して正解WAVと比較する")


if __name__ == "__main__":
    main()
