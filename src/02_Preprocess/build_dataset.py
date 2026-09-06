"""
build_dataset.py
================
AudioFeatureExtractor と VSQXExtractor を利用して、
WAV/VSQX ペアから学習用データセット (.npz) を構築するスクリプト。

従来の preprocess.py の build_dataset メソッドと同等の処理を、
分離されたユーティリティクラスを呼び出す形で再実装したもの。
"""

import os
import sys
import json
import numpy as np
import librosa
import gc

# --- パス設定 ---
# util フォルダをインポートパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "util"))

import config
from feature_extractor import AudioFeatureExtractor
from vsqx_dnn_formatter import VSQXDNNFormatter


class ConfigParameterError(Exception):
    pass


def build_dataset(wav_dir, vsqx_dir, output_dir):
    """
    WAV/VSQX ペアを走査し、音響特徴量とVSQXパラメータを抽出して
    .npz 形式のデータセットを構築する。

    処理フロー:
      1. config の解析（入力/出力パラメータの決定、音素スキャンの要否判定）
      2. WAV/VSQX ペアの収集
      3. 第1パス: 全WAVから音響特徴量の全体統計量 (mean, std) を計算
      4. 第2パス: 各ペアについて特徴量を抽出・正規化し .npz で保存
      5. メタデータ (features.json) の保存
    """
    os.makedirs(output_dir, exist_ok=True)

    # スキップされたファイルとその理由を記録するリスト (エラーログ用)
    skipped_files = []

    # --- Extractor の初期化 ---
    audio_extractor = AudioFeatureExtractor(sr=config.SR, shift_length=config.SHIFT_LENGTH)
    vsqx_extractor = VSQXDNNFormatter(sr=config.SR, shift_length=config.SHIFT_LENGTH)

    # 高速化のため、実際に使用するパラメータ設定をパーサーに注入（オプトイン）
    vsqx_extractor.vsqx_config["INPUT_VSQX_PARAMS"] = config.INPUT_VSQX_PARAMS
    vsqx_extractor.vsqx_config["TARGET_PARAM_CONFIG"] = config.TARGET_PARAM_CONFIG

    # --- config の解析 ---
    audio_param_ids = []
    for p_list in config.INPUT_AUDIO_PARAMS.values():
        audio_param_ids.extend(p_list)

    input_param_ids = []
    for p_list in config.INPUT_VSQX_PARAMS.values():
        input_param_ids.extend(p_list)
    target_param_ids = []
    for cat, p_list in config.TARGET_PARAM_CONFIG.items():
        if isinstance(p_list, list):
            target_param_ids.extend(p_list)
        else:
            target_param_ids.append(p_list)

    # 音素("PHONEME_P")が入力・出力のいずれかに含まれているかチェック
    use_phoneme = ("PHONEME_P" in input_param_ids) or ("PHONEME_P" in target_param_ids)

    # ある場合、全曲で音素次元数を一致させるためにスキャンを行う
    if use_phoneme:
        vsqx_extractor.scan_all_phonemes(vsqx_dir)
    else:
        print("Skipping phoneme scan (PHONEME_P not found in config).")

    # ノート位置特徴量で使用する最大ノート継続フレーム数を事前走査
    vsqx_extractor.scan_max_note_duration(vsqx_dir)

    # --- WAV/VSQX ペアの収集 ---
    from collections import defaultdict

    wav_paths_dict = defaultdict(list)
    vsqx_paths_dict = defaultdict(list)

    for r, _, fs in os.walk(wav_dir):
        for f in fs:
            if f.endswith(".wav"):
                stem = os.path.splitext(f)[0]
                wav_paths_dict[stem].append(os.path.join(r, f))

    for r, _, fs in os.walk(vsqx_dir):
        for f in fs:
            if f.endswith(".vsqx"):
                stem = os.path.splitext(f)[0]
                vsqx_paths_dict[stem].append(os.path.join(r, f))

    wav_files = {}
    vsqx_files = {}

    # 重複チェックとログ登録
    for stem, paths in wav_paths_dict.items():
        if len(paths) > 1:
            print(f"Warning: Skipping duplicate WAV file: {stem}")
            skipped_files.append(
                {
                    "file": stem,
                    "error_type": "DuplicateFileError",
                    "message": f"同名のWAVファイルが複数存在します:\n"
                    + "\n".join([f"      - {p}" for p in paths]),
                }
            )
        else:
            wav_files[stem] = paths[0]

    for stem, paths in vsqx_paths_dict.items():
        if len(paths) > 1:
            print(f"Warning: Skipping duplicate VSQX file: {stem}")
            if stem not in [
                s["file"] for s in skipped_files
            ]:  # すでにWAV側でスキップ済みでなければ追加
                skipped_files.append(
                    {
                        "file": stem,
                        "error_type": "DuplicateFileError",
                        "message": f"同名のVSQXファイルが複数存在します:\n"
                        + "\n".join([f"      - {p}" for p in paths]),
                    }
                )
        else:
            vsqx_files[stem] = paths[0]

    # スキップされたものは辞書から削除（ペア成立を阻止）
    for skipped in skipped_files:
        wav_files.pop(skipped["file"], None)
        vsqx_files.pop(skipped["file"], None)

    file_pairs = sorted(
        [(k, wav_files[k], vsqx_files[k]) for k in wav_files if k in vsqx_files]
    )
    print(f"Processing {len(file_pairs)} pairs...")

    # =====================================================================
    # 第1パス: 全体統計量の計算 (Global Mean / Std for Z-score Normalization)
    # =====================================================================
    print("Calculating global statistics for audio features...")
    sum_x_dict = {branch: None for branch in config.INPUT_AUDIO_PARAMS}
    sum_sq_x_dict = {branch: None for branch in config.INPUT_AUDIO_PARAMS}
    total_frames_all = 0

    for idx, (name_stem, wav_path, _) in enumerate(file_pairs):
        try:
            frames_in_file = 0
            for branch_name, features_list in config.INPUT_AUDIO_PARAMS.items():
                if not features_list:
                    continue
                features = audio_extractor.get_features(wav_path, features_list)
                if sum_x_dict.get(branch_name) is None:
                    dim = features.shape[1]
                    sum_x_dict[branch_name] = np.zeros(dim, dtype=np.float64)
                    sum_sq_x_dict[branch_name] = np.zeros(dim, dtype=np.float64)

                sum_x_dict[branch_name] += np.sum(features, axis=0)
                sum_sq_x_dict[branch_name] += np.sum(features**2, axis=0)
                frames_in_file = features.shape[0]

            total_frames_all += frames_in_file

            if (idx + 1) % 100 == 0:
                print(f"Stats scanning: {idx + 1}/{len(file_pairs)}")
        except Exception as e:
            print(f"Error reading {name_stem} for stats: {e}")

    # 平均・標準偏差の算出
    audio_global_mean = {}
    audio_global_std = {}

    if total_frames_all > 0:
        for branch_name in config.INPUT_AUDIO_PARAMS:
            mean = sum_x_dict[branch_name] / total_frames_all
            var = (sum_sq_x_dict[branch_name] / total_frames_all) - (mean**2)
            std = np.sqrt(np.maximum(var, 1e-7))
            audio_global_mean[branch_name] = mean
            audio_global_std[branch_name] = std
    else:
        print("Warning: No valid audio frames found. Using default normalization.")
        for branch_name in config.INPUT_AUDIO_PARAMS:
            audio_global_mean[branch_name] = 0.0
            audio_global_std[branch_name] = 1.0

    print("Global statistics calculated.")

    # =====================================================================
    # メタデータ構造体の初期化
    # =====================================================================
    feature_info = {
        "x_dims": {},
        "y_dims": {},
        "x_feature_map": {},
        "y_feature_map": {},
        "phoneme_map": vsqx_extractor.phoneme_map,
        "num_phoneme_classes": vsqx_extractor.num_phoneme_classes,
        "num_evec_classes": vsqx_extractor.num_evec_classes,
        "file_lengths": {},
        "feature_types": audio_param_ids,
        "input_vsqx_params": input_param_ids,
        "target_vsqx_params": target_param_ids,
        "sr": config.SR,
        "shift_length": config.SHIFT_LENGTH,
        "max_note_frames": int(vsqx_extractor.max_note_frames),
        # 統計データ (JSON保存用に辞書内のNumPy配列をリスト変換)
        "audio_global_mean": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in audio_global_mean.items()
        },
        "audio_global_std": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in audio_global_std.items()
        },
    }
    # 最初の処理用変数　次元数記述等で使用
    is_first = True

    # =====================================================================
    # 第2パス: データセット生成と保存
    # =====================================================================

    for name_stem, wav_path, vsqx_path in file_pairs:
        try:
            # --- Audio Processing ---
            x_dict = {}
            total_frames = 0

            # check_timelag.py 用に純粋なRMSを別途計算
            y_raw = audio_extractor._load_audio(wav_path)
            audio_env_raw = librosa.feature.rms(y=y_raw, hop_length=config.SHIFT_LENGTH)[
                0
            ]

            # 音響特徴量をブランチごとに抽出・正規化
            for branch_name, features_list in config.INPUT_AUDIO_PARAMS.items():
                if not features_list:
                    continue
                features = audio_extractor.get_features(wav_path, features_list)
                total_frames = features.shape[0]  # 全ブランチで同一フレーム数と仮定

                # ブランチ固有の統計量で正規化
                mean = audio_global_mean[branch_name]
                std = audio_global_std[branch_name]
                features = (features - mean) / (std + 1e-7)
                # NaNエラー対策
                features = np.nan_to_num(features)

                x_dict[branch_name] = features.astype(np.float32)

                if is_first:
                    #  各音響特徴の名前と次元数を登録
                    feature_info["x_feature_map"][branch_name] = [
                        {"name": f, "dim": audio_extractor.get_feature_dim(f)}
                        for f in features_list
                    ]
                    # ブランチごとの合計次元数を登録
                    feature_info["x_dims"][branch_name] = features.shape[1]

            audio_env_raw = audio_env_raw[:total_frames]  # フレーム長を合わせる

            # --- VSQX Processing ---
            raw_vsqx = vsqx_extractor.parse_vsqx(vsqx_path, total_frames)
            if raw_vsqx is None:
                raise ValueError(
                    "VSQXのパースに失敗しました（ファイルが空、またはボーカルトラックが存在しません）"
                )
            norm_vsqx = vsqx_extractor.normalize_features(raw_vsqx)

            # --- Organize Features by Branch ---
            y_dict = {}

            # X (入力パラメータ) の処理
            for branch_name, p_list in config.INPUT_VSQX_PARAMS.items():
                if not p_list:
                    continue
                branch_data_list = []
                if is_first:
                    feature_info["x_feature_map"][branch_name] = []

                for pid in p_list:
                    if pid not in norm_vsqx:
                        raise ConfigParameterError(
                            f"設定ミス: 学習に必要な入力パラメータ '{pid}' が見つかりません。config.py の記述を確認してください。"
                        )
                    data = norm_vsqx[pid]
                    dim = data.shape[1]
                    branch_data_list.append(data)
                    if is_first:
                        feature_info["x_feature_map"][branch_name].append(
                            {"name": pid, "dim": dim}
                        )

                branch_arr = np.concatenate(branch_data_list, axis=-1).astype(
                    np.float32
                )

                x_dict[branch_name] = branch_arr
                if is_first:
                    feature_info["x_dims"][branch_name] = branch_arr.shape[1]

            # Y (ターゲットパラメータ) の処理
            for branch_name, p_list in config.TARGET_PARAM_CONFIG.items():
                if not p_list:
                    raise ConfigParameterError(
                        f"設定ミス: ターゲットパラメータが空です。予測する対象が存在しません。"
                    )
                branch_data_list = []
                if is_first:
                    feature_info["y_feature_map"][branch_name] = []

                for pid in p_list:
                    if pid not in norm_vsqx:
                        raise ConfigParameterError(
                            f"設定ミス: 学習に必要なターゲットパラメータ '{pid}' が見つかりません。config.py の記述を確認してください。"
                        )
                    data = norm_vsqx[pid]
                    dim = data.shape[1]
                    branch_data_list.append(data)
                    if is_first:
                        feature_info["y_feature_map"][branch_name].append(
                            {"name": pid, "dim": dim}
                        )

                branch_arr = np.concatenate(branch_data_list, axis=-1).astype(
                    np.float32
                )

                y_dict[branch_name] = branch_arr
                if is_first:
                    feature_info["y_dims"][branch_name] = branch_arr.shape[1]

            if is_first:
                is_first = False

            voicing_mask_data = norm_vsqx["voicing_mask"]

            # NPZへの保存引数を準備
            save_kwargs = {
                "voicing_mask": voicing_mask_data,
                "audio_env": audio_env_raw,
            }
            # Xの各ブランチを "x_branchName" のキーで保存
            for k, v in x_dict.items():
                save_kwargs[f"x_{k}"] = v
            # Yの各ブランチを "y_branchName" のキーで保存
            for k, v in y_dict.items():
                save_kwargs[f"y_{k}"] = v

            np.savez(os.path.join(output_dir, f"{name_stem}.npz"), **save_kwargs)
            feature_info["file_lengths"][name_stem] = total_frames
            print(f"Saved: {name_stem} (Frames: {total_frames})")

            # メモリを解放
            del raw_vsqx, norm_vsqx, x_dict, y_dict, save_kwargs
            gc.collect()

        except ConfigParameterError as e:
            # 設定ミスの場合は、即座にプログラム全体を停止する
            print(f"\n【致命的なエラー】{e}")
            print("全ファイルの処理を中断します。")
            sys.exit(1)
        except Exception as e:
            print(f"Skipped: {name_stem} ({type(e).__name__}: {e})")
            skipped_files.append(
                {
                    "file": name_stem,
                    "error_type": type(e).__name__,
                    "message": str(e),
                }
            )

    # --- メタデータの保存 ---
    with open(os.path.join(output_dir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=4, ensure_ascii=False)

    # --- スキップレポートの出力 ---
    num_success = len(file_pairs) - len(skipped_files)
    print(
        f"\nDataset generation completed. "
        f"Success: {num_success}/{len(file_pairs)}, "
        f"Skipped: {len(skipped_files)}/{len(file_pairs)}"
    )

    if skipped_files:
        report_path = os.path.join(output_dir, "skipped_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Skipped Files Report\n")
            f.write(f"====================\n")
            f.write(f"Total: {len(skipped_files)} file(s) skipped\n\n")
            for i, entry in enumerate(skipped_files, 1):
                f.write(f"[{i}] {entry['file']}\n")
                f.write(f"    Error: {entry['error_type']}\n")
                f.write(f"    Detail: {entry['message']}\n\n")
        print(f"Skipped report saved: {report_path}")


if __name__ == "__main__":
    build_dataset(
        wav_dir=config.WAV_DIR,
        vsqx_dir=config.VSQX_DIR,
        output_dir=config.OUTPUT_DIR,
    )
