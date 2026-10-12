"""
母音別パラメータ影響差異解析ツール (簡易版)
============================================
7つの母音条件から全ペアを比較し、パラメータの影響が母音によって異なるかを判定する。

計算:
  各母音条件について diff = (パラメータ付きWAV) - (デフォルトWAV) を算出し、
  母音ペアごとに以下の2枚の画像を出力する:

  画像B (diff_of_diffs): (あの差分) - (いの差分) を描画。
    → 真っ白(=ゼロ)なら母音による違いなし。色がつけば違いあり。

  画像A (comparison_overlay): あの差分 と いの差分 を重ねて/並べて描画。
    → 画像Bで違いが見えた場合に、具体的にどう異なるかを確認する。

特徴量の選出:
  FEATURE_GROUPS の各グループから、パラメータカーブとの相関 |r| が最大の
  特徴量を1つずつ選出する (compare.py と同じロジック)。

使い方:
  1. vowel_compare_config.py の設定を編集
  2. python compare_vowels.py を実行

出力 (母音ペア × パラメータ):
  - comparison_overlay_{PARAM}_{vowelA}_vs_{vowelB}.png   (画像A)
  - diff_of_diffs_{PARAM}_{vowelA}_vs_{vowelB}.png        (画像B)
"""

import os
import sys
import numpy as np
import traceback
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "util"))

import vowel_compare_config as cfg
from vsqx_parser import VSQXParser
from vsqx_compare import VSQXCompare
from audio_analyzer import AudioAnalyzer, FEATURE_DISPLAY_NAMES, FEATURE_PLOT_TYPES
from vowel_diff_visualizer import VowelDiffVisualizer



def _build_feature_list():
    """除外リストを適用した特徴量リストを構築"""
    exclude = set(getattr(cfg, "ALWAYS_EXCLUDE_FEATURES", []))
    return [f for f in cfg.ALL_FEATURE_TYPES if f not in exclude]


def _fuzzy_find_file(directory, prefix, suffix):
    """
    ファイル名のドット重複等に対応するファジー検索。
    {prefix}_default{suffix} を探し、見つからなければ
    {prefix}_default.{suffix} (ドット重複) も探す。
    """
    exact = os.path.join(directory, f"{prefix}_default{suffix}")
    if os.path.exists(exact):
        return exact

    # ドット重複パターン (.wav → ..wav 等) に対応
    double_dot = os.path.join(directory, f"{prefix}_default.{suffix}")
    if os.path.exists(double_dot):
        return double_dot

    # ディレクトリ内を部分一致で探索（最終手段）
    if os.path.isdir(directory):
        target_start = f"{prefix}_default"
        for fname in os.listdir(directory):
            if fname.startswith(target_start) and fname.endswith(suffix.lstrip(".")):
                return os.path.join(directory, fname)

    return None


def _find_default_files(prefix, data_dir):
    """母音条件に対応するデフォルトファイルを探索する（default/サブディレクトリ対応）"""
    default_wav_dir = getattr(cfg, "DEFAULT_DIR_WAV", "data/wav/default")
    default_vsqx_dir = getattr(cfg, "DEFAULT_DIR_VSQX", "data/vsqx/default")

    # 個別デフォルトを探索（default/ サブディレクトリ内）
    found_wav = _fuzzy_find_file(default_wav_dir, prefix, ".wav")
    found_vsqx = _fuzzy_find_file(default_vsqx_dir, prefix, ".vsqx")

    if found_wav and found_vsqx:
        return found_wav, found_vsqx

    # フォールバック: 共通デフォルトプレフィクスで再探索
    fb_prefix = getattr(cfg, "FALLBACK_DEFAULT_PREFIX", "あ")
    fb_wav = _fuzzy_find_file(default_wav_dir, fb_prefix, ".wav")
    fb_vsqx = _fuzzy_find_file(default_vsqx_dir, fb_prefix, ".vsqx")

    if fb_wav and fb_vsqx:
        print(f"    INFO: Using fallback default '{fb_prefix}' for prefix '{prefix}'")
        return fb_wav, fb_vsqx

    # 最終フォールバック（旧方式互換）
    legacy_wav_dir = os.path.join(data_dir, "wav")
    legacy_vsqx_dir = os.path.join(data_dir, "vsqx")
    legacy_wav = _fuzzy_find_file(legacy_wav_dir, prefix, ".wav")
    legacy_vsqx = _fuzzy_find_file(legacy_vsqx_dir, prefix, ".vsqx")
    if legacy_wav and legacy_vsqx:
        return legacy_wav, legacy_vsqx

    print(f"    WARNING: No default files found for prefix '{prefix}'")
    return None, None


def _find_param_files(prefix, param_name, data_dir):
    """母音条件×パラメータに対応するファイルを探索する"""
    # まず _1 付きを探す（旧形式）
    wav_path = os.path.join(data_dir, "wav", param_name, f"{prefix}_{param_name}_1.wav")
    vsqx_path = os.path.join(data_dir, "vsqx", param_name, f"{prefix}_{param_name}_1.vsqx")
    if os.path.exists(wav_path) and os.path.exists(vsqx_path):
        return wav_path, vsqx_path
    # _1 なしを探す（新形式）
    wav_path2 = os.path.join(data_dir, "wav", param_name, f"{prefix}_{param_name}.wav")
    vsqx_path2 = os.path.join(data_dir, "vsqx", param_name, f"{prefix}_{param_name}.vsqx")
    if os.path.exists(wav_path2) and os.path.exists(vsqx_path2):
        return wav_path2, vsqx_path2
    # どちらも見つからない場合は旧形式パスを返す（エラーメッセージ用）
    return wav_path, vsqx_path


def _get_target_params():
    """解析対象パラメータのリストを返す"""
    if "ALL" in cfg.TARGET_PARAMS:
        parser = VSQXParser(sr=cfg.SR, shift_length=cfg.SHIFT_LENGTH)
        return parser.vsqx_config["DYNAMIC_PARAM_IDS"]
    return cfg.TARGET_PARAMS


# =====================================================================
# 特徴量選出 (compare.py と同じロジック)
# =====================================================================

def _select_top_features(all_condition_results, feature_list):
    """
    全条件の平均相関 |r| に基づいて、各 FEATURE_GROUP から
    最も相関が高い特徴量を1つずつ選出する。
    compare.py の _select_top_features と同じ排他制御ロジック。
    """
    feature_groups = getattr(cfg, "FEATURE_GROUPS", {})
    max_per_group = getattr(cfg, "MAX_PER_GROUP", 1)
    top_n = getattr(cfg, "TOP_N_FEATURES", 5)

    # 逆引きマップ: feature_name -> group_name
    feat_to_group = {}
    for group_name, members in feature_groups.items():
        for member in members:
            feat_to_group[member] = group_name

    # 各特徴量の全条件平均 |r| を計算
    avg_corr = {}
    for fname in feature_list:
        if fname in ("waveform", "waveform_diff"):
            continue
        rs = []
        for cond_label, result in all_condition_results.items():
            corr = result.get("correlations", {})
            if fname in corr:
                rs.append(abs(corr[fname]["r"]))
        if rs:
            avg_corr[fname] = np.mean(rs)

    # 相関順にソート
    ranked = sorted(avg_corr.items(), key=lambda x: x[1], reverse=True)

    # グループ排他制御 + TOP_N 制限
    group_counts = {}
    selected = []
    for fname, avg_r in ranked:
        group = feat_to_group.get(fname)
        if group:
            cnt = group_counts.get(group, 0)
            if cnt >= max_per_group:
                continue
            group_counts[group] = cnt + 1
        selected.append(fname)
        if len(selected) >= top_n:
            break

    return selected


# =====================================================================
# 単一条件の解析
# =====================================================================

def _analyze_single_condition(condition, param_name, data_dir,
                              analyzer, reader, feature_list):
    """1つの母音条件に対する解析を実行し、結果辞書を返す"""
    prefix = condition["prefix"]
    label = condition["label"]

    # ファイルパスの解決
    default_wav, default_vsqx = _find_default_files(prefix, data_dir)
    param_wav, param_vsqx = _find_param_files(prefix, param_name, data_dir)

    # ファイル存在チェック
    if default_wav is None or default_vsqx is None:
        print(f"    WARNING: Default files not found for '{prefix}' → skipping '{label}'")
        return None
    for p in [default_wav, default_vsqx, param_wav, param_vsqx]:
        if not os.path.exists(p):
            print(f"    WARNING: File not found: {p} → skipping '{label}'")
            return None

    # VSQX解析
    vsqx_a = reader.read(default_vsqx)
    vsqx_b = reader.read(param_vsqx)
    if vsqx_a is None or vsqx_b is None:
        print(f"    WARNING: VSQX parse failed → skipping '{label}'")
        return None

    # 特徴量抽出（ペア）
    extract_features = feature_list.copy()
    if "waveform" not in extract_features:
        extract_features.append("waveform")

    features_a, features_b = analyzer.extract_features_paired(
        default_wav, param_wav, extract_features
    )

    # 差分計算
    diff_features = analyzer.compute_diff(features_a, features_b)

    # フレーム数決定
    all_lens = [f.shape[0] for f in features_a.values()]
    all_lens += [f.shape[0] for f in features_b.values()]
    total_frames = min(all_lens) if all_lens else 0

    # パラメータカーブ
    default_value = cfg.DEFAULT_DYNAMIC_VALUES.get(param_name, 64)
    control_points = vsqx_b["dynamic_params"].get(param_name, [])
    param_curve = reader.interpolate_to_frames(
        control_points, total_frames, default_value=default_value
    )

    # 相関分析
    correlations = {}
    if cfg.COMPUTE_CORRELATION:
        correlations = analyzer.compute_correlations(
            param_curve, diff_features, features_a, features_b
        )

    return {
        "label": label,
        "prefix": prefix,
        "condition": condition,
        "correlations": correlations,
        "features_a": features_a,
        "features_b": features_b,
        "diff_features": diff_features,
        "param_curve": param_curve,
        "default_value": default_value,
        "default_wav": default_wav,
        "param_wav": param_wav,
        "notes": vsqx_b.get("notes", []),
    }


# =====================================================================
# メイン
# =====================================================================

def main():
    print("=" * 60)
    print("  Vowel-wise Parameter Effect Comparison Tool (Simplified)")
    print("=" * 60)

    data_dir = getattr(cfg, "DATA_DIR", "data")
    output_dir = getattr(cfg, "OUTPUT_DIR", "data/output/vowel_compare")
    os.makedirs(output_dir, exist_ok=True)

    feature_list = _build_feature_list()
    target_params = _get_target_params()

    print(f"\n[Config]")
    print(f"  Data Dir:    {data_dir}")
    print(f"  Default WAV: {getattr(cfg, 'DEFAULT_DIR_WAV', 'data/wav/default')}")
    print(f"  Default VSQX:{getattr(cfg, 'DEFAULT_DIR_VSQX', 'data/vsqx/default')}")
    print(f"  Output Dir:  {output_dir}")
    print(f"  Parameters:  {target_params}")
    print(f"  Features:    {feature_list}")
    print(f"  Conditions:  {len(cfg.VOWEL_CONDITIONS)}")

    reader = VSQXCompare(sr=cfg.SR, shift_length=cfg.SHIFT_LENGTH)
    analyzer = AudioAnalyzer(sr=cfg.SR, shift_length=cfg.SHIFT_LENGTH)
    visualizer = VowelDiffVisualizer(sr=cfg.SR, shift_length=cfg.SHIFT_LENGTH)

    for param_name in target_params:
        print(f"\n{'='*60}")
        print(f"  Analyzing parameter: {param_name}")
        print(f"{'='*60}")

        # --- 全条件の解析 ---
        all_results = {}
        for cond in cfg.VOWEL_CONDITIONS:
            label = cond["label"]
            print(f"\n  [{label}]")
            try:
                result = _analyze_single_condition(
                    cond, param_name, data_dir,
                    analyzer, reader, feature_list
                )
                if result is not None:
                    all_results[label] = result
                    # 相関サマリーを表示
                    n_corr = len(result["correlations"])
                    top_corr = sorted(
                        [(f, abs(info["r"]))
                         for f, info in result["correlations"].items()
                         if f not in ("waveform", "waveform_diff")],
                        key=lambda x: x[1], reverse=True
                    )[:3]
                    top_str = ", ".join(
                        f"{FEATURE_DISPLAY_NAMES.get(f,f)}={r:.3f}"
                        for f, r in top_corr
                    )
                    print(f"    → {n_corr} features analyzed. Top |r|: {top_str}")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()

        if not all_results:
            print(f"\n  WARNING: No valid conditions for {param_name}. Skipping.")
            continue

        print(f"\n  Successfully analyzed {len(all_results)}/{len(cfg.VOWEL_CONDITIONS)} conditions.")

        # --- フォーカス特徴量の選出 ---
        focus_features = _select_top_features(all_results, feature_list)
        print(f"\n  Selected features (1 per group, by avg |r|):")
        feature_groups = getattr(cfg, "FEATURE_GROUPS", {})
        feat_to_group = {}
        for gn, members in feature_groups.items():
            for m in members:
                feat_to_group[m] = gn

        for fname in focus_features:
            group = feat_to_group.get(fname, "-")
            display = FEATURE_DISPLAY_NAMES.get(fname, fname)
            # 全条件の平均 |r|
            rs = []
            for result in all_results.values():
                corr = result.get("correlations", {})
                if fname in corr:
                    rs.append(abs(corr[fname]["r"]))
            avg_r = np.mean(rs) if rs else 0.0
            print(f"    [{group:>12}] {display:<25} avg|r|={avg_r:.4f}")

        # --- 全ペアの画像生成 ---
        cond_labels = list(all_results.keys())
        n_pairs = 0

        for label_1, label_2 in combinations(cond_labels, 2):
            result_1 = all_results[label_1]
            result_2 = all_results[label_2]

            # タイプが異なる場合（単母音 vs 連続歌唱など）は長さや性質が違うため比較をスキップ
            if result_1["condition"]["type"] != result_2["condition"]["type"]:
                continue

            # ファイル名用の短い識別子
            tag_1 = result_1["condition"]["vowel"]
            tag_2 = result_2["condition"]["vowel"]
            pair_tag = f"{tag_1}_vs_{tag_2}"

            print(f"\n  --- Comparing: {label_1} vs {label_2} ---")

            # 画像B: 差分の差分 (真っ白なら違いなし)
            path_b = os.path.join(
                output_dir,
                f"diff_of_diffs_{param_name}_{pair_tag}.png"
            )
            visualizer.plot_diff_of_diffs(
                result_1, result_2, focus_features, param_name,
                path_b, dpi=cfg.DPI
            )

            # 画像A: 比較オーバーレイ (どう違うかの詳細)
            path_a = os.path.join(
                output_dir,
                f"comparison_overlay_{param_name}_{pair_tag}.png"
            )
            visualizer.plot_comparison_overlay(
                result_1, result_2, focus_features, param_name,
                path_a, dpi=cfg.DPI
            )

            n_pairs += 1

        print(f"\n  Generated {n_pairs} pair comparisons × 2 images = {n_pairs * 2} files.")

    print("\n" + "=" * 60)
    print("  All analyses complete!")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
