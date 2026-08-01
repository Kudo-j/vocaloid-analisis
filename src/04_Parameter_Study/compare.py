"""
VOCALOIDパラメータ差分解析ツール
=================================
2つの歌唱音声（パラメータなし/あり）とVSQXファイルを入力し、
VSQXパラメータの時系列と音響特徴量の差分を可視化する。

使い方:
  1. compare_config.py の設定を編集
  2. python compare.py を実行
"""

import os
import sys
import json
import csv
import numpy as np

# srcをパスに追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "util"))

import compare_config as cfg
from vsqx_reader import VSQXReader
from audio_analyzer import AudioAnalyzer
from diff_visualizer import DiffVisualizer


def _build_feature_list():
    """ALL_FEATURE_TYPES から ALWAYS_EXCLUDE を除いた特徴量リストを構築"""
    exclude = set(getattr(cfg, "ALWAYS_EXCLUDE_FEATURES", []))
    features = [f for f in cfg.ALL_FEATURE_TYPES if f not in exclude]
    return features


def _select_top_features(sorted_features, correlations):
    """
    相関順にソートされた特徴量リストから、出力対象を選出する。
    - waveform は除外（別途先頭に表示するため）
    - FEATURE_GROUPS によるグループ内排他制御（多様性の確保）
    - TOP_N_FEATURES で上位N個を選出
    - ALWAYS_INCLUDE_FEATURES は順位・グループ制限を無視して追加
    """
    # waveform を除外
    ranked = [
        (f, info)
        for f, info in sorted_features
        if f not in ("waveform", "waveform_diff")
    ]

    top_n = getattr(cfg, "TOP_N_FEATURES", None)
    always_include = set(getattr(cfg, "ALWAYS_INCLUDE_FEATURES", []))
    feature_groups = getattr(cfg, "FEATURE_GROUPS", {})
    max_per_group = getattr(cfg, "MAX_PER_GROUP", 1)

    # 逆引きマップ: feature_name -> group_name
    feat_to_group = {}
    for group_name, members in feature_groups.items():
        for member in members:
            feat_to_group[member] = group_name

    # Step 1: グループ制限を適用（相関順を保持）
    group_counts = {}
    group_filtered = []

    for fname, info in ranked:
        # ALWAYS_INCLUDE はグループ制限をバイパス
        if fname in always_include:
            group_filtered.append((fname, info))
            continue

        group = feat_to_group.get(fname)
        if group:
            current = group_counts.get(group, 0)
            if current >= max_per_group:
                continue  # グループ枠超過 → スキップ
            group_counts[group] = current + 1

        group_filtered.append((fname, info))

    # Step 2: TOP_N 制限を適用
    if top_n is not None:
        selected = []
        count = 0
        for fname, info in group_filtered:
            if fname in always_include:
                # ALWAYS_INCLUDE はTOP_N枠を消費しない
                selected.append((fname, info))
            elif count < top_n:
                selected.append((fname, info))
                count += 1

        # ALWAYS_INCLUDE でまだ追加されていないものを補填
        selected_names = {f for f, _ in selected}
        for fname in always_include:
            if fname not in selected_names and fname in correlations:
                for f, info in ranked:
                    if f == fname:
                        selected.append((f, info))
                        break
        return selected
    else:
        return group_filtered


def main():
    print("=" * 60)
    print("  VOCALOID Parameter Diff Analysis Tool")
    print("=" * 60)

    # 設定ファイルから直接パスを取得
    path_wav_a = cfg.WAV_A
    path_wav_b = cfg.WAV_B
    path_vsqx_a = cfg.VSQX_A
    path_vsqx_b = cfg.VSQX_B
    output_dir = cfg.OUTPUT_DIR

    # 特徴量リストを構築
    compute_features = _build_feature_list()

    # --- 1. 設定確認 ---
    print(f"\n[Config]")
    print(f"  WAV_A:  {path_wav_a}")
    print(f"  WAV_B:  {path_wav_b}")
    print(f"  VSQX_A: {path_vsqx_a}")
    print(f"  VSQX_B: {path_vsqx_b}")
    print(f"  Compute Features: {compute_features}")
    print(f"  Top N Output: {cfg.TOP_N_FEATURES or 'All'}")
    print(f"  Always Include: {cfg.ALWAYS_INCLUDE_FEATURES}")
    print(f"  Always Exclude: {cfg.ALWAYS_EXCLUDE_FEATURES}")
    print(f"  Target Param: {cfg.TARGET_PARAM or 'Auto-detect'}")

    # ファイル存在チェック
    for path in [path_wav_a, path_wav_b, path_vsqx_a, path_vsqx_b]:
        if not os.path.exists(path):
            print(f"\nERROR: File not found: {path}")
            sys.exit(1)

    # --- 2. VSQX解析 ---
    print("\n[Step 1] Parsing VSQX files...")
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    vsqx_data_a = reader.read_sparse(path_vsqx_a)
    vsqx_data_b = reader.read_sparse(path_vsqx_b)

    if vsqx_data_a is None or vsqx_data_b is None:
        print("ERROR: Failed to parse VSQX files.")
        sys.exit(1)

    # read_sparse の結果に可視化用フィールドを追加
    # (diff_visualizer は start_sec / end_sec / tempo を参照するため)
    for data in [vsqx_data_a, vsqx_data_b]:
        data["tempo"] = data["tempo_map"][0][1] if data["tempo_map"] else 120.0
        for note in data["notes"]:
            note["start_sec"] = reader._tick_to_second(
                note["abs_start_tick"], data["tempo_map"], data["resolution"]
            )
            note["end_sec"] = reader._tick_to_second(
                note["abs_end_tick"], data["tempo_map"], data["resolution"]
            )

    print(
        f"  VSQX_A: tempo={vsqx_data_a['tempo_map'][0][1]}BPM, "
        f"notes={len(vsqx_data_a['notes'])}, "
        f"dynamic_params={list(vsqx_data_a['dynamic_params'].keys())}"
    )
    print(
        f"  VSQX_B: tempo={vsqx_data_b['tempo_map'][0][1]}BPM, "
        f"notes={len(vsqx_data_b['notes'])}, "
        f"dynamic_params={list(vsqx_data_b['dynamic_params'].keys())}"
    )

    # --- 3. パラメータ決定 ---
    if cfg.TARGET_PARAM is not None:
        target_params = [cfg.TARGET_PARAM]
        print(f"\n[Step 2] Target parameter: {cfg.TARGET_PARAM} (user-specified)")
    else:
        target_params = reader.find_changed_params(vsqx_data_a, vsqx_data_b)
        if not target_params:
            print("\nWARNING: No parameter differences found between VSQX files.")
            print("Using all dynamic params from VSQX_B for analysis.")
            target_params = list(vsqx_data_b["dynamic_params"].keys())

        if not target_params:
            print("ERROR: No dynamic parameters found in either VSQX file.")
            sys.exit(1)

        print(f"\n[Step 2] Auto-detected changed params: {target_params}")

    # 静的パラメータ差分
    static_diffs = reader.find_static_param_diffs(vsqx_data_a, vsqx_data_b)
    if static_diffs:
        print(f"  Static param diffs: {static_diffs}")
    else:
        print("  Static params: no differences")

    # --- 4. 音響特徴量抽出 ---
    # waveform は常に抽出（表示専用、相関ランキング対象外）
    extract_features = compute_features.copy()
    if "waveform" not in extract_features:
        extract_features.append("waveform")

    print(f"\n[Step 3] Extracting audio features (Paired for consistent dB ref)...")
    print(f"  Target features ({len(extract_features)}): {extract_features}")
    analyzer = AudioAnalyzer(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    print(f"  Processing both files to find common reference...")
    features_a, features_b = analyzer.extract_features_paired(
        path_wav_a, path_wav_b, extract_features
    )
    print(f"    -> Extracted {len(features_a)} features from each file")

    # --- 5. 差分計算 ---
    print("\n[Step 4] Computing feature differences...")
    diff_features = analyzer.compute_diff(features_a, features_b)
    print(f"  Computed diffs for: {list(diff_features.keys())}")

    # --- 6. 各パラメータについて解析・可視化 ---
    os.makedirs(output_dir, exist_ok=True)
    visualizer = DiffVisualizer(
        sr=cfg.SR, hop_length=cfg.HOP_LENGTH, figsize=cfg.FIGSIZE, dpi=cfg.DPI
    )

    for param_name in target_params:
        print(f"\n[Step 5] Analyzing parameter: {param_name}")

        # パラメータカーブを取得
        default_value = cfg.DEFAULT_DYNAMIC_VALUES.get(param_name, 64)

        # 総フレーム数を決定（特徴量の最短に合わせる）
        all_lens = []
        for feat in features_a.values():
            all_lens.append(feat.shape[0])
        for feat in features_b.values():
            all_lens.append(feat.shape[0])
        total_frames = min(all_lens) if all_lens else 0

        # VSQXパラメータをフレーム単位に補間
        control_points_b = vsqx_data_b["dynamic_params"].get(param_name, [])
        param_curve = reader.interpolate_to_frames(
            control_points_b, vsqx_data_b["tempo_map"], vsqx_data_b["resolution"],
            total_frames, default_value=default_value
        )

        # --- 6a. 相関分析 ---
        correlations = {}
        sorted_features = []

        if cfg.COMPUTE_CORRELATION:
            print(f"  Computing correlations for {param_name}...")
            correlations = analyzer.compute_correlations(
                param_curve, diff_features, features_a, features_b
            )

            if cfg.SORT_BY_CORRELATION:
                all_sorted = analyzer.sort_by_correlation(correlations)
            else:
                all_sorted = list(correlations.items())

            # 全相関結果を表示
            from audio_analyzer import FEATURE_DISPLAY_NAMES

            feature_groups = getattr(cfg, "FEATURE_GROUPS", {})
            feat_to_group = {}
            for gn, members in feature_groups.items():
                for m in members:
                    feat_to_group[m] = gn

            print(
                f"\n  {'Feature':<25} {'Group':<14} {'r':>8} {'|r|':>8} {'Strength':>10} {'Output':>8}"
            )
            print(f"  {'-'*80}")

            # 出力対象を選出
            sorted_features = _select_top_features(all_sorted, correlations)
            output_names = {f for f, _ in sorted_features}

            for fname, info in all_sorted:
                if fname in ("waveform", "waveform_diff"):
                    continue
                display = FEATURE_DISPLAY_NAMES.get(fname, fname)
                group = feat_to_group.get(fname, "-")
                r = info["r"]
                r_abs = abs(r)
                strength = (
                    "Strong"
                    if r_abs >= 0.7
                    else ("Moderate" if r_abs >= 0.4 else "Weak")
                )
                mark = "  ✓" if fname in output_names else ""
                print(
                    f"  {display:<25} {group:<14} {r:>8.4f} {r_abs:>8.4f} {strength:>10} {mark:>8}"
                )

            # waveform を先頭に挿入（波形オーバーレイのみ、差分なし）
            sorted_features.insert(
                0, ("waveform", {"r": 0.0, "p": 1.0, "r_per_band": None})
            )

        else:
            # 相関なしの場合はconfig順で表示
            sorted_features = [
                (fname, {"r": 0.0, "p": 1.0, "r_per_band": None})
                for fname in compute_features
                if fname in diff_features
            ]
            # waveform を先頭に挿入
            sorted_features.insert(
                0, ("waveform", {"r": 0.0, "p": 1.0, "r_per_band": None})
            )

        # --- 6b. 可視化 ---
        base_b = os.path.splitext(os.path.basename(cfg.WAV_B))[0]
        output_path = os.path.join(output_dir, f"{base_b}_diff_{param_name}.png")
        print(f"  Generating visualization ({len(sorted_features)} plots)...")

        visualizer.plot(
            param_name=param_name,
            vsqx_data_a=vsqx_data_a,
            vsqx_data_b=vsqx_data_b,
            features_a=features_a,
            features_b=features_b,
            diff_features=diff_features,
            correlations=correlations,
            sorted_features=sorted_features,
            param_curve=param_curve,
            default_value=default_value,
            static_diffs=static_diffs,
            file_a_name=os.path.basename(cfg.WAV_A),
            file_b_name=os.path.basename(cfg.WAV_B),
            output_path=output_path,
        )

        # --- 6c. CSV出力 ---
        if cfg.COMPUTE_CORRELATION:
            csv_path = os.path.join(
                output_dir, f"{base_b}_correlation_{param_name}.csv"
            )
            _save_correlation_csv(csv_path, sorted_features, param_name)
            print(f"  Saved CSV: {csv_path}")

        # --- 6d. JSONメタデータ ---
        json_path = os.path.join(output_dir, f"{base_b}_analysis_{param_name}.json")
        _save_analysis_json(
            json_path,
            param_name,
            cfg,
            vsqx_data_a,
            vsqx_data_b,
            correlations,
            static_diffs,
            compute_features,
        )
        print(f"  Saved JSON: {json_path}")

    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    print("=" * 60)


def _save_correlation_csv(csv_path, sorted_features, param_name):
    """相関係数テーブルをCSV出力"""
    from audio_analyzer import FEATURE_DISPLAY_NAMES

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["parameter", "feature", "r_value", "abs_r", "p_value", "strength"]
        )

        for fname, info in sorted_features:
            if fname in ("waveform", "waveform_diff"):
                continue
            display = FEATURE_DISPLAY_NAMES.get(fname, fname)
            r = info["r"]
            r_abs = abs(r)
            p = info["p"]
            strength = (
                "Strong" if r_abs >= 0.7 else ("Moderate" if r_abs >= 0.4 else "Weak")
            )
            writer.writerow(
                [param_name, display, f"{r:.6f}", f"{r_abs:.6f}", f"{p:.6f}", strength]
            )


def _save_analysis_json(
    json_path,
    param_name,
    cfg,
    vsqx_a,
    vsqx_b,
    correlations,
    static_diffs,
    computed_features,
):
    """解析メタデータをJSON出力"""
    meta = {
        "param_name": param_name,
        "wav_a": cfg.WAV_A,
        "wav_b": cfg.WAV_B,
        "vsqx_a": cfg.VSQX_A,
        "vsqx_b": cfg.VSQX_B,
        "sr": cfg.SR,
        "hop_length": cfg.HOP_LENGTH,
        "feature_types": computed_features,
        "top_n": getattr(cfg, "TOP_N_FEATURES", None),
        "always_include": getattr(cfg, "ALWAYS_INCLUDE_FEATURES", []),
        "always_exclude": getattr(cfg, "ALWAYS_EXCLUDE_FEATURES", []),
        "tempo": vsqx_b["tempo_map"][0][1] if vsqx_b.get("tempo_map") else 0,
        "voice_a": vsqx_a.get("voice_info", {}),
        "voice_b": vsqx_b.get("voice_info", {}),
        "static_param_diffs": {
            k: {"a": v[0], "b": v[1]} for k, v in static_diffs.items()
        },
        "correlations": {
            fname: {"r": info["r"], "p": info["p"]}
            for fname, info in correlations.items()
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
