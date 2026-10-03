"""
randomize_pitch.py
==================
VOCALOID (vsq4) ファイル内の音高（ノート番号）をランダマイズして
Step 3（音高多様化実験）用のVSQXデータセットを一括生成するスクリプト。

主な機能:
  1. サブトラック自動削除 (Track 0 のみ保持):
     remove_subtracks.py のロジックを取り込み、入力VSQX内の Track 0 (メイン) のみを
     抽出して Track 1 以降を安全に削除（1ファイル = 1トラック化）。
  2. ラテン方陣に基づく音素×音高の直交・層化配分:
     - Train: 同一DYNパターン番号間で5母音の音高が重複しないよう排他制御しつつ、
       各母音内で音高を均等配分（37音階中30音高を使用、7音高が自動的に未網羅/Unseen化）。
     - Valid / Test: Trainで網羅できなかった未知音高（Unseen）を指定割合（unseen_ratio）
       割り当て、残りを既知音高（Seen）から割り当て。同一DYNパターン内での重複も完全回避。
  3. 音高網羅ベンチマークデータ（Benchmark Test）の一括生成:
     - 代表DYNパターン（デフォルト: Test_8系）× 5母音 × 全音階（37音高）の計185曲を
       別枠（benchmark/ サブフォルダ）に一括生成。
  4. ファイル名への音名付与:
     - 通常データ: 例 Step3_Train1_あ_1_C3.vsqx
     - ベンチマークデータ: 例 Step3_Benchmark_Test_あ_8_C3.vsqx
  5. ノートタグ (<n>, <noteNum>) を正規表現で安全に置換（CDATA・XML書式を完全維持）。
  6. 乱数シード固定による完全な再現性。
  7. 詳細メタデータログ (CSV) 出力（Seen / Unseen フラグ付き）。

使い方:
  1. 下記の CONFIG 領域を確認・編集する
  2. 実行ボタンを押す (python randomize_pitch.py)
"""

import os
import re
import io
import csv
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict


# =============================================================================
# 設定領域 (CONFIG) - 実行ボタンを押すだけで動作可能
# =============================================================================
CONFIG = {
    # (必須) 入力VSQXフォルダ (Step2の分割済みVSQX)
    "input_dir": "./data/Step2/vsqx",

    # (必須) 出力先フォルダ (Step 3のVSQXデータ保存先)
    "output_dir": "./data/Step_org/Step3_60-96",

    # 音高範囲 (Yamaha規格: C3=60, C6=96, 両端含む37段階)
    "min_note_num": 60,  # C3
    "max_note_num": 96,  # C6

    # 未網羅パターンの配分割合 (Valid/Testで未網羅音高を割り当てる割合, 例: 0.3 = 30%)
    "unseen_ratio": 0.3,

    # ベンチマーク代表DYNパターン番号 (例: 8 -> Test_8系のデータ)
    "benchmark_rep_pattern": 8,

    # 乱数シード (再現性確保)
    "random_seed": 42,

    # 通常データのファイル名接頭辞 (例: "Step3_")
    "file_prefix": "Step3_",

    # ベンチマークデータのファイル名接頭辞
    "benchmark_prefix": "Step3_Benchmark_",

    # ベンチマークデータの保存サブフォルダ名
    "benchmark_subdir": "benchmark",

    # ログCSVのファイル名 (出力フォルダ直下に保存)
    "log_csv_name": "pitch_randomize_log.csv",

    # ドライラン設定 (Trueの場合はファイル書き込みを行わずログ・配分のみ表示)
    "dry_run": False,
}
# =============================================================================


# 音名リスト (Yamaha表記: ノート番号 60 = C3, 69 = A3)
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
VOWEL_LIST = ["あ", "い", "う", "え", "お"]
VSQ4_XMLNS = "http://www.yamaha.co.jp/vocaloid/schema/vsq4/"


def _resolve_dir(path_str: str, base_ref_dir: str = None) -> str:
    """実行環境に左右されず、dataディレクトリ等を自動探索して絶対パスを返す。"""
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # 基準ディレクトリ（input_dir等）がある場合、その共通親ディレクトリから相対パスを解決
    if base_ref_dir and os.path.exists(base_ref_dir):
        norm_ref = os.path.normpath(base_ref_dir)
        parts = norm_ref.split(os.sep)
        if "data" in parts:
            data_idx = len(parts) - 1 - parts[::-1].index("data")
            data_root = os.sep.join(parts[: data_idx + 1])
            norm_path = os.path.normpath(path_str)
            sub_parts = norm_path.split(os.sep)
            if "data" in sub_parts:
                sub_idx = sub_parts.index("data")
                candidate = os.path.join(data_root, *sub_parts[sub_idx + 1 :])
                return os.path.abspath(candidate)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for start in [os.getcwd(), script_dir]:
        curr = start
        for _ in range(4):
            clean_rel = path_str.lstrip("./\\")
            candidate = os.path.normpath(os.path.join(curr, clean_rel))
            if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
                return os.path.abspath(candidate)
            curr = os.path.dirname(curr)

    return os.path.abspath(path_str)


def note_num_to_name(note_num: int) -> str:
    """MIDIノート番号をYamaha音名表記 (例: 60 -> C3, 69 -> A3) に変換する。"""
    pitch = PITCH_NAMES[note_num % 12]
    octave = (note_num // 12) - 2  # 60 // 12 - 2 = 3 (C3)
    return f"{pitch}{octave}"


def note_num_to_hz(note_num: int) -> float:
    """MIDIノート番号を周波数 (Hz, A4=440Hz基準) に変換する。"""
    return 440.0 * (2.0 ** ((note_num - 69) / 12.0))


def remove_subtracks_from_vsqx_content(
    xml_content: str, xmlns: str = VSQ4_XMLNS
) -> str:
    """
    remove_subtracks.py のロジックを取り込み環境適応:
    VSQX XMLテキストから Track 0 以外の vsTrack および vsUnit を削除し、
    Track 0 のみを含む VSQX XML テキストを返す。
    """
    ET.register_namespace("", xmlns)
    ns = {"vsq": xmlns}
    root = ET.fromstring(xml_content)

    # vsTrackタグをすべて探し、tNoが0でないものを削除
    tracks_to_remove = []
    for track in root.findall("vsq:vsTrack", ns):
        tno_elem = track.find("vsq:tNo", ns)
        if tno_elem is not None and tno_elem.text != "0":
            tracks_to_remove.append(track)

    for track in tracks_to_remove:
        root.remove(track)

    # mixerタグの中のvsUnitもtNoが0でないものを削除
    mixer = root.find("vsq:mixer", ns)
    if mixer is not None:
        units_to_remove = []
        for unit in mixer.findall("vsq:vsUnit", ns):
            tno_elem = unit.find("vsq:tNo", ns)
            if tno_elem is not None and tno_elem.text != "0":
                units_to_remove.append(unit)

        for unit in units_to_remove:
            mixer.remove(unit)

    out_io = io.BytesIO()
    tree = ET.ElementTree(root)
    tree.write(out_io, encoding="utf-8", xml_declaration=True)
    return out_io.getvalue().decode("utf-8")


def parse_vsqx_metadata(filename: str) -> dict:
    """ファイル名から split, vowel, dyn_pattern_id を正確に抽出する。"""
    stem = Path(filename).stem

    # 母音 (あ〜お)
    vowel = "Other"
    for v in VOWEL_LIST:
        if v in stem:
            vowel = v
            break

    # Split (Train / Valid / Test)
    f_lower = stem.lower()
    if "test" in f_lower:
        split = "Test"
    elif "valid" in f_lower:
        split = "Valid"
    elif "train" in f_lower:
        split = "Train"
    else:
        split = "Other"

    # DYNパターンID: 母音部分を除去した共通パターン名
    # 例: "Train1_あ_1" -> "Train1_1", "Test_あ_8" -> "Test_8"
    if vowel != "Other":
        dyn_id = re.sub(rf"_{re.escape(vowel)}_", "_", stem)
        dyn_id = re.sub(rf"_{re.escape(vowel)}$", "", dyn_id)
        dyn_id = re.sub(rf"^{re.escape(vowel)}_", "", dyn_id)
    else:
        dyn_id = stem

    return {
        "filename": filename,
        "stem": stem,
        "split": split,
        "vowel": vowel,
        "dyn_pattern_id": dyn_id,
    }


def replace_note_number(content: str, target_note: int) -> tuple[str, int]:
    """
    VSQX内のノートタグ (<n>, <noteNum>) を指定した音高で置換する。
    """
    pattern_note = re.compile(r"<(n|noteNum)>(-?\d+)</\1>")
    replaced_count = 0

    def _sub_note(m_note: re.Match) -> str:
        nonlocal replaced_count
        replaced_count += 1
        tag = m_note.group(1)
        return f"<{tag}>{target_note}</{tag}>"

    new_content = pattern_note.sub(_sub_note, content)
    return new_content, replaced_count


def _allocate_split_with_unseen(
    split_files: list[dict],
    unseen_ratio: float,
    seen_notes_per_vowel: dict[str, set[int]],
    unseen_notes_per_vowel: dict[str, set[int]],
    rng: random.Random,
) -> dict[str, int]:
    """
    Valid または Test ファイル群に対し、各母音ごとに
    unseen_ratio の割合で未網羅音高（Unseen）をサンプリングし、
    残りを既知音高（Seen）からサンプリングした上で、
    同一DYNパターン内での母音間重複をスワップで解消して割り当てる。
    """
    dyn_pattern_ids = sorted(list(set(f["dyn_pattern_id"] for f in split_files)))
    num_pats = len(dyn_pattern_ids)
    unseen_k = round(num_pats * unseen_ratio)
    seen_k = num_pats - unseen_k

    # 各母音ごとに選択する音高リスト
    vowel_notes = {}
    for v in VOWEL_LIST:
        avail_u = sorted(unseen_notes_per_vowel.get(v, set()))
        avail_s = sorted(seen_notes_per_vowel.get(v, set()))

        u_count = min(unseen_k, len(avail_u))
        s_count = min(seen_k, len(avail_s))

        u_chosen = rng.sample(avail_u, u_count) if u_count > 0 else []
        s_chosen = rng.sample(avail_s, s_count) if s_count > 0 else []

        combined = u_chosen + s_chosen
        # 足りない場合はSeenから補充
        while len(combined) < num_pats and avail_s:
            rem = [n for n in avail_s if n not in combined]
            if rem:
                combined.append(rng.choice(rem))
            else:
                combined.append(rng.choice(avail_s))

        rng.shuffle(combined)
        vowel_notes[v] = combined

    # パターン行への初期配置
    # grid: [{母音: note}, ...] (行数: num_pats)
    grid = [
        {v: vowel_notes[v][p_idx] for v in VOWEL_LIST} for p_idx in range(num_pats)
    ]

    # 行内の重複をスワップで解消
    for _ in range(300):
        has_conflict = False
        for p_idx in range(num_pats):
            row_notes = [grid[p_idx][v] for v in VOWEL_LIST]
            if len(row_notes) != len(set(row_notes)):
                has_conflict = True
                seen = {}
                for v in VOWEL_LIST:
                    n = grid[p_idx][v]
                    if n in seen:
                        v_conflict = v
                        for p_other in range(num_pats):
                            if p_other != p_idx:
                                cand = grid[p_other][v_conflict]
                                row_other = [
                                    grid[p_other][v2]
                                    for v2 in VOWEL_LIST
                                    if v2 != v_conflict
                                ]
                                row_this = [
                                    grid[p_idx][v2]
                                    for v2 in VOWEL_LIST
                                    if v2 != v_conflict
                                ]
                                if cand not in row_this and n not in row_other:
                                    grid[p_idx][v_conflict], grid[p_other][v_conflict] = cand, n
                                    break
                        break
                    else:
                        seen[n] = v
        if not has_conflict:
            break

    # ファイル名への割り当てマップを構築
    # pattern_map: {dyn_pattern_id: {vowel: note}}
    pattern_map = {
        dyn_pattern_ids[p_idx]: grid[p_idx] for p_idx in range(num_pats)
    }

    assignments = {}
    for f in split_files:
        assigned = pattern_map[f["dyn_pattern_id"]][f["vowel"]]
        assignments[f["filename"]] = assigned

    return assignments


def allocate_all_pitches(
    files_metadata: list[dict],
    available_notes: list[int],
    unseen_ratio: float,
    rng: random.Random,
) -> tuple[dict[str, int], dict[str, set[int]], dict[str, set[int]]]:
    """
    全データ分割 (Train/Valid/Test) に対して音高を配分する。
    - Train: ラテン方陣的シフトで同一DYNパターン間の重複を完全回避。
    - Valid / Test: unseen_ratio に応じた未網羅音高と既知音高を配分。
    """
    assignments = {}
    seen_notes_per_vowel = {v: set() for v in VOWEL_LIST}
    unseen_notes_per_vowel = {v: set() for v in VOWEL_LIST}

    split_files = defaultdict(list)
    for fmeta in files_metadata:
        split_files[fmeta["split"]].append(fmeta)

    # 1. Trainデータの割り当て (ラテン方陣配分)
    train_files = split_files.get("Train", [])
    if train_files:
        train_dyn_ids = sorted(list(set(f["dyn_pattern_id"] for f in train_files)))
        num_train_pats = len(train_dyn_ids)
        num_notes = len(available_notes)

        # 37音階のランダム順列
        perm = list(available_notes)
        rng.shuffle(perm)

        # 37と互いに素なステップで各母音にシフト適用 (例: 7)
        shift_step = 7

        train_pattern_map = {}
        for k_idx, dyn_id in enumerate(train_dyn_ids):
            train_pattern_map[dyn_id] = {}
            for v_idx, v in enumerate(VOWEL_LIST):
                note_idx = (k_idx + v_idx * shift_step) % num_notes
                note = perm[note_idx]
                train_pattern_map[dyn_id][v] = note
                seen_notes_per_vowel[v].add(note)

        for f in train_files:
            assignments[f["filename"]] = train_pattern_map[f["dyn_pattern_id"]][f["vowel"]]

        for v in VOWEL_LIST:
            unseen_notes_per_vowel[v] = set(available_notes) - seen_notes_per_vowel[v]

    # 2. Validデータの割り当て
    valid_files = split_files.get("Valid", [])
    if valid_files:
        valid_assigns = _allocate_split_with_unseen(
            valid_files, unseen_ratio, seen_notes_per_vowel, unseen_notes_per_vowel, rng
        )
        assignments.update(valid_assigns)

    # 3. Testデータの割り当て
    test_files = split_files.get("Test", [])
    if test_files:
        test_assigns = _allocate_split_with_unseen(
            test_files, unseen_ratio, seen_notes_per_vowel, unseen_notes_per_vowel, rng
        )
        assignments.update(test_assigns)

    return assignments, seen_notes_per_vowel, unseen_notes_per_vowel


def process_randomize_pitch(config: dict):
    """メイン処理関数"""
    input_dir = _resolve_dir(config["input_dir"])
    output_dir = _resolve_dir(config["output_dir"], base_ref_dir=input_dir)
    min_note = config.get("min_note_num", 60)
    max_note = config.get("max_note_num", 96)
    unseen_ratio = config.get("unseen_ratio", 0.3)
    benchmark_rep_pattern = config.get("benchmark_rep_pattern", 8)
    seed = config.get("random_seed", 42)
    file_prefix = config.get("file_prefix", "Step3_")
    benchmark_prefix = config.get("benchmark_prefix", "Step3_Benchmark_")
    benchmark_subdir = config.get("benchmark_subdir", "benchmark")
    log_csv_name = config.get("log_csv_name", "pitch_randomize_log.csv")
    dry_run = config.get("dry_run", False)

    print("=" * 68)
    print("  VOCALOID VSQX Track-Level Pitch Randomizer (Step 3)")
    print("=" * 68)
    print(f"Input Directory       : {input_dir}")
    print(f"Output Directory      : {output_dir}")
    print(f"Benchmark Subdir      : {benchmark_subdir}")
    print(
        f"Pitch Range           : Note {min_note} ({note_num_to_name(min_note)}) 〜 "
        f"Note {max_note} ({note_num_to_name(max_note)}) [{max_note - min_note + 1} steps]"
    )
    print(f"Unseen Ratio          : {unseen_ratio * 100:.1f}%")
    print(f"Benchmark Rep Pattern : Test_{benchmark_rep_pattern} 系")
    print(f"Random Seed           : {seed}")
    print(f"Dry Run Mode          : {dry_run}")
    print("=" * 68)

    if not os.path.exists(input_dir):
        print(f"[Error] 入力ディレクトリが存在しません: {input_dir}")
        return False

    vsqx_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".vsqx")
        ]
    )

    if not vsqx_files:
        print(f"[Warning] 入力ディレクトリ内に .vsqx ファイルが見つかりませんでした: {input_dir}")
        return False

    # 1. 各ファイルのメタデータを走査
    files_metadata = []
    for path in vsqx_files:
        filename = os.path.basename(path)
        meta = parse_vsqx_metadata(filename)
        meta["path"] = path
        files_metadata.append(meta)

    print(f"対象VSQXファイル数: {len(vsqx_files)} 件\n")

    # 2. 音高の割り当て (ラテン方陣配分 + Unseen配分)
    available_notes = list(range(min_note, max_note + 1))
    rng = random.Random(seed)

    assignments, seen_notes_per_vowel, unseen_notes_per_vowel = allocate_all_pitches(
        files_metadata, available_notes, unseen_ratio, rng
    )

    # 3. 出力先ディレクトリの作成
    benchmark_dir = os.path.join(output_dir, benchmark_subdir)
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(benchmark_dir, exist_ok=True)

    # 4. 通常データおよびベンチマークデータの書き出し
    log_rows = []
    success_count = 0
    fail_count = 0

    print("VSQXファイル処理中 (Sub-track削除 & 音高置換)...")

    # (A) 通常データの処理
    for f_idx, fmeta in enumerate(files_metadata):
        orig_filename = fmeta["filename"]
        stem = fmeta["stem"]
        split = fmeta["split"]
        vowel = fmeta["vowel"]
        dyn_id = fmeta["dyn_pattern_id"]
        path = fmeta["path"]

        assigned_note = assignments[orig_filename]
        assigned_name = note_num_to_name(assigned_note)
        hz = note_num_to_hz(assigned_note)
        is_seen = (assigned_note in seen_notes_per_vowel.get(vowel, set()))

        out_filename = f"{file_prefix}{stem}_{assigned_name}.vsqx"
        out_path = os.path.join(output_dir, out_filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # Sub-track削除 (Track 0 のみ保持)
            single_track_content = remove_subtracks_from_vsqx_content(raw_content)

            # 音高置換
            new_content, replaced_count = replace_note_number(
                single_track_content, assigned_note
            )

            if not dry_run:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

            log_rows.append(
                {
                    "file_index": len(log_rows) + 1,
                    "original_file": orig_filename,
                    "output_file": out_filename,
                    "is_benchmark": False,
                    "split": split,
                    "vowel": vowel,
                    "dyn_pattern_id": dyn_id,
                    "assigned_note": assigned_note,
                    "assigned_note_name": assigned_name,
                    "frequency_hz": round(hz, 2),
                    "is_seen": is_seen,
                }
            )
            success_count += 1

        except Exception as e:
            print(f"  [Error] 失敗: {orig_filename} -> {e}")
            fail_count += 1

    # (B) 音高網羅ベンチマークデータの処理 (Test_8系 代表曲 × 37音階)
    rep_pattern_str = f"Test_{benchmark_rep_pattern}"
    benchmark_source_files = [
        f for f in files_metadata
        if f["split"] == "Test" and f["dyn_pattern_id"] == rep_pattern_str
    ]

    print(f"\n音高網羅ベンチマークデータ生成中 (代表: {rep_pattern_str}, 5母音 × {len(available_notes)}音高)...")
    for fmeta in benchmark_source_files:
        orig_filename = fmeta["filename"]
        stem = fmeta["stem"]
        vowel = fmeta["vowel"]
        dyn_id = fmeta["dyn_pattern_id"]
        path = fmeta["path"]

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            single_track_content = remove_subtracks_from_vsqx_content(raw_content)

            for note in available_notes:
                note_name = note_num_to_name(note)
                hz = note_num_to_hz(note)
                is_seen = (note in seen_notes_per_vowel.get(vowel, set()))

                out_filename = f"{benchmark_prefix}{stem}_{note_name}.vsqx"
                out_path = os.path.join(benchmark_dir, out_filename)

                new_content, replaced_count = replace_note_number(
                    single_track_content, note
                )

                if not dry_run:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

                log_rows.append(
                    {
                        "file_index": len(log_rows) + 1,
                        "original_file": orig_filename,
                        "output_file": out_filename,
                        "is_benchmark": True,
                        "split": "Benchmark",
                        "vowel": vowel,
                        "dyn_pattern_id": dyn_id,
                        "assigned_note": note,
                        "assigned_note_name": note_name,
                        "frequency_hz": round(hz, 2),
                        "is_seen": is_seen,
                    }
                )
                success_count += 1

        except Exception as e:
            print(f"  [Error] ベンチマーク生成失敗: {orig_filename} -> {e}")
            fail_count += 1

    # 5. ログCSV出力
    log_path = os.path.join(output_dir, log_csv_name)
    if not dry_run and log_rows:
        with open(log_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nログ保存完了: {log_path}")

    # 6. 統計サマリー表示
    print("\n" + "=" * 68)
    print("  処理結果サマリー")
    print("=" * 68)
    print(
        f"生成ファイル数: 成功 {success_count} 件, 失敗 {fail_count} 件 (総レコード: {len(log_rows)})"
    )

    # Split別のSeen/Unseen統計
    split_seen_stats = defaultdict(lambda: {"Seen": 0, "Unseen": 0, "Total": 0})
    for row in log_rows:
        s = row["split"]
        status = "Seen" if row["is_seen"] else "Unseen"
        split_seen_stats[s][status] += 1
        split_seen_stats[s]["Total"] += 1

    for s, st in sorted(split_seen_stats.items()):
        seen_pct = (st["Seen"] / st["Total"] * 100) if st["Total"] > 0 else 0
        unseen_pct = (st["Unseen"] / st["Total"] * 100) if st["Total"] > 0 else 0
        print(
            f"  - [{s}] 計 {st['Total']} 件 | Seen: {st['Seen']} 件 ({seen_pct:.1f}%), "
            f"Unseen: {st['Unseen']} 件 ({unseen_pct:.1f}%)"
        )

    # 同一DYNパターン内の音高重複チェック (通常データ)
    pattern_note_map = defaultdict(lambda: defaultdict(list))
    for row in log_rows:
        if not row["is_benchmark"]:
            pattern_note_map[row["split"]][row["dyn_pattern_id"]].append(row["assigned_note"])

    has_dup = False
    for s, pmap in pattern_note_map.items():
        for pid, notes in pmap.items():
            if len(notes) != len(set(notes)):
                has_dup = True
                break

    if not has_dup:
        print("  ★ 同一DYNパターン内の全母音間で音高重複なし (完全排他・多様性を保証)")
    else:
        print("  ※ 一部パターンで音高重複が検出されました")

    print("=" * 68)
    return True


if __name__ == "__main__":
    process_randomize_pitch(CONFIG)
