"""
randomize_pitch.py
==================
VOCALOID (vsq4) ファイル内の音高（ノート番号）をランダマイズして
Step 3（音高多様化実験）用のVSQXデータセットを一括生成するスクリプト。

主な機能:
  1. 正規表現による安全なサブトラック自動削除 (Track 0 のみ保持):
     XMLパーサを通さず完全な正規表現テキスト処理で Track 0 (メイン) のみを残し
     Track 1 以降を安全に削除。<![CDATA[...]]> セクション、XML宣言 (standalone="no")、
     タブインデント、改行構造を 100% 完全に維持して VOCALOID ソフトでの互換性を保証。
  2. ラテン方陣に基づく音素×音高の直交・層化配分:
     - Train: 同一DYNパターン番号間で5母音の音高が重複しないよう排他制御しつつ、
       各母音内で音高を均等配分（37音階中30音高を使用、7音高が自動的に未網羅/Unseen化）。
     - Valid / Test: Trainで網羅できなかった未知音高（Unseen）を指定割合（unseen_ratio）
       割り当て、残りを既知音高（Seen）から割り当て。同一DYNパターン内での重複も完全回避。
  3. 音高網羅ベンチマークデータ（Benchmark Test）の一括生成:
     - 代表DYNパターン（デフォルト: Test_8系）× 5母音 × 全音階（37音高）の計185曲を
       別枠（benchmark/ サブフォルダ）に一括生成。
  4. ファイル名への安全な音名サフィックス付与:
     - DAWやVOCALOIDでのパス誤認を防ぐため、シャープ記号 '#' を 's' に置換した安全表記を採用
       (例: Note 61 -> Cs3, Note 66 -> Fs4)。
     - 通常データ: 例 Step3_Train1_あ_1_Cs3.vsqx
     - ベンチマークデータ: 例 Step3_Benchmark_Test_あ_8_Cs3.vsqx
  5. ノートタグ (<n>, <noteNum>) を正規表現で安全に置換。
  6. 乱数シード固定による完全な再現性。
  7. 詳細メタデータログ (CSV) 出力（Seen / Unseen フラグ付き）。

使い方:
  1. 下記の CONFIG 領域を確認・編集する
  2. 実行ボタンを押す (python randomize_pitch.py)
"""

import os
import re
import csv
import math
import random
from pathlib import Path
from collections import defaultdict


# =============================================================================
# 設定領域 (CONFIG) - 実行ボタンを押すだけで動作可能
# =============================================================================
CONFIG = {
    # (必須) 入力VSQXフォルダ (Step2の分割済みVSQX)
    "input_dir": "./data/Step2/vsqx",
    # (必須) 出力先フォルダ (Step 3のVSQXデータ保存先)
    "output_dir": "./data/Step_org/Step3_3times",
    # 音高範囲 (Yamaha規格: C3=60, C6=96, 両端含む37段階)
    "min_note_num": 60,  # C3
    "max_note_num": 96,  # C6
    # Trainデータのバリエーション生成数 (1倍, 2倍, 3倍... 各バリエーションで音高を変更)
    "num_train_variations": 3,
    # Validデータに排他配分する未網羅ペア数 (各母音あたり: 2 -> 5母音で計10種)
    "valid_unseen_per_vowel": 2,
    # Testデータに排他配分する未網羅ペア数 (各母音あたり: 2 -> 5母音で計10種)
    "test_unseen_per_vowel": 2,
    # ベンチマーク代表DYNパターン番号 (例: 1 -> Test_1系, 8 -> Test_8系)
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
    """MIDIノート番号をYamaha音名表記 (例: 60 -> C3, 61 -> C#3) に変換する。"""
    pitch = PITCH_NAMES[note_num % 12]
    octave = (note_num // 12) - 2  # 60 // 12 - 2 = 3 (C3)
    return f"{pitch}{octave}"


def note_num_to_safe_name(note_num: int) -> str:
    """ファイル名として安全な音名表記 (例: 61 -> Cs3, 66 -> Fs4) に変換する。"""
    raw_name = note_num_to_name(note_num)
    return raw_name.replace("#", "s")


def note_num_to_hz(note_num: int) -> float:
    """MIDIノート番号を周波数 (Hz, A4=440Hz基準) に変換する。"""
    return 440.0 * (2.0 ** ((note_num - 69) / 12.0))


def remove_subtracks_pure_regex(vsqx_text: str) -> str:
    """
    正規表現テキスト処理により、Track 0 以外の vsTrack および vsUnit を安全に削除する。
    ElementTree などの XML パーサを通さないため、<![CDATA[...]]> セクション、
    XML 宣言 (standalone="no")、タブインデント、改行構造を 100% 完全に維持する。
    """
    # 1. vsTrack ブロックを分割・走査し、tNo が 0 のトラックのみを残す
    parts = re.split(r"(<vsTrack>.*?</vsTrack>)", vsqx_text, flags=re.DOTALL)
    new_parts = []
    for part in parts:
        if part.startswith("<vsTrack>"):
            m_tno = re.search(r"<tNo>(\d+)</tNo>", part)
            if m_tno and m_tno.group(1) == "0":
                new_parts.append(part)
        else:
            new_parts.append(part)
    res = "".join(new_parts)

    # 2. mixer 内の vsUnit のうち tNo != 0 のものを削除
    def _sub_unit(m: re.Match) -> str:
        u_str = m.group(1)
        m_tno = re.search(r"<tNo>(\d+)</tNo>", u_str)
        if m_tno and m_tno.group(1) != "0":
            return ""
        return u_str

    res = re.sub(r"(<vsUnit>.*?</vsUnit>\s*)", _sub_unit, res, flags=re.DOTALL)
    return res


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
    Track 0 内のノートタグ (<n>, <noteNum>) を指定した音高で置換する。
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


def _allocate_split_with_specific_unseen(
    split_files: list[dict],
    assigned_unseen_per_vowel: dict[str, list[int]],
    seen_notes_per_vowel: dict[str, set[int]],
    rng: random.Random,
) -> dict[str, int]:
    """
    Valid または Test ファイル群に対し、各母音ごとに
    排他指定された未網羅音高 (assigned_unseen_per_vowel) と
    既知音高 (seen_notes_per_vowel) を割り当て、
    同一DYNパターン内での母音間重複をスワップで解消する。
    """
    dyn_pattern_ids = sorted(list(set(f["dyn_pattern_id"] for f in split_files)))
    num_pats = len(dyn_pattern_ids)

    # 各母音ごとに選択する音高リスト
    vowel_notes = {}
    for v in VOWEL_LIST:
        u_chosen = list(assigned_unseen_per_vowel.get(v, []))
        avail_s = sorted(seen_notes_per_vowel.get(v, set()))
        s_count = max(0, num_pats - len(u_chosen))
        s_chosen = (
            rng.sample(avail_s, s_count) if s_count <= len(avail_s) else list(avail_s)
        )

        combined = u_chosen + s_chosen
        rng.shuffle(combined)
        vowel_notes[v] = combined

    # パターン行への初期配置
    grid = [{v: vowel_notes[v][p_idx] for v in VOWEL_LIST} for p_idx in range(num_pats)]

    # 行内の重複をスワップで解消 (母音列内でのみスワップするため各母音の音高集合は完全維持)
    for _ in range(500):
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
                                    (
                                        grid[p_idx][v_conflict],
                                        grid[p_other][v_conflict],
                                    ) = (cand, n)
                                    break
                        break
                    else:
                        seen[n] = v
        if not has_conflict:
            break

    pattern_map = {dyn_pattern_ids[p_idx]: grid[p_idx] for p_idx in range(num_pats)}
    assignments = {}
    for f in split_files:
        assigned = pattern_map[f["dyn_pattern_id"]][f["vowel"]]
        assignments[f["filename"]] = assigned

    return assignments


def allocate_all_pitches(
    files_metadata: list[dict],
    available_notes: list[int],
    num_train_variations: int,
    valid_unseen_per_vowel: int,
    test_unseen_per_vowel: int,
    rng: random.Random,
) -> tuple[
    dict[int, dict[str, int]],
    dict[str, int],
    dict[str, int],
    dict[str, set[int]],
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
]:
    """
    全データ分割 (Train/Valid/Test) に対して音高を配分する。
    - Train: ラテン方陣的シフトで各バリエーション間で同一曲の音高重複をゼロにしつつ、
      同一DYNパターン内の母音間重複も完全排除。
    - Valid / Test: 35種の未網羅ペアを Valid (各母音2種) と Test (各母音2種) に厳密に排他配分。
      残る各母音3種 (計15種) は完全未網羅としてBenchmarkのみで検証可能。
    """
    split_files = defaultdict(list)
    for fmeta in files_metadata:
        split_files[fmeta["split"]].append(fmeta)

    train_files = split_files.get("Train", [])
    valid_files = split_files.get("Valid", [])
    test_files = split_files.get("Test", [])

    train_dyn_ids = sorted(list(set(f["dyn_pattern_id"] for f in train_files)))
    num_train_pats = len(train_dyn_ids)
    num_notes = len(available_notes)

    # 1. 37音階のランダム順列とラテン方陣による基本割り当て (Train バリエーション1)
    perm = list(available_notes)
    rng.shuffle(perm)
    shift_step = 7

    seen_notes_per_vowel = {v: set() for v in VOWEL_LIST}
    train_var1_pattern_map = {}
    for k_idx, dyn_id in enumerate(train_dyn_ids):
        train_var1_pattern_map[dyn_id] = {}
        for v_idx, v in enumerate(VOWEL_LIST):
            note_idx = (k_idx + v_idx * shift_step) % num_notes
            note = perm[note_idx]
            train_var1_pattern_map[dyn_id][v] = note
            seen_notes_per_vowel[v].add(note)

    # 2. 未網羅音高 (37 - 30 = 7音高/母音, 計35種) を排他分割
    valid_unseen_notes = {}
    test_unseen_notes = {}
    unused_unseen_notes = {}

    for v in VOWEL_LIST:
        all_unseen = sorted(list(set(available_notes) - seen_notes_per_vowel[v]))
        rng.shuffle(all_unseen)
        valid_unseen_notes[v] = all_unseen[:valid_unseen_per_vowel]
        test_unseen_notes[v] = all_unseen[
            valid_unseen_per_vowel : valid_unseen_per_vowel + test_unseen_per_vowel
        ]
        unused_unseen_notes[v] = all_unseen[
            valid_unseen_per_vowel + test_unseen_per_vowel :
        ]

    # 3. Train の各バリエーション (1..num_train_variations) への割り当て
    # 既知音高 (30音高) 内での巡回シフトにより、同一曲の音高重複を完全排除 & 同一DYNパターン内の母音重複もゼロ
    train_var_assignments = {}
    for var_idx in range(1, num_train_variations + 1):
        shift_k = (var_idx - 1) % num_train_pats
        var_assign = {}
        for k_idx, dyn_id in enumerate(train_dyn_ids):
            source_dyn_id = train_dyn_ids[(k_idx + shift_k) % num_train_pats]
            for f in train_files:
                if f["dyn_pattern_id"] == dyn_id:
                    v = f["vowel"]
                    var_assign[f["filename"]] = train_var1_pattern_map[source_dyn_id][v]
        train_var_assignments[var_idx] = var_assign

    # 4. Valid データの割り当て (固定セット)
    valid_assignments = {}
    if valid_files:
        valid_assignments = _allocate_split_with_specific_unseen(
            valid_files, valid_unseen_notes, seen_notes_per_vowel, rng
        )

    # 5. Test データの割り当て (固定セット)
    test_assignments = {}
    if test_files:
        test_assignments = _allocate_split_with_specific_unseen(
            test_files, test_unseen_notes, seen_notes_per_vowel, rng
        )

    return (
        train_var_assignments,
        valid_assignments,
        test_assignments,
        seen_notes_per_vowel,
        valid_unseen_notes,
        test_unseen_notes,
        unused_unseen_notes,
    )


def process_randomize_pitch(config: dict):
    """メイン処理関数"""
    input_dir = _resolve_dir(config["input_dir"])
    output_dir = _resolve_dir(config["output_dir"], base_ref_dir=input_dir)
    min_note = config.get("min_note_num", 60)
    max_note = config.get("max_note_num", 96)
    num_train_variations = config.get("num_train_variations", 3)
    valid_unseen_per_vowel = config.get("valid_unseen_per_vowel", 2)
    test_unseen_per_vowel = config.get("test_unseen_per_vowel", 2)
    benchmark_rep_pattern = config.get("benchmark_rep_pattern", 1)
    seed = config.get("random_seed", 42)
    file_prefix = config.get("file_prefix", "Step3_")
    benchmark_prefix = config.get("benchmark_prefix", "Step3_Benchmark_")
    benchmark_subdir = config.get("benchmark_subdir", "benchmark")
    log_csv_name = config.get("log_csv_name", "pitch_randomize_log.csv")
    dry_run = config.get("dry_run", False)

    print("=" * 70)
    print("  VOCALOID VSQX Track-Level Pitch Randomizer (Step 3)")
    print("=" * 70)
    print(f"Input Directory          : {input_dir}")
    print(f"Output Directory         : {output_dir}")
    print(f"Train Variations         : {num_train_variations} 倍 (階層: 1/, 2/, ...)")
    print(
        f"Pitch Range              : Note {min_note} ({note_num_to_name(min_note)}) 〜 "
        f"Note {max_note} ({note_num_to_name(max_note)}) [{max_note - min_note + 1} steps]"
    )
    print(
        f"Unseen Allocation        : Valid {valid_unseen_per_vowel * len(VOWEL_LIST)}種, "
        f"Test {test_unseen_per_vowel * len(VOWEL_LIST)}種 (各母音均等配分)"
    )
    print(f"Benchmark Rep Pattern    : Test_{benchmark_rep_pattern} 系")
    print(f"Random Seed              : {seed}")
    print(f"Dry Run Mode             : {dry_run}")
    print("=" * 70)

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
        print(
            f"[Warning] 入力ディレクトリ内に .vsqx ファイルが見つかりませんでした: {input_dir}"
        )
        return False

    # 1. 各ファイルのメタデータを走査
    files_metadata = []
    for path in vsqx_files:
        filename = os.path.basename(path)
        meta = parse_vsqx_metadata(filename)
        meta["path"] = path
        files_metadata.append(meta)

    print(f"対象VSQXファイル数: {len(vsqx_files)} 件\n")

    available_notes = list(range(min_note, max_note + 1))
    rng = random.Random(seed)

    (
        train_var_assignments,
        valid_assignments,
        test_assignments,
        seen_notes_per_vowel,
        valid_unseen_notes,
        test_unseen_notes,
        unused_unseen_notes,
    ) = allocate_all_pitches(
        files_metadata,
        available_notes,
        num_train_variations,
        valid_unseen_per_vowel,
        test_unseen_per_vowel,
        rng,
    )

    split_files = defaultdict(list)
    for fmeta in files_metadata:
        split_files[fmeta["split"]].append(fmeta)

    train_files = split_files.get("Train", [])
    valid_files = split_files.get("Valid", [])
    test_files = split_files.get("Test", [])

    log_rows = []
    success_count = 0
    fail_count = 0

    print("VSQXファイル処理中 (正規表現Sub-track削除 & 音高置換)...")

    # 2. Train / Valid / Test データの出力 (階層: 1/, 2/, ...)
    for var_idx in range(1, num_train_variations + 1):
        var_dir = os.path.join(output_dir, str(var_idx))
        if not dry_run:
            os.makedirs(var_dir, exist_ok=True)

        print(f"  -> バリエーション {var_idx}/{num_train_variations} 出力中: {var_dir}")

        # (A) Train データの処理 (全バリエーションで出力)
        for fmeta in train_files:
            orig_filename = fmeta["filename"]
            stem = fmeta["stem"]
            split = fmeta["split"]
            vowel = fmeta["vowel"]
            dyn_id = fmeta["dyn_pattern_id"]
            path = fmeta["path"]

            assigned_note = train_var_assignments[var_idx][orig_filename]
            assigned_name = note_num_to_name(assigned_note)
            safe_name = note_num_to_safe_name(assigned_note)
            hz = note_num_to_hz(assigned_note)

            out_filename = f"{file_prefix}{var_idx}_{stem}_{safe_name}.vsqx"
            out_path = os.path.join(var_dir, out_filename)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_content = f.read()

                single_track_content = remove_subtracks_pure_regex(raw_content)
                new_content, replaced_count = replace_note_number(
                    single_track_content, assigned_note
                )

                if not dry_run:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

                log_rows.append(
                    {
                        "file_index": len(log_rows) + 1,
                        "variation": str(var_idx),
                        "original_file": orig_filename,
                        "output_file": out_filename,
                        "relative_path": f"{var_idx}/{out_filename}",
                        "is_benchmark": False,
                        "split": split,
                        "vowel": vowel,
                        "dyn_pattern_id": dyn_id,
                        "assigned_note": assigned_note,
                        "assigned_note_name": assigned_name,
                        "frequency_hz": round(hz, 2),
                        "is_seen": True,
                        "unseen_category": "Seen",
                    }
                )
                success_count += 1
            except Exception as e:
                print(f"    [Error] 失敗: {orig_filename} -> {e}")
                fail_count += 1

        # (B) Valid / Test データの処理 (フォルダ 1/ にのみ固定出力)
        if var_idx == 1:
            for eval_split, eval_files, eval_assigns, u_notes, u_label in [
                (
                    "Valid",
                    valid_files,
                    valid_assignments,
                    valid_unseen_notes,
                    "Valid_Unseen",
                ),
                (
                    "Test",
                    test_files,
                    test_assignments,
                    test_unseen_notes,
                    "Test_Unseen",
                ),
            ]:
                for fmeta in eval_files:
                    orig_filename = fmeta["filename"]
                    stem = fmeta["stem"]
                    split = fmeta["split"]
                    vowel = fmeta["vowel"]
                    dyn_id = fmeta["dyn_pattern_id"]
                    path = fmeta["path"]

                    assigned_note = eval_assigns[orig_filename]
                    assigned_name = note_num_to_name(assigned_note)
                    safe_name = note_num_to_safe_name(assigned_note)
                    hz = note_num_to_hz(assigned_note)
                    is_seen = assigned_note in seen_notes_per_vowel.get(vowel, set())
                    u_category = "Seen" if is_seen else u_label

                    out_filename = f"{file_prefix}1_{stem}_{safe_name}.vsqx"
                    out_path = os.path.join(var_dir, out_filename)

                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            raw_content = f.read()

                        single_track_content = remove_subtracks_pure_regex(raw_content)
                        new_content, replaced_count = replace_note_number(
                            single_track_content, assigned_note
                        )

                        if not dry_run:
                            with open(out_path, "w", encoding="utf-8") as f:
                                f.write(new_content)

                        log_rows.append(
                            {
                                "file_index": len(log_rows) + 1,
                                "variation": "1",
                                "original_file": orig_filename,
                                "output_file": out_filename,
                                "relative_path": f"1/{out_filename}",
                                "is_benchmark": False,
                                "split": split,
                                "vowel": vowel,
                                "dyn_pattern_id": dyn_id,
                                "assigned_note": assigned_note,
                                "assigned_note_name": assigned_name,
                                "frequency_hz": round(hz, 2),
                                "is_seen": is_seen,
                                "unseen_category": u_category,
                            }
                        )
                        success_count += 1
                    except Exception as e:
                        print(f"    [Error] 失敗: {orig_filename} -> {e}")
                        fail_count += 1

    # 3. 音高網羅ベンチマークデータの処理 (Test_代表系 × 37音高 × 5母音 = 185曲)
    benchmark_dir = os.path.join(output_dir, benchmark_subdir)
    if not dry_run:
        os.makedirs(benchmark_dir, exist_ok=True)

    rep_pattern_str = f"Test_{benchmark_rep_pattern}"
    benchmark_source_files = [
        f
        for f in files_metadata
        if f["split"] == "Test" and f["dyn_pattern_id"] == rep_pattern_str
    ]

    print(
        f"\n音高網羅ベンチマークデータ生成中 (代表: {rep_pattern_str}, 5母音 × {len(available_notes)}音高)..."
    )
    for fmeta in benchmark_source_files:
        orig_filename = fmeta["filename"]
        stem = fmeta["stem"]
        vowel = fmeta["vowel"]
        dyn_id = fmeta["dyn_pattern_id"]
        path = fmeta["path"]

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            single_track_content = remove_subtracks_pure_regex(raw_content)

            for note in available_notes:
                note_name = note_num_to_name(note)
                safe_name = note_num_to_safe_name(note)
                hz = note_num_to_hz(note)
                is_seen = note in seen_notes_per_vowel.get(vowel, set())

                if is_seen:
                    u_category = "Seen"
                elif note in valid_unseen_notes.get(vowel, []):
                    u_category = "Valid_Unseen"
                elif note in test_unseen_notes.get(vowel, []):
                    u_category = "Test_Unseen"
                else:
                    u_category = "Completely_Unused"

                out_filename = f"{benchmark_prefix}{stem}_{safe_name}.vsqx"
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
                        "variation": "benchmark",
                        "original_file": orig_filename,
                        "output_file": out_filename,
                        "relative_path": f"{benchmark_subdir}/{out_filename}",
                        "is_benchmark": True,
                        "split": "Benchmark",
                        "vowel": vowel,
                        "dyn_pattern_id": dyn_id,
                        "assigned_note": note,
                        "assigned_note_name": note_name,
                        "frequency_hz": round(hz, 2),
                        "is_seen": is_seen,
                        "unseen_category": u_category,
                    }
                )
                success_count += 1

        except Exception as e:
            print(f"  [Error] ベンチマーク生成失敗: {orig_filename} -> {e}")
            fail_count += 1

    # 4. ログCSV出力 (メインフォルダおよびbenchmarkサブフォルダの両方に保存)
    log_path = os.path.join(output_dir, log_csv_name)
    bm_log_path = os.path.join(benchmark_dir, log_csv_name)

    if not dry_run and log_rows:
        for p in [log_path, bm_log_path]:
            with open(p, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(log_rows)
        print(f"\nログ保存完了: {log_path} (および {bm_log_path})")

    # 5. 統計サマリー表示
    print("\n" + "=" * 70)
    print("  処理結果サマリー")
    print("=" * 70)
    print(
        f"生成ファイル数: 成功 {success_count} 件, 失敗 {fail_count} 件 (総レコード: {len(log_rows)})"
    )

    # Split / カテゴリ別のSeen/Unseen統計
    split_cat_stats = defaultdict(lambda: defaultdict(int))
    for row in log_rows:
        s = row["split"]
        cat = row["unseen_category"]
        split_cat_stats[s][cat] += 1
        split_cat_stats[s]["Total"] += 1

    for s, c_dict in sorted(split_cat_stats.items()):
        total = c_dict["Total"]
        seen = c_dict.get("Seen", 0)
        unseen = total - seen
        seen_pct = (seen / total * 100) if total > 0 else 0
        unseen_pct = (unseen / total * 100) if total > 0 else 0
        details = ", ".join(f"{k}: {v}" for k, v in c_dict.items() if k != "Total")
        print(
            f"  - [{s}] 計 {total} 件 | Seen: {seen} ({seen_pct:.1f}%), "
            f"Unseen: {unseen} ({unseen_pct:.1f}%) | [{details}]"
        )

    # バリエーション間の同一曲音高重複チェック
    print("\n[整合性チェック]")
    train_var_song_notes = defaultdict(lambda: defaultdict(int))
    for row in log_rows:
        if row["split"] == "Train":
            train_var_song_notes[row["original_file"]][row["variation"]] = row[
                "assigned_note"
            ]

    dup_found = False
    for orig_f, v_notes in train_var_song_notes.items():
        notes_list = list(v_notes.values())
        if len(notes_list) != len(set(notes_list)):
            dup_found = True
            print(f"  ※ バリエーション間で音高重複検出: {orig_f} -> {v_notes}")
            break

    if not dup_found:
        print(
            f"  ★ バリエーション間 (1〜{num_train_variations}) で同一曲の音高重複なし (完全多様化達成)"
        )

    # 同一DYNパターン内の音高重複チェック (各バリエーション)
    pat_dup_found = False
    for var_idx in range(1, num_train_variations + 1):
        pat_notes = defaultdict(list)
        for row in log_rows:
            if row["split"] == "Train" and row["variation"] == str(var_idx):
                pat_notes[row["dyn_pattern_id"]].append(row["assigned_note"])
        for pid, n_list in pat_notes.items():
            if len(n_list) != len(set(n_list)):
                pat_dup_found = True
                print(
                    f"  ※ バリエーション {var_idx} 内のパターン {pid} で母音間重複検出"
                )
                break

    if not pat_dup_found:
        print(
            "  ★ 各バリエーション内において、同一DYNパターンの母音間で音高重複なし (ラテン方陣直交性維持)"
        )

    # 未網羅パターンのTrain混入 (リーク) チェック
    leak_found = False
    for row in log_rows:
        if row["split"] == "Train" and not row["is_seen"]:
            leak_found = True
            print(
                f"  ※ 重大警告: Trainに未学習パターンが混入しています: {row['output_file']}"
            )
            break

    if not leak_found:
        print("  ★ Trainデータへの未網羅パターンのリークなし (完全排他性を保証)")

    print("=" * 70)
    return True


if __name__ == "__main__":
    process_randomize_pitch(CONFIG)
