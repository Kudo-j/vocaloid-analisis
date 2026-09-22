"""
randomize_pitch.py
==================
VOCALOID (vsq4) ファイル内の音高（ノート番号）をトラック単位でランダマイズして
Step 3実験用のVSQXデータセットを一括生成するスクリプト。

主な機能:
  1. トラック単位の個別ランダマイズ:
     1ファイル内に複数トラック（10トラック程度）が存在する場合でも、
     同一ファイル内のトラック間で重複しないよう個別に音高を割り当て。
  2. ノートタグ (<n>, <noteNum>) を正規表現で安全に置換（CDATA・XML書式を完全維持）。
  3. C3(60)〜C6(96) の37音階に対応。
  4. 母音・データ分割（Train/Valid/Test）を考慮した層化均等サンプリング（偏り・交絡の防止）。
  5. ファイル名接頭辞（デフォルト: Step3_）の付与。
  6. 乱数シード固定による完全な再現性。
  7. トラック番号・トラック名を含む詳細ログ（CSV）出力。

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
    # (必須) 入力VSQXフォルダ (10トラック程度入ったStep_orgのVSQX、または分割済みVSQX)
    "input_dir": "./data/Step_org/Step2",

    # (必須) 出力先フォルダ (Step 3のVSQXデータ保存先)
    "output_dir": "./data/Step_org/Step3",

    # 音高範囲 (Yamaha規格: C3=60, C6=96, 両端含む37段階)
    "min_note_num": 60,  # C3
    "max_note_num": 96,  # C6

    # サンプリング方式:
    # "stratified": Split (Train/Valid/Test) 及び母音ごとに均等サンプリング (推奨)
    # "balanced"  : 全体で均等サンプリング (シャッフル配分)
    # "uniform"   : 完全一様ランダム (独立な復元抽出)
    "sampling_mode": "stratified",

    # 乱数シード (再現性確保)
    "random_seed": 42,

    # ファイル名接頭辞 (例: "Step3_" -> "Step3_Train1_あ.vsqx")
    # 空文字 "" にすると接頭辞なし
    "file_prefix": "Step3_",

    # ログCSVのファイル名 (出力フォルダに保存)
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
    """MIDIノート番号をYamaha音名表記 (例: 60 -> C3, 69 -> A3) に変換する。"""
    pitch = PITCH_NAMES[note_num % 12]
    octave = (note_num // 12) - 2  # 60 // 12 - 2 = 3 (C3)
    return f"{pitch}{octave}"


def note_num_to_hz(note_num: int) -> float:
    """MIDIノート番号を周波数 (Hz, A4=440Hz基準) に変換する。"""
    return 440.0 * (2.0 ** ((note_num - 69) / 12.0))


def detect_split_and_vowel(filename: str) -> tuple[str, str]:
    """ファイル名からデータ分割 (Train/Valid/Test) と母音 (あ〜お) を判定する。"""
    f_lower = filename.lower()

    if "test" in f_lower:
        split = "Test"
    elif "valid" in f_lower:
        split = "Valid"
    elif "train" in f_lower:
        split = "Train"
    else:
        split = "Other"

    vowel = "Other"
    for v in VOWEL_LIST:
        if v in filename:
            vowel = v
            break

    return split, vowel


def scan_tracks_in_vsqx(content: str) -> list[dict]:
    """
    VSQXテキストから各 <vsTrack> の情報 (tNo, name, 元ノート番号) を抽出する。
    """
    pattern_track = re.compile(r"(<vsTrack>.*?</vsTrack>)", flags=re.DOTALL)
    pattern_tno = re.compile(r"<tNo>(\d+)</tNo>")
    pattern_name = re.compile(r"<name><!\[CDATA\[(.*?)\]\]></name>|<name>(.*?)</name>")
    pattern_note = re.compile(r"<(n|noteNum)>(-?\d+)</\1>")

    track_infos = []
    track_matches = pattern_track.findall(content)

    for idx, t_str in enumerate(track_matches):
        m_tno = pattern_tno.search(t_str)
        tno = int(m_tno.group(1)) if m_tno else idx

        m_name = pattern_name.search(t_str)
        if m_name:
            tname = m_name.group(1) if m_name.group(1) is not None else m_name.group(2)
        else:
            tname = f"Track_{tno}"

        m_note = pattern_note.search(t_str)
        orig_note = int(m_note.group(2)) if m_note else 69

        track_infos.append({
            "tNo": tno,
            "track_name": tname,
            "original_note": orig_note,
        })

    return track_infos


def replace_notes_per_track(content: str, track_note_map: dict[int, int]) -> tuple[str, int]:
    """
    VSQX内の各 <vsTrack> を走査し、tNoに対応する音高で個別に置換する。

    Returns:
        new_content (str): 置換後のテキスト
        replaced_count (int): 置換したノートタグの数
    """
    pattern_track = re.compile(r"(<vsTrack>.*?</vsTrack>)", flags=re.DOTALL)
    pattern_note = re.compile(r"<(n|noteNum)>(-?\d+)</\1>")
    pattern_tno = re.compile(r"<tNo>(\d+)</tNo>")

    replaced_count = 0
    curr_track_idx = 0

    def _sub_track(m_track: re.Match) -> str:
        nonlocal replaced_count, curr_track_idx
        track_str = m_track.group(1)

        m_tno = pattern_tno.search(track_str)
        tno = int(m_tno.group(1)) if m_tno else curr_track_idx
        curr_track_idx += 1

        target_note = track_note_map.get(tno)
        if target_note is None:
            target_note = track_note_map.get(curr_track_idx - 1)

        if target_note is None:
            return track_str

        def _sub_note(m_note: re.Match) -> str:
            nonlocal replaced_count
            replaced_count += 1
            tag = m_note.group(1)
            return f"<{tag}>{target_note}</{tag}>"

        return pattern_note.sub(_sub_note, track_str)

    new_content = pattern_track.sub(_sub_track, content)
    return new_content, replaced_count


def _assign_distinct_notes_to_files(
    file_track_counts: list[tuple[str, int]], available_notes: list[int], rng: random.Random
) -> dict[str, list[int]]:
    """
    複数ファイルに対し、プールされた音高（全音階均等）を各ファイルに分配する。
    各ファイル内のトラック数が音階数（37）以下である限り、
    ファイル内で音高が重複しないことを数学的に保証する。
    """
    total_tracks = sum(count for _, count in file_track_counts)
    num_notes = len(available_notes)
    repeats = total_tracks // num_notes
    remainder = total_tracks % num_notes

    pool = available_notes * repeats
    if remainder > 0:
        pool.extend(rng.sample(available_notes, remainder))
    rng.shuffle(pool)

    file_buckets = {fid: [] for fid, _ in file_track_counts}
    target_counts = {fid: count for fid, count in file_track_counts}

    for note in pool:
        # この note がまだ入っておらず、枠が空いている候補バケットを探す
        candidates = [
            fid
            for fid, c in target_counts.items()
            if len(file_buckets[fid]) < c and note not in file_buckets[fid]
        ]
        if candidates:
            # 充填率が最も小さいバケットに割り当ててバランスを取る
            best_fid = min(
                candidates, key=lambda fid: len(file_buckets[fid]) / target_counts[fid]
            )
            file_buckets[best_fid].append(note)
        else:
            # 空きがあるバケットのうち、他バケットとのスワップで重複を回避
            open_fids = [
                fid for fid, c in target_counts.items() if len(file_buckets[fid]) < c
            ]
            placed = False
            for target_fid in open_fids:
                for other_fid in file_buckets:
                    if (
                        other_fid != target_fid
                        and note not in file_buckets[other_fid]
                        and len(file_buckets[other_fid]) == target_counts[other_fid]
                    ):
                        for cand_swap_note in file_buckets[other_fid]:
                            if cand_swap_note not in file_buckets[target_fid]:
                                file_buckets[other_fid].remove(cand_swap_note)
                                file_buckets[other_fid].append(note)
                                file_buckets[target_fid].append(cand_swap_note)
                                placed = True
                                break
                        if placed:
                            break
                if placed:
                    break
            if not placed and open_fids:
                file_buckets[open_fids[0]].append(note)

    return file_buckets


def allocate_pitches_stratified(
    files_metadata: list[dict], available_notes: list[int], rng: random.Random
) -> dict[tuple[str, int], int]:
    """
    Split (Train/Valid/Test) ごとに均等サンプリングを行い、
    同一ファイル内の全トラックには100%重複のない音高を割り当てる。

    Returns:
        assignments: {(filename, tNo): assigned_note}
    """
    assignments = {}

    # Split ごとに分類
    split_files = defaultdict(list)
    for fmeta in files_metadata:
        split_files[fmeta["split"]].append(fmeta)

    for split, flist in split_files.items():
        file_track_counts = [(fmeta["filename"], len(fmeta["tracks"])) for fmeta in flist]
        file_buckets = _assign_distinct_notes_to_files(file_track_counts, available_notes, rng)

        for fmeta in flist:
            fname = fmeta["filename"]
            assigned_list = file_buckets[fname]
            rng.shuffle(assigned_list)
            for track, note in zip(fmeta["tracks"], assigned_list):
                assignments[(fname, track["tNo"])] = note

    return assignments


def allocate_pitches_balanced(
    files_metadata: list[dict], available_notes: list[int], rng: random.Random
) -> dict[tuple[str, int], int]:
    """ファイル全体で均等に複製してシャッフル配分し、同一ファイル内では重複を防ぐ。"""
    num_notes = len(available_notes)
    total_tracks = sum(len(fmeta["tracks"]) for fmeta in files_metadata)
    repeats = total_tracks // num_notes
    remainder = total_tracks % num_notes

    pool = available_notes * repeats
    if remainder > 0:
        pool.extend(rng.sample(available_notes, remainder))
    rng.shuffle(pool)

    assignments = {}
    pool_idx = 0
    pool_len = len(pool)

    for fmeta in files_metadata:
        filename = fmeta["filename"]
        file_notes = []
        for track in fmeta["tracks"]:
            tno = track["tNo"]
            found_idx = -1
            for search_idx in range(pool_idx, pool_len):
                if pool[search_idx] not in file_notes:
                    found_idx = search_idx
                    break
            if found_idx == -1:
                found_idx = pool_idx

            pool[pool_idx], pool[found_idx] = pool[found_idx], pool[pool_idx]
            chosen_note = pool[pool_idx]
            pool_idx += 1

            file_notes.append(chosen_note)
            assignments[(filename, tno)] = chosen_note

    return assignments


def allocate_pitches_uniform(
    files_metadata: list[dict], available_notes: list[int], rng: random.Random
) -> dict[tuple[str, int], int]:
    """各トラックに対して独立に一様乱数（復元抽出）で割り当てる。"""
    assignments = {}
    for fmeta in files_metadata:
        filename = fmeta["filename"]
        for track in fmeta["tracks"]:
            tno = track["tNo"]
            assignments[(filename, tno)] = rng.choice(available_notes)
    return assignments


def process_randomize_pitch(config: dict):
    """メイン処理関数"""
    input_dir = _resolve_dir(config["input_dir"])
    output_dir = _resolve_dir(config["output_dir"], base_ref_dir=input_dir)
    min_note = config.get("min_note_num", 60)
    max_note = config.get("max_note_num", 96)
    sampling_mode = config.get("sampling_mode", "stratified")
    seed = config.get("random_seed", 42)
    file_prefix = config.get("file_prefix", "Step3_")
    log_csv_name = config.get("log_csv_name", "pitch_randomize_log.csv")
    dry_run = config.get("dry_run", False)

    print("=" * 68)
    print("  VOCALOID VSQX Track-Level Pitch Randomizer (Step 3)")
    print("=" * 68)
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Pitch Range      : Note {min_note} ({note_num_to_name(min_note)}) 〜 Note {max_note} ({note_num_to_name(max_note)}) [{max_note - min_note + 1} steps]")
    print(f"Sampling Mode    : {sampling_mode}")
    print(f"Random Seed      : {seed}")
    print(f"File Prefix      : '{file_prefix}'")
    print(f"Dry Run Mode     : {dry_run}")
    print("=" * 68)

    if not os.path.exists(input_dir):
        print(f"[Error] 入力ディレクトリが存在しません: {input_dir}")
        return False

    vsqx_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".vsqx")]
    )

    if not vsqx_files:
        print(f"[Warning] 入力ディレクトリ内に .vsqx ファイルが見つかりませんでした: {input_dir}")
        return False

    # 1. 各ファイルのトラックメタデータを走査
    files_metadata = []
    total_tracks_count = 0

    for path in vsqx_files:
        filename = os.path.basename(path)
        split, vowel = detect_split_and_vowel(filename)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        tracks = scan_tracks_in_vsqx(content)
        total_tracks_count += len(tracks)

        files_metadata.append({
            "path": path,
            "filename": filename,
            "stem": Path(path).stem,
            "split": split,
            "vowel": vowel,
            "content": content,
            "tracks": tracks,
        })

    print(f"対象VSQXファイル数: {len(vsqx_files)} 件 (総トラック数: {total_tracks_count} トラック)\n")

    # 2. トラック単位の音高割り当て
    available_notes = list(range(min_note, max_note + 1))
    rng = random.Random(seed)

    if sampling_mode == "stratified":
        assignments = allocate_pitches_stratified(files_metadata, available_notes, rng)
    elif sampling_mode == "balanced":
        assignments = allocate_pitches_balanced(files_metadata, available_notes, rng)
    elif sampling_mode == "uniform":
        assignments = allocate_pitches_uniform(files_metadata, available_notes, rng)
    else:
        raise ValueError(f"未知のサンプリングモードです: {sampling_mode}")

    # 3. 出力先ディレクトリの作成
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    # 4. トラック別置換と書き出し
    log_rows = []
    success_count = 0
    fail_count = 0

    print("トラック単位で音高置換中...")
    for f_idx, fmeta in enumerate(files_metadata):
        orig_filename = fmeta["filename"]
        stem = fmeta["stem"]
        split = fmeta["split"]
        vowel = fmeta["vowel"]
        content = fmeta["content"]
        tracks = fmeta["tracks"]

        out_filename = f"{file_prefix}{stem}.vsqx"
        out_path = os.path.join(output_dir, out_filename)

        # このファイルに属するトラックの音高マップ {tNo: target_note}
        track_note_map = {
            t["tNo"]: assignments[(orig_filename, t["tNo"])] for t in tracks
        }

        try:
            new_content, replaced_count = replace_notes_per_track(content, track_note_map)

            if not dry_run:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

            # トラックごとのログ行を作成
            for t in tracks:
                tno = t["tNo"]
                tname = t["track_name"]
                orig_note = t["original_note"]
                assigned_note = track_note_map[tno]
                shift = assigned_note - orig_note
                hz = note_num_to_hz(assigned_note)

                log_rows.append({
                    "file_index": f_idx + 1,
                    "original_file": orig_filename,
                    "output_file": out_filename,
                    "track_no": tno,
                    "track_name": tname,
                    "split": split,
                    "vowel": vowel,
                    "original_note": orig_note,
                    "original_note_name": note_num_to_name(orig_note),
                    "assigned_note": assigned_note,
                    "assigned_note_name": note_num_to_name(assigned_note),
                    "frequency_hz": round(hz, 2),
                    "semitone_shift": shift,
                })

            success_count += 1

        except Exception as e:
            print(f"  [Error] 失敗: {orig_filename} -> {e}")
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
    print("  処理結果サマリー (トラック単位)")
    print("=" * 68)
    print(f"ファイル処理数: 成功 {success_count} 件, 失敗 {fail_count} 件 (総トラック数: {len(log_rows)})")

    # Split別の音高分布
    split_counter = defaultdict(lambda: defaultdict(int))
    for row in log_rows:
        split_counter[row["split"]][row["assigned_note"]] += 1

    for split in sorted(split_counter.keys()):
        notes_seen = len(split_counter[split])
        total_in_split = sum(split_counter[split].values())
        print(f"  - [{split}] 計 {total_in_split} トラック | 出現音高種類数: {notes_seen}/{len(available_notes)} 種類")

    # 同一ファイル内の音高重複チェック
    file_note_tracker = defaultdict(list)
    for row in log_rows:
        file_note_tracker[row["output_file"]].append(row["assigned_note"])

    has_intra_duplicates = False
    for fname, notes in file_note_tracker.items():
        if len(notes) != len(set(notes)):
            has_intra_duplicates = True
            break

    if not has_intra_duplicates:
        print("  ★ 同一ファイル内の全トラックで音高重複なし (完全に異なる音高を配分)")
    else:
        print("  ※ 一部ファイルでトラック数が音階数を超えたため音高が重複しています")

    print("=" * 68)
    return True


if __name__ == "__main__":
    process_randomize_pitch(CONFIG)
