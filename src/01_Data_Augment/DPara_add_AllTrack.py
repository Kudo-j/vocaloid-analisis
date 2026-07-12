import os
import glob
import logging
from pathlib import Path
import re  # 正規表現ライブラリを使用
import shutil
import random
import math
from collections import defaultdict

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- vsqxファイル内のタグIDとパラメータ名のマッピング ---
# [ID] = ([Name], [Default Value], (Min Fluctuation, Max Fluctuation), Jitter)
PARAM_MAP = {
    "D": ("DYN", 64, (10, 60), 0),
    "B": ("BRE", 0, (10, 60), 0),
    "R": ("BRI", 64, (10, 60), 0),
    "C": ("CLE", 0, (10, 60), 0),
    "G": ("GEN", 64, (10, 60), 0),
    "P": ("PIT", 0, (100, 4000), 0),
    "S": ("PBS", 1, (1, 3), 0),
    "W": ("GWL", 0, (10, 60), 0),
    "X": ("XSY", 0, (10, 60), 0),
    "T": ("POR", 64, (10, 60), 0),
}

# パラメータごとの絶対的な値域
PARAM_VALID_RANGES = {
    "D": (0, 127),
    "B": (0, 127),
    "R": (0, 127),
    "C": (0, 127),
    "G": (0, 127),
    "P": (-8192, 8191),
    "S": (0, 24),
    "W": (0, 127),
    "X": (0, 127),
    "T": (0, 127),
}

# --- 正規表現パターン ---
# ★ 修正: 任意のトラック全体にマッチするパターンに変更
RE_ALL_TRACKS = re.compile(r"<vsTrack>.*?</vsTrack>", flags=re.DOTALL)
RE_VS_PART = re.compile(r"<vsPart>.*?</vsPart>", flags=re.DOTALL)
RE_NOTE = re.compile(r"<note>.*?</note>", flags=re.DOTALL)
RE_NOTE_T = re.compile(r"<t>(\d+)</t>")
RE_NOTE_DUR = re.compile(r"<dur>(\d+)</dur>")
RE_NOTE_N = re.compile(r"<n>(\d+)</n>")
RE_SINGER = re.compile(r"(<singer>.*?</singer>)", flags=re.DOTALL)
RE_T_NO = re.compile(r"<tNo>(\d+)</tNo>")  # トラック番号取得用
RE_CC_TAG = re.compile(r'<cc>.*?<v id="([A-Z]+)">.*?</cc>', flags=re.DOTALL)

# --- 設定値 ---
PHRASE_GAP_THRESHOLD = 480
PHRASE_MAX_TICKS = 30720  # 16小節 (= 16 * 1920)


def ease_in_out(t: float) -> float:
    return 0.5 * (1 - math.cos(t * math.pi))


def ease_out_curve(t: float) -> float:
    return 1 - (1 - t) ** 2


def ease_in_curve(t: float) -> float:
    return t**2


def clamp(v, param_id):
    v_int = int(round(v))
    valid_range = PARAM_VALID_RANGES.get(param_id, (0, 127))
    min_val, max_val = valid_range
    return max(min_val, min(max_val, v_int))


def extract_notes_info(part_content: str):
    notes = []
    for note_match in RE_NOTE.finditer(part_content):
        note_text = note_match.group(0)
        t_match = RE_NOTE_T.search(note_text)
        dur_match = RE_NOTE_DUR.search(note_text)
        n_match = RE_NOTE_N.search(note_text)

        if t_match and dur_match and n_match:
            notes.append(
                {
                    "t": int(t_match.group(1)),
                    "dur": int(dur_match.group(1)),
                    "n": int(n_match.group(1)),
                }
            )
    if not notes:
        return []
    notes.sort(key=lambda n: n["t"])
    return notes


def remove_existing_track_ccs(part_content: str, target_ids: list) -> str:
    placeholders = []

    def _mask_note(match):
        placeholders.append(match.group(0))
        return f"__NOTE_PLACEHOLDER_{len(placeholders)-1}__"

    masked_content = RE_NOTE.sub(_mask_note, part_content)

    def _remove_cc(match):
        cc_id = match.group(1)
        if cc_id in target_ids:
            return ""
        return match.group(0)

    cleaned_content = RE_CC_TAG.sub(_remove_cc, masked_content)

    for i, note_str in enumerate(placeholders):
        cleaned_content = cleaned_content.replace(f"__NOTE_PLACEHOLDER_{i}__", note_str)

    return cleaned_content


def add_curve_to_dict(param_id, points, final_ccs):
    if not points:
        return
    times = [p[0] for p in points]
    min_t = min(times)
    max_t = max(times)
    existing_ticks = list(final_ccs[param_id].keys())
    for t in existing_ticks:
        if min_t <= t <= max_t:
            del final_ccs[param_id][t]
    for t, v in points:
        final_ccs[param_id][t] = v


def _generate_attack_cc(
    start_time: int,
    param_id: str,
    mode: str,
    attack_pre_min: int,
    attack_pre_max: int,
    attack_post_min: int,
    attack_post_max: int,
    target_base: int,  # ★追加: このトラックの基準値
    return_noise: int,  # ★追加: 戻るときのブレ幅 (m)
) -> list:
    if param_id not in PARAM_MAP:
        return []
    param_info = PARAM_MAP[param_id]
    # base_value = param_info[1]
    min_fluctuation, max_fluctuation = param_info[2]
    jitter_amount = param_info[3]

    points = []
    change_amount = random.randint(min_fluctuation, max_fluctuation)
    pre_duration = random.randint(attack_pre_min, attack_pre_max)
    post_duration = random.randint(attack_post_min, attack_post_max)

    kf1_t = max(0, start_time - pre_duration)
    kf1_v = target_base
    kf2_t = start_time
    kf2_v = 0
    kf3_t = start_time + post_duration
    noise = random.randint(-return_noise, return_noise)
    kf3_v = clamp(target_base + noise, param_id)

    if mode == "attack_dip":
        kf2_v = clamp(target_base - change_amount, param_id)
    elif mode == "attack_rise":
        kf2_v = clamp(target_base + change_amount, param_id)
    else:
        return []

    points.append((kf1_t, kf1_v))

    total_duration_1 = kf2_t - kf1_t
    if total_duration_1 > 0:
        total_delta_1 = kf2_v - kf1_v
        max_points = min(total_duration_1, 500)
        num_points_1 = (
            random.randint(min(100, max_points), max_points)
            if max_points > 100
            else max_points
        )
        time_fractions_1 = sorted(
            [random.uniform(0.1, 0.9) for _ in range(num_points_1)]
        )
        for fraction in time_fractions_1:
            intermediate_t = int(kf1_t + total_duration_1 * fraction)
            curved_fraction = ease_out_curve(fraction)
            ideal_val = kf1_v + total_delta_1 * curved_fraction
            jitter = random.randint(-jitter_amount, jitter_amount)
            points.append((intermediate_t, clamp(ideal_val + jitter, param_id)))

    points.append((kf2_t, kf2_v))

    total_duration_2 = kf3_t - kf2_t
    if total_duration_2 > 0:
        total_delta_2 = kf3_v - kf2_v
        max_points = min(total_duration_2, 500)
        num_points_2 = (
            random.randint(min(100, max_points), max_points)
            if max_points > 100
            else max_points
        )
        time_fractions_2 = sorted(
            [random.uniform(0.1, 0.9) for _ in range(num_points_2)]
        )
        for fraction in time_fractions_2:
            intermediate_t = int(kf2_t + total_duration_2 * fraction)
            curved_fraction = ease_in_curve(fraction)
            ideal_val = kf2_v + total_delta_2 * curved_fraction
            jitter = random.randint(-jitter_amount, jitter_amount)
            points.append((intermediate_t, clamp(ideal_val + jitter, param_id)))

    points.append((kf3_t, kf3_v))
    return points


def _generate_crescendo_cc(
    start_time: int,
    end_time: int,
    param_id: str,
    mode: str,
    target_base: int,  # ★追加
    return_noise: int,  # ★追加
) -> list:
    if param_id not in PARAM_MAP:
        return []
    param_info = PARAM_MAP[param_id]
    # base_value = param_info[1]
    min_fluctuation, max_fluctuation = param_info[2]
    jitter_amount = param_info[3]

    points = []
    start_t = max(0, start_time - 50)
    phrase_end_t = end_time
    fluctuation_up = random.randint(min_fluctuation, max_fluctuation)
    fluctuation_down = random.randint(min_fluctuation, max_fluctuation)

    start_val = 0
    end_val = 0

    if mode == "crescendo":
        start_val = clamp(target_base - fluctuation_down, param_id)
        end_val = clamp(target_base + fluctuation_up, param_id)
    elif mode == "decrescendo":
        start_val = clamp(target_base + fluctuation_up, param_id)
        end_val = clamp(target_base - fluctuation_down, param_id)
    else:
        return []

    points.append((start_t, start_val))

    total_duration = phrase_end_t - start_t
    if total_duration == 0:
        total_duration = 1
    total_delta = end_val - start_val

    max_points = min(total_duration, 500)
    num_intermediate_points = (
        random.randint(min(100, max_points), max_points)
        if max_points > 100
        else max_points
    )
    time_fractions = sorted(
        [random.uniform(0.1, 0.9) for _ in range(num_intermediate_points)]
    )

    for fraction in time_fractions:
        intermediate_t = int(start_t + total_duration * fraction)
        curved_fraction = ease_in_out(fraction)
        ideal_val = start_val + total_delta * curved_fraction
        jitter = random.randint(-jitter_amount, jitter_amount)
        points.append((intermediate_t, clamp(ideal_val + jitter, param_id)))

    reset_t = phrase_end_t + 10
    noise = random.randint(-return_noise, return_noise)
    final_val = clamp(target_base + noise, param_id)
    points.append((reset_t, final_val))
    return points


# --- ★ 追加: Swell (山型・谷型) 生成関数 ---
def _generate_swell_cc(
    start_time: int,
    end_time: int,
    param_id: str,
    mode: str,
    target_base: int,  # ★追加
    return_noise: int,  # ★追加
) -> list:
    if param_id not in PARAM_MAP:
        return []
    param_info = PARAM_MAP[param_id]
    # base_value = param_info[1]
    min_fluctuation, max_fluctuation = param_info[2]
    jitter_amount = param_info[3]

    points = []
    # 中心点を計算
    mid_time = int((start_time + end_time) / 2)

    # 変動幅
    delta = random.randint(min_fluctuation, max_fluctuation)

    start_val = target_base
    end_val = target_base
    mid_val = 0

    if mode == "swell_convex":  # 山型 (< >)
        mid_val = clamp(target_base + delta, param_id)
    elif mode == "swell_concave":  # 谷型 (> <)
        mid_val = clamp(target_base - delta, param_id)
    else:
        return []

    # 前半 (Start -> Mid)
    dur1 = mid_time - start_time
    if dur1 > 0:
        points.append((start_time, start_val))
        num_p1 = max(2, min(dur1, 250))  # 点の数
        for i in range(1, num_p1):
            rate = i / num_p1
            t = int(start_time + dur1 * rate)
            # Ease-Out (山へ向かう) or Ease-In (谷へ向かう)
            f = ease_out_curve(rate) if mode == "swell_convex" else ease_in_curve(rate)
            val = start_val + (mid_val - start_val) * f
            points.append(
                (
                    t,
                    clamp(
                        val + random.randint(-jitter_amount, jitter_amount), param_id
                    ),
                )
            )

    points.append((mid_time, mid_val))

    # 後半 (Mid -> End)
    dur2 = end_time - mid_time
    if dur2 > 0:
        num_p2 = max(2, min(dur2, 250))
        for i in range(1, num_p2):
            rate = i / num_p2
            t = int(mid_time + dur2 * rate)
            # Ease-In (山から降りる) or Ease-Out (谷から戻る)
            f = ease_in_curve(rate) if mode == "swell_convex" else ease_out_curve(rate)
            val = mid_val + (end_val - mid_val) * f
            points.append(
                (
                    t,
                    clamp(
                        val + random.randint(-jitter_amount, jitter_amount), param_id
                    ),
                )
            )

    noise = random.randint(-return_noise, return_noise)
    final_val = clamp(target_base + noise, param_id)
    points.append((end_time, final_val))
    return points


# --- ★ 改善: Natural Curve (手描き風ランダムカーブ) 生成関数 ---
# Sin波を廃止し、ノートの先頭付近にピーク/谷を偏らせる。
def _generate_natural_curve_cc(
    start_time: int,
    end_time: int,
    param_id: str,
    target_base: int,
    return_noise: int,
    phrase_notes: list = None,
) -> list:
    """少数のランダム制御点をease曲線で繋ぐ手描き風カーブ。
    phrase_notesが与えられた場合、ノートの先頭付近に制御点を偏らせる。"""
    if param_id not in PARAM_MAP:
        return []
    min_fluctuation, max_fluctuation = PARAM_MAP[param_id][2]
    jitter_amount = PARAM_MAP[param_id][3]

    duration = end_time - start_time
    if duration <= 0:
        return []

    # --- 制御点（キーフレーム）の配置 ---
    control_points = [(start_time, target_base)]

    if phrase_notes:
        # ノートの先頭を制御点候補として、70%の確率で採用
        for note in phrase_notes:
            nt = note["t"]
            if start_time < nt < end_time and random.random() < 0.7:
                delta = random.randint(min_fluctuation, max_fluctuation)
                direction = random.choice([1, -1])
                val = clamp(target_base + delta * direction, param_id)
                control_points.append((nt, val))
    else:
        # ノート情報なし → ランダム間隔で配置
        interval = random.randint(480, 960)
        num_cp = max(2, duration // interval)
        for i in range(1, num_cp):
            base_t = start_time + int((duration / num_cp) * i)
            jitter_t = random.randint(-120, 120)
            t = max(start_time + 1, min(end_time - 1, base_t + jitter_t))
            delta = random.randint(min_fluctuation, max_fluctuation)
            direction = random.choice([1, -1])
            control_points.append(
                (t, clamp(target_base + delta * direction, param_id))
            )

    # 終点
    noise = random.randint(-return_noise, return_noise)
    control_points.append((end_time, clamp(target_base + noise, param_id)))
    control_points.sort(key=lambda x: x[0])

    # --- 制御点間をease_in_outで補間（疎な密度: 30〜60 tick） ---
    result = []
    for i in range(len(control_points) - 1):
        t1, v1 = control_points[i]
        t2, v2 = control_points[i + 1]
        seg_dur = t2 - t1
        if seg_dur <= 0:
            continue
        step = max(30, seg_dur // 8)
        for t in range(t1, t2, step):
            rate = (t - t1) / seg_dur
            f = ease_in_out(rate)
            v = v1 + (v2 - v1) * f
            j = random.randint(-jitter_amount, jitter_amount) if jitter_amount > 0 else 0
            result.append((t, clamp(v + j, param_id)))

    result.append((end_time, control_points[-1][1]))
    return result


# --- 追加: 重なり判定用関数 ---
def is_overlapping(start_t, end_t, occupied_ranges):
    """指定区間が予約済み区間と被っているか判定"""
    for o_start, o_end in occupied_ranges:
        if start_t < o_end and end_t > o_start:
            return True
    return False


# --- ★ 追加: 長すぎるフレーズの安全分割 ---
def _split_long_phrase(phrase, max_ticks):
    """フレーズがmax_ticksを超える場合、最も長い休符位置で安全に分割する。
    音符の途中では絶対に切らない。"""
    if not phrase:
        return [phrase]

    phrase_start = phrase[0]["t"]
    phrase_end = phrase[-1]["t"] + phrase[-1]["dur"]
    phrase_duration = phrase_end - phrase_start

    if phrase_duration <= max_ticks:
        return [phrase]

    # 8小節〜16小節の範囲内で最も長いノート間ギャップを探す
    half_max = max_ticks // 2
    best_split_idx = -1
    best_gap = -1

    for i in range(1, len(phrase)):
        note_start = phrase[i]["t"]
        prev_end = phrase[i - 1]["t"] + phrase[i - 1]["dur"]
        gap = note_start - prev_end
        elapsed = note_start - phrase_start
        if half_max <= elapsed <= max_ticks and gap > best_gap:
            best_gap = gap
            best_split_idx = i

    # 見つからなければ半分以降で最大ギャップを探す
    if best_split_idx < 0:
        for i in range(1, len(phrase)):
            note_start = phrase[i]["t"]
            prev_end = phrase[i - 1]["t"] + phrase[i - 1]["dur"]
            gap = note_start - prev_end
            elapsed = note_start - phrase_start
            if elapsed > half_max and gap > best_gap:
                best_gap = gap
                best_split_idx = i

    # それでも見つからなければ中間で強制分割
    if best_split_idx < 0:
        best_split_idx = len(phrase) // 2

    left = phrase[:best_split_idx]
    right = phrase[best_split_idx:]

    # 再帰的に分割
    result = []
    if left:
        result.extend(_split_long_phrase(left, max_ticks))
    if right:
        result.extend(_split_long_phrase(right, max_ticks))
    return result


# --- メイン処理関数 (★ 全トラック対応) ---


def augment_vsqx_params(
    vsqx_path: str,
    output_path: str,
    augmentation_targets: list,
    attack_pre_min: int,
    attack_pre_max: int,
    attack_post_min: int,
    attack_post_max: int,
    jump_rise_probability: float,
    jump_rise_param_ids: list,
    note_attack_probability: float,
    note_attack_param_ids: list,
    base_val_noise_range: int,
    return_val_noise_range: int,
    flat_phrase_probability: float = 0.1,  # ★追加: ベタ打ち確率
    note_attack_min_dur: int = 240,  # ★追加: スパイク対象最小ノート長
):
    try:
        with open(vsqx_path, "r", encoding="utf-8") as f:
            content = f.read()

        # ★ 変更: コールバック関数でトラックごとの処理を定義
        def process_track(track_match):
            track_content = track_match.group(0)

            # トラック番号の取得（ログ用）
            tno_match = RE_T_NO.search(track_content)
            tno = tno_match.group(1) if tno_match else "?"

            part_match = RE_VS_PART.search(track_content)
            if not part_match:
                # パートがないトラックはそのまま返す
                return track_content
            part_content = part_match.group(0)

            # 処理対象ID収集
            all_target_ids = set()
            for t in augmentation_targets:
                all_target_ids.add(t["id"])
            if jump_rise_probability > 0:
                all_target_ids.update(jump_rise_param_ids)
            if note_attack_probability > 0:
                all_target_ids.update(note_attack_param_ids)

            # 既存CC削除
            part_content = remove_existing_track_ccs(part_content, list(all_target_ids))

            notes = extract_notes_info(part_content)
            if not notes:
                return track_content

            final_ccs = defaultdict(dict)
            # ★ 追加: 予約済み区間リスト
            occupied_ranges = []
            # ★変更: トラックごとの基準値を決定する
            # IDごとにランダムな基準値を持つ辞書を作成
            track_base_values = {}
            for t_conf in augmentation_targets:
                pid = t_conf["id"]
                if pid in PARAM_MAP:
                    default_val = PARAM_MAP[pid][1]
                    # 64 ± n の範囲で決定
                    noise_n = random.randint(
                        -base_val_noise_range, base_val_noise_range
                    )
                    track_base_values[pid] = clamp(default_val + noise_n, pid)

            # 1. フレーズベース拡張
            phrases = []
            if notes:
                current_phrase = [notes[0]]
                for i in range(1, len(notes)):
                    prev_note_end = current_phrase[-1]["t"] + current_phrase[-1]["dur"]
                    current_note_start = notes[i]["t"]
                    gap = current_note_start - prev_note_end
                    if gap < PHRASE_GAP_THRESHOLD:
                        current_phrase.append(notes[i])
                    else:
                        phrases.append(current_phrase)
                        current_phrase = [notes[i]]
                phrases.append(current_phrase)

            # ★ 改善: 長すぎるフレーズを安全に分割
            split_phrases = []
            for phrase in phrases:
                split_phrases.extend(_split_long_phrase(phrase, PHRASE_MAX_TICKS))
            phrases = split_phrases

            for phrase in phrases:
                if not phrase:
                    continue
                phrase_start_t = phrase[0]["t"]
                phrase_end_t = phrase[-1]["t"] + phrase[-1]["dur"]

                for target in augmentation_targets:
                    param_id = target.get("id")
                    config_mode = target.get("mode")
                    current_base = track_base_values.get(param_id, 64)
                    if param_id not in PARAM_MAP:
                        continue

                    current_mode = ""
                    if config_mode == "random_attack":
                        current_mode = random.choice(["attack_dip", "attack_rise"])
                    elif config_mode == "random_all":
                        current_mode = random.choice(
                            [
                                "crescendo",
                                "decrescendo",
                                "attack_dip",
                                "attack_rise",
                                "swell_convex",
                                "swell_concave",
                                "natural_curve",  # ★改善: Sin波廃止
                            ]
                        )
                    else:
                        current_mode = config_mode

                    # ★ 改善: 10%の確率で「何もしない(ベタ打ち)」
                    apply_chance = random.random()
                    if config_mode.startswith("random") and apply_chance < flat_phrase_probability:
                        continue

                    points = []
                    if current_mode in ["attack_dip", "attack_rise"]:
                        points = _generate_attack_cc(
                            phrase_start_t,
                            param_id,
                            current_mode,
                            attack_pre_min,
                            attack_pre_max,
                            attack_post_min,
                            attack_post_max,
                            current_base,  # ★引数追加
                            return_val_noise_range,  # ★引数追加
                        )
                    elif current_mode in ["crescendo", "decrescendo"]:
                        points = _generate_crescendo_cc(
                            phrase_start_t,
                            phrase_end_t,
                            param_id,
                            current_mode,
                            current_base,  # ★引数追加
                            return_val_noise_range,  # ★引数追加
                        )
                    elif current_mode == "crescendo_decrescendo":
                        mode = random.choice(["crescendo", "decrescendo"])
                        points = _generate_crescendo_cc(
                            phrase_start_t, phrase_end_t, param_id, mode
                        )
                        current_mode = mode
                    # ★ 追加: 新しいパターンの呼び出し
                    elif current_mode in ["swell_convex", "swell_concave"]:
                        points = _generate_swell_cc(
                            phrase_start_t,
                            phrase_end_t,
                            param_id,
                            current_mode,
                            current_base,  # ★追加
                            return_val_noise_range,  # ★追加
                        )
                    elif current_mode == "natural_curve":
                        points = _generate_natural_curve_cc(
                            phrase_start_t,
                            phrase_end_t,
                            param_id,
                            current_base,
                            return_val_noise_range,
                            phrase_notes=phrase,  # ★改善: ノート先頭にピーク偏り
                        )

                    if points:
                        add_curve_to_dict(param_id, points, final_ccs)
                        # ★ 変更: 予約対象に新しいモードも追加
                        # ほとんど全てのモードが「区間予約」すべき対象になったので条件を広げる
                        if current_mode not in ["attack_dip", "attack_rise"]:
                            occupied_ranges.append((phrase_start_t, phrase_end_t))

            # 2. ノートベース拡張
            # ★ 改善: 短い音符にはスパイクを打たない + 間隔を空ける
            if jump_rise_probability > 0 or note_attack_probability > 0:
                last_attack_t = -9999  # スパイク間隔管理用
                for i in range(1, len(notes)):
                    prev_n = notes[i - 1]["n"]
                    curr_n = notes[i]["n"]
                    curr_t = notes[i]["t"]
                    curr_dur = notes[i]["dur"]
                    curr_end = curr_t + curr_dur

                    # 予約済み区間と被っていたらスキップ
                    if is_overlapping(curr_t, curr_end, occupied_ranges):
                        continue
                    # ★ 改善: 短い音符(note_attack_min_dur未満)はスキップ
                    if curr_dur < note_attack_min_dur:
                        continue
                    # ★ 改善: 前回のスパイクから近すぎる場合はスキップ
                    if curr_t - last_attack_t < 480:
                        continue

                    applied_note_attack = False
                    # ★ バグ修正: param_idごとの正しい基準値を使用
                    if (
                        curr_n - prev_n
                    ) >= 3 and random.random() < jump_rise_probability:
                        for param_id in jump_rise_param_ids:
                            if param_id not in PARAM_MAP:
                                continue
                            pid_base = track_base_values.get(param_id, 64)
                            points = _generate_attack_cc(
                                curr_t,
                                param_id,
                                "attack_rise",
                                attack_pre_min,
                                attack_pre_max,
                                attack_post_min,
                                attack_post_max,
                                pid_base,
                                return_val_noise_range,
                            )
                            add_curve_to_dict(param_id, points, final_ccs)
                        applied_note_attack = True
                        last_attack_t = curr_t

                    if (
                        not applied_note_attack
                        and random.random() < note_attack_probability
                    ):
                        for param_id in note_attack_param_ids:
                            if param_id not in PARAM_MAP:
                                continue
                            pid_base = track_base_values.get(param_id, 64)
                            mode = random.choice(["attack_dip", "attack_rise"])
                            points = _generate_attack_cc(
                                curr_t,
                                param_id,
                                mode,
                                attack_pre_min,
                                attack_pre_max,
                                attack_post_min,
                                attack_post_max,
                                pid_base,
                                return_val_noise_range,
                            )
                            add_curve_to_dict(param_id, points, final_ccs)
                        last_attack_t = curr_t

            if not final_ccs:
                return track_content

            final_cc_strings = []
            for param_id, tick_value_map in final_ccs.items():
                if not tick_value_map:
                    continue
                for t in sorted(tick_value_map.keys()):
                    v = tick_value_map[t]
                    final_cc_strings.append(
                        f'\t\t\t<cc>\n\t\t\t\t<t>{t}</t>\n\t\t\t\t<v id="{param_id}">{v}</v>\n\t\t\t</cc>'
                    )

            final_cc_string = "\n".join(final_cc_strings)
            augmented_part_content, count = RE_SINGER.subn(
                rf"\1\n{final_cc_string}", part_content, count=1
            )

            if count == 0:
                return track_content

            logging.info(f"    -> Track {tno}: 拡張完了")
            # パート部分を置き換えてトラックを再構築
            return track_content.replace(part_content, augmented_part_content, 1)

        # ★ RE_ALL_TRACKS.sub で全てのトラックに対して process_track を実行
        final_content = RE_ALL_TRACKS.sub(process_track, content)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        logging.info(f"  -> {Path(vsqx_path).name} 保存完了")
        return True

    except Exception as e:
        logging.error(f"処理中に予期せぬエラーが発生しました ({vsqx_path}): {e}")
        import traceback

        traceback.print_exc()
        return False


# --- 実行関数 (変更なし) ---


def process_files(config: dict):
    input_dir = config.get("input_dir")
    output_base_dir = config.get("output_base_dir")
    augmentation_targets = config.get("augmentation_targets", [])

    attack_pre_min = config.get("attack_pre_duration_min", 60)
    attack_pre_max = config.get("attack_pre_duration_max", 120)
    attack_post_min = config.get("attack_post_duration_min", 60)
    attack_post_max = config.get("attack_post_duration_max", 120)
    jump_rise_probability = config.get("jump_rise_probability", 0.0)
    jump_rise_param_ids = config.get("jump_rise_param_ids", ["P"])
    note_attack_probability = config.get("note_attack_probability", 0.0)
    note_attack_param_ids = config.get("note_attack_param_ids", ["P"])
    base_n = config.get("base_value_noise", 0)
    return_m = config.get("return_value_noise", 0)
    flat_prob = config.get("flat_phrase_probability", 0.1)  # ★追加
    min_dur = config.get("note_attack_min_dur", 240)  # ★追加

    if not all([input_dir, output_base_dir]):
        logging.error("エラー: config不足")
        return

    folder_name_parts = []
    for target in augmentation_targets:
        param_name = PARAM_MAP.get(target.get("id"), ("Unk", 0))[0].lower()
        mode_name = target.get("mode", "unk")
        folder_name_parts.append(f"{param_name}_{mode_name}")

    if jump_rise_probability > 0:
        param_names = "_".join(
            [PARAM_MAP.get(pid, ("Unk", 0))[0].lower() for pid in jump_rise_param_ids]
        )
        folder_name_parts.append(f"{param_names}_jump")
    if note_attack_probability > 0:
        param_names = "_".join(
            [PARAM_MAP.get(pid, ("Unk", 0))[0].lower() for pid in note_attack_param_ids]
        )
        folder_name_parts.append(f"{param_names}_note_attack")

    output_dir_name = "write_aug"
    output_dir = os.path.join(output_base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"出力先フォルダ: {output_dir}")

    vsqx_files = glob.glob(os.path.join(input_dir, "*.vsqx"))
    logging.info(f"{len(vsqx_files)} 件処理...")

    success_count = 0
    fail_count = 0
    for vsqx_path in vsqx_files:
        original_filename = Path(vsqx_path).stem
        output_filename = f"{original_filename}_aug3.vsqx"
        output_path = os.path.join(output_dir, output_filename)
        if augment_vsqx_params(
            vsqx_path,
            output_path,
            augmentation_targets,
            attack_pre_min,
            attack_pre_max,
            attack_post_min,
            attack_post_max,
            jump_rise_probability,
            jump_rise_param_ids,
            note_attack_probability,
            note_attack_param_ids,
            base_n,
            return_m,
            flat_prob,  # ★追加
            min_dur,  # ★追加
        ):
            success_count += 1
        else:
            fail_count += 1

    logging.info(f"成功: {success_count}件, 失敗: {fail_count}件")


# --- スクリプト実行 ---
if __name__ == "__main__":

    # ▼▼▼ 設定をここで行う ▼▼▼
    execution_config = {
        "input_dir": "./data/vsqx_ALL/original/got_original/vspart_merged/Dynamic",
        "output_base_dir": "./data/vsqx_ALL/original/got_original/augment/Dynamic",
        # ★ モード一覧:
        # "crescendo" / "decrescendo" : フレーズ全体の抑揚
        # "attack_dip"                : フレーズの頭で「下がって上がる」
        # "attack_rise"               : フレーズの頭で「上がって下がる」
        # "swell_convex"              : 山型 (< >)
        # "swell_concave"             : 谷型 (> <)
        # "natural_curve"             : ★改善 ノート先頭にピークが偏る手描き風カーブ
        # "random_attack"             : attack_dip / attack_rise ランダム
        # "random_all"                : 全モードからランダム
        # (必須) フレーズ単位の拡張ターゲット
        "augmentation_targets": [
            {"id": "D", "mode": "random_all"},  # DYN (ダイナミクス)
            {"id": "B", "mode": "random_all"},  # BRE (ブレシネス)
            {"id": "R", "mode": "random_all"},  # BRI (ブライトネス)
            {"id": "C", "mode": "random_all"},  # CLE (クリアネス)
            {"id": "G", "mode": "random_all"},  # GEN (ジェンダーファクター)
            {"id": "P", "mode": "random_attack"},  # PIT (ピッチベンド)
            {"id": "S", "mode": "random_all"},  # PBS (ピッチベンドセンシティビティ)
            {"id": "W", "mode": "random_all"},  # GWL (グロウル)
            {"id": "X", "mode": "random_all"},  # XSY (クロスシンセシス)
            {"id": "T", "mode": "random_all"},  # POR (ポルタメント)
        ],
        # ジャンプ時に適用するパラメータID (リスト)
        "jump_rise_param_ids": ["D", "P", "B"],
        "jump_rise_probability": 0.8,
        # ランダムアタックで適用するパラメータID (リスト)
        "note_attack_param_ids": ["D", "B", "R", "C", "G", "P", "S", "W", "X", "T"],
        "note_attack_probability": 0.2,
        "attack_pre_duration_min": 30,
        "attack_pre_duration_max": 100,
        "attack_post_duration_min": 240,
        "attack_post_duration_max": 480,
        "base_value_noise": 20,  # n: 初期値が 64±20 (44~84) の間で決まる
        "return_value_noise": 15,  # m: 戻るときに 基準値±15 の誤差が乗る
        # ★ 追加: 改善用パラメータ
        "flat_phrase_probability": 0.1,  # フレーズを「何もしない」確率 (10%)
        "note_attack_min_dur": 240,  # スパイク対象の最小ノート長 (8分音符以上)
    }
    # ▲▲▲ 設定はここまで ▲▲▲

    process_files(execution_config)
