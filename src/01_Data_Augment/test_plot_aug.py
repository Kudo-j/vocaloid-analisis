import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math

# ==========================================
# 共通関数群（DPara_add_AllTrack.pyから再利用）
# ==========================================
PARAM_MAP = {"D": ("DYN", 64, (10, 60), 0)}
PARAM_VALID_RANGES = {"D": (0, 127)}


def ease_in_out(t: float) -> float:
    return 0.5 * (1 - math.cos(t * math.pi))


def clamp(v, param_id="D"):
    v_int = int(round(v))
    min_val, max_val = PARAM_VALID_RANGES.get(param_id, (0, 127))
    return max(min_val, min(max_val, v_int))


# ==========================================
# 既存のデータ拡張ロジック（問題点再現用）
# ==========================================
def _generate_periodic_cc_old(
    start_time: int, end_time: int, mode: str, target_base: int
) -> list:
    """既存の機械的なSin波による揺れ"""
    points = []
    duration = end_time - start_time
    if duration <= 0:
        return []
    freq = 0.01 if mode == "drift" else 0.1
    amp = 15 if mode == "drift" else 5
    step = 10  # 10tickごとの超精密な点（人間離れしている原因）

    for t in range(start_time, end_time, step):
        rel_t = t - start_time
        wave = math.sin(rel_t * freq)
        if mode == "drift":
            wave = (math.sin(rel_t * freq) + math.cos(rel_t * (freq * 1.5))) / 2.0
        val = target_base + (wave * amp)
        points.append((t, clamp(val)))
    points.append((end_time, target_base))
    return points


def _generate_attack_cc_old(start_time: int, target_base: int) -> list:
    """既存の単一スパイク（ノートごとに呼ばれて連続する）"""
    points = []
    change_amount = 20
    points.append((max(0, start_time - 60), target_base))
    points.append((start_time, clamp(target_base + change_amount)))
    points.append((start_time + 240, target_base))
    return points


# ==========================================
# ★ 改善案のデータ拡張ロジック
# ==========================================
def _generate_natural_curve_cc_new(
    start_time: int,
    end_time: int,
    target_base: int,
    phrase_notes: list = None,
) -> list:
    """改善案：ノートの先頭付近にピーク/谷を配置する手描き風カーブ。
    少数のランダム制御点をease曲線で繋ぐ。"""
    min_fluctuation, max_fluctuation = 10, 60

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
                val = clamp(target_base + delta * direction)
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
            control_points.append((t, clamp(target_base + delta * direction)))

    # 終点
    noise = random.randint(-5, 5)
    control_points.append((end_time, clamp(target_base + noise)))
    control_points.sort(key=lambda x: x[0])

    # --- 制御点間をease_in_outで補間（疎な密度） ---
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
            result.append((t, clamp(v)))

    result.append((end_time, control_points[-1][1]))
    return result, control_points  # 制御点も返す（可視化用）


def _generate_sparse_attacks_new(
    notes: list, target_base: int, min_dur: int = 240
) -> list:
    """改善案：ノートの長さを考慮し、短い音符にはスパイクを打たない。
    スパイク間に最低480tickの間隔を空ける。"""
    points = []
    last_attack_t = -9999

    for note in notes:
        t = note["t"]
        dur = note["dur"]

        # 短い音符はスキップ
        if dur < min_dur:
            continue
        # 直前のスパイクから近すぎる場合はスキップ
        if t - last_attack_t < 480:
            continue

        # 480tick以上ならスパイクを打つ判定
        if dur >= 480:
            max_spikes = 1 if dur < 960 else random.randint(1, 2)

            used_times = []
            for _ in range(max_spikes):
                spike_t = t + random.randint(0, dur - 120)
                if any(abs(spike_t - ut) < 360 for ut in used_times):
                    continue
                used_times.append(spike_t)

                change_amount = random.randint(15, 30) * random.choice([1, -1])
                points.append((max(0, spike_t - 60), target_base))
                points.append((spike_t, clamp(target_base + change_amount)))
                points.append(
                    (spike_t + random.randint(120, 240), target_base)
                )

            last_attack_t = t

    points.sort(key=lambda x: x[0])
    return points


# ==========================================
# プロット用関数
# ==========================================
def plot_simulation(notes, curve_points, title, control_pts=None):
    """ノート（ピアノロール風）とパラメータカーブを同時にプロット"""
    fig, ax = plt.subplots(figsize=(14, 5))

    # ノート（音符）の描画
    for note in notes:
        t = note["t"]
        dur = note["dur"]
        n = note["n"]
        rect = patches.Rectangle(
            (t, n - 0.5),
            dur,
            1,
            linewidth=1,
            edgecolor="black",
            facecolor="lightblue",
            alpha=0.7,
        )
        ax.add_patch(rect)
        # ノートの先頭に縦線（視認性向上）
        ax.axvline(t, color="steelblue", linestyle=":", linewidth=0.5, alpha=0.5)

    # パラメータカーブの描画（第2軸）
    times = [p[0] for p in curve_points]
    values = [p[1] for p in curve_points]

    ax2 = ax.twinx()
    ax2.plot(
        times,
        values,
        marker="o",
        markersize=2,
        color="red",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label="DYN Parameter",
    )

    # 制御点を大きい丸で強調表示
    if control_pts:
        cp_times = [p[0] for p in control_pts]
        cp_values = [p[1] for p in control_pts]
        ax2.scatter(
            cp_times,
            cp_values,
            s=100,
            color="gold",
            edgecolors="black",
            zorder=5,
            label="Control Points (Note Starts)",
        )
        ax2.legend(loc="upper right")

    ax2.set_ylim(0, 127)
    ax2.set_ylabel("CC Value (0-127)", color="red")
    ax2.axhline(64, color="gray", linestyle="--", linewidth=0.5)

    # 軸の設定
    max_t = max([n["t"] + n["dur"] for n in notes]) + 480
    ax.set_xlim(0, max_t)
    ax.set_ylim(55, 75)
    ax.set_xlabel("Time (ticks) [480 ticks = 1 Quarter Note]")
    ax.set_ylabel("Note Number")

    # 点の数をタイトルに追加
    plt.title(f"{title}\n(Total CC points: {len(curve_points)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 実行部（コンフィグ領域）
# ==========================================
if __name__ == "__main__":

    # 1. テスト用のダミーノート配置（フレーズ）
    dummy_notes = [
        {"t": 480, "dur": 240, "n": 60},  # 8分音符
        {"t": 720, "dur": 240, "n": 62},  # 8分音符
        {"t": 960, "dur": 240, "n": 64},  # 8分音符
        {"t": 1200, "dur": 240, "n": 62},  # 8分音符
        {"t": 1440, "dur": 960, "n": 60},  # 2分音符（長い）
        {"t": 2880, "dur": 1920, "n": 67},  # 全音符（とても長い）
    ]

    target_base = 64
    phrase_start = dummy_notes[0]["t"]
    phrase_end = dummy_notes[-1]["t"] + dummy_notes[-1]["dur"]

    # ▼ 実行したいテストモードをTrueにする ▼

    # --- A: 既存の問題点確認 ---
    TEST_OLD_DRIFT = True  # 機械的な完全Sinカーブ
    TEST_OLD_ATTACK = True  # 短い音符にも構わずスパイクが入るチェーンソー状態

    # --- B: 改善案の確認 ---
    TEST_NEW_CURVE = True  # ノート先頭にピーク/谷が偏る手描き風カーブ
    TEST_NEW_ATTACK = True  # 長い音符にだけスパイクが入る

    # ----------------------------------------------------
    # テストA-1: 既存の機械的ドリフト
    if TEST_OLD_DRIFT:
        pts = _generate_periodic_cc_old(phrase_start, phrase_end, "drift", target_base)
        plot_simulation(
            dummy_notes,
            pts,
            "OLD: Perfect Sine Wave Drift (Mechanical, Too many points)",
        )

    # テストA-2: 既存の連続スパイク
    if TEST_OLD_ATTACK:
        pts = []
        for n in dummy_notes:
            pts.extend(_generate_attack_cc_old(n["t"], target_base))
        pts.sort(key=lambda x: x[0])
        plot_simulation(
            dummy_notes, pts, "OLD: Note Attacks on Every Note (Chainsaw Effect)"
        )

    # テストB-1: 改善案のナチュラルカーブ（ノート先頭偏り）
    if TEST_NEW_CURVE:
        pts, ctrl_pts = _generate_natural_curve_cc_new(
            phrase_start, phrase_end, target_base, phrase_notes=dummy_notes
        )
        plot_simulation(
            dummy_notes,
            pts,
            "NEW: Natural Curve (Peaks biased toward note starts)",
            control_pts=ctrl_pts,
        )

    # テストB-2: 改善案の疎なスパイク
    if TEST_NEW_ATTACK:
        pts = _generate_sparse_attacks_new(dummy_notes, target_base)
        plot_simulation(
            dummy_notes,
            pts,
            "NEW: Sparse Spikes (Only on long notes, min 480 tick spacing)",
        )
