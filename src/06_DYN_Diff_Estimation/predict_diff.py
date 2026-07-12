"""
predict_diff.py
===============
学習済みの時間差分モデルを使い、DYN加工済み音声からDYNカーブを推定する。
推定結果はグラフ画像(PNG)と、VSQXファイルへの書き込みとして出力する。

処理フロー:
  1. 対象音声から Mel を抽出
  2. DIFF_STEPフレーム間隔でMelの時間差分を計算
  3. 学習済みモデルで ΔDYN を予測
  4. 漏れ積分 (Leaky Integrator) で絶対DYN値を復元
  5. ノート境界を補間、ノート間の無音区間を補間
  6. Savitzky-Golayフィルタで平滑化
  7. グラフ出力 & VSQX書き込み

使い方:
  1. train_diff.py を先に実行してモデルを作成
  2. config_diff.py の TARGET_WAV, TARGET_VSQX を設定
  3. このファイルを実行
  4. output/ にグラフとVSQXが生成される
"""

import os
import sys
import re
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa
from scipy.signal import savgol_filter

# 日本語フォントの設定（Windows環境向け）
plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'sans-serif']

# 既存モジュールへのパス追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "05_DYN_Estimation"))

import config_diff as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader
from pair_finder import find_pairs

# --- 正規表現パターン (VSQX書き込み用) ---
RE_RESOLUTION = re.compile(r"<resolution>(\d+)</resolution>")
RE_TEMPO_V = re.compile(
    r"<tempo>\s*<t>0</t>\s*<v>(\d+)</v>\s*</tempo>", re.DOTALL
)
RE_VS_PART = re.compile(r"<vsPart>.*?</vsPart>", re.DOTALL)
RE_SINGER = re.compile(r"(<singer>.*?</singer>)", re.DOTALL)
RE_CC_TAG = re.compile(r'<cc>.*?<v id="([A-Z]+)">.*?</cc>', re.DOTALL)
RE_NOTE = re.compile(r"<note>.*?</note>", re.DOTALL)


def estimate_dyn_diff(mel, model_data, notes, total_frames):
    """
    Melスペクトログラムからフレーム間差分を計算し、
    漏れ積分でDYNカーブを復元する。

    Args:
        mel: np.ndarray (N, n_mels) - Melスペクトログラム
        model_data: dict - 学習済みモデルデータ
        notes: list of dict - ノート情報
        total_frames: int - 総フレーム数

    Returns:
        dyn_curve: np.ndarray (N,) - 推定DYNカーブ
        delta_dyn_raw: np.ndarray (N,) - 生のΔDYN予測値 (デバッグ用)
    """
    model = model_data["model"]
    scaler = model_data["scaler"]
    step = model_data["diff_step"]
    skip = cfg.NOTE_BOUNDARY_SKIP_FRAMES

    # ノート区間をフレームインデックスに変換
    note_regions = []
    for note in notes:
        s_idx = librosa.time_to_frames(
            note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
        )
        e_idx = librosa.time_to_frames(
            note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
        )
        s_idx = max(0, s_idx)
        e_idx = min(total_frames, e_idx)
        note_regions.append((s_idx, e_idx))

    # ΔDYN を格納する配列 (0で初期化 = 変化なし)
    delta_dyn_raw = np.zeros(total_frames, dtype=np.float32)

    # Mel全帯域平均パワー (音量ガード・動的アンカー用)
    mel_mean_power = np.mean(mel, axis=1)

    # ノート区間ごとに予測
    for s_idx, e_idx in note_regions:
        inner_start = s_idx + skip
        inner_end = e_idx - skip

        if inner_end - inner_start < step + 1:
            continue

        # 有効区間のΔMelを一括計算
        frame_indices = np.arange(inner_start + step, inner_end)
        if len(frame_indices) == 0:
            continue

        delta_mel = mel[frame_indices] - mel[frame_indices - step]
        delta_mel_scaled = scaler.transform(delta_mel)
        delta_dyn_pred = model.predict(delta_mel_scaled)

        # 出力振幅倍率の適用
        delta_dyn_pred *= cfg.DIFF_OUTPUT_SCALE

        # ステップリミッター: 音素変化による極端な変動をクランプ
        delta_dyn_pred = np.clip(delta_dyn_pred, -cfg.DELTA_DYN_CLAMP, cfg.DELTA_DYN_CLAMP)

        # 音量ガード: Melパワーが閾値以下のフレームはΔDYN=0（変化なし）とする
        # 無声子音や息継ぎの瞬間的な音量急降下によるスパイクを防止
        power_at_frames = mel_mean_power[frame_indices]
        silence_mask = power_at_frames < cfg.MEL_POWER_THRESHOLD_DB
        delta_dyn_pred[silence_mask] = 0.0

        delta_dyn_raw[frame_indices] = delta_dyn_pred

    # ---- 漏れ積分 (Leaky Integrator) でDYNカーブを復元 ----
    dyn_curve = np.full(total_frames, float(cfg.DYN_DEFAULT), dtype=np.float32)
    leak = cfg.LEAK_RATE
    anchor_model = model_data.get("anchor_model")

    for s_idx, e_idx in note_regions:
        # 動的アンカー: 学習済みのアンカーモデルでノート開始時のDYNを推定
        if cfg.DYNAMIC_ANCHOR_ENABLED and anchor_model is not None:
            # ノート開始付近（境界スキップ後）の安定した区間のパワーを取得
            anchor_start = min(s_idx + skip, e_idx - 1)
            anchor_end = min(anchor_start + step, e_idx)
            if anchor_end > anchor_start:
                note_start_power = np.mean(mel_mean_power[anchor_start:anchor_end])
            else:
                note_start_power = mel_mean_power[anchor_start]

            # 学習済みモデルでパワーから開始DYNを予測
            predicted_start_dyn = anchor_model.predict([[note_start_power]])[0]
            current_dyn = float(np.clip(predicted_start_dyn, 0, 127))
        else:
            current_dyn = float(cfg.DYN_DEFAULT)

        for f in range(s_idx, e_idx):
            # 漏れ積分: DYNの変化量を加算しつつ、デフォルト値に少しずつ引き戻す
            current_dyn = current_dyn + delta_dyn_raw[f] - leak * (current_dyn - cfg.DYN_DEFAULT)
            # フレームごとにクリッピング（マイナス借金の蓄積を防止）
            current_dyn = max(0.0, min(127.0, current_dyn))
            dyn_curve[f] = current_dyn

    return dyn_curve, delta_dyn_raw


def interpolate_note_gaps(dyn_curve, notes, total_frames):
    """
    ノート間の無音区間を補間処理する。

    - ノート有区間: 推定値をそのまま使用
    - ノート無区間 (短い休符): 前後のDYN値を線形補間
    - ノート無区間 (長い休符): DYN_DEFAULT (64) に自然に戻す
    """
    dyn_interpolated = dyn_curve.copy()

    if not notes:
        return dyn_interpolated

    note_regions = []
    for note in notes:
        s_idx = librosa.time_to_frames(
            note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
        )
        e_idx = librosa.time_to_frames(
            note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
        )
        s_idx = max(0, s_idx)
        e_idx = min(total_frames, e_idx)
        note_regions.append((s_idx, e_idx))

    # 最初のノートより前の区間
    if note_regions[0][0] > 0:
        dyn_interpolated[:note_regions[0][0]] = cfg.DYN_DEFAULT

    # ノート間の補間
    for i in range(len(note_regions) - 1):
        _, prev_end = note_regions[i]
        next_start, _ = note_regions[i + 1]

        if next_start <= prev_end:
            continue

        gap_frames = next_start - prev_end
        gap_sec = librosa.frames_to_time(
            gap_frames, sr=cfg.SR, hop_length=cfg.HOP_LENGTH
        )

        prev_dyn = dyn_curve[min(prev_end - 1, total_frames - 1)]
        next_dyn = dyn_curve[min(next_start, total_frames - 1)]

        if gap_sec >= cfg.REST_THRESHOLD_SEC:
            # 長い休符: デフォルトに戻す
            mid = prev_end + gap_frames // 2
            dyn_interpolated[prev_end:mid] = np.linspace(
                prev_dyn, cfg.DYN_DEFAULT, mid - prev_end
            )
            dyn_interpolated[mid:next_start] = np.linspace(
                cfg.DYN_DEFAULT, next_dyn, next_start - mid
            )
        else:
            # 短い休符: 直接線形補間
            dyn_interpolated[prev_end:next_start] = np.linspace(
                prev_dyn, next_dyn, next_start - prev_end
            )

    # 最後のノートより後の区間
    if note_regions[-1][1] < total_frames:
        last_end = note_regions[-1][1]
        last_dyn = dyn_curve[min(last_end - 1, total_frames - 1)]
        remaining = total_frames - last_end
        dyn_interpolated[last_end:] = np.linspace(
            last_dyn, cfg.DYN_DEFAULT, remaining
        )

    return dyn_interpolated


def smooth_dyn(dyn_curve):
    """Savitzky-Golayフィルタによる平滑化を適用する。"""
    window = cfg.SAVGOL_WINDOW
    polyorder = cfg.SAVGOL_POLYORDER

    if len(dyn_curve) < window:
        return dyn_curve

    dyn_smoothed = savgol_filter(dyn_curve, window, polyorder)
    return np.clip(dyn_smoothed, 0, 127)


def plot_results(time_axis, mel, delta_dyn_raw, dyn_curve,
                 gt_dyn_curve, notes, output_path):
    """推定結果のグラフを描画・保存する"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # 1. Mel Power と ΔDYN予測値
    ax = axes[0]
    mel_mean = np.mean(mel, axis=1)
    ax.plot(time_axis, mel_mean, label="Mel Power (全帯域平均)", alpha=0.7)
    ax.set_ylabel("Mel Power (dB)")
    ax.set_title("Input: Mel Spectrogram Power")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ノート区間の表示
    for note in notes:
        for a in axes:
            a.axvspan(note["start_sec"], note["end_sec"],
                      color="green", alpha=0.05)

    # 2. 推定DYN vs 正解DYN
    ax = axes[1]
    ax.plot(time_axis, dyn_curve, label="推定DYN", color="tab:blue")
    if gt_dyn_curve is not None:
        min_len = min(len(time_axis), len(gt_dyn_curve))
        ax.plot(time_axis[:min_len], gt_dyn_curve[:min_len],
                label="正解DYN", color="tab:red", alpha=0.7, linestyle="--")
    ax.axhline(y=64, color="gray", linestyle=":", alpha=0.5, label="DYN=64")
    ax.set_ylabel("DYN Value")
    ax.set_ylim(-5, 132)
    ax.set_title("Estimated DYN Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ノートのテキスト表示
    for note in notes:
        mid = (note["start_sec"] + note["end_sec"]) / 2
        ax.text(mid, 130, note.get("lyric", ""),
                ha='center', va='top', fontsize=8, color='darkgreen')

    # 3. 誤差 (正解がある場合)
    ax = axes[2]
    if gt_dyn_curve is not None:
        min_len = min(len(dyn_curve), len(gt_dyn_curve))
        error = dyn_curve[:min_len] - gt_dyn_curve[:min_len]
        mae = np.mean(np.abs(error))
        ax.plot(time_axis[:min_len], error, color="tab:orange", alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle=":")
        ax.set_title(f"Estimation Error (MAE: {mae:.2f})")
    else:
        # 正解がない場合はΔDYNの生値を表示
        ax.plot(time_axis, delta_dyn_raw, color="tab:purple", alpha=0.5)
        ax.axhline(y=0, color="gray", linestyle=":")
        ax.set_title("Raw ΔDYN Prediction")
    ax.set_ylabel("Error (Est - GT)")
    ax.set_xlabel("Time (sec)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  グラフ保存: {output_path}")


def write_dyn_to_vsqx(vsqx_path, output_path, dyn_values, sr, hop_length):
    """
    推定したDYNカーブをVSQXファイルに書き込む。
    """
    try:
        with open(vsqx_path, "r", encoding="utf-8") as f:
            content = f.read()

        # メタ情報の取得
        res_match = RE_RESOLUTION.search(content)
        resolution = int(res_match.group(1)) if res_match else 480

        tempo_match = RE_TEMPO_V.search(content)
        bpm = int(tempo_match.group(1)) / 100.0 if tempo_match else 120.0

        # vsPart の取得
        part_match = RE_VS_PART.search(content)
        if not part_match:
            print("  [エラー] vsPart が見つかりません。")
            return
        part_content = part_match.group(0)

        # 既存の DYN CC タグを削除
        placeholders = []

        def _mask_note(match):
            placeholders.append(match.group(0))
            return f"__NOTE_PLACEHOLDER_{len(placeholders)-1}__"

        masked_part = RE_NOTE.sub(_mask_note, part_content)

        removed_count = 0

        def _remove_dyn_cc(match):
            nonlocal removed_count
            cc_id = match.group(1)
            if cc_id == "D":  # D = DYN
                removed_count += 1
                return ""
            return match.group(0)

        cleaned_part = RE_CC_TAG.sub(_remove_dyn_cc, masked_part)

        # プレースホルダ復元
        for i, note_str in enumerate(placeholders):
            cleaned_part = cleaned_part.replace(
                f"__NOTE_PLACEHOLDER_{i}__", note_str
            )

        print(f"  既存DYN CC {removed_count} 個を削除")

        # 新しいCC文字列の生成
        # vsqx_reader.py の読み込みロジックと対称:
        #   読み込み: パート内相対Tick → _tick_to_second → 秒
        #   書き込み: 秒 → 相対Tick (逆変換)
        cc_strings = []
        last_val = -9999

        for frame_idx, val in enumerate(dyn_values):
            int_val = int(np.clip(val, 0, 127))

            if int_val == last_val:
                continue

            time_sec = librosa.frames_to_time(
                frame_idx, sr=sr, hop_length=hop_length
            )
            # 秒 → パート内相対Tick (vsqx_readerの逆変換)
            rel_tick = int(time_sec * bpm * resolution / 60.0)

            if rel_tick < 0:
                continue

            cc_strings.append(
                f'\t\t\t<cc>\n'
                f'\t\t\t\t<t>{rel_tick}</t>\n'
                f'\t\t\t\t<v id="D">{int_val}</v>\n'
                f'\t\t\t</cc>'
            )
            last_val = int_val

        # singer直後にCC文字列を挿入
        if cc_strings:
            final_cc_block = "\n".join(cc_strings)
            augmented_part = RE_SINGER.sub(
                rf"\1\n{final_cc_block}", cleaned_part, count=1
            )
        else:
            augmented_part = cleaned_part

        # 書き出し
        final_content = content.replace(part_content, augmented_part, 1)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"  VSQX保存: {output_path} ({len(cc_strings)} CC points)")

    except Exception as e:
        print(f"  [エラー] VSQX書き込み失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("  DYN Estimation (Time-Diff Model) — Prediction")
    print("=" * 60)

    # モデルの読み込み
    if not os.path.exists(cfg.MODEL_PATH):
        print(f"\n[エラー] モデルファイルが見つかりません: {cfg.MODEL_PATH}")
        print("  先に train_diff.py を実行してください。")
        return

    print(f"\n[Step 1] モデルの読み込み...")
    model_data = joblib.load(cfg.MODEL_PATH)
    print(f"  R²={model_data['r_squared']:.4f}, "
          f"RMSE={model_data['rmse']:.4f}, "
          f"学習ファイル数={model_data['n_training_files']}")

    # 対象ファイルの確認
    wav_path = cfg.TARGET_WAV
    vsqx_path = cfg.TARGET_VSQX

    if not os.path.exists(wav_path):
        print(f"\n[エラー] WAVファイルが見つかりません: {wav_path}")
        return
    if not os.path.exists(vsqx_path):
        print(f"\n[エラー] VSQXファイルが見つかりません: {vsqx_path}")
        return

    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    print(f"  対象: {base_name}")

    # 特徴量抽出
    print(f"\n[Step 2] 特徴量抽出...")
    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    mel = extractor.get_features(wav_path, ["mel"])
    total_frames = len(mel)

    vsqx_data = reader.read(vsqx_path)
    if vsqx_data is None:
        print("  [エラー] VSQX読み込み失敗")
        return

    notes = vsqx_data.get("notes", [])
    print(f"  フレーム数: {total_frames}, ノート数: {len(notes)}")

    # 正解DYNの取得 (あれば)
    gt_dyn_points = vsqx_data["dynamic_params"].get("DYN", [])
    gt_dyn_curve = None
    if gt_dyn_points:
        gt_dyn_curve = reader.interpolate_param_to_frames(
            gt_dyn_points, total_frames, default_value=cfg.DYN_DEFAULT
        )

    # DYN推定
    print(f"\n[Step 3] DYN推定 (時間差分モデル)...")
    dyn_raw, delta_dyn_raw = estimate_dyn_diff(mel, model_data, notes, total_frames)

    # ノート間補間
    print(f"[Step 4] ノート間補間...")
    dyn_interpolated = interpolate_note_gaps(dyn_raw, notes, total_frames)

    # 平滑化
    print(f"[Step 5] 平滑化...")
    dyn_smooth = smooth_dyn(dyn_interpolated)

    # 精度表示 (正解がある場合)
    if gt_dyn_curve is not None:
        min_len = min(len(dyn_smooth), len(gt_dyn_curve))
        error = dyn_smooth[:min_len] - gt_dyn_curve[:min_len]
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # グラフ出力
    print(f"\n[Step 6] 結果出力...")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    time_axis = librosa.frames_to_time(
        np.arange(total_frames), sr=cfg.SR, hop_length=cfg.HOP_LENGTH
    )

    plot_path = os.path.join(cfg.OUTPUT_DIR, f"{base_name}_diff_dyn.png")
    plot_results(time_axis, mel, delta_dyn_raw, dyn_smooth,
                 gt_dyn_curve, notes, plot_path)

    # VSQX書き込み
    vsqx_output = os.path.join(cfg.OUTPUT_DIR, f"{base_name}_diff_dyn.vsqx")
    write_dyn_to_vsqx(vsqx_path, vsqx_output, dyn_smooth,
                       cfg.SR, cfg.HOP_LENGTH)

    print(f"\n{'=' * 60}")
    print(f"  推定完了!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
