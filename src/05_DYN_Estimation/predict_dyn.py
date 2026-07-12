"""
predict_dyn.py
==============
学習済みの2段階モデルを使い、DYN加工済み音声からDYNカーブを推定する。
推定結果はグラフ画像(PNG)と、VSQXファイルへの書き込みとして出力する。

処理フロー:
  1. 対象音声から MFCC と Mel を抽出
  2. ベースラインモデルで「本来のMel（DYN=64相当）」を予測
  3. ΔMel = 実際のMel - ベースラインMel を算出
  4. DYNスケールモデルで ΔMel → DYN値 を推定
  5. VSQXのノート情報に基づき、ノート間を補間処理
  6. Savitzky-Golayフィルタで平滑化
  7. グラフ出力 & VSQX書き込み

使い方:
  1. train_baseline.py を先に実行してモデルを作成
  2. config_dyn.py の TARGET_WAV, TARGET_VSQX を設定
  3. このファイルをVSCode上で実行 (F5 or ▶)
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

import config_dyn as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader

# --- 正規表現パターン (VSQX書き込み用, predict_en.py準拠) ---
RE_RESOLUTION = re.compile(r"<resolution>(\d+)</resolution>")
RE_TEMPO_V = re.compile(
    r"<tempo>\s*<t>0</t>\s*<v>(\d+)</v>\s*</tempo>", re.DOTALL
)
RE_VS_PART = re.compile(r"<vsPart>.*?</vsPart>", re.DOTALL)
RE_PART_T = re.compile(r"<vsPart>\s*<t>(\d+)</t>")
RE_SINGER = re.compile(r"(<singer>.*?</singer>)", re.DOTALL)
RE_CC_TAG = re.compile(r'<cc>.*?<v id="([A-Z]+)">.*?</cc>', re.DOTALL)
RE_NOTE = re.compile(r"<note>.*?</note>", re.DOTALL)


def estimate_dyn(mfcc, mel, baseline_model_data, dyn_model_data, voicing_mask):
    """
    MFCCとMelスペクトログラムからDYNカーブを推定する。

    Args:
        mfcc: np.ndarray (N, n_mfcc)
        mel: np.ndarray (N, n_mels)
        baseline_model_data: dict - Phase 1のモデルデータ
        dyn_model_data: dict - Phase 2のモデルデータ
        voicing_mask: np.ndarray (N,) - 発声区間がTrueのマスク

    Returns:
        dyn_curve: np.ndarray (N,) - 推定DYN (0-127)
        baseline_mel: np.ndarray (N, n_mels) - ベースラインMel
        delta_mel: np.ndarray (N, n_mels) - ΔMel
    """
    bl_model = baseline_model_data["model"]
    bl_scaler = baseline_model_data["scaler"]

    dyn_model = dyn_model_data["model"]
    dyn_scaler = dyn_model_data["scaler"]
    poly_transformer = dyn_model_data.get("poly_transformer", None)

    N = len(mfcc)
    dyn_raw = np.full(N, float(cfg.DYN_DEFAULT))
    baseline_mel = np.zeros_like(mel)
    delta_mel = np.zeros_like(mel)

    # 発声区間のみを予測
    if np.any(voicing_mask):
        mfcc_voiced = mfcc[voicing_mask]
        mel_voiced = mel[voicing_mask]

        # Phase 1: ベースラインMelの予測
        mfcc_scaled = bl_scaler.transform(mfcc_voiced)
        bl_mel_voiced = bl_model.predict(mfcc_scaled)

        # ΔMelの算出
        delta_mel_voiced = mel_voiced - bl_mel_voiced

        # Phase 2: ΔMel → DYN値
        delta_scaled = dyn_scaler.transform(delta_mel_voiced)
        if poly_transformer is not None:
            delta_poly = poly_transformer.transform(delta_scaled)
        else:
            delta_poly = delta_scaled

        dyn_voiced = dyn_model.predict(delta_poly)
        
        # 出力スケーリングの適用 (64を基準に変動幅を圧縮/拡大)
        if hasattr(cfg, 'DYN_OUTPUT_SCALE'):
            dyn_voiced = 64 + (dyn_voiced - 64) * cfg.DYN_OUTPUT_SCALE

        # 結果を格納
        dyn_raw[voicing_mask] = dyn_voiced
        baseline_mel[voicing_mask] = bl_mel_voiced
        delta_mel[voicing_mask] = delta_mel_voiced

    # 無音区間のベースラインは実際の値に合わせる（描画用）
    baseline_mel[~voicing_mask] = mel[~voicing_mask]

    return dyn_raw, baseline_mel, delta_mel


def interpolate_note_gaps(dyn_raw, notes, total_frames, sr, hop_length):
    """
    ノート間の無音区間を補間処理する。

    - ノート有区間: 推定値をそのまま使用
    - ノート無区間: 直前のノート末尾のDYN値と次のノート先頭のDYN値を線形補間
    - 長い休符: DYN_DEFAULT (64) に自然に戻す

    Args:
        dyn_raw: np.ndarray (N,) - 生の推定DYN値
        notes: list of dict - ノート情報
        total_frames: int - 総フレーム数
        sr: int - サンプリングレート
        hop_length: int - ホップ長

    Returns:
        dyn_interpolated: np.ndarray (N,) - 補間済みDYN値
    """
    dyn_interpolated = dyn_raw.copy()

    if not notes:
        return dyn_interpolated

    # ノート区間をフレームインデックスに変換
    note_regions = []
    for note in notes:
        s_idx = librosa.time_to_frames(
            note["start_sec"], sr=sr, hop_length=hop_length
        )
        e_idx = librosa.time_to_frames(
            note["end_sec"], sr=sr, hop_length=hop_length
        )
        s_idx = max(0, s_idx)
        e_idx = min(total_frames, e_idx)
        note_regions.append((s_idx, e_idx))

    # 最初のノートより前の区間
    if note_regions[0][0] > 0:
        first_start = note_regions[0][0]
        first_dyn = dyn_raw[first_start]
        dyn_interpolated[:first_start] = np.linspace(
            cfg.DYN_DEFAULT, first_dyn, first_start
        )

    # ノート間の補間
    for i in range(len(note_regions) - 1):
        _, prev_end = note_regions[i]
        next_start, _ = note_regions[i + 1]

        if next_start <= prev_end:
            continue  # ノートが隣接または重複

        gap_frames = next_start - prev_end
        gap_sec = librosa.frames_to_time(
            gap_frames, sr=sr, hop_length=hop_length
        )

        # 直前のノート末尾のDYN値
        prev_dyn = dyn_raw[min(prev_end - 1, total_frames - 1)]
        # 次のノート先頭のDYN値
        next_dyn = dyn_raw[min(next_start, total_frames - 1)]

        if gap_sec >= cfg.REST_THRESHOLD_SEC:
            # 長い休符: 前半はデフォルトに戻し、後半はデフォルトから次のDYNへ
            mid = prev_end + gap_frames // 2
            dyn_interpolated[prev_end:mid] = np.linspace(
                prev_dyn, cfg.DYN_DEFAULT, mid - prev_end
            )
            dyn_interpolated[mid:next_start] = np.linspace(
                cfg.DYN_DEFAULT, next_dyn, next_start - mid
            )
        else:
            # 短い休符: 線形補間
            dyn_interpolated[prev_end:next_start] = np.linspace(
                prev_dyn, next_dyn, next_start - prev_end
            )

    # 最後のノートより後の区間
    if note_regions[-1][1] < total_frames:
        last_end = note_regions[-1][1]
        last_dyn = dyn_raw[min(last_end - 1, total_frames - 1)]
        remaining = total_frames - last_end
        dyn_interpolated[last_end:] = np.linspace(
            last_dyn, cfg.DYN_DEFAULT, remaining
        )

    return dyn_interpolated


def smooth_dyn(dyn_curve):
    """
    Savitzky-Golayフィルタによる平滑化を適用する。

    Args:
        dyn_curve: np.ndarray (N,) - DYNカーブ

    Returns:
        dyn_smoothed: np.ndarray (N,) - 平滑化済みDYNカーブ
    """
    window = cfg.SAVGOL_WINDOW
    polyorder = cfg.SAVGOL_POLYORDER

    # データ長がウィンドウより短い場合はスキップ
    if len(dyn_curve) < window:
        return dyn_curve

    dyn_smoothed = savgol_filter(dyn_curve, window, polyorder)
    return np.clip(dyn_smoothed, 0, 127)


def plot_results(time_axis, mel, baseline_mel, delta_mel, dyn_curve,
                 gt_dyn_curve, notes, output_path):
    """推定結果のグラフを描画・保存する"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # 1. RMS比較（Melの全バンド平均をRMS代わりに使用）
    ax = axes[0]
    actual_rms = np.mean(mel, axis=1)
    baseline_rms = np.mean(baseline_mel, axis=1)
    ax.plot(time_axis, actual_rms, label="実際のMel平均", alpha=0.7)
    ax.plot(time_axis, baseline_rms, label="ベースラインMel平均", alpha=0.7,
            linestyle="--")
    ax.fill_between(time_axis, actual_rms, baseline_rms, alpha=0.15,
                    color="orange", label="差分 (= DYN効果)")
    ax.set_ylabel("Mel Power (dB)")
    ax.set_title("Mel Power: Actual vs Predicted Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 推定DYN vs 正解DYN
    ax = axes[1]
    ax.plot(time_axis, dyn_curve, label="推定DYN", color="tab:blue")
    if gt_dyn_curve is not None:
        min_len = min(len(time_axis), len(gt_dyn_curve))
        ax.plot(time_axis[:min_len], gt_dyn_curve[:min_len],
                label="正解DYN", color="tab:red", alpha=0.7,
                linestyle="--")
    ax.axhline(y=64, color="gray", linestyle=":", alpha=0.5, label="DYN=64")
    ax.set_ylabel("DYN Value")
    ax.set_ylim(-5, 132)
    ax.set_title("Estimated DYN Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 誤差
    ax = axes[2]
    if gt_dyn_curve is not None:
        min_len = min(len(dyn_curve), len(gt_dyn_curve))
        error = dyn_curve[:min_len] - gt_dyn_curve[:min_len]
        ax.plot(time_axis[:min_len], error, color="tab:orange", alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        mae = np.mean(np.abs(error))
        ax.set_title(f"Estimation Error (MAE: {mae:.2f})")
    else:
        ax.text(0.5, 0.5, "正解データなし", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Estimation Error")
    ax.set_ylabel("Error (Est - GT)")
    ax.set_xlabel("Time (sec)")
    ax.grid(True, alpha=0.3)

    # 全てのグラフにノートの区切り線と歌詞を描画
    if notes:
        for ax in axes:
            for note in notes:
                s_sec = note["start_sec"]
                e_sec = note["end_sec"]
                lyric = note["lyric"]

                # ノートの開始・終了ライン
                ax.axvline(x=s_sec, color="green", linestyle=":", alpha=0.6)
                ax.axvline(x=e_sec, color="red", linestyle=":", alpha=0.6)

                # 歌詞のテキスト表示（Y軸の上端付近に配置）
                ymin, ymax = ax.get_ylim()
                text_y = ymax - (ymax - ymin) * 0.05
                ax.text(s_sec + (e_sec - s_sec) / 2, text_y, lyric,
                        ha='center', va='top', color='green', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.5,
                                  edgecolor='none', pad=1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  グラフ保存: {output_path}")


def write_dyn_to_vsqx(vsqx_path, output_path, dyn_values, sr, hop_length):
    """
    推定したDYNカーブをVSQXファイルに書き込む。
    predict_en.py の update_vsqx_xml() を簡略化したDYN専用版。
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
        #   読み込み: パート内相対Tick → _tick_to_second → 秒 (WAVの0秒 = 相対Tick 0)
        #   書き込み: 秒 → 相対Tick (逆変換)
        cc_strings = []
        last_val = -9999

        for frame_idx, val in enumerate(dyn_values):
            int_val = int(np.clip(val, 0, 127))

            # 間引き: 値が変わった時だけ書き込む
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
    print("  DYN Estimation from Audio (2-Phase Model)")
    print("=" * 60)

    # 設定確認
    print(f"\n[Config]")
    print(f"  ベースラインモデル: {cfg.BASELINE_MODEL_PATH}")
    print(f"  DYNスケールモデル: {cfg.DYN_MODEL_PATH}")
    print(f"  対象WAV: {cfg.TARGET_WAV}")
    print(f"  対象VSQX: {cfg.TARGET_VSQX}")
    print(f"  平滑化: Savitzky-Golay (window={cfg.SAVGOL_WINDOW}, "
          f"poly={cfg.SAVGOL_POLYORDER})")
    print(f"  休符閾値: {cfg.REST_THRESHOLD_SEC} sec")

    # モデルの読み込み
    if not os.path.exists(cfg.BASELINE_MODEL_PATH):
        print(f"\n[エラー] ベースラインモデルが見つかりません: "
              f"{cfg.BASELINE_MODEL_PATH}")
        print("  先に train_baseline.py を実行してください。")
        return

    if not os.path.exists(cfg.DYN_MODEL_PATH):
        print(f"\n[エラー] DYNスケールモデルが見つかりません: "
              f"{cfg.DYN_MODEL_PATH}")
        print("  先に train_baseline.py を実行してください。")
        return

    baseline_model_data = joblib.load(cfg.BASELINE_MODEL_PATH)
    dyn_model_data = joblib.load(cfg.DYN_MODEL_PATH)
    print(f"\n[Step 1] モデル読み込み完了")
    print(f"  ベースライン R²: {baseline_model_data['r_squared']:.4f}, "
          f"RMSE: {baseline_model_data['rmse']:.4f} dB")
    print(f"  DYNスケール R²: {dyn_model_data['r_squared']:.4f}, "
          f"MAE: {dyn_model_data['mae']:.4f}")

    # 対象ファイルの存在確認
    if not os.path.exists(cfg.TARGET_WAV):
        print(f"\n[エラー] WAVが見つかりません: {cfg.TARGET_WAV}")
        return
    if not os.path.exists(cfg.TARGET_VSQX):
        print(f"\n[エラー] VSQXが見つかりません: {cfg.TARGET_VSQX}")
        return

    # 特徴量の抽出
    print(f"\n[Step 2] 特徴量抽出中...")
    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    mfcc = extractor.get_features(cfg.TARGET_WAV, ["mfcc"])
    mel = extractor.get_features(cfg.TARGET_WAV, ["mel"])

    min_len = min(len(mfcc), len(mel))
    mfcc = mfcc[:min_len]
    mel = mel[:min_len]

    print(f"  フレーム数: {min_len}")

    # 正解DYNカーブとノート情報の取得
    print(f"\n[Step 3] VSQXから正解DYNとノート情報の読み込み...")
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    vsqx_data = reader.read(cfg.TARGET_VSQX)

    gt_dyn_curve = None
    voicing_mask = np.zeros(min_len, dtype=bool)

    if vsqx_data:
        if "DYN" in vsqx_data["dynamic_params"]:
            gt_dyn_curve = reader.interpolate_param_to_frames(
                vsqx_data["dynamic_params"]["DYN"],
                total_frames=min_len,
                default_value=64
            )
            print(f"  正解DYN範囲: [{gt_dyn_curve.min():.1f}, "
                  f"{gt_dyn_curve.max():.1f}]")
        else:
            print("  正解DYNが見つかりません（比較なしで出力します）")

        # 発声マスクの作成
        for note in vsqx_data["notes"]:
            s_idx = librosa.time_to_frames(
                note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
            )
            e_idx = librosa.time_to_frames(
                note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
            )
            voicing_mask[max(0, s_idx):min(min_len, e_idx)] = True
    else:
        print("  VSQXの読み込みに失敗しました")
        return

    # DYNの推定
    print(f"\n[Step 4] DYN推定中...")
    dyn_raw, baseline_mel, delta_mel = estimate_dyn(
        mfcc, mel, baseline_model_data, dyn_model_data, voicing_mask
    )

    # ノート間補間
    print(f"  ノート間補間を適用中...")
    dyn_interpolated = interpolate_note_gaps(
        dyn_raw, vsqx_data["notes"], min_len, cfg.SR, cfg.HOP_LENGTH
    )

    # 平滑化
    print(f"  Savitzky-Golayフィルタを適用中...")
    dyn_curve = smooth_dyn(dyn_interpolated)

    print(f"  推定DYN範囲: [{dyn_curve.min():.1f}, {dyn_curve.max():.1f}]")

    # 出力
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(cfg.TARGET_WAV))[0]
    time_axis = librosa.frames_to_time(
        np.arange(min_len), sr=cfg.SR, hop_length=cfg.HOP_LENGTH
    )

    # グラフの描画・保存
    print(f"\n[Step 5] 結果の出力...")
    plot_path = os.path.join(cfg.OUTPUT_DIR, f"{base_name}_dyn_estimation.png")

    notes = vsqx_data["notes"] if vsqx_data else []
    plot_results(time_axis, mel, baseline_mel, delta_mel, dyn_curve,
                 gt_dyn_curve, notes, plot_path)

    # VSQXへの書き込み
    vsqx_output_path = os.path.join(
        cfg.OUTPUT_DIR, f"{base_name}_estimated_DYN.vsqx"
    )
    write_dyn_to_vsqx(
        cfg.TARGET_VSQX, vsqx_output_path, dyn_curve, cfg.SR, cfg.HOP_LENGTH
    )

    # 精度の要約
    if gt_dyn_curve is not None:
        min_eval = min(len(dyn_curve), len(gt_dyn_curve))
        error = dyn_curve[:min_eval] - gt_dyn_curve[:min_eval]
        mae = np.mean(np.abs(error))

        # 発声区間のみの精度
        voiced_error = dyn_curve[voicing_mask] - gt_dyn_curve[voicing_mask]
        voiced_mae = np.mean(np.abs(voiced_error))

        print(f"\n  === 精度要約 ===")
        print(f"  MAE (全体): {mae:.2f}")
        print(f"  MAE (発声区間のみ): {voiced_mae:.2f}")
        print(f"  誤差標準偏差: {np.std(error):.2f}")
        print(f"  最大過大推定: {error.max():.2f}")
        print(f"  最大過小推定: {error.min():.2f}")

    print(f"\n{'='*60}")
    print(f"  推定完了！")
    print(f"  出力先: {os.path.abspath(cfg.OUTPUT_DIR)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
