import os
import sys
import re
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'sans-serif']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "05_DYN_Estimation"))

import config_direct as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader

RE_RESOLUTION = re.compile(r"<resolution>(\d+)</resolution>")
RE_TEMPO_V = re.compile(r"<tempo>\s*<t>0</t>\s*<v>(\d+)</v>\s*</tempo>", re.DOTALL)
RE_VS_PART = re.compile(r"<vsPart>.*?</vsPart>", re.DOTALL)
RE_SINGER = re.compile(r"(<singer>.*?</singer>)", re.DOTALL)
RE_CC_TAG = re.compile(r'<cc>.*?<v id="([A-Z]+)">.*?</cc>', re.DOTALL)
RE_NOTE = re.compile(r"<note>.*?</note>", re.DOTALL)

def estimate_dyn_direct(mel, model_data):
    model = model_data["model"]
    scaler = model_data["scaler"]
    mean_dyn = model_data.get("mean_dyn", 64.0)

    # 全フレームを一括で予測
    mel_scaled = scaler.transform(mel)
    dyn_pred = model.predict(mel_scaled)

    # 振幅倍率の適用 (平均値を中心にスケールを拡大)
    scale = cfg.DIRECT_OUTPUT_SCALE
    dyn_pred = mean_dyn + (dyn_pred - mean_dyn) * scale

    # 範囲制限
    dyn_pred = np.clip(dyn_pred, 0, 127)
    
    # 音量ガード：無音・無声子音区間をデフォルト値にリセット
    mel_power = np.mean(mel, axis=1)
    silence_mask = mel_power < cfg.MEL_POWER_THRESHOLD_DB
    dyn_pred[silence_mask] = cfg.DYN_DEFAULT

    return dyn_pred

def interpolate_note_gaps(dyn_curve, notes, total_frames):
    dyn_interpolated = dyn_curve.copy()
    if not notes:
        return dyn_interpolated

    note_regions = []
    for note in notes:
        s_idx = librosa.time_to_frames(note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
        e_idx = librosa.time_to_frames(note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
        note_regions.append((max(0, s_idx), min(total_frames, e_idx)))

    if note_regions[0][0] > 0:
        dyn_interpolated[:note_regions[0][0]] = cfg.DYN_DEFAULT

    for i in range(len(note_regions) - 1):
        _, prev_end = note_regions[i]
        next_start, _ = note_regions[i + 1]
        if next_start <= prev_end: continue

        gap_frames = next_start - prev_end
        gap_sec = librosa.frames_to_time(gap_frames, sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
        prev_dyn = dyn_curve[min(prev_end - 1, total_frames - 1)]
        next_dyn = dyn_curve[min(next_start, total_frames - 1)]

        if gap_sec >= cfg.REST_THRESHOLD_SEC:
            mid = prev_end + gap_frames // 2
            dyn_interpolated[prev_end:mid] = np.linspace(prev_dyn, cfg.DYN_DEFAULT, mid - prev_end)
            dyn_interpolated[mid:next_start] = np.linspace(cfg.DYN_DEFAULT, next_dyn, next_start - mid)
        else:
            dyn_interpolated[prev_end:next_start] = np.linspace(prev_dyn, next_dyn, gap_frames)

    last_end = note_regions[-1][1]
    if last_end < total_frames:
        last_dyn = dyn_curve[min(last_end - 1, total_frames - 1)]
        dyn_interpolated[last_end:] = np.linspace(last_dyn, cfg.DYN_DEFAULT, total_frames - last_end)

    return dyn_interpolated

def smooth_dyn(dyn_curve):
    if len(dyn_curve) < cfg.SAVGOL_WINDOW: return dyn_curve
    return np.clip(savgol_filter(dyn_curve, cfg.SAVGOL_WINDOW, cfg.SAVGOL_POLYORDER), 0, 127)

def plot_results(time_axis, mel, dyn_curve, gt_dyn_curve, notes, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    mel_mean = np.mean(mel, axis=1)
    axes[0].plot(time_axis, mel_mean, label="Mel Power", alpha=0.7)
    axes[0].set_title("Input: Mel Spectrogram Power")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 無声区間（ノート間）の計算とグレー背景の描画
    unvoiced_regions = []
    last_sec = 0.0
    for note in notes:
        if note["start_sec"] > last_sec:
            unvoiced_regions.append((last_sec, note["start_sec"]))
        last_sec = note["end_sec"]
    if len(time_axis) > 0 and time_axis[-1] > last_sec:
        unvoiced_regions.append((last_sec, time_axis[-1]))

    for start_sec, end_sec in unvoiced_regions:
        for ax in axes:
            ax.axvspan(start_sec, end_sec, color="gray", alpha=0.15)

    # 歌詞の描画
    for note in notes:
        mid = (note["start_sec"] + note["end_sec"]) / 2
        axes[1].text(mid, 130, note.get("lyric", ""), ha='center', va='top', fontsize=8, color='darkgreen')

    axes[1].plot(time_axis, dyn_curve, label="推定DYN", color="tab:blue")
    if gt_dyn_curve is not None:
        min_len = min(len(time_axis), len(gt_dyn_curve))
        axes[1].plot(time_axis[:min_len], gt_dyn_curve[:min_len], label="正解DYN", color="tab:red", alpha=0.7, linestyle="--")
    axes[1].axhline(y=64, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_ylim(-5, 132)
    axes[1].set_title("Estimated DYN Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if gt_dyn_curve is not None:
        error = dyn_curve[:min_len] - gt_dyn_curve[:min_len]
        axes[2].plot(time_axis[:min_len], error, color="tab:orange", alpha=0.7)
        
        # 有声区間（ノート内）のみの誤差を計算
        voiced_errors = []
        for note in notes:
            s_idx = librosa.time_to_frames(note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
            e_idx = librosa.time_to_frames(note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
            s_idx = max(0, min(s_idx, min_len))
            e_idx = max(0, min(e_idx, min_len))
            if s_idx < e_idx:
                voiced_errors.extend(error[s_idx:e_idx])
                
        valid_mae = np.mean(np.abs(voiced_errors)) if voiced_errors else 0.0
        axes[2].set_title(f"Estimation Error (Voiced MAE: {valid_mae:.2f})")
    axes[2].axhline(y=0, color="gray", linestyle=":")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def write_dyn_to_vsqx(vsqx_path, output_path, dyn_values, sr, hop_length):
    try:
        with open(vsqx_path, "r", encoding="utf-8") as f:
            content = f.read()

        res_match = RE_RESOLUTION.search(content)
        resolution = int(res_match.group(1)) if res_match else 480
        
        bpm_match = RE_TEMPO_V.search(content)
        bpm = int(bpm_match.group(1)) / 100.0 if bpm_match else 120.0
        part_match = RE_VS_PART.search(content)
        if not part_match: return
        part_content = part_match.group(0)

        placeholders = []
        def _mask_note(m):
            placeholders.append(m.group(0))
            return f"__NOTE_PLACEHOLDER_{len(placeholders)-1}__"
        masked_part = RE_NOTE.sub(_mask_note, part_content)
        cleaned_part = RE_CC_TAG.sub(lambda m: "" if m.group(1) == "D" else m.group(0), masked_part)

        for i, note_str in enumerate(placeholders):
            cleaned_part = cleaned_part.replace(f"__NOTE_PLACEHOLDER_{i}__", note_str)

        cc_strings = []
        last_val = -9999
        for frame_idx, val in enumerate(dyn_values):
            int_val = int(np.clip(val, 0, 127))
            if int_val == last_val: continue
            
            time_sec = librosa.frames_to_time(frame_idx, sr=sr, hop_length=hop_length)
            rel_tick = int(time_sec * bpm * resolution / 60.0)
            if rel_tick >= 0:
                cc_strings.append(f'\t\t\t<cc>\n\t\t\t\t<t>{rel_tick}</t>\n\t\t\t\t<v id="D">{int_val}</v>\n\t\t\t</cc>')
                last_val = int_val

        if cc_strings:
            augmented_part = RE_SINGER.sub(rf"\1\n{chr(10).join(cc_strings)}", cleaned_part, count=1)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content.replace(part_content, augmented_part, 1))
    except Exception as e:
        print(f"  [エラー] VSQX書き込み失敗: {e}")

def main():
    print("=" * 60)
    print("  DYN Direct Estimation (Ridge) - Prediction")
    print("=" * 60)

    if not os.path.exists(cfg.MODEL_PATH):
        print("[エラー] モデルが見つかりません。train_direct.pyを実行してください。")
        return

    model_data = joblib.load(cfg.MODEL_PATH)
    wav_path = cfg.TARGET_WAV
    vsqx_path = cfg.TARGET_VSQX
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    mel = extractor.get_features(wav_path, ["mel"])
    total_frames = len(mel)
    
    vsqx_data = reader.read(vsqx_path)
    notes = vsqx_data.get("notes", []) if vsqx_data else []
    
    gt_dyn_points = vsqx_data["dynamic_params"].get("DYN", []) if vsqx_data else []
    gt_dyn_curve = reader.interpolate_param_to_frames(gt_dyn_points, total_frames, cfg.DYN_DEFAULT) if gt_dyn_points else None

    print("\n[Step 1] DYN推定...")
    dyn_raw = estimate_dyn_direct(mel, model_data)
    
    print("[Step 2] 補間と平滑化...")
    dyn_interp = interpolate_note_gaps(dyn_raw, notes, total_frames)
    dyn_smooth = smooth_dyn(dyn_interp)

    print("\n[Step 3] 出力...")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    time_axis = librosa.frames_to_time(np.arange(total_frames), sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    plot_results(time_axis, mel, dyn_smooth, gt_dyn_curve, notes, os.path.join(cfg.OUTPUT_DIR, f"{base_name}_direct_dyn.png"))
    write_dyn_to_vsqx(vsqx_path, os.path.join(cfg.OUTPUT_DIR, f"{base_name}_direct_dyn.vsqx"), dyn_smooth, cfg.SR, cfg.HOP_LENGTH)
    print("  完了！")

if __name__ == "__main__":
    main()
