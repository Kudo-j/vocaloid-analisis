"""
eval_diff.py
============
時間差分モデルによるDYN推定 — 一括評価スクリプト

テストデータフォルダ内の全ペアに対して推定を実行し、
精度メトリクス (MAE, RMSE, 相関係数) を算出してExcelに出力する。

使い方:
  1. train_diff.py を先に実行してモデルを作成
  2. config_diff.py の EVAL_WAV_DIR, EVAL_VSQX_DIR を設定
  3. このファイルを実行
  4. output/ に評価結果のExcelとグラフが生成される
"""

import os
import sys
import numpy as np
import joblib
import librosa
import matplotlib.pyplot as plt
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

# predict_diff.py の関数を再利用
from predict_diff import (
    estimate_dyn_diff,
    interpolate_note_gaps,
    smooth_dyn,
    plot_results,
)


def main():
    print("=" * 60)
    print("  DYN Estimation Evaluation (Time-Diff Model)")
    print("=" * 60)

    # モデルの読み込み
    if not os.path.exists(cfg.MODEL_PATH):
        print(f"\n[エラー] モデルファイルが見つかりません: {cfg.MODEL_PATH}")
        print("  先に train_diff.py を実行してください。")
        return

    print(f"\n[Step 1] モデルの読み込み...")
    model_data = joblib.load(cfg.MODEL_PATH)
    print(f"  R²={model_data['r_squared']:.4f}, "
          f"RMSE={model_data['rmse']:.4f}")

    # テストデータのペア検索
    print(f"\n[Step 2] テストデータの検索...")
    print(f"  WAVフォルダ: {cfg.EVAL_WAV_DIR}")
    print(f"  VSQXフォルダ: {cfg.EVAL_VSQX_DIR}")

    if not os.path.isdir(cfg.EVAL_WAV_DIR):
        print(f"\n[エラー] WAVフォルダが存在しません: {cfg.EVAL_WAV_DIR}")
        return
    if not os.path.isdir(cfg.EVAL_VSQX_DIR):
        print(f"\n[エラー] VSQXフォルダが存在しません: {cfg.EVAL_VSQX_DIR}")
        return

    pairs = find_pairs(cfg.EVAL_WAV_DIR, cfg.EVAL_VSQX_DIR)
    if not pairs:
        print("\n[エラー] ペアが1つも見つかりませんでした。")
        return

    # 評価の実行
    print(f"\n[Step 3] 評価の実行...")
    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    results = []

    for i, (base_name, wav_path, vsqx_path) in enumerate(pairs):
        print(f"\n  [{i+1}/{len(pairs)}] {base_name}")

        try:
            # 特徴量抽出
            mel = extractor.get_features(wav_path, ["mel"])
            total_frames = len(mel)

            vsqx_data = reader.read(vsqx_path)
            if vsqx_data is None:
                print(f"    → VSQX読み込み失敗。スキップ")
                continue

            notes = vsqx_data.get("notes", [])
            if not notes:
                print(f"    → ノートなし。スキップ")
                continue

            # 正解DYNの取得
            gt_dyn_points = vsqx_data["dynamic_params"].get("DYN", [])
            if not gt_dyn_points:
                print(f"    → 正解DYNなし。スキップ")
                continue

            gt_dyn_curve = reader.interpolate_param_to_frames(
                gt_dyn_points, total_frames, default_value=cfg.DYN_DEFAULT
            )

            # DYN推定
            dyn_raw, delta_dyn_raw = estimate_dyn_diff(
                mel, model_data, notes, total_frames
            )
            dyn_interpolated = interpolate_note_gaps(dyn_raw, notes, total_frames)
            dyn_smooth = smooth_dyn(dyn_interpolated)

            # 精度計算 (発声区間のみ)
            voicing_mask = np.zeros(total_frames, dtype=bool)
            for note in notes:
                s_idx = librosa.time_to_frames(
                    note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
                )
                e_idx = librosa.time_to_frames(
                    note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
                )
                s_idx = max(0, s_idx)
                e_idx = min(total_frames, e_idx)
                voicing_mask[s_idx:e_idx] = True

            min_len = min(len(dyn_smooth), len(gt_dyn_curve))
            mask = voicing_mask[:min_len]

            if mask.sum() == 0:
                print(f"    → 発声区間が0フレーム。スキップ")
                continue

            est_voiced = dyn_smooth[:min_len][mask]
            gt_voiced = gt_dyn_curve[:min_len][mask]

            error = est_voiced - gt_voiced
            mae = np.mean(np.abs(error))
            rmse = np.sqrt(np.mean(error ** 2))

            # 相関係数
            if np.std(gt_voiced) > 0 and np.std(est_voiced) > 0:
                corr = np.corrcoef(est_voiced, gt_voiced)[0, 1]
            else:
                corr = 0.0

            results.append({
                "name": base_name,
                "frames": int(mask.sum()),
                "mae": mae,
                "rmse": rmse,
                "corr": corr,
                "gt_mean": gt_voiced.mean(),
                "gt_std": gt_voiced.std(),
                "est_mean": est_voiced.mean(),
                "est_std": est_voiced.std(),
            })

            print(f"    → MAE={mae:.2f}, RMSE={rmse:.2f}, Corr={corr:.3f}")

            # 個別グラフの保存
            time_axis = librosa.frames_to_time(
                np.arange(total_frames), sr=cfg.SR, hop_length=cfg.HOP_LENGTH
            )
            plot_path = os.path.join(cfg.OUTPUT_DIR, f"eval_{base_name}.png")
            plot_results(time_axis, mel, delta_dyn_raw, dyn_smooth,
                         gt_dyn_curve, notes, plot_path)

        except Exception as e:
            print(f"    → エラー: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 集計結果の出力
    if not results:
        print("\n[エラー] 有効な評価結果が0件です。")
        return

    print(f"\n{'=' * 60}")
    print(f"  評価結果サマリ ({len(results)} ファイル)")
    print(f"{'=' * 60}")

    all_mae = [r["mae"] for r in results]
    all_rmse = [r["rmse"] for r in results]
    all_corr = [r["corr"] for r in results]

    print(f"  MAE  — 平均: {np.mean(all_mae):.2f}, "
          f"中央: {np.median(all_mae):.2f}, "
          f"最良: {np.min(all_mae):.2f}, 最悪: {np.max(all_mae):.2f}")
    print(f"  RMSE — 平均: {np.mean(all_rmse):.2f}, "
          f"中央: {np.median(all_rmse):.2f}")
    print(f"  相関 — 平均: {np.mean(all_corr):.3f}, "
          f"中央: {np.median(all_corr):.3f}")

    # Excel出力
    try:
        import openpyxl
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "DYN Diff Estimation"

        # ヘッダ
        headers = ["ファイル名", "フレーム数", "MAE", "RMSE", "相関係数",
                    "正解 平均", "正解 標準偏差", "推定 平均", "推定 標準偏差"]
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4",
                                  fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)

        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        # データ行
        for row_idx, r in enumerate(results, 2):
            ws.cell(row=row_idx, column=1, value=r["name"])
            ws.cell(row=row_idx, column=2, value=r["frames"])
            ws.cell(row=row_idx, column=3, value=round(r["mae"], 2))
            ws.cell(row=row_idx, column=4, value=round(r["rmse"], 2))
            ws.cell(row=row_idx, column=5, value=round(r["corr"], 3))
            ws.cell(row=row_idx, column=6, value=round(r["gt_mean"], 2))
            ws.cell(row=row_idx, column=7, value=round(r["gt_std"], 2))
            ws.cell(row=row_idx, column=8, value=round(r["est_mean"], 2))
            ws.cell(row=row_idx, column=9, value=round(r["est_std"], 2))

        # サマリ行
        summary_row = len(results) + 3
        ws.cell(row=summary_row, column=1, value="=== 平均 ===")
        ws.cell(row=summary_row, column=1).font = Font(bold=True)
        ws.cell(row=summary_row, column=3, value=round(np.mean(all_mae), 2))
        ws.cell(row=summary_row, column=4, value=round(np.mean(all_rmse), 2))
        ws.cell(row=summary_row, column=5, value=round(np.mean(all_corr), 3))

        # 列幅の自動調整
        for col in range(1, len(headers) + 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 16

        wb.save(cfg.EVAL_OUTPUT_EXCEL)
        print(f"\n  Excel保存: {cfg.EVAL_OUTPUT_EXCEL}")

    except ImportError:
        print("\n  [情報] openpyxl がインストールされていないため、Excel出力をスキップしました。")
        print("  pip install openpyxl でインストール可能です。")

    print(f"\n{'=' * 60}")
    print(f"  評価完了!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
