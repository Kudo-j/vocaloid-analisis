"""
eval_dyn.py
===========
学習済みの2段階モデルを使い、複数の楽曲に対してDYNを推定し、
正解VSQXのDYNカーブとの誤差を一括で計算してExcelに出力する。

使い方:
  1. train_baseline.py を先に実行してモデルを作成
  2. config_dyn.py の EVAL_WAV_DIR, EVAL_VSQX_DIR を設定
  3. このファイルをVSCode上で実行 (F5 or ▶)
  4. output/dyn_eval_results.xlsx が生成される
"""

import os
import sys
import numpy as np
import joblib
import librosa
import pandas as pd
from scipy.signal import savgol_filter
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

# 既存モジュールへのパス追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "04_Parameter_Study", "src"))

import config_dyn as cfg
from feature_extractor import AudioFeatureExtractor
from vsqx_reader import VSQXReader
from pair_finder import find_pairs


def estimate_dyn_for_eval(mfcc, mel, baseline_model_data, dyn_model_data,
                          voicing_mask, notes, total_frames):
    """
    MFCCとMelスペクトログラムからDYNカーブを推定する（評価用）。

    Returns:
        dyn_curve: np.ndarray (N,) - 推定DYN (0-127)
    """
    bl_model = baseline_model_data["model"]
    bl_scaler = baseline_model_data["scaler"]

    dyn_model = dyn_model_data["model"]
    dyn_scaler = dyn_model_data["scaler"]
    poly_transformer = dyn_model_data.get("poly_transformer", None)

    N = len(mfcc)
    dyn_raw = np.full(N, float(cfg.DYN_DEFAULT))

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
            
        # 出力スケーリングの適用
        if hasattr(cfg, 'DYN_OUTPUT_SCALE'):
            dyn_voiced = 64 + (dyn_voiced - 64) * cfg.DYN_OUTPUT_SCALE
        
        dyn_raw[voicing_mask] = dyn_voiced

    # ノート間補間
    from predict_dyn import interpolate_note_gaps
    dyn_interpolated = interpolate_note_gaps(
        dyn_raw, notes, total_frames, cfg.SR, cfg.HOP_LENGTH
    )

    # 平滑化
    if len(dyn_interpolated) >= cfg.SAVGOL_WINDOW:
        dyn_smoothed = savgol_filter(
            dyn_interpolated, cfg.SAVGOL_WINDOW, cfg.SAVGOL_POLYORDER
        )
    else:
        dyn_smoothed = dyn_interpolated

    # 0-127 にクリップ
    dyn_curve = np.clip(dyn_smoothed, 0, 127)

    return dyn_curve


def evaluate_single(base_name, wav_path, vsqx_path,
                    extractor, reader, baseline_model_data, dyn_model_data):
    """
    1曲分のDYN推定と精度計算を実行する。

    Returns:
        dict: 精度メトリクスの辞書。正解DYNがない場合はNone。
    """
    # VSQXの読み込み
    vsqx_data = reader.read(vsqx_path)
    if vsqx_data is None:
        return None

    # 正解DYNの取得
    if "DYN" not in vsqx_data["dynamic_params"]:
        return None

    if not vsqx_data["notes"]:
        return None

    # 特徴量の抽出
    try:
        mfcc = extractor.get_features(wav_path, ["mfcc"])
        mel = extractor.get_features(wav_path, ["mel"])
    except Exception as e:
        print(f"    [警告] 特徴量抽出失敗: {e}")
        return None

    min_len = min(len(mfcc), len(mel))
    mfcc = mfcc[:min_len]
    mel = mel[:min_len]

    # 正解DYNカーブの取得
    gt_dyn_curve = reader.interpolate_param_to_frames(
        vsqx_data["dynamic_params"]["DYN"],
        total_frames=min_len,
        default_value=64
    )

    # 発声マスクの作成
    voicing_mask = np.zeros(min_len, dtype=bool)
    for note in vsqx_data["notes"]:
        s_idx = librosa.time_to_frames(
            note["start_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
        )
        e_idx = librosa.time_to_frames(
            note["end_sec"], sr=cfg.SR, hop_length=cfg.HOP_LENGTH
        )
        voicing_mask[max(0, s_idx):min(min_len, e_idx)] = True

    voiced_frames = np.sum(voicing_mask)
    if voiced_frames == 0:
        return None

    # DYNの推定
    dyn_curve = estimate_dyn_for_eval(
        mfcc, mel, baseline_model_data, dyn_model_data,
        voicing_mask, vsqx_data["notes"], min_len
    )

    # === 精度計算（発声区間のみ） ===
    est_voiced = dyn_curve[voicing_mask]
    gt_voiced = gt_dyn_curve[voicing_mask]
    error_voiced = est_voiced - gt_voiced

    mae = np.mean(np.abs(error_voiced))
    rmse = np.sqrt(np.mean(error_voiced ** 2))
    mean_error = np.mean(error_voiced)  # バイアス（正：過大推定傾向）
    std_error = np.std(error_voiced)
    max_overestimate = np.max(error_voiced)
    max_underestimate = np.min(error_voiced)

    # 相関係数
    if np.std(gt_voiced) > 1e-8 and np.std(est_voiced) > 1e-8:
        correlation = np.corrcoef(gt_voiced, est_voiced)[0, 1]
    else:
        correlation = 0.0

    # 正解DYNの変動幅 (どの程度DYNが動いているか)
    gt_range = np.max(gt_voiced) - np.min(gt_voiced)

    return {
        "Song Name": base_name,
        "MAE": mae,
        "RMSE": rmse,
        "Mean Error": mean_error,
        "Std Error": std_error,
        "Correlation": correlation,
        "Max Overest.": max_overestimate,
        "Max Underest.": max_underestimate,
        "GT DYN Range": gt_range,
        "Voiced Frames": int(voiced_frames),
    }


def save_results_to_excel(results, output_path):
    """
    結果をExcelファイルに保存する（eval_Test_enco.py準拠のフォーマット）。
    """
    df = pd.DataFrame(results)

    # 列の並び順を整理
    col_order = [
        "Song Name", "MAE", "RMSE", "Mean Error", "Std Error",
        "Correlation", "Max Overest.", "Max Underest.",
        "GT DYN Range", "Voiced Frames",
    ]
    final_cols = [c for c in col_order if c in df.columns]
    df = df[final_cols]

    # MAEが悪い順にソート (初期表示用)
    df_sorted = df.sort_values(by="MAE", ascending=False).reset_index(drop=True)

    # --- 平均行の作成 ---
    numeric_cols = [c for c in final_cols if c != "Song Name"]
    avg_row = {"Song Name": "AVERAGE"}
    for col in numeric_cols:
        avg_row[col] = df_sorted[col].mean()
    df_avg = pd.concat([df_sorted, pd.DataFrame([avg_row])], ignore_index=True)

    try:
        # pandasでExcelに書き出す
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_avg.to_excel(output_path, index=False)

        # --- openpyxl でテーブル機能と太字装飾を適用 ---
        wb = load_workbook(output_path)
        ws = wb.active

        num_cols = len(final_cols)
        num_data_rows = len(df_sorted)  # 平均行を含まないデータ行数

        # テーブル範囲: ヘッダー(1行目) + データ行 (平均行は含めない)
        table_ref = f"A1:{get_column_letter(num_cols)}{num_data_rows + 1}"
        table = Table(displayName="DYNEvalResults", ref=table_ref)
        style = TableStyleInfo(
            name="TableStyleMedium9",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        table.tableStyleInfo = style
        ws.add_table(table)

        # 平均行 (テーブル外の最下行) を太字にする
        avg_row_idx = num_data_rows + 2  # ヘッダー(1) + データ行 + 1
        bold_font = Font(bold=True)
        for col_idx in range(1, num_cols + 1):
            cell = ws.cell(row=avg_row_idx, column=col_idx)
            cell.font = bold_font

        # 列幅を自動調整 (簡易)
        for col_idx in range(1, num_cols + 1):
            col_letter = get_column_letter(col_idx)
            max_len = len(str(ws.cell(row=1, column=col_idx).value or ""))
            for row_idx in range(2, avg_row_idx + 1):
                cell_val = ws.cell(row=row_idx, column=col_idx).value
                if cell_val is not None:
                    max_len = max(max_len, len(str(cell_val)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 30)

        wb.save(output_path)

        print(f"\n  Excel保存完了: {output_path}")
        return df_sorted

    except Exception as e:
        print(f"\n  [警告] Excel保存失敗: {e}")
        csv_path = output_path.replace(".xlsx", ".csv")
        df_avg.to_csv(csv_path, index=False)
        print(f"  CSVとして保存: {csv_path}")
        return df_sorted


def main():
    print("=" * 60)
    print("  DYN Estimation - Batch Evaluation (2-Phase Model)")
    print("=" * 60)

    # 設定確認
    print(f"\n[Config]")
    print(f"  ベースラインモデル: {cfg.BASELINE_MODEL_PATH}")
    print(f"  DYNスケールモデル: {cfg.DYN_MODEL_PATH}")
    print(f"  テスト用WAVフォルダ: {cfg.EVAL_WAV_DIR}")
    print(f"  テスト用VSQXフォルダ: {cfg.EVAL_VSQX_DIR}")
    print(f"  出力先: {cfg.EVAL_OUTPUT_EXCEL}")

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

    # ペアの検索
    print(f"\n[Step 2] WAV-VSQXペアの検索...")
    if not os.path.isdir(cfg.EVAL_WAV_DIR):
        print(f"  [エラー] WAVフォルダが存在しません: {cfg.EVAL_WAV_DIR}")
        return
    if not os.path.isdir(cfg.EVAL_VSQX_DIR):
        print(f"  [エラー] VSQXフォルダが存在しません: {cfg.EVAL_VSQX_DIR}")
        return

    pairs = find_pairs(cfg.EVAL_WAV_DIR, cfg.EVAL_VSQX_DIR)

    if not pairs:
        print("\n[エラー] ペアが1つも見つかりませんでした。")
        return

    # 評価実行
    print(f"\n[Step 3] 一括評価開始 ({len(pairs)} 曲)...")
    extractor = AudioFeatureExtractor(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)
    reader = VSQXReader(sr=cfg.SR, hop_length=cfg.HOP_LENGTH)

    results = []
    skipped = 0

    for i, (base_name, wav_path, vsqx_path) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {base_name}...", end=" ")

        result = evaluate_single(
            base_name, wav_path, vsqx_path,
            extractor, reader, baseline_model_data, dyn_model_data
        )

        if result is not None:
            results.append(result)
            print(f"MAE: {result['MAE']:.2f}, Corr: {result['Correlation']:.3f}")
        else:
            skipped += 1
            print("スキップ (正解DYNなし or エラー)")

    # 結果の出力
    print(f"\n[Step 4] 結果の保存...")
    print(f"  評価完了: {len(results)} 曲")
    if skipped > 0:
        print(f"  スキップ: {skipped} 曲")

    if not results:
        print("\n[警告] 有効な結果が得られませんでした。")
        return

    df = save_results_to_excel(results, cfg.EVAL_OUTPUT_EXCEL)

    # サマリ表示
    print(f"\n{'='*60}")
    print(f"  === 全体サマリ ===")
    print(f"  平均MAE:  {df['MAE'].mean():.2f}")
    print(f"  平均RMSE: {df['RMSE'].mean():.2f}")
    print(f"  平均相関: {df['Correlation'].mean():.3f}")
    print(f"\n  Top 5 worst (MAE):")
    top5 = df[["Song Name", "MAE", "RMSE", "Correlation"]].head(5)
    print(top5.to_string(index=False))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
