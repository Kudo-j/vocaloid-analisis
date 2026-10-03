import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 日本語フォントの設定（文字化け対策）
# Windows環境で一般的なメイリオ、またはMSゴシックを指定
mpl.rcParams["font.family"] = ["Meiryo", "MS Gothic", "sans-serif"]
from openpyxl import Workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font, PatternFill, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

# パス解決 (呼び出し元の位置に関わらず losses.py をインポート可能にする)
util_dir = os.path.dirname(os.path.abspath(__file__))
if util_dir not in sys.path:
    sys.path.append(util_dir)
from losses import AdvancedLoss


def denormalize_array(arr, pname):
    """
    正規化された値(-1.0 ~ 1.0)を、VSQX用の元のスケール(0~127など)に戻す。
    """
    if pname == "PIT":
        # PITは -8192 ~ 8191 (中心0)
        return arr * 8192.0
    elif pname == "PBS":
        # PBSは 0 ~ 24
        return (arr + 1.0) / 2.0 * 24.0
    elif pname in ["fallPort", "risePort"]:
        # 0.0 ~ 1.0 の場合は 0~127にマッピングされていることが多い
        return np.clip(arr * 127.0, 0, 127)
    elif pname == "vibType":
        return arr
    else:
        # DYN, BRI, CLE 等の標準CC (0~127)
        val = (arr + 1.0) / 2.0 * 127.0
        return np.clip(val, 0, 127)


class Evaluator:
    """
    あらゆるモデル（線形、DNN）から呼び出せる共通評価・可視化クラス。
    モデルの構造や推論の仕組みには一切依存せず、純粋な「実測値と予測値のNumPy配列」を受け取って評価する。
    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.loss_fn = AdvancedLoss().to(self.device)
        self.results = []
        self.all_y_true = []
        self.all_y_pred = []

    def add_song_result(
        self, song_name, y_true, y_pred, param_names, voicing_mask=None
    ):
        """
        1曲分の推論結果を追加し、内部で誤差を計算・蓄積する。
        イントロの無音部分は削除し、誤差計算は有声部分のみで行う。
            song_name (str): 曲名
            y_true (np.ndarray): 実測値 (フレーム数, 次元数)
            y_pred (np.ndarray): 予測値 (フレーム数, 次元数)
            param_names (list): 出力次元の名前リスト (例: ["DYN", "PIT"])
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return

        # イントロの無音（最初の有声フレームより前）を削除
        if voicing_mask is not None:
            voiced_indices = np.where(voicing_mask > 0.5)[0]
            if len(voiced_indices) > 0:
                start_idx = voiced_indices[0]
                y_true = y_true[start_idx:]
                y_pred = y_pred[start_idx:]
                voicing_mask = voicing_mask[start_idx:]
            else:
                return  # 有声区間がない場合
        else:
            voicing_mask = np.ones(y_true.shape[0], dtype=np.float32)

        # PyTorchの計算グラフを利用するため、NumPy配列をテンソルに変換
        t_y_true = (
            torch.tensor(y_true, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        t_y_pred = (
            torch.tensor(y_pred, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        t_v_mask = (
            torch.tensor(voicing_mask, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(self.device)
        )

        lengths = torch.tensor([t_y_true.size(1)], dtype=torch.long).to(self.device)

        with torch.no_grad():
            # MSE と各サブメトリクスを共通関数で計算 (voicing_maskを渡して無音を無視)
            total_loss, metrics = self.loss_fn(
                t_y_pred, t_y_true, lengths, voicing_mask=t_v_mask
            )

            # 各次元ごとのMSE (無音区間をマスク)
            squared_diff = (t_y_pred - t_y_true) ** 2
            masked_sq_diff = squared_diff * t_v_mask
            param_mse = (
                (masked_sq_diff.sum(dim=(0, 1)) / (t_v_mask.sum() + 1e-7)).cpu().numpy()
            )

        res_dict = {
            "Song Name": song_name,
            "Total Loss": total_loss.item(),
            "MSE Loss": metrics.get("mse", 0.0),
            "Diff Loss": metrics.get("diff", 0.0),
            "Valid Frames": int(voicing_mask.sum()),
        }

        for i, pname in enumerate(param_names):
            if i < len(param_mse):
                res_dict[f"MSE_{pname}"] = param_mse[i]

        self.results.append(res_dict)

        # 散布図用に「有声フレーム」のみを蓄積
        valid_idx = voicing_mask > 0.5
        self.all_y_true.append(y_true[valid_idx])
        self.all_y_pred.append(y_pred[valid_idx])

    def _shade_unvoiced(self, ax, time_axis, voicing_mask_1d):
        """無声区間をグラフ背景に灰色で塗る関数"""
        in_unvoiced = False
        start_t = 0.0
        for i, v in enumerate(voicing_mask_1d):
            if v < 0.5 and not in_unvoiced:
                in_unvoiced = True
                start_t = time_axis[i]
            elif v >= 0.5 and in_unvoiced:
                in_unvoiced = False
                ax.axvspan(start_t, time_axis[i], alpha=0.18, color="#9E9E9E")
        if in_unvoiced:
            ax.axvspan(start_t, time_axis[-1], alpha=0.18, color="#9E9E9E")

    def plot_waveform(
        self,
        song_name,
        y_true,
        y_pred,
        save_path,
        param_names=None,
        voicing_mask=None,
        extra_features=None,
        sr=44100,
        shift_length=512,
        remove_intro_silence=True,
    ):
        """
        時系列波形をプロットして保存する。
        無音区間をグレー背景にし、時間軸を秒単位にする。
        """
        # イントロの無音削除
        if remove_intro_silence and voicing_mask is not None:
            voiced_indices = np.where(voicing_mask > 0.5)[0]
            if len(voiced_indices) > 0:
                start_idx = voiced_indices[0]
                y_true = y_true[start_idx:]
                y_pred = y_pred[start_idx:]
                voicing_mask = voicing_mask[start_idx:]
                if extra_features is not None:
                    for k in extra_features.keys():
                        extra_features[k] = extra_features[k][start_idx:]
        elif voicing_mask is None:
            voicing_mask = np.ones(y_true.shape[0], dtype=np.float32)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        num_dims = y_true.shape[1]
        num_extra = len(extra_features) if extra_features else 0
        num_plots = num_dims + num_extra

        # 時間軸 (秒) の計算
        # librosa.frames_to_time を使わず直接計算してlibrosa依存を避ける
        time_axis = np.arange(y_true.shape[0]) * shift_length / sr

        fig, axes = plt.subplots(
            num_plots, 1, figsize=(12, 3.5 * num_plots), sharex=True
        )
        if num_plots == 1:
            axes = [axes]

        for i in range(num_dims):
            ax = axes[i]
            pname = (
                param_names[i] if param_names and i < len(param_names) else f"Dim_{i}"
            )

            # 逆正規化
            y_t_denorm = denormalize_array(y_true[:, i], pname)
            y_p_denorm = denormalize_array(y_pred[:, i], pname)

            # 有声区間のみでMAEの計算
            voiced_idx = voicing_mask > 0.5
            if np.any(voiced_idx):
                mae = np.mean(np.abs(y_t_denorm[voiced_idx] - y_p_denorm[voiced_idx]))
            else:
                mae = 0.0

            # 無音区間のグレー背景
            self._shade_unvoiced(ax, time_axis, voicing_mask)

            # プロット（予測値を点線にして重なりを見やすくする）
            ax.plot(
                time_axis,
                y_t_denorm,
                label="実測値 (Ground Truth)",
                color="steelblue",
                linewidth=1.5,
            )
            ax.plot(
                time_axis,
                y_p_denorm,
                label="予測値 (Predicted)",
                color="coral",
                linewidth=1.5,
                linestyle="--",
            )

            ax.set_title(f"波形比較: {song_name} - {pname}  |  MAE = {mae:.2f}")
            if pname == "PIT":
                ax.set_ylabel("Pitch (-8192~8191)")
                ax.set_ylim(-8192, 8191)
            elif pname == "PBS":
                ax.set_ylabel("PBS (0~24)")
                ax.set_ylim(0, 24)
            else:
                ax.set_ylabel("Value (0~127)")
                ax.set_ylim(0, 127)

            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        # --- 追加特徴量 (RMS、Melスペクトログラムなど) のプロット ---
        colorbar_configs = []
        has_2d_feat = False
        if extra_features:
            for idx, (feat_name, feat_data) in enumerate(extra_features.items()):
                ax = axes[num_dims + idx]

                # 無音区間のグレー背景
                self._shade_unvoiced(ax, time_axis, voicing_mask)

                # 1次元特徴量 (RMS, 全ラウドネス, F0 など) は折れ線グラフ
                if feat_data.ndim == 1 or (feat_data.ndim == 2 and feat_data.shape[1] == 1):
                    y_plot = feat_data.squeeze()
                    ax.plot(
                        time_axis,
                        y_plot,
                        label=feat_name,
                        color="forestgreen",
                        linewidth=1.5,
                    )
                    ax.set_title(f"入力特徴量 (Input Feature): {feat_name}")
                    ax.set_ylabel("Value")
                    ax.legend(loc="upper right")
                    ax.grid(True, alpha=0.3)
                # 2次元特徴量 (Mel, Bark, CQT, STFT など) はスペクトログラム（ヒートマップ）表示
                elif feat_data.ndim == 2 and feat_data.shape[1] > 1:
                    has_2d_feat = True
                    n_bins = feat_data.shape[1]
                    mesh = ax.pcolormesh(
                        time_axis,
                        np.arange(n_bins),
                        feat_data.T,
                        shading="auto",
                        cmap="magma",
                    )
                    ax.set_title(
                        f"入力特徴量 (Input Spectrogram): {feat_name} ({n_bins} bins)"
                    )
                    ax.set_ylabel(f"{feat_name} Bin")
                    colorbar_configs.append((ax, mesh))

        axes[-1].set_xlabel("Time (s)")

        if has_2d_feat:
            # 2次元特徴量がある場合、右側にカラーバー用の余白を確保してtight_layout
            plt.tight_layout(rect=[0, 0, 0.88, 1.0])
            for target_ax, mesh_obj in colorbar_configs:
                pos = target_ax.get_position()
                cax = fig.add_axes([0.90, pos.y0, 0.018, pos.height])
                cbar = fig.colorbar(mesh_obj, cax=cax)
                cbar.set_label("Intensity", fontsize=9)
        else:
            plt.tight_layout()

        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"波形グラフを保存しました: {save_path}")

    def plot_scatter(self, save_dir, prefix, param_names=None):
        """
        蓄積された全曲のデータを用いて、次元ごとに散布図をプロットして保存する。
        """
        if not self.all_y_true:
            print("散布図を描画するためのデータがありません。")
            return

        os.makedirs(save_dir, exist_ok=True)
        full_y_true = np.concatenate(self.all_y_true, axis=0)
        full_y_pred = np.concatenate(self.all_y_pred, axis=0)
        num_dims = full_y_true.shape[1]

        for dim_idx in range(num_dims):
            pname = (
                param_names[dim_idx]
                if param_names and dim_idx < len(param_names)
                else f"Dim_{dim_idx}"
            )
            save_path = os.path.join(save_dir, f"{prefix}_scatter_{pname}.png")

            y_t = denormalize_array(full_y_true[:, dim_idx], pname)
            y_p = denormalize_array(full_y_pred[:, dim_idx], pname)

            plt.figure(figsize=(8, 8))
            plt.scatter(y_t, y_p, alpha=0.05, s=1, color="steelblue")

            if pname == "PIT":
                val_min, val_max = -8192, 8191
                label_suffix = ""
            elif pname == "PBS":
                val_min, val_max = 0, 24
                label_suffix = ""
            else:
                val_min, val_max = 0, 127
                label_suffix = "(0~127)"

            plt.xlim(val_min, val_max)
            plt.ylim(val_min, val_max)

            plt.plot(
                [val_min, val_max],
                [val_min, val_max],
                "r--",
                linewidth=1.5,
                label="Ideal",
            )

            plt.xlabel(f"実測値 (Ground Truth) {label_suffix}")
            plt.ylabel(f"予測値 (Predicted) {label_suffix}")
            plt.title(f"散布図 (Scatter Plot): {prefix} - {pname}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"散布図を保存しました: {save_path}")

    def export_report(self, save_excel_path):
        """
        蓄積された全曲の評価結果をExcelとして整形して保存する。
        上部にサマリーテーブル（平均・中央値・最大値・最小値＋データ名）を配置し、
        下部にカラースケール付きのメインデータテーブルを配置する。
        """
        if not self.results:
            print("エクスポートする評価結果がありません。")
            return

        df = pd.DataFrame(self.results)

        # 列の並び順の整理
        base_cols = ["Song Name", "Total Loss", "MSE Loss", "Diff Loss", "Valid Frames"]
        param_cols = [c for c in df.columns if c.startswith("MSE_")]
        final_cols = [c for c in base_cols + param_cols if c in df.columns]
        df = df[final_cols]

        # Total Loss が悪い順（降順）にソート
        df_sorted = df.sort_values(by="Total Loss", ascending=False).reset_index(
            drop=True
        )

        numeric_cols = [c for c in final_cols if c != "Song Name"]

        # 各統計行の「データ名」を Total Loss 基準で決定
        max_song = df_sorted.loc[df_sorted["Total Loss"].idxmax(), "Song Name"]
        min_song = df_sorted.loc[df_sorted["Total Loss"].idxmin(), "Song Name"]
        median_val = df_sorted["Total Loss"].median()
        median_idx = (df_sorted["Total Loss"] - median_val).abs().idxmin()
        median_song = df_sorted.loc[median_idx, "Song Name"]

        os.makedirs(os.path.dirname(save_excel_path), exist_ok=True)

        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Evaluation"

            # ==========================================================
            # サマリーテーブル (Rows 1-5)
            # ==========================================================
            summary_header = ["指標名", "データ名"] + numeric_cols
            summary_rows = [
                ["平均値 (Mean)", ""] + [df_sorted[c].mean() for c in numeric_cols],
                ["中央値 (Median)", median_song]
                + [df_sorted[c].median() for c in numeric_cols],
                ["最大値 (Max)", max_song] + [df_sorted[c].max() for c in numeric_cols],
                ["最小値 (Min)", min_song] + [df_sorted[c].min() for c in numeric_cols],
            ]

            # ヘッダー書き込み
            header_fill = PatternFill(
                start_color="4472C4", end_color="4472C4", fill_type="solid"
            )
            header_font = Font(bold=True, color="FFFFFF")
            for col_idx, val in enumerate(summary_header, 1):
                cell = ws.cell(row=1, column=col_idx, value=val)
                cell.fill = header_fill
                cell.font = header_font

            # 定義: 中央値・最大値・最小値用のソフトカラー背景
            fill_median = PatternFill(
                start_color="FFF2CC", end_color="FFF2CC", fill_type="solid"
            )  # 淡い黄
            fill_max = PatternFill(
                start_color="FCE4D6", end_color="FCE4D6", fill_type="solid"
            )  # 淡い赤
            fill_min = PatternFill(
                start_color="E2EFDA", end_color="E2EFDA", fill_type="solid"
            )  # 淡い緑

            # データ行書き込み＆各指標に応じた行の色付け
            summary_row_fills = [None, fill_median, fill_max, fill_min]
            bold_font = Font(bold=True)
            for row_offset, row_data in enumerate(summary_rows, 2):
                row_fill = summary_row_fills[row_offset - 2]
                for col_idx, val in enumerate(row_data, 1):
                    cell = ws.cell(row=row_offset, column=col_idx, value=val)
                    if row_fill:
                        cell.fill = row_fill
                # 指標名列を太字に
                ws.cell(row=row_offset, column=1).font = bold_font

            # サマリーブロックに罫線を適用
            thin_border = Border(
                left=Side(style="thin"),
                right=Side(style="thin"),
                top=Side(style="thin"),
                bottom=Side(style="thin"),
            )
            for row_idx in range(1, 6):
                for col_idx in range(1, len(summary_header) + 1):
                    ws.cell(row=row_idx, column=col_idx).border = thin_border

            # ==========================================================
            # メインデータテーブル (Row 7+)
            # ==========================================================
            main_start_row = 7

            # ヘッダー書き込み
            for col_idx, col_name in enumerate(final_cols, 1):
                ws.cell(row=main_start_row, column=col_idx, value=col_name)

            # データ行書き込み（中央値・最大値・最小値の行のみハイライト）
            for row_offset, (_, row_data) in enumerate(df_sorted.iterrows(), 1):
                row_idx = main_start_row + row_offset
                song_name = row_data["Song Name"]

                # 対象の行に応じた色付け判定
                target_fill = None
                if song_name == max_song:
                    target_fill = fill_max
                elif song_name == min_song:
                    target_fill = fill_min
                elif song_name == median_song:
                    target_fill = fill_median

                for col_idx, col_name in enumerate(final_cols, 1):
                    cell = ws.cell(
                        row=row_idx,
                        column=col_idx,
                        value=row_data[col_name],
                    )
                    if target_fill:
                        cell.fill = target_fill

            # テーブル書式の適用
            num_data_rows = len(df_sorted)
            table_end_row = main_start_row + num_data_rows
            table_ref = (
                f"A{main_start_row}:{get_column_letter(len(final_cols))}{table_end_row}"
            )
            table = Table(displayName="EvalResults", ref=table_ref)
            style = TableStyleInfo(
                name="TableStyleMedium9",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            table.tableStyleInfo = style
            ws.add_table(table)

            # ==========================================================
            # 列幅の自動調整
            # ==========================================================
            total_cols = max(len(summary_header), len(final_cols))
            for col_idx in range(1, total_cols + 1):
                col_letter = get_column_letter(col_idx)
                max_len = 0
                for row_idx in range(1, table_end_row + 1):
                    cell_val = ws.cell(row=row_idx, column=col_idx).value
                    if cell_val is not None:
                        if isinstance(cell_val, float):
                            cell_str = f"{cell_val:.6f}"
                        else:
                            cell_str = str(cell_val)
                        max_len = max(max_len, len(cell_str))
                ws.column_dimensions[col_letter].width = min(max_len + 3, 30)

            wb.save(save_excel_path)
            print(f"\n評価レポートを保存しました: {save_excel_path}")

        except Exception as e:
            print(f"Excelの保存に失敗しました: {e}")
            csv_path = save_excel_path.replace(".xlsx", ".csv")
            df_sorted.to_csv(csv_path, index=False)
            print(f"代わりにCSVとして保存しました: {csv_path}")
