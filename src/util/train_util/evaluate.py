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
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font
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
            # MSE と Diff Loss を共通関数で計算 (voicing_maskを渡して無音を無視)
            total_loss, mse_val, diff_val = self.loss_fn(
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
            "MSE Loss": mse_val,
            "Diff Loss": diff_val,
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
        sr=44100,
        hop_length=256,
    ):
        """
        時系列波形をプロットして保存する。
        無音区間をグレー背景にし、時間軸を秒単位にする。
        """
        # イントロの無音削除
        if voicing_mask is not None:
            voiced_indices = np.where(voicing_mask > 0.5)[0]
            if len(voiced_indices) > 0:
                start_idx = voiced_indices[0]
                y_true = y_true[start_idx:]
                y_pred = y_pred[start_idx:]
                voicing_mask = voicing_mask[start_idx:]
        else:
            voicing_mask = np.ones(y_true.shape[0], dtype=np.float32)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        num_dims = y_true.shape[1]

        # 時間軸 (秒) の計算
        # librosa.frames_to_time を使わず直接計算してlibrosa依存を避ける
        time_axis = np.arange(y_true.shape[0]) * hop_length / sr

        fig, axes = plt.subplots(num_dims, 1, figsize=(12, 3.5 * num_dims), sharex=True)
        if num_dims == 1:
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
            elif pname == "PBS":
                ax.set_ylabel("PBS (0~24)")
            else:
                ax.set_ylabel("Value (0~127)")

            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
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

            val_min = min(y_t.min(), y_p.min())
            val_max = max(y_t.max(), y_p.max())
            plt.plot(
                [val_min, val_max],
                [val_min, val_max],
                "r--",
                linewidth=1.5,
                label="Ideal",
            )

            label_suffix = "(0~127)" if pname not in ["PIT", "PBS"] else ""
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
        Pandasとopenpyxlを用いて、テーブル化とソートを行う。
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

        # --- 平均行の作成 ---
        numeric_cols = [c for c in final_cols if c != "Song Name"]
        avg_row = {"Song Name": "AVERAGE"}
        for col in numeric_cols:
            avg_row[col] = df_sorted[col].mean()

        df_avg = pd.concat([df_sorted, pd.DataFrame([avg_row])], ignore_index=True)

        os.makedirs(os.path.dirname(save_excel_path), exist_ok=True)

        try:
            # まずPandasでExcelにベタ書き
            df_avg.to_excel(save_excel_path, index=False)

            # openpyxlで装飾
            wb = load_workbook(save_excel_path)
            ws = wb.active

            num_cols = len(final_cols)
            num_data_rows = len(df_sorted)

            # テーブル化
            table_ref = f"A1:{get_column_letter(num_cols)}{num_data_rows + 1}"
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

            # 平均行を太字に
            avg_row_idx = num_data_rows + 2
            bold_font = Font(bold=True)
            for col_idx in range(1, num_cols + 1):
                ws.cell(row=avg_row_idx, column=col_idx).font = bold_font

            # 列幅の自動調整
            for col_idx in range(1, num_cols + 1):
                col_letter = get_column_letter(col_idx)
                max_len = len(str(ws.cell(row=1, column=col_idx).value or ""))
                for row_idx in range(2, avg_row_idx + 1):
                    cell_val = ws.cell(row=row_idx, column=col_idx).value
                    if cell_val is not None:
                        max_len = max(max_len, len(str(cell_val)))
                ws.column_dimensions[col_letter].width = min(max_len + 2, 30)

            wb.save(save_excel_path)
            print(f"\n評価レポートを保存しました: {save_excel_path}")

        except Exception as e:
            print(f"Excelの保存に失敗しました: {e}")
            csv_path = save_excel_path.replace(".xlsx", ".csv")
            df_avg.to_csv(csv_path, index=False)
            print(f"代わりにCSVとして保存しました: {csv_path}")
