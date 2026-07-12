import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

# =============================================================================
# ユーザー設定エリア (CONFIG) - ここを変更して「実行」を押してください
# =============================================================================
CONFIG = {
    # 解析対象のVSQXが入っているフォルダパス (Windowsのパス区切り \ は / にするか、r"..." で囲む)
    "input_dir": "./data/vsqx_ALL/PAM/got_merged/augment/aug_DYN",
    # 結果を出力するフォルダパス
    "output_dir": "./data/analysis_result/augments",
    # 解析・仕分けを行いたいパラメータのリスト
    # 選択肢: "DYN", "BRE", "BRI", "CLE", "GEN", "POR", "PIT", "PBS", "GWL", "XSY"
    "target_params": ["DYN"],
    # 判定閾値 (標準偏差 Std Dev)
    # ※注意: PITなどは値の範囲が広いため、DYNと同じ閾値だと「Low」になりやすい可能性があります。
    "thresholds": {
        "dead": 5, # ほぼ一直線
        "low_variance": 15  # 動きが少ない
    },
}

# =============================================================================
# 定数定義 (ID Mapping)
# =============================================================================
ID_MAPPING = {
    "D": "DYN",  # Dynamics
    "B": "BRE",  # Breathiness
    "R": "BRI",  # Brightness
    "C": "CLE",  # Clearness
    "G": "GEN",  # Gender Factor
    "T": "POR",  # Portamento Timing
    "P": "PIT",  # Pitch Bend
    "S": "PBS",  # Pitch Bend Sensitivity
    "W": "GWL",  # Growl
    "X": "XSY",  # Cross-Synthesis
}

# =============================================================================
# クラス定義: パラメータ別アナライザー
# =============================================================================


class ParameterAnalyzer:
    def __init__(self, config):
        self.cfg = config
        self.input_dir = config["input_dir"]
        self.output_dir = config["output_dir"]
        self.targets = config["target_params"]
        self.th_dead = config["thresholds"]["dead"]
        self.th_low = config["thresholds"]["low_variance"]

        self.results = []

    def _setup_directories(self):
        """出力用フォルダツリーの作成 (毎回リセット)"""
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                print(f"[Warning] Could not delete old output dir: {e}")

        categories = ["Active", "Low_Variance", "Dead_Flat"]

        for cat in categories:
            for param in self.targets:
                path = os.path.join(self.output_dir, cat, param)
                os.makedirs(path, exist_ok=True)

        os.makedirs(os.path.join(self.output_dir, "_graphs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "_reports"), exist_ok=True)

    def _parse_vsqx(self, file_path):
        """
        VSQXから【Track 0】の指定パラメータ値を抽出する
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # データ格納用辞書
            extracted_data = {name: [] for name in self.targets}

            # --- 変更点: Track 0 の特定 ---
            # VSQXの構造は通常 <vsq4> -> <vsTrack> ... </vsTrack> となっている
            # 名前空間(xmlns)が含まれる場合があるため、タグ名の末尾で判定する
            tracks = []
            for child in root:
                if child.tag.endswith("vsTrack"):
                    tracks.append(child)

            if not tracks:
                # トラックが見つからない場合は空データを返す
                return extracted_data

            # 最初のトラック (Track 0) を取得
            target_track = tracks[0]

            # --- Track 0 の内部のみを走査 ---
            for elem in target_track.iter():
                if "id" in elem.attrib:
                    eid = elem.attrib["id"]
                    if eid in ID_MAPPING:
                        p_name = ID_MAPPING[eid]
                        # ターゲットに含まれているパラメータのみ抽出
                        if p_name in self.targets:
                            text = elem.text
                            if text is not None:
                                try:
                                    val = int(text.strip())
                                    extracted_data[p_name].append(val)
                                except ValueError:
                                    continue

            return extracted_data

        except Exception as e:
            print(f"[Error] Failed to parse {os.path.basename(file_path)}: {e}")
            return {name: [] for name in self.targets}

    def _analyze_file(self, file_path):
        """1ファイルの統計計算とカテゴリ判定"""
        raw_data = self._parse_vsqx(file_path)
        fname = os.path.basename(file_path)

        file_result = {"file": fname, "path": file_path, "params": {}}

        for param_name in self.targets:
            values = np.array(raw_data[param_name])

            if len(values) > 0:
                std = np.std(values)
                rng = np.max(values) - np.min(values)
                count = len(values)
            else:
                std = 0.0
                rng = 0
                count = 0

            if std < self.th_dead or rng == 0:
                category = "Dead_Flat"
            elif std < self.th_low:
                category = "Low_Variance"
            else:
                category = "Active"

            file_result["params"][param_name] = {
                "std": std,
                "range": rng,
                "count": count,
                "category": category,
            }

        return file_result

    def _rename_folders(self):
        """フォルダ名にファイル数を付与する (Active -> Active_120)"""
        print("\nRenaming folders with file counts...")
        categories = ["Active", "Low_Variance", "Dead_Flat"]

        for cat in categories:
            cat_path = os.path.join(self.output_dir, cat)
            if not os.path.exists(cat_path):
                continue

            total_cat_count = 0

            # 1. サブフォルダ(パラメータ)のリネーム
            for param in self.targets:
                param_path = os.path.join(cat_path, param)
                if os.path.exists(param_path):
                    count = len(glob.glob(os.path.join(param_path, "*.vsqx")))
                    total_cat_count += count

                    new_param_name = f"{param}_{count}"
                    new_param_path = os.path.join(cat_path, new_param_name)
                    try:
                        os.rename(param_path, new_param_path)
                    except Exception as e:
                        print(f"[Warning] Failed to rename subfolder {param}: {e}")

            # 2. カテゴリフォルダのリネーム
            new_cat_name = f"{cat}_{total_cat_count}"
            new_cat_path = os.path.join(self.output_dir, new_cat_name)
            try:
                os.rename(cat_path, new_cat_path)
                print(f" -> Renamed: {cat} \t-> {new_cat_name}")
            except Exception as e:
                print(f"[Warning] Failed to rename category {cat}: {e}")

    def run(self):
        print(f"\n{'='*60}")
        print(f" VSQX Parameter Analyzer (Track 0 Only)")
        print(f"{'='*60}")
        print(f"Input : {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Targets: {self.targets}")
        print(f"{'-'*60}")

        vsqx_files = glob.glob(
            os.path.join(self.input_dir, "**/*.vsqx"), recursive=True
        )
        if not vsqx_files:
            print("[Error] No .vsqx files found.")
            return

        self._setup_directories()

        print(f"Analyzing {len(vsqx_files)} files...")

        summary_counts = {
            p: {"Active": 0, "Low_Variance": 0, "Dead_Flat": 0} for p in self.targets
        }

        for fpath in tqdm(vsqx_files):
            res = self._analyze_file(fpath)
            self.results.append(res)

            for param_name, info in res["params"].items():
                cat = info["category"]
                summary_counts[param_name][cat] += 1

                dest_path = os.path.join(self.output_dir, cat, param_name, res["file"])
                shutil.copy2(res["path"], dest_path)

        self._export_csv()
        self._visualize()
        self._rename_folders()

        print(f"\n{'='*60}")
        print(" Final Summary")
        print(f"{'='*60}")
        for param in self.targets:
            print(f"[{param}]")
            for cat, count in summary_counts[param].items():
                print(f"  - {cat:<12}: {count} files")
            print("-" * 20)
        print(f"Done. Check folder: {self.output_dir}")

    def _export_csv(self):
        rows = []
        for r in self.results:
            row = {"filename": r["file"]}
            for param in self.targets:
                info = r["params"][param]
                row[f"{param}_std"] = round(info["std"], 2)
                row[f"{param}_range"] = info["range"]
                row[f"{param}_cat"] = info["category"]
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "_reports", "analysis_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nReport saved: {csv_path}")

    def _visualize(self):
        print("Generating histograms...")
        for param in self.targets:
            stds = [r["params"][param]["std"] for r in self.results]

            plt.figure(figsize=(10, 6))
            plt.hist(stds, bins=50, color="skyblue", edgecolor="black", alpha=0.7)

            plt.axvline(
                x=self.th_dead,
                color="red",
                linestyle="--",
                label=f"Dead ({self.th_dead})",
            )
            plt.axvline(
                x=self.th_low,
                color="orange",
                linestyle="--",
                label=f"Low ({self.th_low})",
            )

            plt.title(f"Distribution of {param} Standard Deviation (Track 0)")
            plt.xlabel("Standard Deviation")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(axis="y", alpha=0.5)

            path = os.path.join(self.output_dir, "_graphs", f"hist_{param}.png")
            plt.savefig(path)
            plt.close()


# =============================================================================
# 実行ブロック
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(CONFIG["input_dir"]):
        print(f"[Error] Input directory not found: {CONFIG['input_dir']}")
        print("Please edit the 'CONFIG' dictionary at the top of the script.")
    else:
        analyzer = ParameterAnalyzer(CONFIG)
        analyzer.run()
