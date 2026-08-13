import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys

# utilフォルダからvsqx_readerをインポートするための設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VSQX_READER_DIR = os.path.join(SRC_DIR, "04_Parameter_Study", "src")
sys.path.insert(0, VSQX_READER_DIR)

try:
    from vsqx_compare import VSQXCompare
except ImportError:
    print(f"[Error] vsqx_reader.py could not be imported from {VSQX_READER_DIR}")
    sys.exit(1)

# =============================================================================
# ユーザー設定エリア (CONFIG)
# =============================================================================
CONFIG = {
    # 解析対象のVSQXが入っているフォルダパス
    "input_dir": "./data/vsqx_ALL/PAM/got_merged/augment/aug0706",
    # 結果を出力するフォルダパス
    "output_dir": "./data/vsqx_ALL/PAM/got_merged/augment/aug0706_chunk_filter",
    # 有効データ量（有音区間の合計秒数）の閾値
    "threshold_sec": 30.0,
    # 短いファイルを移動・コピーする設定 ("copy", "move", "none")
    "action_for_short_files": "copy",
}

# =============================================================================
# クラス定義: 有効データ量アナライザー
# =============================================================================
class ValidDurationAnalyzer:
    def __init__(self, config):
        self.cfg = config
        self.input_dir = config["input_dir"]
        self.output_dir = config["output_dir"]
        self.threshold = config["threshold_sec"]
        self.action = config["action_for_short_files"]
        
        self.results = []
        self.skipped_files = []  # スキップされたファイルとその理由を記録
        # VSQXCompareの初期化 (サンプリングレートは計算用に適当に設定)
        self.reader = VSQXCompare(sr=44100, shift_length=512)

    def _setup_directories(self):
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                print(f"[Warning] Could not delete old output dir: {e}")

        os.makedirs(os.path.join(self.output_dir, "Passed"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "Short_Excluded"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "_graphs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "_reports"), exist_ok=True)

    def _analyze_file(self, file_path):
        fname = os.path.basename(file_path)
        try:
            vsqx_data = self.reader.read(file_path)
            notes = vsqx_data.get("notes", [])
            
            # 有音区間の合計秒数を計算
            total_duration_sec = sum(note["end_sec"] - note["start_sec"] for note in notes)
            
            # ノート数
            note_count = len(notes)
            
            # チャンク数換算（例えば200フレーム=約2.3秒のチャンクがいくつ取れるか）
            # ここでは単純に「合計秒数 / 2.3秒」を理論上のチャンク数とする
            chunk_200 = total_duration_sec / 2.32
            
            category = "Passed" if total_duration_sec >= self.threshold else "Short_Excluded"
            
            return {
                "file": fname,
                "path": file_path,
                "duration_sec": total_duration_sec,
                "note_count": note_count,
                "est_chunks": chunk_200,
                "category": category
            }
        except Exception as e:
            print(f"[Skipped] {fname}: {type(e).__name__}: {e}")
            self.skipped_files.append({
                "file": fname,
                "error_type": type(e).__name__,
                "message": str(e),
            })
            return None

    def run(self):
        print(f"\n{'='*60}")
        print(f" VSQX Valid Duration Analyzer")
        print(f"{'='*60}")
        print(f"Input : {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Threshold: {self.threshold} sec")
        print(f"{'-'*60}")

        vsqx_files = glob.glob(os.path.join(self.input_dir, "**/*.vsqx"), recursive=True)
        if not vsqx_files:
            print("[Error] No .vsqx files found.")
            return

        self._setup_directories()
        print(f"Analyzing {len(vsqx_files)} files...")

        summary_counts = {"Passed": 0, "Short_Excluded": 0}

        for fpath in tqdm(vsqx_files):
            res = self._analyze_file(fpath)
            if res:
                self.results.append(res)
                cat = res["category"]
                summary_counts[cat] += 1

                # ファイルのコピー/移動処理
                if self.action in ["copy", "move"]:
                    dest_path = os.path.join(self.output_dir, cat, res["file"])
                    if self.action == "copy":
                        shutil.copy2(res["path"], dest_path)
                    elif self.action == "move":
                        shutil.move(res["path"], dest_path)

        self._export_csv()
        self._visualize()

        print(f"\n{'='*60}")
        print(" Final Summary")
        print(f"{'='*60}")
        for cat, count in summary_counts.items():
            print(f"  - {cat:<15}: {count} files")
        if self.skipped_files:
            print(f"  - {'Skipped':<15}: {len(self.skipped_files)} files")
        print("-" * 20)

        # スキップレポートの出力
        if self.skipped_files:
            report_path = os.path.join(self.output_dir, "_reports", "skipped_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Skipped Files Report\n")
                f.write(f"====================\n")
                f.write(f"Total: {len(self.skipped_files)} file(s) skipped\n\n")
                for i, entry in enumerate(self.skipped_files, 1):
                    f.write(f"[{i}] {entry['file']}\n")
                    f.write(f"    Error: {entry['error_type']}\n")
                    f.write(f"    Detail: {entry['message']}\n\n")
            print(f"Skipped report saved: {report_path}")

        print(f"Done. Check folder: {self.output_dir}")

    def _export_csv(self):
        df = pd.DataFrame(self.results)
        # 不要なpath列はCSVから削除
        if "path" in df.columns:
            df = df.drop(columns=["path"])
            
        csv_path = os.path.join(self.output_dir, "_reports", "duration_analysis.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nReport saved: {csv_path}")

    def _visualize(self):
        print("Generating histograms...")
        durations = [r["duration_sec"] for r in self.results]

        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=min(50, max(10, len(durations))), color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(x=self.threshold, color="red", linestyle="--", label=f"Threshold ({self.threshold}s)")

        plt.title("Distribution of Valid Singing Duration")
        plt.xlabel("Total Duration (seconds)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(axis="y", alpha=0.5)

        path = os.path.join(self.output_dir, "_graphs", "duration_hist.png")
        plt.savefig(path)
        plt.close()

if __name__ == "__main__":
    if not os.path.exists(CONFIG["input_dir"]):
        print(f"[Error] Input directory not found: {CONFIG['input_dir']}")
    else:
        analyzer = ValidDurationAnalyzer(CONFIG)
        analyzer.run()
