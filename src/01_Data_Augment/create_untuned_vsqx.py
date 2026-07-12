import os
import glob
import re
from tqdm import tqdm
from pathlib import Path

# =============================================================================
# ユーザー設定エリア (CONFIG)
# =============================================================================
CONFIG = {
    # 処理対象のフォルダ
    "input_dir": "./data/vsqx_ALL/original/got_original/vspart_merged/non_dynamic",
    # 結果を出力するフォルダ
    "output_dir": "./data/vsqx_ALL/original/got_original/untuned/non_dynamic",
}

# =============================================================================
# クラス定義: パラメータ消去 (Untuned作成)
# =============================================================================
class UntunedVsqxCreator:
    def __init__(self, config):
        self.input_dir = config["input_dir"]
        self.output_dir = config["output_dir"]

    def run(self):
        print(f"\n{'='*60}")
        print(f" VSQX All Parameters Remover (Create Untuned)")
        print(f"{'='*60}")
        print(f"Input : {self.input_dir}")
        print(f"Output: {self.output_dir}")
        
        vsqx_files = glob.glob(os.path.join(self.input_dir, "**/*.vsqx"), recursive=True)
        if not vsqx_files:
            print("[Error] No .vsqx files found.")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        
        processed_count = 0
        error_count = 0

        # <cc> タグ全体をマッチさせる正規表現
        re_cc_tag = re.compile(r'\s*<cc>.*?</cc>', flags=re.DOTALL)

        for fpath in tqdm(vsqx_files, desc="Removing parameters"):
            try:
                rel_path = os.path.relpath(fpath, self.input_dir)
                base, ext = os.path.splitext(rel_path)
                out_path = os.path.join(self.output_dir, f"{base}_untuned{ext}")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # すべての <cc> タグを削除
                cleaned_content = re_cc_tag.sub("", content)
                
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
                    
                processed_count += 1
            except Exception as e:
                print(f"\n[Error] Failed on {fpath}: {e}")
                error_count += 1

        print(f"\nDone! Processed: {processed_count}, Errors: {error_count}")


if __name__ == "__main__":
    if not os.path.exists(CONFIG["input_dir"]):
        print(f"[Error] Input directory not found: {CONFIG['input_dir']}")
    else:
        creator = UntunedVsqxCreator(CONFIG)
        creator.run()
