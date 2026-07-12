import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil

# =============================================================================
# ユーザー設定エリア (CONFIG)
# =============================================================================
CONFIG = {
    # 処理対象のフォルダ
    "input_dir":"./data/vsqx_ALL/PAM/got_merged/original",
    # 処理結果を保存するフォルダ (同じにすると上書きされます)
    "output_dir": "./data/vsqx_ALL/PAM/got_merged/remove_subtracks",
    # 名前空間の処理用 (vsqxファイルのxmlnsに合わせる必要があります)
    "xmlns": "http://www.yamaha.co.jp/vocaloid/schema/vsq4/"
}

# =============================================================================
# クラス定義: サブトラック削除
# =============================================================================
class SubTrackRemover:
    def __init__(self, config):
        self.input_dir = config["input_dir"]
        self.output_dir = config["output_dir"]
        self.xmlns = config["xmlns"]
        # namespaceを含めてタグを検索するための設定
        self.ns = {'vsq': self.xmlns}
        ET.register_namespace('', self.xmlns)

    def run(self):
        print(f"\n{'='*60}")
        print(f" VSQX Sub-Track (Track != 0) Remover")
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

        for fpath in tqdm(vsqx_files, desc="Removing sub-tracks"):
            try:
                rel_path = os.path.relpath(fpath, self.input_dir)
                out_path = os.path.join(self.output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                self._process_file(fpath, out_path)
                processed_count += 1
            except Exception as e:
                print(f"\n[Error] Failed on {fpath}: {e}")
                error_count += 1

        print(f"\nDone! Processed: {processed_count}, Errors: {error_count}")

    def _process_file(self, input_path, output_path):
        tree = ET.parse(input_path)
        root = tree.getroot()

        # vsTrackタグをすべて探し、tNoが0でないものを削除
        tracks_to_remove = []
        for track in root.findall('vsq:vsTrack', self.ns):
            tno_elem = track.find('vsq:tNo', self.ns)
            if tno_elem is not None and tno_elem.text != "0":
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            root.remove(track)

        # mixerタグの中のvsUnitもtNoが0でないものを削除
        mixer = root.find('vsq:mixer', self.ns)
        if mixer is not None:
            units_to_remove = []
            for unit in mixer.findall('vsq:vsUnit', self.ns):
                tno_elem = unit.find('vsq:tNo', self.ns)
                # MasterTrackは tNo がないため、エラーにならないようにする
                if tno_elem is not None and tno_elem.text != "0":
                    units_to_remove.append(unit)
            
            for unit in units_to_remove:
                mixer.remove(unit)

        # 保存（xml宣言とエンコーディングを指定）
        tree.write(output_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    if not os.path.exists(CONFIG["input_dir"]):
        print(f"[Error] Input directory not found: {CONFIG['input_dir']}")
    else:
        remover = SubTrackRemover(CONFIG)
        remover.run()
