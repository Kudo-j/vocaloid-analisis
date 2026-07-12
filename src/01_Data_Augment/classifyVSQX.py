import os
import glob
import logging
from pathlib import Path
import shutil  # ファイルコピー用

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- メイン処理関数 ---


def get_vsq_version(vsqx_path: str) -> str:
    """
    VSQXファイルの先頭を読み込み、バージョンを判定する。
    (vsq3 / vsq4 / unknown のいずれかを返す)
    """
    try:
        with open(vsqx_path, "r", encoding="utf-8") as f:
            # ファイルの先頭512バイト（XML宣言とルートタグを含むには十分）だけ読み込む
            # これにより、巨大なファイルでも高速に判定できる
            head_content = f.read(512)

            if "vsq4 xmlns" in head_content:
                return "vsq4"
            elif "vsq3 xmlns" in head_content:
                return "vsq3"
            else:
                return "unknown"

    except Exception as e:
        logging.warning(f"警告: {vsqx_path} の読み込み中にエラーが発生しました: {e}")
        return "error"


# --- 実行関数 ---


def process_files(config: dict):
    """
    設定辞書に基づき、VSQXファイルのバージョン仕分けを実行します。
    """

    # --- 1. 設定の読み込み ---
    input_dir = config.get("input_dir")
    output_base_dir = config.get("output_base_dir")

    if not all([input_dir, output_base_dir]):
        logging.error(
            "エラー: configに 'input_dir' または 'output_base_dir' が不足しています。"
        )
        return

    # --- 2. 出力先フォルダの準備 ---
    output_vsq3_dir = os.path.join(output_base_dir, "vsq3")
    output_vsq4_dir = os.path.join(output_base_dir, "vsq4")
    output_unknown_dir = os.path.join(output_base_dir, "unknown")

    os.makedirs(output_vsq3_dir, exist_ok=True)
    os.makedirs(output_vsq4_dir, exist_ok=True)
    os.makedirs(output_unknown_dir, exist_ok=True)

    logging.info(f"入力元: {input_dir}")
    logging.info(f"仕分け先 (V3): {output_vsq3_dir}")
    logging.info(f"仕分け先 (V4): {output_vsq4_dir}")

    # --- 3. ファイル処理の実行 ---
    vsqx_files = glob.glob(os.path.join(input_dir, "*.vsqx"))

    if not vsqx_files:
        logging.warning(
            f"警告: 入力ディレクトリ '{input_dir}' に .vsqx ファイルが見つかりません。"
        )
        return

    logging.info(f"{len(vsqx_files)} 件のVSQXファイルを仕分けします...")

    count_v3 = 0
    count_v4 = 0
    count_unknown = 0

    for vsqx_path in vsqx_files:
        original_filename = Path(vsqx_path).name

        version = get_vsq_version(vsqx_path)

        if version == "vsq4":
            output_path = os.path.join(output_vsq4_dir, original_filename)
            shutil.copy(vsqx_path, output_path)
            count_v4 += 1
        elif version == "vsq3":
            output_path = os.path.join(output_vsq3_dir, original_filename)
            shutil.copy(vsqx_path, output_path)
            count_v3 += 1
        else:
            output_path = os.path.join(output_unknown_dir, original_filename)
            shutil.copy(vsqx_path, output_path)
            count_unknown += 1

    logging.info("--- 処理が完了しました。 ---")
    logging.info(f"VSQ4: {count_v4}件, VSQ3: {count_v3}件, 不明: {count_unknown}件")


# --- スクリプト実行 ---
if __name__ == "__main__":

    # ▼▼▼ 設定をここで行う ▼▼▼
    execution_config = {
        # (必須) 仕分けしたいvsqxファイルが全て入っているフォルダ
        "input_dir": "./data/classified_VSQX/original",
        # (必須) 仕分け先のベースフォルダ
        # (この下に 'vsq3', 'vsq4', 'unknown' フォルダが自動作成されます)
        "output_base_dir": "./data/classified_VSQX/classified_by_version",
    }
    # ▲▲▲ 設定はここまで ▲▲▲

    process_files(execution_config)
