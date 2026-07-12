import os
import glob
import logging
from pathlib import Path
import re  # 正規表現ライブラリを使用

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- 正規表現置換用のコールバック関数 ---


def _transpose_callback(match: re.Match, transpose_value: int) -> str:
    """
    re.subによって呼び出される関数。
    <n> または <noteNum> タグにマッチし、中の数値を移調する。
    """
    tag_name = match.group(1)  # ( n または noteNum )
    original_note_str = match.group(2)  # ( -?\d+ )

    try:
        # 元のノートナンバーを整数に変換
        original_note_num = int(original_note_str)

        # 移調値を加算
        new_note_num = original_note_num + transpose_value

        # 新しいタグを文字列として返す (元のタグ名を維持)
        return f"<{tag_name}>{new_note_num}</{tag_name}>"

    except ValueError:
        logging.warning(f"警告: 不正なノートナンバーを検出しました: {match.group(0)}")
        return match.group(0)


# --- メイン処理関数 ---


def transpose_vsqx_notes(vsqx_path: str, output_path: str, transpose_value: int):
    """VSQXファイルをテキストとして読み込み、ノートナンバーを移調して保存します。"""

    try:
        # 1. ファイルをテキストとして読み込む
        with open(vsqx_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. <n> または <noteNum> タグを検索し、コールバック関数で置換
        #    パターン: <(n|noteNum)>(-?\d+)</\1>
        #    - グループ1: タグ名 (n または noteNum)
        #    - グループ2: ノートナンバー
        #    - \1 : グループ1と同じタグ名で閉じる

        pattern = re.compile(r"<(n|noteNum)>(-?\d+)</\1>")
        content = pattern.sub(
            lambda m: _transpose_callback(m, transpose_value), content
        )

        # 3. 変更した内容をファイルに書き込む
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return True

    except Exception as e:
        logging.error(f"処理中に予期せぬエラーが発生しました ({vsqx_path}): {e}")
        import traceback

        traceback.print_exc()
        return False


# --- 実行関数 ---


def process_files(config: dict):
    """
    設定辞書に基づき、VSQXファイルの一括移調処理を実行します。
    """
    input_dir = config.get("input_dir")
    output_base_dir = config.get("output_base_dir")
    transpose_semitones = config.get("transpose_semitones", 0)

    if not all([input_dir, output_base_dir]):
        logging.error(
            "エラー: configに 'input_dir' または 'output_base_dir' が不足しています。"
        )
        return

    if not isinstance(transpose_semitones, int):
        logging.error(
            f"エラー: 'transpose_semitones' の値 ({transpose_semitones}) は整数である必要があります。"
        )
        return

    # 出力パスの準備
    file_suffix = f"_T{transpose_semitones:+.0f}".replace("+-", "-")
    output_folder_name = f"Transposed{file_suffix}"
    final_output_dir = os.path.join(output_base_dir, output_folder_name)

    os.makedirs(final_output_dir, exist_ok=True)
    logging.info(f"出力先フォルダ: {final_output_dir}")

    # ファイル処理
    vsqx_files = glob.glob(os.path.join(input_dir, "*.vsqx"))

    if not vsqx_files:
        logging.warning(
            f"警告: 入力ディレクトリ '{input_dir}' に .vsqx ファイルが見つかりません。"
        )
        return

    logging.info(
        f"{len(vsqx_files)} 件のVSQXファイルを処理します (移調: {transpose_semitones} 半音)..."
    )

    success_count = 0
    fail_count = 0

    for vsqx_path in vsqx_files:
        original_filename = Path(vsqx_path).stem
        output_filename = f"{original_filename}{file_suffix}.vsqx"
        output_path = os.path.join(final_output_dir, output_filename)

        logging.info(f"処理中: {original_filename}.vsqx -> {output_filename}")

        if transpose_vsqx_notes(vsqx_path, output_path, transpose_semitones):
            success_count += 1
        else:
            fail_count += 1

    logging.info("--- 処理が完了しました。 ---")
    logging.info(f"成功: {success_count}件, 失敗: {fail_count}件")


# --- スクリプト実行 ---
if __name__ == "__main__":

    # ▼▼▼ 設定をここで行う ▼▼▼
    execution_config1 = {
        # (必須) 変換したいvsqxファイルが入っているフォルダ
        "input_dir": "./data/vsqx",
        # (必須) 変換後のファイルを出力するベースフォルダ
        "output_base_dir": "./data/vsqx/augment/transposed",
        # (必須) 移調する半音の数 (整数)
        # プラスの値でキーが上がり、マイナスの値でキーが下がります。
        "transpose_semitones": -1,
    }

    # ▲▲▲ 設定はここまで ▲▲▲

    process_files(execution_config1)
