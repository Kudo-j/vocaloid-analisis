import os
import glob
import random
import logging
from pathlib import Path
import xml.dom.minidom as minidom  # ElementTreeの代わりに使用
from xml.dom.minidom import Node  # ノードタイプ判別用

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- ボーカルマッピング ---
VOICE_LIB_ID_TO_NAME = {
    "BCNFCY43LB2LZCD4": "MIKU_V4X_Original_EVEC",
    "BFPL93T7GE3RWFC9": "MIKU_V4X_Soft_EVEC",
    "BKGHF6Y5PHTN4KD6": "MIKU_V4X_Solid_EVEC",
    "BHNME578NG2S3HD8": "MIKU_V4X_Dark",
    "BDGE8ZB2FCTKTDBA": "MIKU_V4X_Sweet",
}

# 逆引きマップ（名前からIDを検索するため）
NAME_TO_ID_MAP = {name: id for id, name in VOICE_LIB_ID_TO_NAME.items()}

# --- ヘルパー関数 ---


def get_vocal_suffix(vocal_name: str) -> str:
    """ボーカル名からファイル名用の短いサフィックスを生成します。"""
    if "MIKU_V4X_" in vocal_name:
        return vocal_name.replace("MIKU_V4X_", "").replace("_EVEC", "")
    return vocal_name


def set_text_or_cdata(parent_element, tag_name, new_text):
    """
    指定されたタグの子ノード(テキストまたはCDATA)の値を書き換える。
    タグが存在しない場合は作成する。
    """
    elements = parent_element.getElementsByTagName(tag_name)
    if not elements:
        # タグが存在しない場合 (例: 元のvVoiceに <id2> がなかった場合)
        new_tag = parent_element.ownerDocument.createElement(tag_name)
        # 新しいノードはCDATAとして作成する
        cdata = parent_element.ownerDocument.createCDATASection(new_text)
        new_tag.appendChild(cdata)
        parent_element.appendChild(new_tag)
        return

    # タグが存在する場合
    target_element = elements[0]

    # 子ノード(テキスト or CDATA)の値を書き換える
    if target_element.hasChildNodes():
        # 既存の子ノードの値を変更する
        target_element.firstChild.data = new_text
    else:
        # 子ノードがない場合は、新しいCDATAノードとして追加する
        cdata = parent_element.ownerDocument.createCDATASection(new_text)
        target_element.appendChild(cdata)


def set_params_to_zero(parent_element, tag_name):
    """vPrm, vPrm2 タグを探し、その子要素の値をすべて '0' にする"""
    prm_elements = parent_element.getElementsByTagName(tag_name)
    if not prm_elements:
        return

    prm_tag = prm_elements[0]
    for param in prm_tag.childNodes:
        # 要素ノード(bre, briなど)のみを対象
        if param.nodeType == Node.ELEMENT_NODE and param.hasChildNodes():
            param.firstChild.data = "0"


def remove_blank_text_nodes(node):
    """
    DOMツリーを再帰的に走査し、空白文字のみのテキストノードを削除する。
    (toprettyxmlが余分な改行を入れないようにするため)
    """
    # 逆順でループ（リストから削除を行うため）
    for child in reversed(node.childNodes):
        if child.nodeType == Node.TEXT_NODE and child.data.strip() == "":
            # 空白文字のみのテキストノードは削除
            node.removeChild(child)
        elif child.nodeType == Node.ELEMENT_NODE:
            # 子要素に対しても再帰的に実行
            remove_blank_text_nodes(child)


# --- メイン処理関数 (minidom版) ---


def modify_vsqx_vocals(
    vsqx_path: str, output_path: str, vocal1_name: str, vocal2_name: str
):
    """VSQXファイルを読み込み、ボーカル設定を変更して保存します。(minidom版)"""

    try:
        vocal1_id = NAME_TO_ID_MAP[vocal1_name]
        vocal2_id = NAME_TO_ID_MAP[vocal2_name]
    except KeyError as e:
        logging.error(
            f"エラー: 指定されたボーカル名 '{e.args[0]}' がマッピングに存在しません。"
        )
        return False

    try:
        # XMLファイルをminidomでパース
        doc = minidom.parse(vsqx_path)
        root = doc.documentElement  # <vsq4> タグ

        # 1. <vVoiceTable> を処理
        vVoiceTable_list = root.getElementsByTagName("vVoiceTable")
        if not vVoiceTable_list:
            logging.warning(
                f"警告: {vsqx_path} に <vVoiceTable> が見つかりません。スキップします。"
            )
            return False

        vVoiceTable = vVoiceTable_list[0]
        all_vVoices = vVoiceTable.getElementsByTagName("vVoice")
        if not all_vVoices:
            logging.warning(
                f"警告: {vsqx_path} の <vVoiceTable> に <vVoice> がありません。スキップします。"
            )
            return False

        vVoice_to_keep = all_vVoices[0]

        for vVoice in reversed(all_vVoices[1:]):
            vVoiceTable.removeChild(vVoice)

        set_text_or_cdata(vVoice_to_keep, "pc", "0")
        set_text_or_cdata(vVoice_to_keep, "id", vocal1_id)
        set_text_or_cdata(vVoice_to_keep, "name", vocal1_name)
        set_text_or_cdata(vVoice_to_keep, "id2", vocal2_id)

        set_params_to_zero(vVoice_to_keep, "vPrm")
        set_params_to_zero(vVoice_to_keep, "vPrm2")

        # 2. 全<vsTrack>を処理
        tracks = root.getElementsByTagName("vsTrack")
        for track in tracks:
            # トラック名(<name>)を変更する
            set_text_or_cdata(track, "name", vocal1_name)

            parts = track.getElementsByTagName("vsPart")
            for part in parts:
                # ★ 改善点: パート名(<name>)も変更する
                set_text_or_cdata(part, "name", vocal1_name)

                # パート内の<singer>の<pc>を変更する
                singers = part.getElementsByTagName("singer")
                if singers:
                    set_text_or_cdata(singers[0], "pc", "0")

        # 3. DOMツリーから空白ノードを削除
        remove_blank_text_nodes(root)

        # 4. 変更をインデント付きでファイルに保存
        with open(output_path, "wb") as f:
            xml_str = doc.toprettyxml(indent="\t", newl="\n", encoding="UTF-8")
            f.write(xml_str)

        return True

    except Exception as e:
        logging.error(f"処理中に予期せぬエラーが発生しました ({vsqx_path}): {e}")
        import traceback

        traceback.print_exc()  # 詳細なエラーを表示
        return False


# --- 実行関数 (変更なし) ---


def process_files(config: dict):
    """
    設定辞書に基づき、VSQXファイルの一括変換処理を実行します。
    """

    # --- 1. 設定の読み込み ---
    input_dir = config.get("input_dir")
    output_base_dir = config.get("output_base_dir")
    vocal1_name = config.get("vocal1")
    vocal2_setting = config.get("vocal2", "random")  # デフォルトは "random"

    if not all([input_dir, output_base_dir, vocal1_name]):
        logging.error(
            "エラー: configに 'input_dir', 'output_base_dir', 'vocal1' のいずれかが不足しています。"
        )
        return

    # --- 2. ボーカル名の決定 ---
    vocal_name_list = list(NAME_TO_ID_MAP.keys())

    if vocal1_name not in NAME_TO_ID_MAP:
        logging.error(
            f"エラー: vocal1 の名前 '{vocal1_name}' がマッピングに存在しません。"
        )
        logging.error(f"利用可能な名前: {vocal_name_list}")
        return

    vocal2_name = ""
    if vocal2_setting == "random":
        vocal2_name = random.choice(vocal_name_list)
        logging.info(
            f"--- 2ndボーカルが 'random' に設定されたため、ランダムに '{vocal2_name}' を選択しました。 ---"
        )
    else:
        vocal2_name = vocal2_setting
        if vocal2_name not in NAME_TO_ID_MAP:
            logging.error(
                f"エラー: vocal2 の名前 '{vocal2_name}' がマッピングに存在しません。"
            )
            logging.error(f"利用可能な名前: {vocal_name_list}")
            return

    # --- 3. 出力パスの準備 ---
    v1_suffix = get_vocal_suffix(vocal1_name)
    v2_suffix = get_vocal_suffix(vocal2_name)

    output_folder_name = f"1st_{v1_suffix}_2nd_{v2_suffix}"
    final_output_dir = os.path.join(output_base_dir, output_folder_name)

    os.makedirs(final_output_dir, exist_ok=True)
    logging.info(f"出力先フォルダ: {final_output_dir}")

    file_suffix = f"_{v1_suffix}_{v2_suffix}"

    # --- 4. ファイル処理の実行 ---
    vsqx_files = glob.glob(os.path.join(input_dir, "*.vsqx"))

    if not vsqx_files:
        logging.warning(
            f"警告: 入力ディレクトリ '{input_dir}' に .vsqx ファイルが見つかりません。"
        )
        return

    logging.info(f"{len(vsqx_files)} 件のVSQXファイルを処理します...")

    success_count = 0
    fail_count = 0

    for vsqx_path in vsqx_files:
        original_filename = Path(vsqx_path).stem
        output_filename = f"{original_filename}{file_suffix}.vsqx"
        output_path = os.path.join(final_output_dir, output_filename)

        logging.info(f"処理中: {original_filename}.vsqx -> {output_filename}")

        if modify_vsqx_vocals(vsqx_path, output_path, vocal1_name, vocal2_name):
            success_count += 1
        else:
            fail_count += 1

    logging.info("--- 処理が完了しました。 ---")
    logging.info(f"成功: {success_count}件, 失敗: {fail_count}件")


# --- スクリプト実行 ---
if __name__ == "__main__":

    # ▼▼▼ 設定をここで行う ▼▼▼
    execution_config = {
        # (必須) 変換したいvsqxファイルが入っているフォルダ
        "input_dir": "./data/vsqx",
        # (必須) 変換後のファイルを出力するベースフォルダ
        "output_base_dir": "./data/vsqx/vocalchange",
        # (必須) 1stボーカルの名前
        "vocal1": "MIKU_V4X_Original_EVEC",
        # (オプション) 2ndボーカルの名前
        "vocal2": "MIKU_V4X_Soft_EVEC",  # "random" または特定の名前に
    }
    # ▲▲▲ 設定はここまで ▲▲▲

    process_files(execution_config)
