import os
import glob
import logging
from copy import deepcopy
from lxml import etree

logging.basicConfig(level=logging.INFO, format="%(message)s")

VSQ4_NS = "http://www.yamaha.co.jp/vocaloid/schema/vsq4/"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
NSMAP = {"vsq4": VSQ4_NS}


def _tag(local_name):
    """名前空間付きのタグ名を生成する"""
    return f"{{{VSQ4_NS}}}{local_name}"


def _local(element):
    """要素のローカルタグ名を取得する"""
    tag = element.tag
    if "}" in tag:
        return tag.split("}")[-1]
    return tag


def _get_t(element):
    """要素内の <t> の値を整数で取得する"""
    t_elem = element.find(_tag("t"))
    if t_elem is not None and t_elem.text is not None:
        return int(t_elem.text)
    return 0


def _set_t(element, value):
    """要素内の <t> の値を設定する"""
    t_elem = element.find(_tag("t"))
    if t_elem is not None:
        t_elem.text = str(value)


# vsPart内で最初のvsPartからのみ採用するタグ (重複不要)
UNIQUE_TAGS = {"t", "playTime", "name", "comment", "sPlug", "pStyle", "singer"}

# タグの並び順 (VOCALOID 4 の標準)
TAG_ORDER = {
    "t": 0,
    "playTime": 1,
    "name": 2,
    "comment": 3,
    "sPlug": 4,
    "pStyle": 5,
    "singer": 6,
    "cc": 7,
    "note": 8,
    "plane": 9,
}


def merge_vsparts_in_file(input_file, output_file):
    """
    1つのVSQXファイルを読み込み、トラック内の複数のvsPartを1つのvsPartに統合する。
    lxml を使用して CDATA セクションとインデントを保持する。
    """
    # strip_cdata=False で CDATA セクションを保持してパース
    parser = etree.XMLParser(
        strip_cdata=False,
        remove_blank_text=False,
        resolve_entities=False,
    )
    try:
        tree = etree.parse(input_file, parser)
    except Exception as e:
        logging.error(f"[Error] Failed to parse {input_file}: {e}")
        return False

    root = tree.getroot()
    modified = False

    for track in root.findall(_tag("vsTrack")):
        parts = track.findall(_tag("vsPart"))
        if len(parts) <= 1:
            continue

        modified = True
        base_part = parts[0]

        # ベースの開始時間
        base_t = _get_t(base_part)

        # 全体の終了時間を計算
        max_end = 0
        for p in parts:
            p_t = _get_t(p)
            p_play = int(p.find(_tag("playTime")).text)
            end = p_t + p_play
            if end > max_end:
                max_end = end

        new_play_time = max_end - base_t

        # ----- 統合する要素を収集 -----
        # unique_elements: 最初のvsPartから取る固有タグ
        # merged_cc: 全vsPartから集めるcc (ソートキー付き)
        # merged_note: 全vsPartから集めるnote (ソートキー付き)
        # plane_elem: planeは最初のvsPartから1つだけ

        unique_elements = []  # (order, element)
        merged_cc = []  # (global_t, element)
        merged_note = []  # (global_t, element)
        plane_elem = None

        for i, p in enumerate(parts):
            p_t = _get_t(p)
            offset = p_t - base_t  # ベースからのオフセット

            for child in p:
                tag_name = _local(child)

                if tag_name == "plane":
                    # plane は最初のvsPartからのみ採用
                    if i == 0:
                        plane_elem = deepcopy(child)
                    continue

                if tag_name in UNIQUE_TAGS:
                    # 固有タグは最初のvsPartからのみ
                    if i == 0:
                        elem_copy = deepcopy(child)
                        unique_elements.append((TAG_ORDER.get(tag_name, 99), elem_copy))
                    continue

                if tag_name == "cc":
                    elem_copy = deepcopy(child)
                    local_t = _get_t(elem_copy)
                    global_t = local_t + offset
                    _set_t(elem_copy, global_t)
                    merged_cc.append((global_t, elem_copy))

                elif tag_name == "note":
                    elem_copy = deepcopy(child)
                    local_t = _get_t(elem_copy)
                    global_t = local_t + offset
                    _set_t(elem_copy, global_t)
                    merged_note.append((global_t, elem_copy))

        # ccとnoteをそれぞれ時間順でソート
        merged_cc.sort(key=lambda x: x[0])
        merged_note.sort(key=lambda x: x[0])

        # ----- playTime を更新 -----
        for elem in unique_elements:
            if _local(elem[1]) == "playTime":
                elem[1].text = str(new_play_time)
                break

        # ----- base_part を再構成 -----
        # 元のインデント情報を取得
        base_text = base_part.text  # 通常は改行+インデント
        base_tail = base_part.tail

        # base_part の子要素を全削除
        for child in list(base_part):
            base_part.remove(child)

        # 固有タグを順番通り追加
        unique_elements.sort(key=lambda x: x[0])
        for _, elem in unique_elements:
            base_part.append(elem)

        # cc を追加
        for _, elem in merged_cc:
            base_part.append(elem)

        # note を追加
        for _, elem in merged_note:
            base_part.append(elem)

        # plane を末尾に追加
        if plane_elem is not None:
            base_part.append(plane_elem)

        # text/tail を復元
        base_part.text = base_text
        base_part.tail = base_tail

        # 2番目以降の vsPart をトラックから削除
        for p in parts[1:]:
            track.remove(p)

    if not modified:
        # 変更がなくてもファイルは出力する
        pass

    # 出力ディレクトリの作成
    dest_dir = os.path.dirname(output_file)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # ファイル書き込み (CDATA 保持、XML宣言付き)
    # lxml は XML宣言にシングルクォートを使うため、手動でダブルクォートの宣言を書く
    xml_bytes = etree.tostring(
        root,
        encoding="UTF-8",
        xml_declaration=False,
        pretty_print=False,
    )
    with open(output_file, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write(xml_bytes)

    return modified


def process_directory(config: dict):
    """
    フォルダ内の.vsqxファイルを検索し、統合処理を実行する
    """
    input_dir = config.get("input_dir")
    output_dir = config.get("output_dir")

    if not all([input_dir, output_dir]):
        logging.error("エラー: config設定が不足しています。")
        return

    logging.info(f"入力フォルダ: {input_dir}")
    logging.info(f"出力フォルダ: {output_dir}")

    search_pattern = os.path.join(input_dir, "**", "*.vsqx")
    vsqx_files = glob.glob(search_pattern, recursive=True)

    if not vsqx_files:
        logging.warning("警告: .vsqx ファイルが見つかりません。")
        return

    logging.info(f"{len(vsqx_files)} 件のファイルを処理します...")

    processed_count = 0

    for vsqx_path in vsqx_files:
        relative_path = os.path.relpath(vsqx_path, start=input_dir)
        dest_path = os.path.join(output_dir, relative_path)

        modified = merge_vsparts_in_file(vsqx_path, dest_path)
        if modified:
            logging.info(f"[統合実行] {relative_path}")
        else:
            logging.info(f"[変更なし] {relative_path}")

        processed_count += 1

    logging.info("-" * 30)
    logging.info(f"処理完了: {processed_count} 件")


# --- スクリプト実行 ---
if __name__ == "__main__":
    test_input = "daze_melo_IAROCKS_Original_Soft.vsqx"
    test_output = "merged_daze_melo.vsqx"
    if os.path.exists(test_input):
        logging.info("--- 統合処理の開始 ---")
        success = merge_vsparts_in_file(test_input, test_output)
        if success:
            logging.info(f"成功: {test_output} を生成しました。")
        else:
            logging.info("変更の必要はありませんでした。")

    # 一括処理用設定
    execution_config = {
        "input_dir": "./data/vsqx",
        "output_dir": "./data/vsqx/vspart_merged",
    }
    process_directory(execution_config)
