# =============================================================================
# VSQX 母音置換ツール
# =============================================================================
# 「あ」のVSQXファイルをベースに、歌詞と音素を書き換えて
# 他の母音（い・う・え・お）のVSQXファイルを一括生成する。
#
# DYNカーブなどのパラメータ領域には一切触れず、
# ノート内の歌詞(<y>)と音素(<p>)のみを安全に置換する。
#
# 使い方:
#   1. 下の設定を確認する
#   2. python generate_vsqx_vowels.py を実行する
# =============================================================================

import os
import re
import shutil

# =============================================================================
# ===== 設定 =====
# =============================================================================

# ベースとなるVSQXファイル（「あ」のデータ）
# ファイル名に含まれる「あ」がターゲット母音に自動置換される
BASE_VSQX_DIR = "./data/Step_org"
BASE_FILES = [
    "あ_DYN_Train.vsqx",
    "あ_DYN_手動.vsqx",
]

# 出力先ディレクトリ（ベースと同じ場所に出力する場合は同じパスを指定）
OUTPUT_DIR = "./data/Step_org"

# 母音マッピング: {入力歌詞: 出力音素}
# ベースの「あ」は生成不要なので含めない
TARGET_VOWELS = {
    "い": "i",
    "う": "M",
    "え": "e",
    "お": "o",
}

# ベースファイルの元の歌詞と音素（置換元）
SOURCE_LYRIC = "あ"
SOURCE_PHONEME = "a"

# ドライラン: Trueにすると実際のファイル書き込みをせず、変更内容のみ表示する
DRY_RUN = False

# =============================================================================
# ===== メイン処理 =====
# =============================================================================


def replace_vsqx_vowel(xml_text, target_lyric, target_phoneme):
    """
    VSQX(XML)テキスト内のノートの歌詞と音素を置換する。

    CDATA セクション内の値のみを対象とするため、
    パラメータ(<cc>)やその他の構造には一切影響しない。

    Args:
        xml_text: VSQXファイルの内容（文字列）
        target_lyric: 置換先の歌詞（例: "い"）
        target_phoneme: 置換先の音素（例: "i"）

    Returns:
        置換後のXMLテキスト
    """
    # <y><![CDATA[あ]]></y> → <y><![CDATA[い]]></y>
    # 正規表現で <y> タグ内の CDATA を安全に置換
    xml_text = re.sub(
        r"(<y><!\[CDATA\[)" + re.escape(SOURCE_LYRIC) + r"(\]\]></y>)",
        r"\g<1>" + target_lyric + r"\2",
        xml_text,
    )

    # <p><![CDATA[a]]></p> → <p><![CDATA[i]]></p>
    xml_text = re.sub(
        r"(<p><!\[CDATA\[)" + re.escape(SOURCE_PHONEME) + r"(\]\]></p>)",
        r"\g<1>" + target_phoneme + r"\2",
        xml_text,
    )

    return xml_text


def main():
    print("=" * 55)
    print("  VSQX Vowel Replacement Tool")
    print("=" * 55)

    if DRY_RUN:
        print("  *** ドライランモード: ファイルは書き込まれません ***\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generated_count = 0
    skipped_count = 0

    for base_filename in BASE_FILES:
        base_path = os.path.join(BASE_VSQX_DIR, base_filename)

        if not os.path.exists(base_path):
            print(f"WARNING: ベースファイルが見つかりません: {base_path}")
            skipped_count += 1
            continue

        print(f"\n[Base] {base_filename}")

        # ベースファイルを読み込む
        with open(base_path, "r", encoding="utf-8") as f:
            original_xml = f.read()

        for target_lyric, target_phoneme in TARGET_VOWELS.items():
            # 出力ファイル名: 「あ」をターゲット母音に置換
            output_filename = base_filename.replace(SOURCE_LYRIC, target_lyric)
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # 置換実行
            new_xml = replace_vsqx_vowel(original_xml, target_lyric, target_phoneme)

            # 変更が実際に行われたか確認
            if new_xml == original_xml:
                print(
                    f"  WARNING: {target_lyric}({target_phoneme}) への置換が"
                    f"行われませんでした。ベースファイルの歌詞を確認してください。"
                )
                skipped_count += 1
                continue

            # 書き込み
            if not DRY_RUN:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(new_xml)

            print(
                f"  -> {output_filename}  "
                f"(歌詞: {SOURCE_LYRIC}->{target_lyric}, "
                f"音素: {SOURCE_PHONEME}->{target_phoneme})"
            )
            generated_count += 1

    # --- サマリー ---
    print("\n" + "=" * 55)
    print(f"  Complete!")
    print(f"  Generated: {generated_count} files")
    if skipped_count > 0:
        print(f"  Skipped:   {skipped_count}")
    print(f"  Output:    {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 55)

    if DRY_RUN:
        print(
            "\n  ドライランモードでした。実際に生成するには DRY_RUN = False にしてください。"
        )


if __name__ == "__main__":
    main()
