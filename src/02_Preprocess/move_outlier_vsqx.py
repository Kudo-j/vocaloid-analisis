"""
move_outlier_vsqx.py
時間同期に失敗した（outlier）ファイルに対応する .vsqx ファイルを
移動元フォルダから移動先フォルダへ、サブフォルダ構造を維持して移動するスクリプト。
"""

import os
import shutil
from pathlib import Path

# ============================================================
# Config 領域 — ここでパスを指定してください
# ============================================================

# 指定フォルダ: ずれたファイル (.npz 等) があるフォルダ
OUTLIER_DIR = r"d:\黒岩研究室\卒研\26Voca_Ana\src_result\augDYN_DYN_srcrms\outlier"

# 移動元フォルダ: 元となる .vsqx が入ったフォルダ（サブフォルダも含めて探索）
SRC_VSQX_DIR = r"d:\黒岩研究室\卒研\26Voca_Ana\src_result\use_vsqx"

# 移動先フォルダ: ずれている .vsqx を移動させる先のフォルダ
DEST_VSQX_DIR = r"d:\黒岩研究室\卒研\26Voca_Ana\src_result\use_vsqx_outlier"

# 見つからなかったファイルが 10 個以上ある場合に出力するテキストファイルのパス
MISSING_LOG_FILE = r"d:\黒岩研究室\卒研\26Voca_Ana\src_result\missing_vsqx_list.txt"

# ============================================================


def main():
    outlier_dir = Path(OUTLIER_DIR)
    src_vsqx_dir = Path(SRC_VSQX_DIR)
    dest_vsqx_dir = Path(DEST_VSQX_DIR)

    # --- 1. 対象ファイル名のリストアップ ---
    if not outlier_dir.exists():
        print(f"[エラー] 指定フォルダが存在しません: {outlier_dir}")
        return

    target_stems = set()
    for f in outlier_dir.iterdir():
        if f.is_file():
            target_stems.add(f.stem)

    if not target_stems:
        print("[情報] 指定フォルダにファイルがありません。処理を終了します。")
        return

    print(f"[情報] outlier フォルダから {len(target_stems)} 個のファイル名を取得しました。")

    # --- 2. VSQX ファイルの探索と移動 ---
    if not src_vsqx_dir.exists():
        print(f"[エラー] 移動元フォルダが存在しません: {src_vsqx_dir}")
        return

    found_stems = set()
    moved_count = 0

    for vsqx_path in src_vsqx_dir.rglob("*.vsqx"):
        if vsqx_path.stem in target_stems:
            # 相対パスを計算してサブフォルダ構造を維持
            rel_path = vsqx_path.relative_to(src_vsqx_dir)
            dest_path = dest_vsqx_dir / rel_path

            # 移動先サブフォルダを作成
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # 移動
            shutil.move(str(vsqx_path), str(dest_path))
            print(f"  移動: {rel_path}")

            found_stems.add(vsqx_path.stem)
            moved_count += 1

    # --- 3. 結果の出力 ---
    print(f"\n{'='*60}")
    print(f"[結果] {moved_count} 個の .vsqx ファイルを移動しました。")

    # 見つからなかったファイルの処理
    missing_stems = sorted(target_stems - found_stems)

    if not missing_stems:
        print("[結果] すべての対応する .vsqx ファイルが見つかりました。")
    elif len(missing_stems) < 10:
        print(f"\n[警告] 以下の {len(missing_stems)} 個のファイルに対応する .vsqx が見つかりませんでした:")
        for stem in missing_stems:
            print(f"  - {stem}")
    else:
        # 10 個以上の場合はテキストファイルに出力
        log_path = Path(MISSING_LOG_FILE)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"対応する .vsqx が見つからなかったファイル一覧 ({len(missing_stems)} 件)\n")
            f.write("=" * 60 + "\n")
            for stem in missing_stems:
                f.write(f"{stem}\n")

        print(f"\n[警告] {len(missing_stems)} 個のファイルに対応する .vsqx が見つかりませんでした。")
        print(f"  一覧をテキストデータとして出力しました: {log_path}")


if __name__ == "__main__":
    main()
