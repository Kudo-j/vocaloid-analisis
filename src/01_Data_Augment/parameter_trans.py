import os
import glob
import logging
from pathlib import Path
import re
import random

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- 定数設定 ---
PARAM_ID = "D"  # 対象パラメータ (D=DYN)
CENTER_VALUE = 64  # DYNの中心値
MIN_VALUE = 0  # 最小値
MAX_VALUE = 127  # 最大値

# --- 正規表現 ---
# <v id="D">数値</v> を見つけるパターン
RE_DYN_TAG = re.compile(rf'(<v id="{PARAM_ID}">)(\d+)(</v>)')


def clamp_int(value: float) -> int:
    """数値を0-127の整数範囲に収める"""
    v = int(round(value))
    return max(MIN_VALUE, min(MAX_VALUE, v))


def linear_transform_vsqx(
    file_path: str, output_dir: str, scale_range: tuple, shift_range: tuple
):
    try:
        # ファイル読み込み
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # ファイルごとにランダムな変換係数を決定
        # scale: 1.0より大きいとメリハリ強調、小さいと平坦化
        scale = random.uniform(scale_range[0], scale_range[1])
        # shift: プラスだと全体的に大きく、マイナスだと小さく
        shift = random.randint(shift_range[0], shift_range[1])

        # 変換ロジック関数
        def _transform_match(match):
            original_val = int(match.group(2))

            # 線形変換計算: (元の値 - 中心) * 倍率 + 中心 + オフセット
            transformed_val = (
                (original_val - CENTER_VALUE) * scale + CENTER_VALUE + shift
            )

            final_val = clamp_int(transformed_val)

            # 元のタグ形式に戻す (<v id="D">新値</v>)
            return f"{match.group(1)}{final_val}{match.group(3)}"

        # 置換実行
        new_content, count = RE_DYN_TAG.subn(_transform_match, content)

        if count == 0:
            logging.warning(
                f"  [Skip] DYNパラメータが見つかりませんでした: {Path(file_path).name}"
            )
            return

        # ファイル名生成 (例: original_linear_s1.2_sh5.vsqx)
        original_name = Path(file_path).stem
        scale_str = f"{scale:.2f}"
        shift_str = f"{shift:+d}"
        output_filename = f"{original_name}_lin_x{scale_str}_{shift_str}.vsqx"
        output_path = os.path.join(output_dir, output_filename)

        # 保存
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        logging.info(
            f"  [OK] {original_name} -> Scale:{scale:.2f}, Shift:{shift:+d} (変更数: {count})"
        )

    except Exception as e:
        logging.error(f"  [Error] {file_path}: {e}")


def main():
    # ▼▼▼ 設定エリア ▼▼▼
    config = {
        # 入力フォルダ（元のvsqxがある場所）
        "input_dir": "./data/vsqx",
        # 出力フォルダ
        "output_dir": "./data/vsqx/augment/linear_aug",
        # 倍率の範囲 (min, max)
        # 例: (0.5, 1.5) -> 半分の平坦さ ～ 1.5倍のメリハリ の間でランダム
        "scale_range": (0.5, 1.5),
        # 加算値の範囲 (min, max)
        # 例: (-20, 20) -> 全体的に-20 ～ +20 の範囲で音量をずらす
        "shift_range": (-20, 20),
    }
    # ▲▲▲ 設定エリア ▲▲▲

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    # 出力フォルダ作成
    os.makedirs(output_dir, exist_ok=True)

    # VSQXファイル取得
    vsqx_files = glob.glob(os.path.join(input_dir, "*.vsqx"))

    if not vsqx_files:
        logging.error("入力フォルダに.vsqxファイルが見つかりません。")
        return

    logging.info(f"処理開始: {len(vsqx_files)} 件のファイルを処理します...")
    logging.info(
        f"Scale範囲: {config['scale_range']}, Shift範囲: {config['shift_range']}"
    )
    logging.info("-" * 40)

    for vsqx_file in vsqx_files:
        linear_transform_vsqx(
            vsqx_file, output_dir, config["scale_range"], config["shift_range"]
        )

    logging.info("-" * 40)
    logging.info("全処理完了")


if __name__ == "__main__":
    main()
