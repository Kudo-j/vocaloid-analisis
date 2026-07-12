"""
pair_finder.py
==============
WAVディレクトリとVSQXディレクトリを再帰的に走査し、
ファイル名（拡張子なし）が一致するペアを自動で検出するユーティリティ。

preprocess.py のペアリング方式を踏襲。
VSQXを基準にし、対応するWAVが見つかったペアのみを返す。
"""

import os


def find_pairs(wav_dir, vsqx_dir, verbose=True):
    """
    WAVディレクトリとVSQXディレクトリを再帰的に検索し、
    ファイル名（拡張子なし）が一致するペアを返す。

    VSQXを基準とし、対応するWAVが見つかったペアのみを返す。
    WAVは別フォルダにあっても良い（名前が一致すれば良い）。

    Args:
        wav_dir: WAVファイルのルートフォルダ
        vsqx_dir: VSQXファイルのルートフォルダ
        verbose: ログ出力するかどうか

    Returns:
        list of tuple: [(ベース名, WAVパス, VSQXパス), ...]
    """
    # WAVファイルの辞書 {basename: filepath}
    wav_files = {}
    for root, dirs, files in os.walk(wav_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                base_name = os.path.splitext(f)[0]
                wav_files[base_name] = os.path.join(root, f)

    # VSQXファイルの辞書 {basename: filepath}
    vsqx_files = {}
    for root, dirs, files in os.walk(vsqx_dir):
        for f in files:
            if f.lower().endswith(".vsqx"):
                base_name = os.path.splitext(f)[0]
                vsqx_files[base_name] = os.path.join(root, f)

    # ペアの作成（VSQXを基準）
    pairs = []
    missing_wav = 0
    for base_name, vsqx_path in sorted(vsqx_files.items()):
        if base_name in wav_files:
            pairs.append((base_name, wav_files[base_name], vsqx_path))
        else:
            missing_wav += 1

    if verbose:
        print(f"  WAVファイル数: {len(wav_files)}")
        print(f"  VSQXファイル数: {len(vsqx_files)}")
        print(f"  ペア成立: {len(pairs)}")
        if missing_wav > 0:
            print(f"  [情報] WAVが見つからないVSQX: {missing_wav} 件")

    return pairs
