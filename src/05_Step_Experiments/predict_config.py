"""
predict_config.py
=================
predict.py と plot_rendered_eval.py が共通で参照する設定ファイル。
対象曲名やパスの変更はこのファイルだけで行う。
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
#  config 領域
# =============================================================================

# --- 推論対象 ---
INPUT_FILE_NAME = "Test_い_9"  # 対象曲のファイル名 (拡張子なし)

# --- 入力データ検索ディレクトリ ---
DATA_SEARCH_DIR = "./data/Step2"  # 正解WAV/VSQXの検索フォルダ (配下を再帰検索)

# --- 使用モデルディレクトリ ---
# best_model.pth, train_config.json, features.json が置かれているフォルダを指定
MODEL_DIR = os.path.join(BASE_DIR, "output", "predict", "config")

# --- 出力先 ---
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "predict", "Step2")

# --- グラフ描画設定 ---
REMOVE_INTRO_SILENCE = True  # イントロの無音区間をカットしてプロットするかどうか (Falseでカットしない)

# --- チャンク推論設定 (スライディングウィンドウ・オーバーラップ加算平均) ---
CHUNK_SIZE = 600  # 1回の推論フレーム数 (dataset.py の CHUNK_SIZE と合わせる)
CHUNK_HOP = 300  # シフト幅 (dataset.py の CHUNK_HOP と合わせる)

# =============================================================================
# config 領域ここまで
# =============================================================================
