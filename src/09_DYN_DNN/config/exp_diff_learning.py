"""
exp_diff_learning.py — 実験B: 差分学習モデル（DiffNet）
Tuned音声・Untuned音声・その差分の3入力からTunedのDYNを推定する。
exp_rms_only.py を完全コピーし、実験B固有の設定を上書き。
"""

from config.config_base import *  # noqa: F401,F403
from config.config_base import _PROJECT_ROOT, _09_ROOT
import os

# =============================================================================
# ===== 実験B固有設定 =====
# =============================================================================
EXPERIMENT_NAME = "exp_diff_learning"

# 使用する入力特徴量
# Tuned / Untuned / Diff の3ブランチそれぞれに同じ特徴量が使われる
# ここを変更するだけで ["rms", "mel"] などの複数特徴量にも対応
INPUT_VSQX_PARAMS = []
INPUT_FEATURES = ["mel"]

# =============================================================================
# ===== モデル設計設定 =====
# =============================================================================
# モデルの種類（DiffNet: Tuned/Untuned/Diff の3系統ブランチ構成）
MODEL_TYPE = "diff_net"

# --- ブランチ設定（結合前: 次元調整・空間特徴抽出） ---
# 各特徴量のブランチタイプ: "linear" or "cnn"
# 未指定のキーは "linear" にフォールバック
FEATURE_BRANCH_TYPES = {"mel": "cnn"}

# ブランチ出力次元（特徴量ごとに個別指定も可能）
BRANCH_HIDDEN_SIZE = 32
# FEATURE_HIDDEN_SIZES = {"mel": 64, "rms": 16}

# CNN ブランチの設定（FEATURE_BRANCH_TYPES で "cnn" を指定した特徴量に適用）
CNN_NUM_FILTERS = 32
CNN_KERNEL_SIZE = 5
CNN_NUM_LAYERS = 2

# --- メインモデル設定（結合後: 時系列学習） ---
# "bilstm" or "transformer"
MAIN_MODEL_TYPE = "bilstm"

# BiLSTM メインモデルの設定
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2

# Transformer メインモデルの設定（MAIN_MODEL_TYPE = "transformer" 時に使用）
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 4

# スパイク防止用の学習可能な平滑化フィルタの範囲
SMOOTHING_KERNEL_SIZE = 9

# =============================================================================
# ===== 学習ハイパーパラメータ =====
# =============================================================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 2e-4

# Untunedデータ混合割合（0.0ならTunedデータのみ）
UNTUNED_MIX_RATIO = 0.0  # 実験BではペアデータでUntunedが学習に組み込まれるため0で良い

# =============================================================================
# ===== パス設定（実験B固有） =====
# =============================================================================
# 前処理済みデータの保存先
# ※ 4点ペア（Tuned WAV/VSQX + Untuned WAV/VSQX）で前処理したデータのパス
PREPROCESSED_DIR = os.path.join(_PROJECT_ROOT, "processed_09_diff_rms")

# 結果保存先
RESULTS_DIR = os.path.join(_09_ROOT, "results", EXPERIMENT_NAME)
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "output")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# 評価結果のExcel保存先
EVAL_OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "eval_results.xlsx")
