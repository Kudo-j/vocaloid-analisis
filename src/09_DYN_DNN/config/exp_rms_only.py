"""
exp_rms_only.py — 実験A: RMSのみ入力・軽量モデル
config_base の全定数を継承し、実験A固有の設定を上書きする。
"""
from config.config_base import *  # noqa: F401,F403
from config.config_base import _PROJECT_ROOT, _09_ROOT
import os

# =============================================================================
# ===== 実験A固有設定 =====
# =============================================================================
EXPERIMENT_NAME = "exp_rms_only"

# 使用する入力特徴量
INPUT_FEATURES = ["rms"]

# =============================================================================
# ===== モデル設計設定 =====
# =============================================================================
# モデルの種類: "small_rms", "baseline_single" など
MODEL_TYPE = "small_rms"

# BiLSTMの設定（SmallRMSNet用）
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 2

# Transformerの設定（将来の実験用）
TF_D_MODEL = 128
TF_NHEAD = 4
TF_NUM_LAYERS = 3

# スパイク防止用の学習可能な平滑化フィルタの範囲
SMOOTHING_KERNEL_SIZE = 15

# =============================================================================
# ===== 学習ハイパーパラメータ（実験A固有の上書き） =====
# =============================================================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 2e-4

# サボり（Lazy Prediction）と過学習を防ぐためのUntunedデータ混合割合
# 0.0 ならTunedデータのみで学習。0.2ならバッチ内の20%をUntunedデータに差し替える
UNTUNED_MIX_RATIO = 0.1

# =============================================================================
# ===== パス設定（実験A固有） =====
# =============================================================================
# 前処理済みデータの保存先（プロジェクトルート直下）
PREPROCESSED_DIR = os.path.join(_PROJECT_ROOT, "processed_09_standard")

# 結果保存先（09_DYN_DNN内）
RESULTS_DIR = os.path.join(_09_ROOT, "results", EXPERIMENT_NAME)
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "output")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# 評価結果のExcel保存先
EVAL_OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "eval_results.xlsx")
