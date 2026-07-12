# config/ — 実験設定ファイル

実験ごとに異なる設定を管理するディレクトリ。  
`train.py` / `eval.py` / `predict.py` の先頭 import を切り替えることで実験を選択する。

## ファイル一覧

| ファイル | 役割 |
|---------|------|
| `config_base.py` | 全実験共通の定数・パス（SR, HOP_LENGTH, データパス等） |
| `exp_rms_only.py` | 実験A: RMSのみ入力・軽量モデル（`SmallRMSNet`） |

## 実験configの追加方法

新しい実験を追加する場合は、`config_base.py` を import して上書きする。

```python
# 新しい実験configの雛形
from config.config_base import *  # noqa

EXPERIMENT_NAME = "exp_new"
INPUT_FEATURES  = ["mel", "rms"]
MODEL_TYPE      = "baseline_single"

PREPROCESSED_DIR = os.path.join(_PROJECT_ROOT, "processed_09_new")
RESULTS_DIR      = os.path.join(_09_ROOT, "results", EXPERIMENT_NAME)
MODEL_DIR        = os.path.join(RESULTS_DIR, "models")
OUTPUT_DIR       = os.path.join(RESULTS_DIR, "output")
MODEL_PATH       = os.path.join(MODEL_DIR, "best_model.pth")
EVAL_OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "eval_results.xlsx")
```

## 主要な設定項目

| 設定項目 | 説明 | 例 |
|---------|------|-----|
| `INPUT_FEATURES` | 使用する入力特徴量のリスト | `["rms"]`, `["mel", "rms"]` |
| `MODEL_TYPE` | 使用するモデル種別 | `"small_rms"`, `"baseline_single"`, `"multibranch"` |
| `PREPROCESSED_DIR` | 前処理済みデータの保存先 | `src_result/processed_09_standard/` |

## パス構造

`config_base.py` が定義するパス変数：

| 変数 | 指す場所 |
|-----|---------|
| `_09_ROOT` | `09_DYN_DNN/` |
| `_SRC_DIR` | `src/` |
| `_SRC_RESULT_DIR` | `src_result/` |
| `_PROJECT_ROOT` | プロジェクトルート（`src_result/` の一つ上） |
| `UTIL_DIR` | `src_result/util/`（`feature_extractor.py` の場所） |
