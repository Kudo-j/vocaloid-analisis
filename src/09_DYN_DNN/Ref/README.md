# Ref/ — 旧バージョンファイル（参照用）

`09_DYN_DNN` ディレクトリの旧実装ファイルを退避したフォルダ。  
現在の実験では使用しないが、過去の実装の参照用として保存。

## 格納ファイル一覧

| ファイル | 旧役割 |
|---------|--------|
| `config.py` | 旧設定ファイル（SR, HOP_LENGTH, パラメータID等の定義） |
| `dataset.py` | 旧データセットクラス |
| `train_encoder.py` | 旧学習スクリプト（エンコーダベースのモデル） |
| `eval_Test_enco.py` | 旧評価スクリプト |
| `predict_en.py` | 旧推論スクリプト |
| `model_encoder.py` | 旧モデル定義（エンコーダ） |
| `preprocess.py` | 旧前処理スクリプト（`09_DYN_DNN/preprocess.py`） |
| `check_timelag.py` | 旧時間ラグ確認スクリプト |

## 注意事項

これらのファイルは現在の `config/`, `models/`, `dataset_09.py` とは**互換性がない**。  
参照のみを目的とし、直接実行することは推奨しない。

現行の実装は `09_DYN_DNN/` 直下の以下のスクリプトを使用する：
- `train.py` / `eval.py` / `predict.py`
- `dataset_09.py`
- `config/exp_rms_only.py`（実験A）
