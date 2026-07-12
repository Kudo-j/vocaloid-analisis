# 09_DYN_DNN — DYN推定実験ディレクトリ

DYNパラメータ推定のための各種モデル・入力特徴・学習方式を試す実験ディレクトリ。

## 実験一覧

| 実験 | config | 概要 |
|------|--------|------|
| **A** | `exp_rms_only` | RMSのみ入力・軽量モデル（`SmallRMSNet`） |
| **B** | `exp_diff_learning` | tuned/untunedペアの差分から学習（未実装） |
| **C** | `exp_multibranch` | 複数特徴量をブランチ分岐で入力（未実装） |

## ディレクトリ構造

```
09_DYN_DNN/
├── config/               実験ごとの設定ファイル
├── models/               モデル定義・Factory
├── preprocess/           前処理スクリプト
├── results/              学習済みモデル・評価結果（実験ごとに分類）
├── Ref/                  旧バージョンのファイル（参照用）
│
├── dataset_09.py         共通データセットクラス（DYNDataset）
├── train.py              学習スクリプト
├── eval.py               評価スクリプト
└── predict.py            推論スクリプト
```

## 実行手順

**1. 実験configの選択**
`train.py`, `eval.py`, `predict.py` それぞれの先頭にある import 文を切り替える。

```python
import config.exp_rms_only as cfg   # ← ここを変更
```

**2. 前処理**
```
python preprocess/preprocess_standard.py
```

**3. 学習**
```
python train.py
```

**4. 評価**
```
python eval.py
```

**5. 推論（単一ファイル）**
```
python predict.py
```

## 依存関係

| モジュール | 場所 |
|-----------|------|
| `feature_extractor.py` | `src_result/util/` |
| `vsqx_reader.py` | `src/04_Parameter_Study/src/` |
| 前処理済みデータ | `src_result/processed_09_*/` |
