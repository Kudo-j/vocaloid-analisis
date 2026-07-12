# models/ — モデル定義

DYN推定に使用するモデルを定義するディレクトリ。  
全モデルは **dict → dict** の統一インターフェースを持つ。

```python
# 呼び出し側は常にこの形式（モデルの内部構造を知らなくてよい）
inputs  = {"rms": tensor}          # または {"mel": tensor, "rms": tensor} など
outputs = model(inputs)            # {"dyn_pred": (B, T, 1)}
```

## ファイル一覧

| ファイル | 役割 |
|---------|------|
| `encoders.py` | 基本エンコーダ群（BiLSTM, Transformer, DYN_Predictor） |
| `model_factory.py` | `get_model(config)` Factory関数、各モデルクラスを定義 |

## 実装済みモデル

### `SmallRMSNet`（実験A用）
- **入力**: `{"rms": (B, T, 1)}`
- **構造**: BiLSTM(hidden=32, 2層) → Conv1d平滑化 → Linear
- **パラメータ数**: 約2万（現行Transformerモデルの約1/25）
- **目的**: 1次元RMS入力に対して過剰なモデル容量を避けた軽量設計

### `BaselineSingleNet`（汎用）
- **入力**: `config.INPUT_FEATURES` で指定した特徴量を結合した dict
- **構造**: BiLSTM or Transformer → DYN_Predictor
- **目的**: 複数特徴量を単純結合して推定する汎用ベースライン

## モデルの追加方法

1. `model_factory.py` に新しいクラスを定義する
2. `get_model()` 内に `elif model_type == "new_model":` の分岐を追加する
3. 対応する実験configで `MODEL_TYPE = "new_model"` を設定する

```python
class NewModel(nn.Module):
    def forward(self, inputs):
        # inputs: dict of {feature_name: (B, T, D)}
        # ...
        return {"dyn_pred": dyn_pred}  # 出力は常にdictで統一
```

## `encoders.py` に定義されているクラス

| クラス | 説明 |
|-------|------|
| `BiLSTM_Encoder` | 双方向LSTM エンコーダ |
| `Transformer_Encoder` | Positional Encoding付き Transformer エンコーダ |
| `DYN_Predictor` | Conv1d平滑化フィルタ付き 線形出力ヘッド |
| `PositionalEncoding` | Transformer用位置エンコーディング |
