# results/ — 実験結果

実験ごとの学習済みモデルと評価出力を保存するディレクトリ。  
フォルダ名は `config/` 内の `EXPERIMENT_NAME` に対応する。

## ディレクトリ構造

```
results/
├── exp_rms_only/               実験A: RMSのみ入力・SmallRMSNet
│   ├── models/
│   │   └── best_model.pth      Valid Loss が最良のモデル重み
│   └── output/
│       ├── training_loss.png   学習曲線グラフ
│       ├── eval_results.xlsx   テストセット評価結果（楽曲ごとの指標）
│       └── plots/              代表楽曲（Best/Q1/Median/Q3/Worst）の可視化
│
├── exp_diff_learning/          実験B（未実装）
│   ├── models/
│   └── output/
│
└── exp_multibranch/            実験C（未実装）
    ├── models/
    └── output/
```

## 評価指標（`eval_results.xlsx`）

| 列 | 説明 |
|----|------|
| `Song Name` | 楽曲名 |
| `Total Loss` | DYN MSE × 重み + Diff Loss × 重み（モデル保存基準） |
| `MSE_DYN` | DYN推定の平均二乗誤差（正規化スケール） |
| `Diff_DYN` | DYNの隣接フレーム差分誤差（変化追従性の指標） |
| `MAE_DYN (Raw)` | 0〜127スケールでの平均絶対誤差（直感的な誤差） |
| `Valid Frames` | 評価に使用した有音フレーム数 |

## モデルの読み込み方

```python
import config.exp_rms_only as cfg
from models.model_factory import get_model
import torch

model = get_model(cfg)
model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
```
