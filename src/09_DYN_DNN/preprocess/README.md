# preprocess/ — 前処理スクリプト

前処理済みデータ（`.npz`）を生成するスクリプトを格納するディレクトリ。  
生成されたデータは **プロジェクトルート直下** に保存される（メモリ対策）。

| 実験 | 使用スクリプト | 保存先 |
|------|--------------|--------|
| A（RMSのみ） | `preprocess_standard.py` | `processed_09_standard/` |
| B（差分学習） | `preprocess_diff.py`（未実装） | `processed_09_diff/` |
| C（マルチブランチ） | `preprocess_standard.py` | `processed_09_standard/` |

## ファイル一覧

| ファイル | 役割 |
|---------|------|
| `preprocess_standard.py` | 実験A/C用。tuned WAV+VSQXペアから特徴量を抽出しnpzに保存 |
| `preprocess_diff.py` | 実験B用（未実装）。tuned/untunedペアの差分データを生成 |

## `preprocess_standard.py` の処理フロー

```
tuned WAV + tuned VSQX
        ↓
[Pass 1] 全ファイルをスキャンして全体統計（平均・標準偏差）を計算
        ↓
scaler.json 保存（推論時にも同じスケールで標準化するため）
        ↓
[Pass 2] Zスコア標準化した特徴量 + DYN正規化 + Voicing Mask を .npz に保存
        ↓
processed_09_standard/{ファイル名}.npz
```

## `.npz` ファイルの構造（`exp_rms_only` の場合）

| キー | 形状 | 内容 |
|-----|------|------|
| `rms` | `(T, 1)` | RMS特徴量（Zスコア標準化済み） |
| `dyn` | `(T, 1)` | DYNパラメータ（`[-1, 1]` に正規化） |
| `voicing_mask` | `(T, 1)` | 有音区間マスク（ノートベース、0 or 1） |

`INPUT_FEATURES` に複数の特徴量が指定された場合は、それぞれのキーで保存される。  
例: `["mel", "rms"]` → `mel`, `rms`, `dyn`, `voicing_mask`

## 実行方法

スクリプト先頭の import を使用したい config に切り替えてから実行する。

```python
# preprocess_standard.py 先頭
import config.exp_rms_only as cfg  # ← ここを変更
```

```
python preprocess/preprocess_standard.py
```

## 注意事項

- 既に処理済みのファイルはスキップされる（再実行時は `processed_09_*/` 内の `.npz` を手動削除して再生成）
- `chunk_cache.json` がある場合はスキャンをスキップする（高速化）
- データ分割ログは `dataset_split_log.txt` に保存される
