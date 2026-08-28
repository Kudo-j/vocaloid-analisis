# 05_Step_Experiments 実行マニュアル

本ディレクトリは、Vocaloidの歌声パラメータ予測モデルの学習、自動ハイパーパラメータ探索、および学習済みモデルを用いた推論とレンダリング後の詳細評価を実行するための作業領域である。

## ディレクトリ構成と役割

- **`train.py`**
  DNNモデル（Bi-LSTM、CNN-Bi-LSTM）の統合学習およびテストデータ評価スクリプト。`train_settings.json` の設定に従って学習を行う。
- **`train_settings.json`**
  学習時のモデル構造、結合戦略、バッチサイズ、エポック数、学習率などのハイパーパラメータを指定する設定ファイル。
- **`_README_train_settings.md`**
  `train_settings.json` の各設定項目（SPLIT_MODE、MODEL_TYPE、FUSION_STRATEGYなど）に関するマニュアル。
- **`tune_optuna.py`**
  Optunaを用いたDNNモデルの自動ハイパーパラメータ探索スクリプト。`search_space.json` で探索空間を定義し、スリープ防止機能を備えつつグリッド/ランダム探索を実行する。
- **`search_space.json`**
  Optunaで最適化するパラメータの範囲や探索対象モデルを定義する設定ファイル。
- **`analyze_linear.py`**
  メモリ節約型（アウトオブコア）の線形回帰学習および評価スクリプト。大規模データを1曲ずつ累積計算することで、メモリ枯渇を防ぎつつ最適解を算出する。
- **`predict_linear.py`**
  学習済みの線形回帰モデル（.pkl）を使用して、WAV + VSQX からパラメータ予測を行い、予測VSQXと評価プロットを出力するスクリプト。
- **`predict_config.py`**
  DNNモデルの推論（`predict.py`）およびレンダリング後評価（`plot_rendered_eval.py`）で共通して使用する設定ファイル。対象曲名やパスを一元管理する。
- **`predict.py`**
  学習済みのDNNモデル（.pth）を用いて推論を行うスクリプト。前処理で生成された `features.json` をロードし、スライディングウィンドウ（Overlap-Add）方式でオーバーラップ加算平均を行いながら推論を実施後、予測値を書き込んだVSQXを出力する。
- **`plot_rendered_eval.py`**
  予測VSQXをVocaloidソフトで音声化（レンダリング）した結果（予測WAV）と正解データを詳細に比較するスクリプト。上段にDYNパラメータ、下段にRMS音響特徴量の比較波形をプロットする。
- **`models/`**
  ニューラルネットワークモデルの定義コード群を格納するディレクトリ。

## 設定ファイルの取り扱いについて（Git管理外）

本ディレクトリ内の設定ファイルのうち、以下のファイルは頻繁に変更されるローカル環境依存の情報を含むため、`.gitignore` によりGitの追跡対象外となっている。
- `train_settings.json`
- `search_space.json`
- `predict_config.py`

リポジトリを新規にクローンした環境で実行する際は、各ファイルのテンプレートである以下を同じディレクトリ内にコピーし、拡張子 `.template` を除いたファイル名に変更した上で使用すること。
- `train_settings.json.template` -> `train_settings.json` にコピー
- `search_space.json.template`  -> `search_space.json` にコピー
- `predict_config.py.template`  -> `predict_config.py` にコピー
