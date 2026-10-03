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
  メモリ節約型の線形回帰学習および評価スクリプト。大規模データを1曲ずつ累積計算することで、メモリ枯渇を防ぎつつ最適解を算出する。
- **`predict_linear.py`**
  学習済みの線形回帰モデル（.pkl）を使用して、WAV + VSQX からパラメータ予測を行い、予測VSQXと評価プロットを出力するスクリプト。
- **`predict_config.py`**
  DNNモデルの推論（`predict.py`）およびレンダリング後評価（`plot_rendered_eval.py`）で共通して使用する設定ファイル。対象曲名やパスを一元管理する。
- **`predict.py`**
  指定された `MODEL_DIR` 内の学習済みDNNモデル（`best_model.pth`）、設定（`train_config.json`）、および特徴量情報（`features.json`）をロードし、スライディングウィンドウ（Overlap-Add）方式でオーバーラップ加算平均を行いながら推論を実施後、予測値を書き込んだVSQXを出力する。
- **`plot_rendered_eval.py`**
  予測VSQXをVocaloidソフトで音声化（レンダリング）した結果（予測WAV）と正解データを詳細に比較するスクリプト。上段にDYNパラメータ、下段にRMS音響特徴量の比較波形をプロットする。
- **`evaluate_pitch_benchmark.py`**
  Step 3（音高多様化実験）専用の音高網羅ベンチマーク評価スクリプト。学習済みモデルを用いて全37音階×5母音（185曲）を推論し、正解vs予測の比較プロット（PNG）を保存するとともに、`pitch_randomize_log.csv` と突合して Seen（既知） vs Unseen（未知）や音高別・母音別の誤差（MSE/MAE/相関係数）を集計し、高品質なExcelレポートを出力する。
- **`models/`**
  ニューラルネットワークモデルの定義コード群を格納するディレクトリ。以下に詳述。

## models/ 内の各プログラムの説明

ニューラルネットワークモデルの構築・結合を行うモジュール群である。

### 1. `fusion_block.py`
複数系統の入力特徴量を結合・高次元化するための前処理モジュール。`pass_through`（単純結合）、`late_concat`（個別モジュール処理後に結合）、`late_add`（次元統一後に加算）の3つの融合戦略（`FUSION_STRATEGY`）に対応している。

### 2. `bilstm_base.py`
Bi-LSTM 単体で構成されるパラメータ予測モデルである。`train_settings.json` 等で `use_gate=True` が指定された場合、Bi-LSTM の出力に対して双方向ゲート層（`DirectionalGate`）を適用し、順方向と逆方向の相補的スケーリングを行う。

### 3. `cnn_bilstm.py`
1D-CNN と Bi-LSTM を組み合わせたモデルである。1D-CNN で局所特徴を抽出した後に Bi-LSTM で時系列の文脈を学習する。`bilstm_base.py` と同様に `use_gate` による双方向ゲート層の挿入に対応している。

### 4. `directional_gate.py`
Bi-LSTM の順方向・逆方向の出力を相補的にスケーリングする共通ゲートモジュールである。隠れ状態からフレームごとのスカラーゲート値（0.0～1.0）をシグモイド関数で算出し、順方向成分を乗算、逆方向成分をその補数（1 - gate）で乗算して再結合する。

### 5. `unified_model.py`
特徴量融合ブロック（`FeatureFusionBlock`）とメイン予測モデル（`core_model`）を1つにまとめる統合ラッパークラス。データの結合から予測値の出力までを一括処理し、モデルの保存や学習管理をシンプルにする。

## 設定ファイルの取り扱いについて（Git管理外）

本ディレクトリ内の設定ファイルのうち、以下のファイルは頻繁に変更されるローカル環境依存の情報を含むため、`.gitignore` によりGitの追跡対象外となっている。
- `train_settings.json`
- `search_space.json`
- `predict_config.py`

リポジトリを新規にクローンした環境で実行する際は、各ファイルのテンプレートである以下を同じディレクトリ内にコピーし、拡張子 `.template` を除いたファイル名に変更した上で使用すること。
- `train_settings.json.template` -> `train_settings.json` にコピー
- `search_space.json.template`  -> `search_space.json` にコピー
- `predict_config.py.template`  -> `predict_config.py` にコピー
