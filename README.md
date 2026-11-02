# VOCALOID パラメータ推定・分析システム

本プロジェクトは、DNNを用いたVOCALOIDの動的パラメータ推定、および音響特徴量と歌唱パラメータの相関関係を分析するための研究・開発プログラム群である。
システムは以下のモジュール（ディレクトリ）から構成されている。

## ディレクトリ構成

### 1. src/01_Data_Augment（データ拡張系）
少数のデータセットから学習用データを増やすためのデータ拡張処理を行う。VSQXの移調やボーカルチェンジ、パラメータの自動書き込みなどを自動化し、データ量を確保する。
詳細は [src/01_Data_Augment/_README_01DataAugment.md](src/01_Data_Augment/_README_01DataAugment.md) を参照。

### 2. src/02_Preprocess（データチェック・前処理系）
データセットの品質を担保するため、VSQXファイルやWAVファイルの形式チェック、マルチブランチ対応の特徴量抽出・テンソル変換（`.npz` 化）を行う。
詳細は [src/02_Preprocess/_README_02Preprocess.md](src/02_Preprocess/_README_02Preprocess.md) を参照。

### 3. src/03_DNN_System（【旧システム】DNNモデリング・学習・推論系）
※現在は使用していない旧プログラム群。初期のTransformerベースのエンコーダ・デコーダモデルおよび学習・推論スクリプトを格納している（現行のモデル開発・実験は `05_Step_Experiments` に移行）。
詳細は [src/03_DNN_System/_README_03DNN.md](src/03_DNN_System/_README_03DNN.md) を参照。

### 4. src/04_Parameter_Study（パラメータ調査・音響分析系）
WAVの音響特徴量（F0, RMS, aperiodicity等）とVSQXパラメータとの相関関係を可視化・分析する。母音ごとの特性差やパラメータ感度を定量的に調査するためのツール群である。
詳細は [src/04_Parameter_Study/_README_04ParameterStudy.md](src/04_Parameter_Study/_README_04ParameterStudy.md) を参照。

### 5. src/05_Step_Experiments（【現行・主力】実験検証・DNNモデリング系）
現在主軸として開発・実験を行っている中核領域。Bi-LSTMやCNN-Bi-LSTM、双方向ゲート、特徴量融合ブロック等を用いたDNNモデルの統合学習、Optunaによる自動ハイパーパラメータ探索、スライディングウィンドウによる推論、およびレンダリング後の詳細精度評価を実行する。
詳細は [src/05_Step_Experiments/_README_05Step.md](src/05_Step_Experiments/_README_05Step.md) を参照。

### 6. src/analysis（音響・物理特性解析系）
歌声の音高とRMSの物理的関係、フォルマント包絡線と倍音共鳴の相関分析（`analyze_formant_harmonics.py` 等）を行う解析スクリプト群を集約している。

### 7. src/util（共通基盤モジュール）
音響特徴量抽出（`feature_extractor.py`）、12平均律の音高螺旋変換やOne-hot化（`vsqx_dnn_formatter.py`）、VSQX解析パーサー、共通損失関数（DiffLoss等）、評価器などの共通ライブラリを格納している。
詳細は [src/util/_README_util.md](src/util/_README_util.md) を参照。

## 大まかな処理の流れ
1. **分析・物理特性調査 (Parameter Study & Analysis)**
   VOCALOIDパラメータと音響特徴量（倍音・フォルマント・RMS等）の相関を調査し、推定モデル・特徴量設計の要件を定義する。
2. **データ作成・前処理 (Data Augment & Preprocess)**
   学習用のVSQXデータ・WAVデータを作成・拡張し、辞書型テンソル形式（`.npz`）に変換する。
3. **学習・実験・評価 (Step Experiments)**
   特徴量融合ブロックやCNN-BiLSTM等を用いたモデルの学習・Optuna探索を行い、未知データに対する推論と精度検証を実行する。
