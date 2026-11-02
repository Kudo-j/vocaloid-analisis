# VOCALOID パラメータ推定・分析システム

本プロジェクトは、DNNを用いたVOCALOID（ボーカロイド）の動的パラメータ推定、および音響特徴量とVSQXパラメータの相関関係を分析するためのプログラム群である。
システムは大きく分けて以下のモジュール（ディレクトリ）から構成されている。

## ディレクトリ構成

### 1. src/01_Data_Augment（データ拡張系）
少数のデータセットから学習用データを増やすためのデータ拡張処理を行う。VSQXの移調やボーカルチェンジ、パラメータの書き込みなどを自動化し、データ量を確保する。
詳細は [src/01_Data_Augment/_README_01DataAugment.md](src/01_Data_Augment/_README_01DataAugment.md) を参照。

### 2. src/02_Preprocess（データチェック・前処理系）
データセットの品質を担保するため、VSQXファイルやWAVファイルの形式チェック、データの前処理（前処理後のテンソルデータや特徴量の検証）を行う。
詳細は [src/02_Preprocess/_README_02Preprocess.md](src/02_Preprocess/_README_02Preprocess.md) を参照。

### 3. src/03_DNN_System（DNNモデリング・学習・推論系）
抽出した音響特徴量からVSQXの動的パラメータ（DYNなど）を推定する深層学習（DNN）モデルの定義、学習、および推論処理を行う。
詳細は [src/03_DNN_System/_README_03DNN.md](src/03_DNN_System/_README_03DNN.md) を参照。

### 4. src/04_Parameter_Study（パラメータ調査・音響分析系）
WAVの音響特徴量（F0, RMS, aperiodicityなど）と、VSQXのパラメータとの相関関係を可視化・分析する。母音ごとの違いやパラメータの効き具合を定量的に調査するためのツール群である。
詳細は [src/04_Parameter_Study/_README_04ParameterStudy.md](src/04_Parameter_Study/_README_04ParameterStudy.md) を参照。

### 5. src/05_Step_Experiments（実験検証・モデル評価系）
特徴量設計やモデル構造（CNN-BiLSTM、双方向ゲート等）の比較実験、ハイパーパラメータ調整、評価指標のプロットを行う。
詳細は [src/05_Step_Experiments/_README_05Step.md](src/05_Step_Experiments/_README_05Step.md) を参照。

## 大まかな処理の流れ
1. **分析・調査 (Parameter Study)**
   VOCALOIDのパラメータが音声にどのような影響を与えるかを調査し、推定モデルの要件を定義する。
2. **データ作成 (Data Augment & Check)**
   学習用のVSQXデータ・WAVデータを作成し、拡張・検証・前処理を行う。
3. **学習・推論・実験 (DNN System & Step Experiments)**
   特徴量からパラメータを推定するモデルを学習し、未知のデータに対して推論・性能評価を実行する。
