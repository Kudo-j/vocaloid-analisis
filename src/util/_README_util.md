# ユーティリティ系 (util)

このディレクトリは、VOCALOIDのプロジェクトにおいて、WAV音声からの音響特徴量抽出と、VSQXファイルからのボーカルパラメータ抽出を担う、独立した中核クラス群（モジュール）を格納している。

これらのクラスは `preprocess.py` や `build_dataset.py` などの前処理スクリプトから呼び出されるだけでなく、推論スクリプト等からも再利用できるように自己完結した設計になっている。

## 各プログラムの説明

### 1. `feature_extractor.py`
WAVファイルなどの音声信号から、様々な音響特徴量を抽出するモジュール

*   **メインクラス**: `AudioFeatureExtractor`
*   **主な機能**:
    *   `librosa`, `pyworld`, `scipy` 等をバックエンドとして統合し、指定された特徴量タイプに応じた計算処理を一手に引き受ける。
    *   **抽出可能な主な特徴量**:
        *   Melスペクトログラム、MFCC、CQT (定Q変換)、STFT
        *   RMS (音量エンベロープ)
        *   F0 (基本周波数) ※ `pyworld` (harvest/dio) と `crepe` (ニューラルネットベース) を切り替え可能
        *   Aperiodicity (非周期成分)、ケプストラム
        *   その他スペクトル特徴量 (Spectral Centroid, Bandwidth, Contrast, HNR, Zero Crossing Rate など)
    *   抽出された各特徴量は、最終的に指定されたフレーム間隔（`hop_length`）に合わせて時間軸が同期（スタック）され、NumPy配列として返却される。

### 2. `vsqx_parser.py`
VSQXファイルのパース処理の基盤となるモジュール
VSQXをパースし、全パラメータを tick 形式の疎な点データ辞書として返す
DNN用の処理などは他のモジュールが行う

*   **メインクラス**: `VSQXParser`
*   **主な機能**:
    *   **疎なデータ (Sparse Data) の抽出**: 
        `read_sparse()` メソッドにより、VSQXファイル内のXMLを解析し、時間（Tick）ベースのコントロールチェンジ (CC) やノートイベント、テンポマップ等をリストや辞書として抽出する。
    *   **設定情報の内包**: 
        パラメータIDのマッピングやデフォルト値、正規化上限値などの設定 (`vsqx_config`) を内包している。

### 3. `vsqx_dnn_formatter.py`
`VSQXParser` を継承し、抽出された疎なデータをDNN（深層学習）向けの密なフレーム同期テンソル（配列）に変換するモジュール。

*   **メインクラス**: `VSQXDNNFormatter`
*   **主な機能**:
    *   **動的パラメータの補間と配列化**: 
        `parse_vsqx()` により、VSQXのCCイベントを音声フレームに合わせた連続値に線形補間し、VOCALOIDのビブラート仕様 (vibLenによるマスク等) を正確に再現する。
    *   **正規化とOne-hot化**:
        `normalize_features()` により、各パラメータを [-1.0, 1.0] 等に正規化し、音素やEVEC等のカテゴリ変数をOne-hotベクトルに変換する。
    *   **フラットな辞書の返却**:
        最終的に、`config.py` で指定される名前 (`PIT`, `PHONEME_P` など) をキーとしたフラットな辞書形式で、WAVフレーム長と完全に同期した学習用データを出力する。

### 4. `vsqx_compare.py`
`VSQXParser` を継承し、2つのVSQXファイル間の差異を比較・分析するためのモジュール。

*   **メインクラス**: `VSQXCompare`
*   **主な機能**:
    *   **時間変換**: 
        `read()` メソッドにより、Tickベースのイベントを物理時間 (秒) に変換した比較しやすいデータ構造を生成する。
    *   **パラメータ差分検出**:
        `find_changed_params()` や `find_static_param_diffs()` により、2つのVSQX間でどの動的・静的パラメータに差分（ユーザーの手による編集など）が発生しているかを特定する。

## 使い方・呼び出し例

これらは単独で実行するスクリプトではなく、他のスクリプトからインポートして使用する。

```python
import sys
sys.path.append("./util")
from feature_extractor import AudioFeatureExtractor
from vsqx_dnn_formatter import VSQXDNNFormatter

# 初期化 (サンプリングレートとホップ長を統一する)
sr = 44100
hop_length = 256

audio_ext = AudioFeatureExtractor(sr=sr, hop_length=hop_length)
vsqx_ext = VSQXDNNFormatter(sr=sr, hop_length=hop_length)

# 特徴量の抽出
audio_features = audio_ext.get_features("sample.wav", ["mel", "f0_world"])
vsqx_data_raw = vsqx_ext.parse_vsqx("sample.vsqx", total_frames=audio_features.shape[0])
vsqx_data_norm = vsqx_ext.normalize_features(vsqx_data_raw) # フラットな辞書で返却される
```
