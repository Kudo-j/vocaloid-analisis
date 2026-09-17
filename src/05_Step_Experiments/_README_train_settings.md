# train_settings.json 設定マニュアル

本ファイルは `train.py` を実行する際の設定ファイルである `train_settings.json` の主要パラメータおよび選択肢についてまとめたドキュメントである。

## 主要設定項目

### SPLIT_MODE (データ分割手法)
学習・検証・テストデータの分割戦略を指定する。
*   `"random"`: 全データをファイル（チャンク）単位で完全にランダムに分割する。
*   `"song"` (または `"random"` 以外の任意の文字列): 曲単位で分割する。同一楽曲がTrainとTestにまたがることがないため、未知の楽曲に対する汎化性能を正しく評価する際に用いる。

### MODEL_TYPE (メインモデル構造)
ネットワークのコアとなるモデルアーキテクチャを指定する。
*   `"bilstm_base"`: シンプルな双方向LSTMモデル。
*   `"cnn_bilstm"`: 音声特徴の局所的なパターンを捉えるための1次元CNN層を前段に配置し、その後ろに双方向LSTMを接続したハイブリッドモデル。
*   `"lstm_base"`: 単方向LSTMモデル。順方向または逆方向の文脈を単独で学習・検証するためのモデルである。CNNや入力線形層を通さず、入力特徴量を直接単方向LSTMに入力する。方向の指定は `MODEL_PARAMS` の `direction` で行う。

### FUSION_STRATEGY (ブランチ結合戦略)
`build_dataset.py` で定義した複数の入力ブランチ（例: `audio_rms`, `phoneme`）をどのように1つにまとめるかを指定する。
*   `"pass_through"`: 各ブランチを入力直後にすべてそのまま結合（Concat）し、メインモデルに渡す。最もシンプルで標準的な手法である。
*   `"late_concat"`: 各ブランチごとに独立した事前ネットワーク（`BRANCH_MODULES` で指定）を通し、それぞれ特徴抽出を行った後で結合してメインモデルに渡す。
*   `"late_add"`: 各ブランチごとに独立した事前ネットワークを通した後、結合ではなく加算（Add）によってまとめる。全ブランチの出力次元数が一致している必要がある。

### BRANCH_MODULES (ブランチごとの事前ネットワーク)
`FUSION_STRATEGY` が `"late_concat"` または `"late_add"` の場合に有効となる設定である。入力ブランチ名に対して、適用するネットワークモジュールを指定する。
*   `"none"`: 何も処理を行わず、そのままの値を渡す。
*   `"linear"`: 全結合層（Linear）と ReLU 活性化関数を通す。
*   `"cnn"`: 1次元の畳み込み層（BranchCNN）を通す。
*   `"embedding"`: カテゴリ変数用の埋め込み（Embedding）層を通す。
*   `"lstm"`: 単層のLSTMを通す。

※ `FUSION_STRATEGY` が `"pass_through"` の場合でも、エラー防止のために存在する入力ブランチ名はすべてこの辞書に定義し、値は `"none"` としておくことが推奨される。

### MODEL_PARAMS (モデル固有ハイパーパラメータ)
各 `MODEL_TYPE` に応じた内部パラメータを指定する辞書である。

#### `cnn_bilstm` の主要パラメータ
*   `num_conv_layers`: 前段に配置する1次元CNN（ConvBlock）の積層数（デフォルト: 1）。多層化することで受容野（Receptive Field）を拡大できる。
*   `conv_channels`: 1次元CNNの出力チャンネル数（デフォルト: 64）。
*   `kernel_size`: 1次元CNNのカーネルサイズ（デフォルト: 9または19）。
*   `hidden_dim`: Bi-LSTM の隠れ層次元数（デフォルト: 64）。
*   `num_layers`: Bi-LSTM の積層数（デフォルト: 1）。
*   `dropout`: Bi-LSTM / Attention のドロップアウト率（デフォルト: 0.3）。
*   `use_attention`: MultiheadAttention 層を挿入するかどうかのフラグ（`true` / `false`）。
*   `num_heads`: MultiheadAttention のヘッド数。

#### `lstm_base` の主要パラメータ
*   `direction`: LSTMの処理方向を指定する。
    *   `"forward"`: 順方向処理（過去から未来へ向けて時系列を処理）。
    *   `"backward"`: 逆方向処理（未来から過去へ向けて時系列を処理）。モデル内部で時間軸を自動反転して処理するため、外部データの反転処理は不要である。
*   `hidden_dim`: LSTMの隠れ層次元数（デフォルト: 64）。
*   `num_layers`: LSTMの積層数（デフォルト: 1）。
*   `dropout`: LSTM層間のドロップアウト率（`num_layers > 1` の場合に適用）。
*   `use_attention`: MultiheadAttention 層を挿入するかどうかのフラグ（`true` / `false`）。
*   `num_heads`: MultiheadAttention のヘッド数（`hidden_dim` がこの値で割り切れる必要がある）。


