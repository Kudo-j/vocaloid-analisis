"""
analyze_linear.py
=================
重回帰分析による線形予測スクリプト。

Scikit-Learn の最小二乗法 (OLS) / Ridge回帰を用い、
エポックを回さず一撃で最適解を求める。
DNN (train.py) との精度比較のベースラインとして使用する。

出力:
    1. 精度指標 (MAE, RMSE, R²)
    2. 各入力パラメータの回帰係数 (影響力の大きさ)
    3. 予測値 vs 実測値の散布図
    4. 結果テキストファイル

使い方:
    1. 下記の config 領域を実験条件に合わせて編集する
    2. python analyze_linear.py を実行する
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- パス設定 ---
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "util"))
from train_util.evaluate import Evaluator
from train_util.dataset import create_datasets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
#  config 領域
# =============================================================================

# --- データパス ---
DATA_DIR = "./processed/Step1"  # build_dataset.py の出力先
OUTPUT_DIR = os.path.join(BASE_DIR, "output_linear")  # 結果の保存先
EXPERIMENT_NAME = "Step1_RMS_Linear"  # 実験名 (ファイル名に使用)

SONGLIST_PATH = ""  # 曲名リストへのパス

# --- データ分割モード ---
# "random"   : 全てのファイルを個別にランダム分割する (推奨: 楽曲に依存しない汎用的な学習)
# "songlist" : 同一楽曲のフレームがTrain/Testに混在しないように楽曲単位で分割する
SPLIT_MODE = "random"
# --- 手法選択 ---
# "ols"   : 通常の最小二乗法 (Ordinary Least Squares)
# "ridge" : Ridge回帰 (L2正則化付き。入力パラメータが多い場合に安定)
METHOD = "ols"
RIDGE_ALPHA = 1.0  # Ridge回帰の正則化強度 (METHOD="ridge" の場合のみ有効)

# --- 文脈窓 (前後フレーム参照) ---
# 0: DYN(t) = a * RMS(t) + b  (単純線形)
# 2: DYN(t) = Σ w_i * x(t+i) + b  (前後2フレームを参照する重回帰)
CONTEXT_SIZE = 0

# --- データ分割比率 ---
# 線形回帰なので分割する必要はないが、学習データをDNNと合わせるために分割する（シードも固定値なのでデータは合致する）
VALID_RATIO = 0.2
TEST_RATIO = 0.2

# --- メモリ安全化: 学習データの上限フレーム数 ---
# 曲単位で順に読み込み、累計が上限を超える曲以降は学習から除外する。
# 0 の場合は制限なし (全曲読み込み)。テストデータには適用されない。
MAX_TRAIN_FRAMES = 5_000_000  # 500万フレーム

# =============================================================================
# config 領域ここまで
# =============================================================================


def _load_all_frames(
    data_dir, file_list, context_size=0, max_frames=0, keep_silence=False
):
    """
    指定されたNPZファイル群から全フレームを読み込み、
    入力(X)と出力(Y)の2次元配列として返す。

    曲単位で以下の順序で処理を行い、時系列の整合性と無音除去を保証する:
      1. 曲のデータを読み込む
      2. 文脈窓を適用する (曲の時系列が保たれた状態で前後フレームを結合)
      3. 発声マスクで無音フレームを除去する
      4. 有効フレームのみをリストに蓄積する
      5. max_frames > 0 の場合、累計が上限を超える曲で読み込みを打ち切る

    Args:
        data_dir: NPZファイルが格納されたディレクトリ。
        file_list: 読み込むファイル名(拡張子なし)のリスト。
        context_size: 前後何フレーム参照するか (0なら文脈窓なし)。
        max_frames: 学習データの上限フレーム数。0なら制限なし。
        keep_silence: Trueなら無音区間を削除せずにそのまま返す。

    Returns:
        X_all: (全有効フレーム数, 全入力次元数) のndarray。
        Y_all: (全有効フレーム数, 全出力次元数) のndarray。
        M_all: (全有効フレーム数,) の発声マスク。
        feature_names: 入力の特徴名リスト (文脈窓拡張前の基本名)。
        used_files: 実際に学習に使用したファイル名と各フレーム数のリスト [(name, frames), ...]。
        excluded_files: 上限により除外されたファイル名のリスト。
    """
    X_chunks = []
    Y_chunks = []
    M_chunks = []
    feature_names = None
    total_frames = 0
    used_files = []
    excluded_files = []

    for name_stem in file_list:
        # 上限チェック (max_frames > 0 の場合のみ)
        if max_frames > 0 and total_frames >= max_frames:
            excluded_files.append(name_stem)
            continue

        npz_path = os.path.join(data_dir, f"{name_stem}.npz")
        if not os.path.exists(npz_path):
            continue

        with np.load(npz_path) as data:
            # Keyを取り出してソート
            keys = list(data.keys())
            x_keys = sorted([k for k in keys if k.startswith("x_")])
            y_keys = sorted([k for k in keys if k.startswith("y_")])

            x_parts = []
            names = []
            for k in x_keys:
                # 値を取り出しつつ結合
                arr = data[k]
                x_parts.append(arr)
                # 後のレポート出力用に見やすい特徴量名を作成
                branch_name = k[2:]  # "x_" を除去
                # 各次元に名前を付ける (例: "mel_0", "mel_1", ...)
                for d in range(arr.shape[1]):
                    names.append(f"{branch_name}_{d}")
            y_parts = [data[k] for k in y_keys]

            # 発声マスクの読み込み
            if "voicing_mask" in data:
                voicing_mask = data["voicing_mask"]  # (N, 1)
            else:
                # マスクが存在しない場合は全フレーム有効とみなす
                first_arr = x_parts[0] if x_parts else y_parts[0]
                voicing_mask = np.ones((len(first_arr), 1), dtype=np.float32)

            if x_parts and y_parts:
                X_song = np.concatenate(x_parts, axis=-1)
                Y_song = np.concatenate(y_parts, axis=-1)

                # 1. 文脈窓の適用 (曲の時系列が保たれた状態で行う)
                if context_size > 0:
                    X_song = _apply_context_window(X_song, context_size)

                # 2. 発声マスクで無音フレームを除去
                if not keep_silence:
                    voiced_idx = voicing_mask.squeeze(-1) > 0.5
                    X_song = X_song[voiced_idx]
                    Y_song = Y_song[voiced_idx]
                    voicing_mask = voicing_mask[voiced_idx]

                if len(X_song) > 0:
                    X_chunks.append(X_song)
                    Y_chunks.append(Y_song)
                    M_chunks.append(voicing_mask.squeeze(-1))
                    song_frames = len(X_song)
                    total_frames += song_frames
                    used_files.append((name_stem, song_frames))
                    if feature_names is None:
                        feature_names = names

    if not X_chunks:
        return None, None, None, None, used_files, excluded_files

    X_all = np.concatenate(X_chunks, axis=0).astype(np.float64)
    Y_all = np.concatenate(Y_chunks, axis=0).astype(np.float64)
    M_all = np.concatenate(M_chunks, axis=0).astype(np.float32)

    return X_all, Y_all, M_all, feature_names, used_files, excluded_files


def _apply_context_window(X, context_size):
    """
    各フレームに前後のフレームの情報を結合する。

    context_size=2 の場合、フレームtの入力は
    [x(t-2), x(t-1), x(t), x(t+1), x(t+2)] に拡張される。

    Args:
        X: (N, D) の入力配列。
        context_size: 前後何フレーム参照するか。

    Returns:
        X_ctx: (N, D * (2*context_size+1)) の拡張配列。
    """
    if context_size == 0:
        return X

    N, D = X.shape
    pad_width = ((context_size, context_size), (0, 0))
    # 端の値を複製して埋める (replicate padding)
    X_padded = np.pad(X, pad_width, mode="edge")

    window_size = 2 * context_size + 1
    X_ctx = np.zeros((N, D * window_size), dtype=X.dtype)

    for i in range(window_size):
        X_ctx[:, i * D : (i + 1) * D] = X_padded[i : i + N]

    return X_ctx


def _plot_scatter(y_true, y_pred, save_path, title):
    """予測値 vs 実測値の散布図を出力する。"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.05, s=1, color="steelblue")

    # 対角線 (完全一致の場合ここに乗る)
    val_min = min(y_true.min(), y_pred.min())
    val_max = max(y_true.max(), y_pred.max())
    plt.plot(
        [val_min, val_max], [val_min, val_max], "r--", linewidth=1.5, label="Ideal"
    )

    plt.xlabel("Actual (Normalized)")
    plt.ylabel("Predicted (Normalized)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"散布図を {save_path} に保存しました。")
    plt.close()


def _plot_coefficients(coef, feature_names, save_path, top_n=20):
    """回帰係数の絶対値を棒グラフで出力する。"""
    abs_coef = np.abs(coef)
    # 上位 top_n 個を表示
    if len(abs_coef) > top_n:
        top_indices = np.argsort(abs_coef)[-top_n:]
    else:
        top_indices = np.argsort(abs_coef)

    top_names = [feature_names[i] for i in top_indices]
    top_values = abs_coef[top_indices]

    plt.figure(figsize=(10, max(6, len(top_indices) * 0.35)))
    colors = ["steelblue" if coef[i] >= 0 else "coral" for i in top_indices]
    plt.barh(range(len(top_indices)), top_values, color=colors)
    plt.yticks(range(len(top_indices)), top_names)
    plt.xlabel("|Coefficient|")
    plt.title(
        f"Top {len(top_indices)} Feature Importance (Blue=Positive, Red=Negative)"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"係数グラフを {save_path} に保存しました。")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- データ分割 (dataset.py 共通ロジックを使用) ---
    print("データ分割を実行中 (DNN共通)...")
    try:
        train_ds, valid_ds, test_ds, feature_info = create_datasets(
            DATA_DIR,
            SONGLIST_PATH,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            split_mode=SPLIT_MODE,
        )
    except FileNotFoundError:
        print("Error: features.json が見つかりません。")
        print("build_dataset.py を先に実行してください。")
        return

    x_dims = feature_info.get("x_dims", {})
    y_dims = feature_info.get("y_dims", {})
    print(f"入力次元数: {sum(x_dims.values())} {dict(x_dims)}")
    print(f"出力次元数: {sum(y_dims.values())} {dict(y_dims)}")

    # Datasetのチャンクから一意な曲名リストを抽出
    train_files = list(dict.fromkeys([c[0] for c in train_ds.chunk_map]))
    valid_files = list(dict.fromkeys([c[0] for c in valid_ds.chunk_map]))
    test_files = list(dict.fromkeys([c[0] for c in test_ds.chunk_map]))

    print(
        f"ファイル数 (DNN共通分割) -> Train: {len(train_files)} | Valid: {len(valid_files)} | Test: {len(test_files)}"
    )

    # --- 全フレーム読み込み (曲単位で文脈窓適用 → 無音除去を行う) ---
    print("Train データ読み込み中 (無音除去あり)...")
    if MAX_TRAIN_FRAMES > 0:
        print(f"  ※ 学習データ上限: {MAX_TRAIN_FRAMES:,} フレーム (曲単位で打ち切り)")
    X_train, Y_train, _, feature_names, used_files, excluded_files = _load_all_frames(
        DATA_DIR, train_files, CONTEXT_SIZE, max_frames=MAX_TRAIN_FRAMES
    )
    print("Test データ読み込み中 (無音除去あり)...")
    X_test, Y_test, _, _, _, _ = _load_all_frames(DATA_DIR, test_files, CONTEXT_SIZE)

    if X_train is None or X_test is None:
        print("Error: データの読み込みに失敗しました。")
        return

    print(
        f"Train: {X_train.shape[0]:,} フレーム (有音のみ) | Test: {X_test.shape[0]:,} フレーム (有音のみ)"
    )

    # --- サンプリングレポートの出力 ---
    sampling_report_path = os.path.join(
        OUTPUT_DIR, f"{EXPERIMENT_NAME}_sampling_report.txt"
    )
    with open(sampling_report_path, "w", encoding="utf-8") as f:
        f.write(f"=== 線形モデル 学習データサンプリングレポート ===\n")
        f.write(
            f"設定上限フレーム数: {'制限なし' if MAX_TRAIN_FRAMES == 0 else f'{MAX_TRAIN_FRAMES:,}'}\n"
        )
        f.write(f"実際の学習フレーム数: {X_train.shape[0]:,}\n")
        f.write(f"テストフレーム数: {X_test.shape[0]:,}\n\n")

        f.write(f"【学習に使用した曲 ({len(used_files)} 曲)】\n")
        for name, frames in used_files:
            f.write(f"  - {name} ({frames:,} フレーム)\n")

        if excluded_files:
            f.write(
                f"\n【メモリ上限により学習から除外された曲 ({len(excluded_files)} 曲)】\n"
            )
            for name in excluded_files:
                f.write(f"  - {name}\n")
        else:
            f.write(f"\n【除外された曲: なし (全曲を学習に使用)】\n")

    print(f"サンプリングレポートを {sampling_report_path} に保存しました。")
    if excluded_files:
        print(f"  ※ {len(excluded_files)} 曲が上限により学習から除外されました。")

    # --- 文脈窓適用時の特徴名拡張 ---
    if CONTEXT_SIZE > 0:
        orig_names = feature_names[:]
        feature_names = []
        for offset in range(-CONTEXT_SIZE, CONTEXT_SIZE + 1):
            suffix = f"(t{offset:+d})" if offset != 0 else "(t)"
            for name in orig_names:
                feature_names.append(f"{name}{suffix}")

        print(
            f"文脈窓 (前後 {CONTEXT_SIZE} フレーム) 適用後の入力次元数: {X_train.shape[1]}"
        )

    # --- モデルの学習 (一撃で完了) ---
    print(f"\n--- {METHOD.upper()} 回帰を実行中 ---")
    if METHOD == "ridge":
        model = Ridge(alpha=RIDGE_ALPHA)
    else:
        model = LinearRegression()

    model.fit(X_train, Y_train)
    print("学習完了。")

    # --- モデルの保存 ---
    model_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_model.pkl")
    joblib.dump(model, model_path)
    print(f"モデルを {model_path} に保存しました。")

    # --- 予測と評価 ---
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    # 出力が多次元の場合、各次元ごとに評価する
    num_outputs = Y_train.shape[1]
    results = []

    for dim_idx in range(num_outputs):
        y_train_d = Y_train[:, dim_idx]
        y_test_d = Y_test[:, dim_idx]
        yp_train_d = Y_pred_train[:, dim_idx]
        yp_test_d = Y_pred_test[:, dim_idx]

        mae_train = mean_absolute_error(y_train_d, yp_train_d)
        mae_test = mean_absolute_error(y_test_d, yp_test_d)
        rmse_train = np.sqrt(mean_squared_error(y_train_d, yp_train_d))
        rmse_test = np.sqrt(mean_squared_error(y_test_d, yp_test_d))
        r2_train = r2_score(y_train_d, yp_train_d)
        r2_test = r2_score(y_test_d, yp_test_d)

        result = {
            "dim": dim_idx,
            "mae_train": mae_train,
            "mae_test": mae_test,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "r2_train": r2_train,
            "r2_test": r2_test,
        }
        results.append(result)

        print(f"\n--- 出力次元 {dim_idx} ---")
        print(f"  Train MAE:  {mae_train:.6f}  |  Test MAE:  {mae_test:.6f}")
        print(f"  Train RMSE: {rmse_train:.6f}  |  Test RMSE: {rmse_test:.6f}")
        print(f"  Train R²:   {r2_train:.6f}  |  Test R²:   {r2_test:.6f}")

    # --- 結果の保存 ---
    # 1. テキストレポート
    report_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== 線形予測分析レポート ===\n")
        f.write(f"実験名: {EXPERIMENT_NAME}\n")
        f.write(f"手法: {METHOD.upper()}\n")
        if METHOD == "ridge":
            f.write(f"Ridge Alpha: {RIDGE_ALPHA}\n")
        f.write(f"文脈窓: {CONTEXT_SIZE} (前後{CONTEXT_SIZE}フレーム)\n")
        f.write(f"入力次元数: {X_train.shape[1]}\n")
        f.write(f"Train フレーム数: {X_train.shape[0]:,}\n")
        f.write(f"Test フレーム数: {X_test.shape[0]:,}\n\n")

        for r in results:
            f.write(f"--- 出力次元 {r['dim']} ---\n")
            f.write(
                f"  Train MAE:  {r['mae_train']:.6f}  |  Test MAE:  {r['mae_test']:.6f}\n"
            )
            f.write(
                f"  Train RMSE: {r['rmse_train']:.6f}  |  Test RMSE: {r['rmse_test']:.6f}\n"
            )
            f.write(
                f"  Train R²:   {r['r2_train']:.6f}  |  Test R²:   {r['r2_test']:.6f}\n\n"
            )

        # 回帰係数一覧
        f.write(f"\n=== 回帰係数一覧 (出力次元 0) ===\n")

        # scikit-learn は Y_train が 2D の場合、coef_ を (n_outputs, n_features) にする
        coef = model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_
        intercept = (
            model.intercept_[0]
            if getattr(model.intercept_, "ndim", 0) > 0
            else model.intercept_
        )

        f.write(f"バイアス (intercept): {float(intercept):.6f}\n\n")

        # 絶対値の大きい順にソート
        sorted_indices = np.argsort(np.abs(coef))[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            f.write(
                f"  {rank:3d}. {feature_names[idx]:30s}  coef={float(coef[idx]):+.6f}\n"
            )

    print(f"\nレポートを {report_path} に保存しました。")

    # 2. 係数重要度グラフ (出力次元0のみ)
    if X_train.shape[1] > 1:
        coef_for_plot = (
            model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_
        )
        coef_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_coefficients.png")
        _plot_coefficients(coef_for_plot, feature_names, coef_path)
    else:
        print("\n※入力データが1次元のため、係数重要度グラフの出力はスキップしました。")

    # 3. 共通評価モジュール(Evaluator)によるテストデータの楽曲単位評価
    print("\n--- 共通評価モジュール(Evaluator)によるテストデータ詳細評価 ---")
    evaluator = Evaluator()

    # feature_infoから出力パラメータ名を取得
    y_param_names = []
    if "y_feature_map" in feature_info:
        for branch_name, features in feature_info["y_feature_map"].items():
            for feat in features:
                dim = feat["dim"]
                name = feat["name"]
                if dim == 1:
                    y_param_names.append(name)
                else:
                    for i in range(dim):
                        y_param_names.append(f"{name}_{i}")
    else:
        y_param_names = [f"Dim_{i}" for i in range(Y_train.shape[1])]

    # テストデータを1曲ずつ推論してEvaluatorに追加 (波形用に無音区間も残す)
    for i, song_name in enumerate(test_files):
        X_song, Y_song, M_song, _, _, _ = _load_all_frames(
            DATA_DIR, [song_name], CONTEXT_SIZE, keep_silence=True
        )
        if X_song is None:
            continue

        Y_pred_song = model.predict(X_song)
        evaluator.add_song_result(
            song_name, Y_song, Y_pred_song, y_param_names, voicing_mask=M_song
        )

        # 波形プロットは最初の5曲のみ出力
        if i < 5:
            wave_path = os.path.join(
                OUTPUT_DIR, f"{EXPERIMENT_NAME}_waveform_{song_name}.png"
            )
            evaluator.plot_waveform(
                song_name,
                Y_song,
                Y_pred_song,
                wave_path,
                y_param_names,
                voicing_mask=M_song,
            )

    # 全テストデータの蓄積結果から散布図とExcelを出力
    evaluator.plot_scatter(OUTPUT_DIR, EXPERIMENT_NAME, y_param_names)
    excel_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_eval.xlsx")
    evaluator.export_report(excel_path)

    print("\n--- 分析完了 ---")


if __name__ == "__main__":
    main()
