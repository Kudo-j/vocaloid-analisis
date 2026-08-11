"""
analyze_linear.py
=======================
メモリを枯渇させずに無限のデータを学習できる、
行列の累積計算による線形予測スクリプト（アウトオブコア学習）。

Scikit-Learnに依存せず、1曲ずつ読み込んで極小の行列(D×D)に
累積足し算を行うことで、メモリ使用量を常に「1曲分」に抑えつつ
全データ一括計算と完全に一致する真の最適解を算出します。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
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
SONGLIST_PATH = ""  # 曲名リストへのパス

# --- データ分割モード ---
SPLIT_MODE = "fixed_name"
# --- 手法選択 ---
# "ols"   : 通常の最小二乗法 (Ordinary Least Squares)
# "ridge" : Ridge回帰 (L2正則化付き。入力パラメータが多い場合に安定)
METHOD = "ols"
RIDGE_ALPHA = 1.0  # Ridge回帰の正則化強度 (METHOD="ridge" の場合のみ有効)

# --- 文脈窓 (前後フレーム参照) ---
CONTEXT_SIZE = 0

# --- データ分割比率 ---
VALID_RATIO = 0.2
TEST_RATIO = 0.2

# --- 評価グラフへの追加プロット ---
PLOT_EXTRA_FEATURES = {"audio_rms": "RMS"}

DATA_DIR = "./processed/Step1"  # build_dataset.py の出力先
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "Step1",
    "linear",
    f"Context_{CONTEXT_SIZE}",
)  # 結果の保存先
EXPERIMENT_NAME = "Step1_RMS_Linear"  # 実験名 (ファイル名に使用)
# =============================================================================
# config 領域ここまで
# =============================================================================


def _load_single_song(data_dir, name_stem, context_size=0, keep_silence=False):
    """
    指定された1曲のNPZファイルからフレームを読み込み、
    入力(X)と出力(Y)の2次元配列として返す。
    """
    npz_path = os.path.join(data_dir, f"{name_stem}.npz")
    if not os.path.exists(npz_path):
        return None, None, None, None

    with np.load(npz_path) as data:
        keys = list(data.keys())
        x_keys = sorted([k for k in keys if k.startswith("x_")])
        y_keys = sorted([k for k in keys if k.startswith("y_")])

        x_parts = []
        names = []
        x_dict_raw = {}
        for k in x_keys:
            arr = data[k]
            x_parts.append(arr)
            branch_name = k[2:]
            x_dict_raw[branch_name] = arr
            for d in range(arr.shape[1]):
                names.append(f"{branch_name}_{d}")
        y_parts = [data[k] for k in y_keys]

        if "voicing_mask" in data:
            voicing_mask = data["voicing_mask"]
        else:
            first_arr = x_parts[0] if x_parts else y_parts[0]
            voicing_mask = np.ones((len(first_arr), 1), dtype=np.float32)

        if not (x_parts and y_parts):
            return None, None, None, None

        X_song = np.concatenate(x_parts, axis=-1)
        Y_song = np.concatenate(y_parts, axis=-1)

        if context_size > 0:
            X_song = _apply_context_window(X_song, context_size)

        if not keep_silence:
            voiced_idx = voicing_mask.squeeze(-1) > 0.5
            X_song = X_song[voiced_idx]
            Y_song = Y_song[voiced_idx]
            voicing_mask = voicing_mask[voiced_idx]
            for k in x_dict_raw:
                x_dict_raw[k] = x_dict_raw[k][voiced_idx]

        if len(X_song) == 0:
            return None, None, None, None, None

        return (
            X_song.astype(np.float64),
            Y_song.astype(np.float64),
            voicing_mask.squeeze(-1).astype(np.float32),
            names,
            x_dict_raw,
        )


def _apply_context_window(X, context_size):
    if context_size == 0:
        return X

    N, D = X.shape
    pad_width = ((context_size, context_size), (0, 0))
    X_padded = np.pad(X, pad_width, mode="edge")

    window_size = 2 * context_size + 1
    X_ctx = np.zeros((N, D * window_size), dtype=X.dtype)

    for i in range(window_size):
        X_ctx[:, i * D : (i + 1) * D] = X_padded[i : i + N]

    return X_ctx


def _plot_scatter(y_true, y_pred, save_path, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.05, s=1, color="steelblue")
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
    abs_coef = np.abs(coef)
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


class LinearAccumulator:
    """Numpyベースの行列累積学習クラス"""

    def __init__(self, method="ols", alpha=1.0):
        self.method = method
        self.alpha = alpha
        self.w = None
        self.coef_ = None
        self.intercept_ = None

    def predict(self, X):
        if self.w is None:
            raise ValueError("Model is not trained yet.")
        # テスト時もバイアス項のダミー列(1)を追加して積を計算
        X_tilde = np.hstack([np.ones((len(X), 1)), X])
        return X_tilde @ self.w


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        return

    x_dims = feature_info.get("x_dims", {})
    y_dims = feature_info.get("y_dims", {})

    # 元の入力次元数
    orig_x_dim = sum(x_dims.values())

    train_files = list(dict.fromkeys([c[0] for c in train_ds.chunk_map]))
    test_files = list(dict.fromkeys([c[0] for c in test_ds.chunk_map]))

    print(f"ファイル数 -> Train: {len(train_files)} | Test: {len(test_files)}")

    print(f"\n--- {METHOD.upper()} 回帰の行列累積計算を開始 ---")

    # 累積用行列 A と B
    A = None
    B = None
    total_train_frames = 0
    feature_names = None

    # Trainループ（1曲ずつ）
    for i, song_name in enumerate(train_files):
        X_song, Y_song, _, names, _ = _load_single_song(
            DATA_DIR, song_name, CONTEXT_SIZE, keep_silence=False
        )
        if X_song is None:
            continue

        if feature_names is None:
            feature_names = names
            if CONTEXT_SIZE > 0:
                orig_names = feature_names[:]
                feature_names = []
                for offset in range(-CONTEXT_SIZE, CONTEXT_SIZE + 1):
                    suffix = f"(t{offset:+d})" if offset != 0 else "(t)"
                    for name in orig_names:
                        feature_names.append(f"{name}{suffix}")

            # 初期化: D' は元の次元数 + バイアス項(1)
            D_prime = X_song.shape[1] + 1
            O = Y_song.shape[1]
            A = np.zeros((D_prime, D_prime), dtype=np.float64)
            B = np.zeros((D_prime, O), dtype=np.float64)

        # 1のダミー列を追加 (バイアス項)
        X_tilde = np.hstack([np.ones((len(X_song), 1), dtype=np.float64), X_song])

        # 行列を累積
        A += X_tilde.T @ X_tilde
        B += X_tilde.T @ Y_song
        total_train_frames += len(X_song)

        if (i + 1) % 10 == 0 or (i + 1) == len(train_files):
            print(
                f"  Processed {i+1}/{len(train_files)} songs... (Total frames: {total_train_frames:,})"
            )

    if A is None:
        print("Error: 有効な学習データがありませんでした。")
        return

    print("行列の累積完了。連立方程式を解いて最適解を算出します...")

    # L2正則化の適用 (Ridge)
    if METHOD == "ridge":
        # バイアス項（インデックス0）にはペナルティをかけないのが一般的
        alpha_matrix = np.eye(A.shape[0]) * RIDGE_ALPHA
        alpha_matrix[0, 0] = 0.0
        A += alpha_matrix

    # w を計算: A * w = B -> w = A^-1 * B
    # np.linalg.solve の方が逆行列を直接計算するより高速・安全
    w = np.linalg.solve(A, B)

    # モデルの構築 (推論用にラップ)
    model = LinearAccumulator(method=METHOD, alpha=RIDGE_ALPHA)
    model.w = w
    model.intercept_ = w[0, :]  # 最初の行がバイアス
    model.coef_ = w[
        1:, :
    ].T  # 残りの行が各特徴量の重み (scikit-learn形式: [n_outputs, n_features])

    # --- 推論時に必要なメタデータをモデルに埋め込む ---
    model.context_size = CONTEXT_SIZE
    model.method = METHOD
    model.data_dir = DATA_DIR
    model.feature_names = feature_names

    # --- モデルの保存 ---
    model_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_model.pkl")
    joblib.dump(model, model_path)
    print(f"モデルを {model_path} に保存しました。")

    print("学習完了。")

    # --- 評価フェーズ ---
    print("\n--- テストデータ評価 (Evaluator) ---")
    evaluator = Evaluator()

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
        y_param_names = [f"Dim_{i}" for i in range(model.coef_.shape[0])]

    total_test_frames = 0
    Y_test_list = []
    Y_pred_test_list = []
    # テストデータを1曲ずつ推論
    for i, song_name in enumerate(test_files):
        X_song, Y_song, M_song, _, x_dict_raw = _load_single_song(
            DATA_DIR, song_name, CONTEXT_SIZE, keep_silence=True
        )
        if X_song is None:
            continue

        # テストデータの無音を除外した場合のフレーム数を集計(表示用)
        voiced_frames = np.sum(M_song > 0.5)
        total_test_frames += voiced_frames

        Y_pred_song = model.predict(X_song)
        evaluator.add_song_result(
            song_name, Y_song, Y_pred_song, y_param_names, voicing_mask=M_song
        )

        extra_features_np = {}
        if x_dict_raw:
            for k, feat_name in PLOT_EXTRA_FEATURES.items():
                if k in x_dict_raw:
                    feat_arr = x_dict_raw[k]
                    if feat_arr.ndim > 1:
                        feat_arr = feat_arr[:, 0]
                    extra_features_np[feat_name] = feat_arr

        # Testデータの全体メトリクス計算用に有声部分のみを抽出して保存
        voiced_idx = M_song > 0.5
        Y_test_list.append(Y_song[voiced_idx])
        Y_pred_test_list.append(Y_pred_song[voiced_idx])

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
                extra_features=extra_features_np,
            )

    # --- メトリクスの計算 ---
    print("\n--- 学習データの全体メトリクス計算中 (2周目) ---")
    Y_train_list = []
    Y_pred_train_list = []
    for song_name in train_files:
        X_song, Y_song, _, _, _ = _load_single_song(
            DATA_DIR, song_name, CONTEXT_SIZE, keep_silence=False
        )
        if X_song is None:
            continue
        Y_train_list.append(Y_song)
        Y_pred_train_list.append(model.predict(X_song))

    Y_train_all = (
        np.concatenate(Y_train_list, axis=0)
        if Y_train_list
        else np.zeros((0, len(y_param_names)))
    )
    Y_pred_train_all = (
        np.concatenate(Y_pred_train_list, axis=0)
        if Y_pred_train_list
        else np.zeros((0, len(y_param_names)))
    )
    Y_test_all = (
        np.concatenate(Y_test_list, axis=0)
        if Y_test_list
        else np.zeros((0, len(y_param_names)))
    )
    Y_pred_test_all = (
        np.concatenate(Y_pred_test_list, axis=0)
        if Y_pred_test_list
        else np.zeros((0, len(y_param_names)))
    )

    results = []
    for dim_idx in range(len(y_param_names)):
        y_train_d = Y_train_all[:, dim_idx]
        y_test_d = Y_test_all[:, dim_idx]
        yp_train_d = Y_pred_train_all[:, dim_idx]
        yp_test_d = Y_pred_test_all[:, dim_idx]

        mae_train = mean_absolute_error(y_train_d, yp_train_d)
        mae_test = mean_absolute_error(y_test_d, yp_test_d)
        mse_train = mean_squared_error(y_train_d, yp_train_d)
        mse_test = mean_squared_error(y_test_d, yp_test_d)
        r2_train = r2_score(y_train_d, yp_train_d)
        r2_test = r2_score(y_test_d, yp_test_d)

        # 逆正規化MAEの計算 (evaluate.py の denormalize_array のスケールに準拠)
        pname = y_param_names[dim_idx].split("_")[0]
        if pname == "PIT":
            scale = 8192.0
        elif pname == "PBS":
            scale = 12.0
        elif pname in ["fallPort", "risePort"]:
            scale = 127.0
        elif pname == "vibType":
            scale = 1.0
        else:
            scale = (
                63.5  # DYN, BRI, CLE 等 ( [-1, 1] -> [0, 127] なのでスケールは 127/2 )
            )

        denorm_mae_train = mae_train * scale
        denorm_mae_test = mae_test * scale

        results.append(
            {
                "dim": dim_idx,
                "mae_train": mae_train,
                "mae_test": mae_test,
                "denorm_mae_train": denorm_mae_train,
                "denorm_mae_test": denorm_mae_test,
                "mse_train": mse_train,
                "mse_test": mse_test,
                "r2_train": r2_train,
                "r2_test": r2_test,
            }
        )

        print(f"\n--- 出力次元 {dim_idx} ({y_param_names[dim_idx]}) ---")
        print(
            f"  Train MAE:  {mae_train:.6f} (逆正規化: {denorm_mae_train:.2f})  |  Test MAE:  {mae_test:.6f} (逆正規化: {denorm_mae_test:.2f})"
        )
        print(f"  Train MSE:  {mse_train:.6f}  |  Test MSE:  {mse_test:.6f}")
        print(f"  Train R²:   {r2_train:.6f}  |  Test R²:   {r2_test:.6f}")

    # --- 結果レポート作成 ---
    report_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== 線形予測分析レポート (行列累積版) ===\n")
        f.write(f"実験名: {EXPERIMENT_NAME}\n")
        f.write(f"手法: {METHOD.upper()} (Scikit-Learn非依存)\n")
        if METHOD == "ridge":
            f.write(f"Ridge Alpha: {RIDGE_ALPHA}\n")
        f.write(f"文脈窓: {CONTEXT_SIZE} (前後{CONTEXT_SIZE}フレーム)\n")
        f.write(f"入力次元数: {model.coef_.shape[1]}\n")
        f.write(f"Train フレーム数: {total_train_frames:,} (完全に使用)\n")
        f.write(f"Test フレーム数: {total_test_frames:,}\n\n")

        for r in results:
            f.write(f"--- 出力次元 {r['dim']} ({y_param_names[r['dim']]}) ---\n")
            f.write(
                f"  Train MAE:  {r['mae_train']:.6f} (逆正規化: {r['denorm_mae_train']:.2f})  |  Test MAE:  {r['mae_test']:.6f} (逆正規化: {r['denorm_mae_test']:.2f})\n"
            )
            f.write(
                f"  Train MSE:  {r['mse_train']:.6f}  |  Test MSE:  {r['mse_test']:.6f}\n"
            )
            f.write(
                f"  Train R²:   {r['r2_train']:.6f}  |  Test R²:   {r['r2_test']:.6f}\n\n"
            )

        # 回帰係数一覧
        f.write(f"=== 回帰係数一覧 (出力次元 0) ===\n")
        coef = model.coef_[0]
        intercept = model.intercept_[0]

        f.write(f"バイアス (intercept): {float(intercept):.6f}\n\n")

        sorted_indices = np.argsort(np.abs(coef))[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            f.write(
                f"  {rank:3d}. {feature_names[idx]:30s}  coef={float(coef[idx]):+.6f}\n"
            )

    print(f"レポートを {report_path} に保存しました。")

    if model.coef_.shape[1] > 1:
        coef_for_plot = model.coef_[0]
        coef_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_coefficients.png")
        _plot_coefficients(coef_for_plot, feature_names, coef_path)

    evaluator.plot_scatter(OUTPUT_DIR, EXPERIMENT_NAME, y_param_names)
    excel_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_eval.xlsx")
    evaluator.export_report(excel_path)

    print("\n--- 分析完了 ---")


if __name__ == "__main__":
    main()
