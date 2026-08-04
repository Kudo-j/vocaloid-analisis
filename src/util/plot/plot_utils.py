"""
plot_utils.py
=============
学習Loss履歴のプロットを行う共通ユーティリティ。

train_encoder.py の save_loss_plot 関数を移植したもの。
各Stepの学習スクリプトから呼び出して使用する。
"""

import matplotlib.pyplot as plt


def save_loss_plot(
    train_history,
    val_history,
    train_mse_history,
    val_mse_history,
    train_diff_history,
    val_diff_history,
    save_path,
    best_epoch,
    best_val_loss,
    hyperparams,
):
    """
    Total, MSE, Diff のLoss履歴をプロットする。
    凡例をグラフ右外・上部に、ハイパーパラメータをグラフ右外・下部に配置する。
    """
    # グラフのサイズ設定
    plt.figure(figsize=(14, 7))

    # --- プロット処理 ---
    # 1. Total Loss (実線)
    plt.plot(train_history, label="Train Total", linewidth=2.0, color="blue", alpha=0.7)
    plt.plot(val_history, label="Val Total", linewidth=2.0, color="orange", alpha=0.7)

    # 2. MSE Loss (破線)
    plt.plot(
        train_mse_history,
        label="Train MSE",
        linewidth=1.5,
        linestyle="--",
        color="red",
        alpha=0.6,
    )
    plt.plot(
        val_mse_history,
        label="Val MSE",
        linewidth=1.5,
        linestyle="--",
        color="cyan",
        alpha=0.8,
    )

    # 3. Diff Loss (点線)
    plt.plot(
        train_diff_history,
        label="Train Diff",
        linewidth=1.5,
        linestyle=":",
        color="magenta",
        alpha=0.6,
    )
    plt.plot(
        val_diff_history,
        label="Val Diff",
        linewidth=1.5,
        linestyle=":",
        color="navy",
        alpha=0.6,
    )

    # Best Point (詳細ラベル付き)
    best_epoch_idx = best_epoch - 1
    best_val_mse = val_mse_history[best_epoch_idx]
    best_val_diff = val_diff_history[best_epoch_idx]

    best_label = (
        f"Best (Ep {best_epoch})\n"
        f"Total: {best_val_loss:.6f}\n"
        f"MSE : {best_val_mse:.6f}\n"
        f"Diff : {best_val_diff:.6f}"
    )

    plt.plot(
        best_epoch_idx,
        best_val_loss,
        "r*",
        markersize=15,
        label=best_label,
        zorder=5,
    )

    # --- 軸とレイアウトの設定 ---
    plt.yscale("log")
    plt.title("Training and Validation Loss Breakdown (Log Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss - Log Scale")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # 右側に余白を大きく確保
    plt.subplots_adjust(right=0.65)

    # --- 1. 凡例をグラフ外（右上）に配置 ---
    # bbox_to_anchor=(x, y): (1.05, 1.0) はグラフ右上の角のすぐ外側を指す
    # loc="upper left": 凡例ボックスの左上を、上記アンカーポイントに合わせる
    plt.legend(
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        borderaxespad=0,
        fontsize="small",
        title="Loss Legends",
    )

    # --- 2. ハイパーパラメータをグラフ外（右下）に配置 ---
    param_str = "--- Hyperparameters ---\n\n"
    for key, value in hyperparams.items():
        param_str += f"{key}: {value}\n"

    # plt.text を使い、transform=plt.gca().transAxes で座標系をグラフ基準にする
    # (1.05, 0.0) はグラフ右下の角のすぐ外側
    # verticalalignment="bottom" で下揃えにする
    plt.text(
        1.05,
        0.0,  # x, y (axes座標)
        param_str,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.savefig(save_path)
    print(f"学習グラフを {save_path} に保存しました。")
    plt.close()
