"""
plot.py
=============
学習Loss履歴のプロットを行う共通ユーティリティ。

metrics_history辞書のキーを動的に読み取り、
任意の数のサブメトリクスを自動でグラフ化する。
"""

import matplotlib.pyplot as plt

# サブメトリクスの描画スタイル定義（自動で循環する）
_SUB_STYLES = [
    {"linestyle": "--", "alpha": 0.6},
    {"linestyle": ":",  "alpha": 0.6},
    {"linestyle": "-.", "alpha": 0.6},
]

# Train / Val のカラーパレット（キーごとに自動で循環する）
_TRAIN_COLORS = ["red", "magenta", "green", "brown", "purple"]
_VAL_COLORS = ["cyan", "navy", "lime", "orange", "teal"]


def save_loss_plot(
    train_total_history,
    val_total_history,
    train_metrics_history,
    val_metrics_history,
    save_path,
    best_epoch,
    best_val_loss,
    hyperparams,
):
    """
    Total Lossと各サブメトリクスの履歴をプロットする。

    Args:
        train_total_history: 各エポックのTrain Total Loss (list[float])
        val_total_history: 各エポックのVal Total Loss (list[float])
        train_metrics_history: {metric_name: list[float]} 形式のサブメトリクス (Train)
        val_metrics_history: {metric_name: list[float]} 形式のサブメトリクス (Val)
        save_path: 保存先パス
        best_epoch: ベストエポック番号 (1-indexed)
        best_val_loss: ベストVal Loss値
        hyperparams: ハイパーパラメータ辞書
    """
    plt.figure(figsize=(14, 7))

    # --- 1. Total Loss (実線) ---
    # サブメトリクスが1つの場合、Totalと完全に重なって見栄えが悪いため非表示にする
    if len(train_metrics_history) > 1:
        plt.plot(train_total_history, label="Train Total", linewidth=2.0, color="blue", alpha=0.7)
        plt.plot(val_total_history, label="Val Total", linewidth=2.0, color="orange", alpha=0.7)

    # --- 2. サブメトリクス (動的描画) ---
    # train_metrics_historyのキー一覧を取得（例: ["mse", "diff"]）
    sub_keys = sorted(train_metrics_history.keys())

    for idx, key in enumerate(sub_keys):
        style = _SUB_STYLES[idx % len(_SUB_STYLES)]
        t_color = _TRAIN_COLORS[idx % len(_TRAIN_COLORS)]
        v_color = _VAL_COLORS[idx % len(_VAL_COLORS)]

        plt.plot(
            train_metrics_history[key],
            label=f"Train {key.upper()}",
            linewidth=1.5,
            color=t_color,
            **style,
        )
        plt.plot(
            val_metrics_history[key],
            label=f"Val {key.upper()}",
            linewidth=1.5,
            color=v_color,
            **style,
        )

    # --- 3. Best Point (詳細ラベル付き) ---
    best_epoch_idx = best_epoch - 1

    best_label = f"Best (Ep {best_epoch})"
    if len(sub_keys) > 1:
        best_label += f"\nTotal: {best_val_loss:.6f}"
        
    for key in sub_keys:
        val_at_best = val_metrics_history[key][best_epoch_idx]
        best_label += f"\n{key.upper()}: {val_at_best:.6f}"

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

    # --- 凡例をグラフ外（右上）に配置 ---
    plt.legend(
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        borderaxespad=0,
        fontsize="small",
        title="Loss Legends",
    )

    # --- ハイパーパラメータをグラフ外（右下）に配置 ---
    param_str = "--- Hyperparameters ---\n\n"
    for key, value in hyperparams.items():
        param_str += f"{key}: {value}\n"

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
