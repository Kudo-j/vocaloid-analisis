import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from dataset import create_datasets, collate_fn
from model_decoder import TransformerModel


# --- 設定 ---
DATA_NAME = "augAll_tagDYN"
DATA_DIR = f"./data/processed_{DATA_NAME}"  # preprocess.py の出力先
MODEL_DIR = "./model"
FEATURES_JSON = os.path.join(DATA_DIR, "features.json")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"{DATA_NAME}_model.pth")
PLOT_SAVE_PATH = os.path.join(MODEL_DIR, f"{DATA_NAME}_loss.png")

SONG_PATH = "songlist.txt"


# --- ハイパーパラメータ ---
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 1024
DROPOUT = 0.25
BATCH_SIZE = 64
ACCUM_STEPS = 16
EPOCHS = 100  # 学習エポック数
LR = 0.0001  # 学習率

# --- Scheduled Sampling 設定 ---
# sampling_ratio: モデル自身の予測値を使う確率 (0.0=完全Teacher Forcing, 1.0=完全自己回帰)
# 学習初期は0.0から開始し、エポックが進むにつれて線形にmax値まで上昇する
SS_START_EPOCH = (
    1  # Scheduled Samplingを開始するエポック (それ以前は純粋なTeacher Forcing)
)
SS_MAX_RATIO = (
    1  # sampling_ratioの最大値 (1.0にすると完全自己回帰になるので0.3~0.5推奨)
)
SS_RAMPUP_EPOCHS = 1  # SS_START_EPOCHからSS_MAX_RATIOに到達するまでのエポック数


def worker_init_fn(worker_id):
    # 各ワーカーが異なるシードを持つように設定
    # 現在の時刻やworker_idを種にする
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


# どこからマスクか判定する関数
def create_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask


# --- 高機能損失関数 (Masked MSE + Derivative Loss) ---
class AdvancedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, lengths, voicing_mask):
        # --- 1. マスク作成 (パディング & 無音) ---
        max_len = target.size(1)
        mask_len = create_mask(lengths, max_len).unsqueeze(-1).float()  # (B, S, 1)

        # 【修正】条件分岐・計算を削除し、そのまま乗算するだけ
        if voicing_mask is not None:
            final_mask = mask_len * voicing_mask
        else:
            final_mask = mask_len

        # --- 2. 通常のMSE Loss ---
        loss_mse_element = self.mse(pred, target)
        loss_mse = (loss_mse_element * final_mask).sum() / (final_mask.sum() + 1e-9)

        # --- 3. 微分Loss ---
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        mask_diff = final_mask[:, 1:, :] * final_mask[:, :-1, :]

        loss_diff_element = self.mse(pred_diff, target_diff)
        movement_weight = torch.abs(target_diff) * 50.0

        loss_diff = (loss_diff_element * mask_diff * movement_weight).sum() / (
            mask_diff.sum() + 1e-9
        )

        total_loss = loss_mse + loss_diff

        return total_loss, loss_mse.item(), loss_diff.item()


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

    # --- プロット処理 (変更なし) ---
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

    # 【重要】右側に余白を大きく確保 (0.7 くらいまで縮める)
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


# --- メイン処理 ---
def main():
    if not os.path.exists(FEATURES_JSON):
        print(f"Error: {FEATURES_JSON} が見つかりません。")
        print("preprocess.py を先に実行してください。")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. features.json から次元数を読み込む
    with open(FEATURES_JSON, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    NUM_X = feature_info["num_x_features"]
    NUM_Y = feature_info["num_y_features"]

    if NUM_Y == 0:
        print(
            "Error: Yの特徴量数が0です。preprocess.py の target_param_config を確認してください。"
        )
        return

    print(f"X features: {NUM_X}, Y features: {NUM_Y}")

    # 2. モデルのインスタンス化
    model = TransformerModel(
        num_x_features=NUM_X,
        num_y_features=NUM_Y,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)
    # モデル作成後、GPUに転送

    # テスト環境では使えず
    # try:
    #     model = torch.compile(model)
    #     print("Model compiled with torch.compile")
    # except Exception as e:
    #     print(f"Skipping torch.compile: {e}")

    # 3. データセットとデータローダーの準備
    print("Preparing datasets...")

    # ★変更: 分割関数の呼び出し
    songlist_path = SONG_PATH  # パスは適宜調整
    train_dataset, val_dataset, test_dataset, feature_info = create_datasets(
        DATA_DIR, songlist_path, valid_ratio=0.10, test_ratio=0.10
    )

    print(f"Dataset Chunks -> Train: {len(train_dataset)} | Valid: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("Error: Train dataset has 0 chunks.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,  # 1回に渡すデータの数 (64個など)
        shuffle=True,  # 毎回順番をシャッフルするか？
        collate_fn=collate_fn,  # バラバラの長さを揃える関数
        num_workers=4,  # 並列処理の数
        worker_init_fn=worker_init_fn,
        pin_memory=True,  # GPU転送の高速化
        # メモリ（RAM）上に、GPU転送専用の固定領域を確保
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 学習に影響しないので、順番は何でも良い
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # 4. 損失関数とオプティマイザ
    # MSE
    # criterion = MaskedMSELoss().to(device)
    # 微分Loss
    criterion = AdvancedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # スケーラーの準備
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))

    # 切り捨て( // )ではなく、切り上げ( math.ceil )を使って余り分をカウントに含める
    steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5.0e-4,  # 到達する最大学習率 (元のLRより少し高めに設定するのが定石です)
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=0.3,  # 全体の30%を使ってWarmup（徐々に上げる）
        anneal_strategy="cos",
        div_factor=10.0,  # 初期LR = max_lr / 25
        final_div_factor=10.0,  # 最終LR = 初期LR / 1000
    )

    # 5. 学習ループ
    best_val_loss = float("inf")  # 初期値、∞に
    best_epoch = 0

    # グラフ描画用に記録するリスト
    train_loss_history = []
    val_loss_history = []

    # ★追加: 成分ごとの履歴リストを作成
    train_mse_history = []
    train_diff_history = []
    val_mse_history = []
    val_diff_history = []

    hyperparams = {
        "D_MODEL": D_MODEL,
        "NHEAD": NHEAD,
        "ENC_LAYERS": NUM_ENCODER_LAYERS,
        "DEC_LAYERS": NUM_DECODER_LAYERS,
        "FFN_DIM": DIM_FEEDFORWARD,
        "DROPOUT": DROPOUT,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
    }

    print("--- 学習開始 ---")
    total_start_time = time.time()

    try:
        for epoch in range(1, EPOCHS + 1):  # エポック（全データの周回）のループ
            epoch_start_time = time.time()

            # --- 訓練モード ---
            model.train()
            train_loss_total = 0  # 初期化

            train_mse_total = 0
            train_diff_total = 0

            print(f"--- Epoch {epoch}/{EPOCHS} ---")
            train_batches = len(train_loader)

            for i, batch in enumerate(train_loader):
                x_batch, y_batch, mask_batch, len_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                len_batch = len_batch.to(device)

                max_len = x_batch.size(1)
                padding_mask = ~create_mask(len_batch, max_len)

                # --- Scheduled Sampling ---
                # sampling_ratio に基づき、正解データとモデル予測をフレーム単位で混合
                if epoch < SS_START_EPOCH:
                    sampling_ratio = 0.0
                else:
                    progress = (epoch - SS_START_EPOCH) / max(SS_RAMPUP_EPOCHS, 1)
                    sampling_ratio = min(SS_MAX_RATIO, progress * SS_MAX_RATIO)

                with torch.amp.autocast("cuda", enabled=(device_type == "cuda")):
                    if sampling_ratio < 1e-6:
                        # 純粋な Teacher Forcing
                        tgt_input = torch.zeros_like(y_batch)
                        tgt_input[:, 1:, :] = y_batch[:, :-1, :]
                        output = model(
                            x_batch,
                            tgt_input,
                            src_key_padding_mask=padding_mask,
                            tgt_key_padding_mask=padding_mask,
                        )
                    else:
                        # Scheduled Sampling: フレームごとにGTか予測値かを選択
                        seq_len = y_batch.size(1)
                        tgt_input = torch.zeros_like(y_batch)
                        # まず1フレーム目は常にゼロ（開始トークン）

                        # エンコーダ出力を事前に計算（繰り返し計算を避ける）
                        memory = model._encode(
                            x_batch, src_key_padding_mask=padding_mask
                        )

                        for t in range(1, seq_len):
                            # フレームごとにランダムに決定
                            if random.random() < sampling_ratio:
                                # モデル自身の予測値を使用
                                with torch.no_grad():
                                    partial_out = model._decode(
                                        tgt_input[:, :t, :],
                                        memory,
                                        tgt_key_padding_mask=(
                                            padding_mask[:, :t]
                                            if padding_mask is not None
                                            else None
                                        ),
                                        memory_key_padding_mask=padding_mask,
                                    )
                                tgt_input[:, t, :] = partial_out[:, -1, :]
                            else:
                                # 正解データを使用 (Teacher Forcing)
                                tgt_input[:, t, :] = y_batch[:, t - 1, :]

                        output = model._decode(
                            tgt_input,
                            memory,
                            tgt_key_padding_mask=padding_mask,
                            memory_key_padding_mask=padding_mask,
                        )

                    # 微分Loss用
                    loss, mse_val, diff_val = criterion(
                        output, y_batch, len_batch, mask_batch
                    )

                train_mse_total += mse_val
                train_diff_total += diff_val

                # ★変更: Lossを蓄積数で割る（勾配が大きくなりすぎないように）
                loss = loss / ACCUM_STEPS
                scaler.scale(loss).backward()

                # ★変更: 一定回数(ACCUM_STEPS)ごとに重みを更新
                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    # 学習率LR変更
                    scheduler.step()

                    torch.cuda.empty_cache()

                    # 現在のLossを計算
                    current_loss = loss.item() * ACCUM_STEPS

                    # メッセージを作成
                    # log_msg = f"Epoch {epoch}/{EPOCHS} | Step {i+1}/{train_batches} | Loss: {current_loss:.4f}"
                    # 微分Loss用
                    log_msg = (
                        f"Epoch {epoch}/{EPOCHS} | Step {i+1}/{train_batches} | "
                        f"Loss: {current_loss:.6f} (M:{mse_val:.6f}, D:{diff_val:.6f}) | "
                        f"SS: {sampling_ratio:.2f}"
                    )

                    # 【工夫1】 status.txt には「毎回」書き込み、「上書き」する (モード 'w')
                    # これならファイルサイズは常に1行分だけです。
                    with open("status.txt", "w") as f:
                        f.write(log_msg + "\n")

                train_loss_total += loss.item() * ACCUM_STEPS

            avg_train_loss = train_loss_total / train_batches
            avg_train_mse = train_mse_total / train_batches
            avg_train_diff = train_diff_total / train_batches

            # --- 検証モード ---
            model.eval()
            val_loss_total = 0
            val_mse_total = 0
            val_diff_total = 0

            print(f"  Validating...")
            with torch.no_grad():
                for batch in val_loader:
                    x_batch, y_batch, mask_batch, len_batch = batch

                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    len_batch = len_batch.to(device)

                    max_len = x_batch.size(1)
                    padding_mask = ~create_mask(len_batch, max_len)

                    # Teacher Forcing: 検証時も同様にデコーダ入力を作成
                    tgt_input = torch.zeros_like(y_batch)
                    tgt_input[:, 1:, :] = y_batch[:, :-1, :]

                    output = model(
                        x_batch,
                        tgt_input,
                        src_key_padding_mask=padding_mask,
                        tgt_key_padding_mask=padding_mask,
                    )
                    loss, mse_val, diff_val = criterion(
                        output, y_batch, len_batch, mask_batch
                    )
                    val_loss_total += loss.item()
                    val_mse_total += mse_val
                    val_diff_total += diff_val

            avg_val_loss = val_loss_total / len(val_loader)
            avg_val_mse = val_mse_total / len(val_loader)
            avg_val_diff = val_diff_total / len(val_loader)

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)

            train_mse_history.append(avg_train_mse)  # 追加
            train_diff_history.append(avg_train_diff)  # 追加
            val_mse_history.append(avg_val_mse)  # 追加
            val_diff_history.append(avg_val_diff)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(
                f"Epoch {epoch} DONE | "
                f"Time: {epoch_duration:.2f}s\n"
                f"Train: {avg_train_loss:.6f} (M:{avg_train_mse:.6f}, D:{avg_train_diff:.6f})\n "
                f"Val: {avg_val_loss:.6f} (M:{avg_val_mse:.6f}, D:{avg_val_diff:.6f})"
            )

            if epoch == 1:
                remaining_epochs = EPOCHS - 1
                estimated_remaining_time_sec = epoch_duration * remaining_epochs
                if estimated_remaining_time_sec > 3600:
                    print(
                        f"  [Info] 残り {remaining_epochs} エポックの推定所要時間: 約 {estimated_remaining_time_sec / 3600:.1f} 時間"
                    )
                elif estimated_remaining_time_sec > 60:
                    print(
                        f"  [Info] 残り {remaining_epochs} エポックの推定所要時間: 約 {estimated_remaining_time_sec / 60:.1f} 分"
                    )
                else:
                    print(
                        f"  [Info] 残り {remaining_epochs} エポックの推定所要時間: 約 {estimated_remaining_time_sec:.0f} 秒"
                    )

            # --- モデルの保存 ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(
                    f"  -> Model saved to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.6f})"
                )

    except KeyboardInterrupt:
        print("\n\n!!! 学習が強制中断されました (KeyboardInterrupt) !!!")
        print("現在の進捗までのグラフとデータを保存します...")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"--- 学習完了 (Total Time: {total_duration / 3600:.2f} 時間) ---")
    if len(train_loss_history) > 0:
        print(f"Best Validation Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

        # グラフを保存 (hyperparams は上で定義済みなのでOK)
        save_loss_plot(
            train_loss_history,
            val_loss_history,
            train_mse_history,  # 追加
            val_mse_history,  # 追加
            train_diff_history,  # 追加
            val_diff_history,  # 追加
            PLOT_SAVE_PATH,
            best_epoch,
            best_val_loss,
            hyperparams,
        )
    else:
        print("学習データが記録されていないため、グラフは保存されませんでした。")


if __name__ == "__main__":
    main()
