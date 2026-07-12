"""
train.py — 09_DYN_DNN 共通学習スクリプト
08_DYN_DNN_Estimation/train_diffDYN.py を完全コピーし、以下を変更:
  - configのインポートをスクリプト先頭で切り替え可能にした
  - dict入出力のモデルインターフェースに対応
  - compute_loss_baseline を dict 出力に対応させた
"""
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt

# =============================================================================
# ===== config 切り替え（ここを書き換えて実験を切り替え） =====
# =============================================================================
import config.exp_diff_learning as cfg
# import config.exp_diff_learning as cfg
# import config.exp_multibranch as cfg

from dataset_09 import create_datasets, collate_fn
from models.model_factory import get_model


def worker_init_fn(worker_id):
    """各ワーカーが異なるシードを持つように設定"""
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def save_loss_plot(
    train_total_hist, val_total_hist,
    train_dyn_hist, val_dyn_hist,
    train_diff_hist, val_diff_hist,
    save_path, best_epoch, best_val_loss
):
    """Train/Valid の各種Loss推移をグラフ化して保存（08完全再現）"""
    plt.figure(figsize=(14, 7))

    # Total Loss (実線)
    plt.plot(train_total_hist, label="Train Total", linewidth=2.0, color="blue", alpha=0.7)
    plt.plot(val_total_hist, label="Val Total", linewidth=2.0, color="orange", alpha=0.7)

    # DYN Loss (点線)
    plt.plot(train_dyn_hist, label="Train DYN", linewidth=1.5, linestyle=":", color="magenta", alpha=0.6)
    plt.plot(val_dyn_hist, label="Val DYN", linewidth=1.5, linestyle=":", color="navy", alpha=0.6)

    # Diff Loss (一点鎖線)
    plt.plot(train_diff_hist, label="Train Diff", linewidth=1.5, linestyle="-.", color="green", alpha=0.6)
    plt.plot(val_diff_hist, label="Val Diff", linewidth=1.5, linestyle="-.", color="brown", alpha=0.6)

    # Best Point
    best_idx = best_epoch - 1
    best_label = f"Best (Ep {best_epoch})\nTotal: {best_val_loss:.6f}"
    plt.plot(best_idx, best_val_loss, "r*", markersize=15, label=best_label, zorder=5)

    plt.yscale("log")
    plt.title(f"Training: {cfg.EXPERIMENT_NAME} ({cfg.MODEL_TYPE})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss - Log Scale")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.subplots_adjust(right=0.65)
    plt.legend(
        bbox_to_anchor=(1.05, 1.0), loc="upper left",
        borderaxespad=0, fontsize="small", title="Loss Legends"
    )

    # ハイパーパラメータ情報
    param_str = "--- Hyperparameters ---\n\n"
    param_str += f"MODEL: {cfg.MODEL_TYPE}\n"
    param_str += f"FEATURES: {cfg.INPUT_FEATURES}\n"
    param_str += f"BATCH_SIZE: {cfg.BATCH_SIZE}\n"
    param_str += f"EPOCHS: {cfg.EPOCHS}\n"
    param_str += f"LR: {cfg.LEARNING_RATE}\n"
    param_str += f"DYN_W: {cfg.LOSS_DYN_WEIGHT}\n"
    param_str += f"DIFF_W: {cfg.LOSS_DIFF_WEIGHT}\n"

    plt.text(
        1.05, 0.0, param_str,
        transform=plt.gca().transAxes, fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.savefig(save_path)
    print(f"学習グラフを {save_path} に保存しました。")
    plt.close()


def compute_loss(outputs, dyns, masks, criterion):
    """
    DYN直接推定用 Loss 計算（08 compute_loss_baseline を完全コピーし dict I/F に対応）
    """
    pred_dyn = outputs["dyn_pred"]

    # --- DYN MSE ---
    loss_dyn_batch = criterion(pred_dyn, dyns) * masks
    mask_sum = masks.sum() + 1e-8
    loss_dyn_mean = loss_dyn_batch.sum() / mask_sum

    # --- 微分 Loss (DYN の隣接フレーム差分) ---
    pred_dyn_diff = pred_dyn[:, 1:, :] - pred_dyn[:, :-1, :]
    target_dyn_diff = dyns[:, 1:, :] - dyns[:, :-1, :]
    mask_diff = masks[:, 1:, :] * masks[:, :-1, :]

    loss_diff_element = criterion(pred_dyn_diff, target_dyn_diff)
    # 動きが大きい箇所を重視する重み
    movement_weight = torch.abs(target_dyn_diff) * 25.0
    loss_diff = (loss_diff_element * mask_diff * movement_weight).sum() / (mask_diff.sum() + 1e-8)

    # 合算
    total = (cfg.LOSS_DYN_WEIGHT * loss_dyn_mean) + (cfg.LOSS_DIFF_WEIGHT * loss_diff)

    return total, loss_dyn_mean, loss_diff


def train():
    print("=" * 60)
    print(f"  09_DYN_DNN Training")
    print(f"  Config: {cfg.EXPERIMENT_NAME}")
    print(f"  Model: {cfg.MODEL_TYPE}")
    print(f"  Features: {cfg.INPUT_FEATURES}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # データセット準備
    train_ds, valid_ds = create_datasets(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=False,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, persistent_workers=False,
    )

    # モデル構築
    model = get_model(cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss(reduction='none')

    steps_per_epoch = math.ceil(len(train_loader) / cfg.ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.LEARNING_RATE * 10,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.EPOCHS,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # AMP (自動混合精度)
    scaler = torch.amp.GradScaler('cuda', enabled=(device_type == "cuda"))

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    best_valid_loss = float('inf')
    best_epoch = 1

    # Loss 履歴
    train_total_hist, val_total_hist = [], []
    train_dyn_hist, val_dyn_hist = [], []
    train_diff_hist, val_diff_hist = [], []

    print("\nTraining Started...")
    total_start_time = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start_time = time.time()

        # --- Training ---
        model.train()
        ep_loss, ep_dyn, ep_diff = 0.0, 0.0, 0.0

        optimizer.zero_grad()

        for i, (features_t_dict, features_ut_dict, dyns, masks, _, vsqx) in enumerate(train_loader):
            dyns = dyns.to(device)
            masks = masks.to(device)

            # Untunedデータの混合（指定割合だけバッチ内のデータをUntunedに差し替える）
            mix_ratio = getattr(cfg, "UNTUNED_MIX_RATIO", 0.0)
            if mix_ratio > 0.0 and len(features_ut_dict) > 0:
                batch_size = dyns.size(0)
                num_untuned = int(batch_size * mix_ratio)
                
                if num_untuned > 0:
                    untuned_indices = torch.randperm(batch_size)[:num_untuned]
                    
                    # 入力をUntunedに差し替え
                    for k in features_t_dict.keys():
                        features_t_dict[k][untuned_indices] = features_ut_dict[k][untuned_indices]
                    
                    # 正解DYNをデフォルト値（64を [-1, 1] に正規化した値）に差し替え
                    dyn_default_norm = (cfg.DYN_DEFAULT / 127.0) * 2.0 - 1.0
                    dyns[untuned_indices] = dyn_default_norm

            # dict内の各テンソルをdeviceに転送（DiffNet用に _tuned / _untuned キーも作成）
            inputs = {}
            for k, v in features_t_dict.items():
                inputs[k] = v.to(device)
                inputs[f"{k}_tuned"] = v.to(device)
            for k, v in features_ut_dict.items():
                inputs[f"{k}_untuned"] = v.to(device)
            if vsqx is not None:
                inputs["vsqx"] = vsqx.to(device)

            with torch.amp.autocast('cuda', enabled=(device_type == "cuda")):
                outputs = model(inputs)
                loss, l_dyn, l_diff = compute_loss(outputs, dyns, masks, criterion)
                # 勾配累積のためLossを割る
                loss = loss / cfg.ACCUM_STEPS

            scaler.scale(loss).backward()

            # ログ記録用に元のスケールの値を足し込む
            ep_loss += (loss.item() * cfg.ACCUM_STEPS)
            ep_dyn += l_dyn.item()
            ep_diff += l_diff.item()

            if (i + 1) % cfg.ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # AMPがNaNを検知して最適化をスキップした場合はSchedulerもスキップ
                skip_lr_sched = (scale_before > scaler.get_scale())
                if not skip_lr_sched:
                    scheduler.step()

        num_batches = len(train_loader)
        ep_loss /= num_batches
        ep_dyn /= num_batches
        ep_diff /= num_batches

        train_total_hist.append(ep_loss)
        train_dyn_hist.append(ep_dyn)
        train_diff_hist.append(ep_diff)

        # --- Validation ---
        model.eval()
        v_loss, v_dyn, v_diff = 0.0, 0.0, 0.0
        with torch.no_grad():
            for features_t_dict, features_ut_dict, dyns, masks, _, vsqx in valid_loader:
                inputs = {}
                for k, v in features_t_dict.items():
                    inputs[k] = v.to(device)
                    inputs[f"{k}_tuned"] = v.to(device)
                for k, v in features_ut_dict.items():
                    inputs[f"{k}_untuned"] = v.to(device)
                if vsqx is not None:
                    inputs["vsqx"] = vsqx.to(device)
                dyns, masks = dyns.to(device), masks.to(device)

                with torch.amp.autocast('cuda', enabled=(device_type == "cuda")):
                    outputs = model(inputs)
                    loss, l_dyn, l_diff = compute_loss(outputs, dyns, masks, criterion)

                v_loss += loss.item()
                v_dyn += l_dyn.item()
                v_diff += l_diff.item()

        num_val_batches = len(valid_loader)
        v_loss /= num_val_batches
        v_dyn /= num_val_batches
        v_diff /= num_val_batches

        val_total_hist.append(v_loss)
        val_dyn_hist.append(v_dyn)
        val_diff_hist.append(v_diff)

        # 時間計測
        epoch_elapsed = time.time() - epoch_start_time
        epoch_mins = int(epoch_elapsed // 60)
        epoch_secs = int(epoch_elapsed % 60)

        print(
            f"Epoch {epoch:02d}/{cfg.EPOCHS} | "
            f"Train: {ep_loss:.4f} (DYN:{ep_dyn:.5f}, Diff:{ep_diff:.6f}) | "
            f"Valid: {v_loss:.4f} (DYN:{v_dyn:.5f}, Diff:{v_diff:.6f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_mins}m {epoch_secs}s"
        )

        # ベストモデルの保存
        if v_loss < best_valid_loss:
            best_valid_loss = v_loss
            best_epoch = epoch
            torch.save(model.state_dict(), cfg.MODEL_PATH)
            print(f"  --> Model saved! (Valid Loss improved)")

    total_elapsed = time.time() - total_start_time
    total_mins = int(total_elapsed // 60)
    total_secs = int(total_elapsed % 60)
    print(f"\nTraining Complete! Total Time: {total_mins}m {total_secs}s")
    print(f"Best Epoch: {best_epoch} | Best Valid Loss: {best_valid_loss:.6f}")

    # Loss グラフ保存
    plot_path = os.path.join(cfg.OUTPUT_DIR, "training_loss.png")
    save_loss_plot(
        train_total_hist, val_total_hist,
        train_dyn_hist, val_dyn_hist,
        train_diff_hist, val_diff_hist,
        plot_path, best_epoch, best_valid_loss,
    )


if __name__ == "__main__":
    train()
