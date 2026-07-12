"""
Step 1: Profile抽出器の独立学習
「DYNが変化すると、各周波数帯がどの程度変化するか」の法則を学習させる。

Loss = MSE( Profile × 正解DYN,  真の差分(Tuned - Untuned) )

Profileが0になるサボりを防ぐため、Untuned Mel全体ではなく「差分のみ」で評価する。
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

import config_diffDYN as cfg
from dataset import create_datasets, collate_fn
from model import get_model


def worker_init_fn(worker_id):
    """各ワーカーが異なるシードを持つように設定"""
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def save_step1_loss_plot(train_hist, val_hist, save_path, best_epoch, best_val_loss):
    """Step 1 の Delta Loss 推移をグラフ化して保存"""
    plt.figure(figsize=(12, 6))
    plt.plot(train_hist, label="Train Delta Loss", linewidth=2.0, color="blue", alpha=0.7)
    plt.plot(val_hist, label="Val Delta Loss", linewidth=2.0, color="orange", alpha=0.7)

    best_idx = best_epoch - 1
    plt.plot(best_idx, best_val_loss, "r*", markersize=15,
             label=f"Best (Ep {best_epoch}) = {best_val_loss:.6f}", zorder=5)

    plt.yscale("log")
    plt.title("Step 1: Profile Extractor Training (Delta Reconstruction Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=11)

    param_str = (
        f"ENCODER: {cfg.ENCODER_TYPE}\n"
        f"BATCH_SIZE: {cfg.BATCH_SIZE}\n"
        f"EPOCHS: {cfg.STEP1_EPOCHS}\n"
        f"LR: {cfg.STEP1_LEARNING_RATE}\n"
    )
    plt.text(
        1.02, 0.5, param_str,
        transform=plt.gca().transAxes, fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Step 1 学習グラフを {save_path} に保存しました。")
    plt.close()


def compute_delta_loss(delta_pred, true_delta, masks, criterion):
    """
    差分のみでLossを計算（Profileが0にサボることを防ぐ）
    delta_pred: (B, T, 80) = Profile × 正解DYN
    true_delta: (B, T, 80) = Tuned Mel - Untuned Mel
    masks: (B, T, 1)
    """
    loss_batch = criterion(delta_pred, true_delta) * masks  # (B, T, 80)
    mask_sum = masks.sum() + 1e-8
    loss_mean = loss_batch.sum() / (mask_sum * true_delta.shape[-1])
    return loss_mean


def train_step1():
    print("=" * 60)
    print(f"  Step 1: Profile Extractor Training ({cfg.ENCODER_TYPE.upper()})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # データセット準備
    train_ds, valid_ds = create_datasets(
        cfg.PREPROCESSED_DIR, songlist_path=cfg.SONGLIST_PATH, valid_ratio=0.1
    )
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

    # モデル初期化 (Factory関数を使用)
    model = get_model(cfg).to(device)
    
    # Step 1 では Encoder + ProfileExtractor のみ最適化対象にする
    step1_params = list(model.encoder.parameters()) + list(model.profile_extractor.parameters())
    optimizer = optim.AdamW(step1_params, lr=cfg.STEP1_LEARNING_RATE)
    criterion = nn.MSELoss(reduction='none')

    steps_per_epoch = math.ceil(len(train_loader) / cfg.ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.STEP1_LEARNING_RATE * 10,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.STEP1_EPOCHS,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device_type == "cuda"))

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.STEP1_OUTPUT_DIR, exist_ok=True)
    best_valid_loss = float('inf')
    best_epoch = 1

    train_hist, val_hist = [], []

    print("\nStep 1 Training Started...")
    total_start_time = time.time()

    for epoch in range(1, cfg.STEP1_EPOCHS + 1):
        epoch_start_time = time.time()

        # --- Training ---
        model.train()
        ep_loss = 0.0
        optimizer.zero_grad()

        for i, (features_t_dict, features_ut_dict, dyns, masks, _) in enumerate(train_loader):
            # 設定された全特徴量をTime軸以降(dim=-1)で結合
            x_t = torch.cat([features_t_dict[f] for f in cfg.INPUT_FEATURES], dim=-1).to(device)
            x_ut = torch.cat([features_ut_dict[f] for f in cfg.INPUT_FEATURES], dim=-1).to(device)
            dyns, masks = dyns.to(device), masks.to(device)

            # 真の差分（ターゲット）
            true_delta = x_t - x_ut  # (B, T, feature_dim)

            with torch.amp.autocast('cuda', enabled=(device_type == "cuda")):
                # forward_step1: Profile × 正解DYN → 予測差分
                delta_pred, profile = model.forward_step1(x_t, dyns)
                loss = compute_delta_loss(delta_pred, true_delta, masks, criterion)
                loss = loss / cfg.ACCUM_STEPS

            scaler.scale(loss).backward()
            ep_loss += (loss.item() * cfg.ACCUM_STEPS)

            if (i + 1) % cfg.ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(step1_params, max_norm=1.0)
                
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                skip_lr_sched = (scale_before > scaler.get_scale())
                if not skip_lr_sched:
                    scheduler.step()

        num_batches = len(train_loader)
        ep_loss /= num_batches
        train_hist.append(ep_loss)

        # --- Validation ---
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for features_t_dict, features_ut_dict, dyns, masks, _ in valid_loader:
                x_t = torch.cat([features_t_dict[f] for f in cfg.INPUT_FEATURES], dim=-1).to(device)
                x_ut = torch.cat([features_ut_dict[f] for f in cfg.INPUT_FEATURES], dim=-1).to(device)
                dyns, masks = dyns.to(device), masks.to(device)

                true_delta = x_t - x_ut

                with torch.amp.autocast('cuda', enabled=(device_type == "cuda")):
                    delta_pred, profile = model.forward_step1(x_t, dyns)
                    loss = compute_delta_loss(delta_pred, true_delta, masks, criterion)

                v_loss += loss.item()

        num_val_batches = len(valid_loader)
        v_loss /= num_val_batches
        val_hist.append(v_loss)

        # 時間計測
        epoch_elapsed = time.time() - epoch_start_time
        epoch_mins = int(epoch_elapsed // 60)
        epoch_secs = int(epoch_elapsed % 60)

        print(
            f"Epoch {epoch:02d}/{cfg.STEP1_EPOCHS} | "
            f"Train Delta: {ep_loss:.6f} | Valid Delta: {v_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_mins}m {epoch_secs}s"
        )

        # ベストモデルの保存（Encoder + ProfileExtractor の重みのみ）
        if v_loss < best_valid_loss:
            best_valid_loss = v_loss
            best_epoch = epoch
            # Encoder + ProfileExtractor の state_dict を保存
            step1_state = {
                'encoder': model.encoder.state_dict(),
                'profile_extractor': model.profile_extractor.state_dict(),
            }
            torch.save(step1_state, cfg.STEP1_MODEL_PATH)
            print(f"  --> Step 1 Model saved! (Valid Loss improved)")

    total_elapsed = time.time() - total_start_time
    total_mins = int(total_elapsed // 60)
    total_secs = int(total_elapsed % 60)
    print(f"\nStep 1 Training Complete! Total Time: {total_mins}m {total_secs}s")
    print(f"Best Epoch: {best_epoch} | Best Valid Delta Loss: {best_valid_loss:.6f}")

    # Loss グラフ保存
    plot_path = os.path.join(cfg.STEP1_OUTPUT_DIR, "step1_delta_loss.png")
    save_step1_loss_plot(train_hist, val_hist, plot_path, best_epoch, best_valid_loss)

    # ===== Profile の健全性チェック =====
    print("\n--- Profile 健全性チェック ---")
    model.eval()
    profile_abs_sum = 0.0
    profile_count = 0
    with torch.no_grad():
        for features_t_dict, features_ut_dict, dyns, masks, _ in valid_loader:
            x_t = torch.cat([features_t_dict[f] for f in cfg.INPUT_FEATURES], dim=-1).to(device)
            dyns, masks = dyns.to(device), masks.to(device)
            _, profile = model.forward_step1(x_t, dyns)
            # マスク付きの平均絶対値
            profile_abs = (torch.abs(profile) * masks).sum()
            profile_abs_sum += profile_abs.item()
            profile_count += masks.sum().item() * profile.shape[-1]
    
    avg_profile_abs = profile_abs_sum / (profile_count + 1e-8)
    print(f"  Profile平均絶対値: {avg_profile_abs:.6f}")
    if avg_profile_abs < 1e-4:
        print("  [WARNING] Profileがほぼ0です。Step 1の学習が不十分な可能性があります。")
    else:
        print("  [OK] Profileは非ゼロです。Step 2に進めます。")


if __name__ == "__main__":
    train_step1()
