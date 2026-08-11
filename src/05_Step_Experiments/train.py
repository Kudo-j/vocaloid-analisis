"""
train.py
========
全Step共通の統合学習スクリプト。

config 領域のパラメータを変更するだけで、
Step1～4 の全ての実験を実行できる。

使い方:
    1. 下記の config 領域を実験条件に合わせて編集する
    2. python train.py を実行する
"""

import os
import sys
import json
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# --- パス設定 ---
# util フォルダをインポートパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "util"))

from train_util.dataset import VocalDataset, create_datasets, collate_fn
from train_util.losses import build_loss, create_mask
from train_util.early_stopping import EarlyStopping
from train_util.evaluate import Evaluator
from plot_util.plot import save_loss_plot

# モデル
from models.cnn_bilstm import CNNBiLSTM
from models.bilstm_base import BiLSTMBase
from models.fusion_block import FeatureFusionBlock
from models.unified_model import UnifiedModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def worker_init_fn(worker_id):
    """各ワーカーが異なるシードを持つように設定する。"""
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def _build_model(model_type, input_dim, output_dim, model_params):
    """
    config の MODEL_TYPE に基づいてモデルを動的にインスタンス化する。

    Args:
        model_type: モデル種別文字列。
        input_dim: 入力次元数。
        output_dim: 出力次元数。
        model_params: モデルパラメータの辞書。

    Returns:
        model: nn.Module のインスタンス。
    """
    model_registry = {
        "cnn_bilstm": CNNBiLSTM,
        "bilstm_base": BiLSTMBase,
    }

    if model_type not in model_registry:
        raise ValueError(f"未知のモデル種別: {model_type}")

    params = model_params.get(model_type, {})
    model_class = model_registry[model_type]

    return model_class(input_dim=input_dim, output_dim=output_dim, **params)


def _concat_branch_tensors(branch_dict, device):
    """
    辞書型のブランチテンソルを結合して1つのテンソルにする。

    Args:
        branch_dict: {"audio": (B,S,D1), "branch_basic": (B,S,D2), ...}
        device: 転送先デバイス。

    Returns:
        concatenated: (B, S, D1+D2+...) の結合テンソル。
    """
    tensors = [v.to(device) for v in branch_dict.values()]
    return torch.cat(tensors, dim=-1)


def train_model(custom_config=None, is_tuning=False):
    import builtins

    _original_print = builtins.print
    if is_tuning:
        builtins.print = lambda *args, **kwargs: None

    if custom_config is None:
        with open(
            os.path.join(BASE_DIR, "train_settings.json"), "r", encoding="utf-8"
        ) as f:
            config = json.load(f)
    else:
        config = custom_config

    DATA_DIR = config["DATA_DIR"]
    OUTPUT_BASE_DIR = config.get("OUTPUT_BASE_DIR", "output")
    EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
    SONGLIST_PATH = config["SONGLIST_PATH"]
    SPLIT_MODE = config["SPLIT_MODE"]
    MODEL_TYPE = config["MODEL_TYPE"]
    FUSION_STRATEGY = config["FUSION_STRATEGY"]
    BRANCH_MODULES = config["BRANCH_MODULES"]
    LOSS_TYPE = config.get("LOSS_TYPE", "advanced")
    BATCH_SIZE = config["BATCH_SIZE"]
    ACCUM_STEPS = config["ACCUM_STEPS"]
    EPOCHS = config["EPOCHS"]
    LR = config["LR"]
    MAX_LR = config["MAX_LR"]
    PATIENCE = config["PATIENCE"]
    VALID_RATIO = config["VALID_RATIO"]
    TEST_RATIO = config["TEST_RATIO"]
    MODEL_PARAMS = config.get("MODEL_PARAMS", {})

    # --- 出力ディレクトリの構築 ---
    # output/Step1_RMS_CNNBiLSTM/cnn_bilstm/
    EXPERIMENT_DIR = os.path.join(
        BASE_DIR, OUTPUT_BASE_DIR, EXPERIMENT_NAME, MODEL_TYPE
    )
    TEST_EVAL_DIR = os.path.join(EXPERIMENT_DIR, "test_evaluate")

    if not is_tuning:
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        os.makedirs(TEST_EVAL_DIR, exist_ok=True)

    model_save_path = os.path.join(EXPERIMENT_DIR, "best_model.pth")
    plot_save_path = os.path.join(EXPERIMENT_DIR, "loss_curve.png")

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- features.json の読み込みと次元数の算出 ---
    features_path = os.path.join(DATA_DIR, "features.json")
    if not os.path.exists(features_path):
        print(f"Error: {features_path} が見つかりません。")
        print("build_dataset.py を先に実行してください。")
        return

    with open(features_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    # 辞書型の x_dims, y_dims
    x_dims = feature_info.get("x_dims", {})
    y_dims = feature_info.get("y_dims", {})
    total_y_dim = sum(y_dims.values())

    if total_y_dim == 0:
        print(
            "Error: 出力次元数が0です。config の TARGET_PARAM_CONFIG を確認してください。"
        )
        return

    # --- Premodel (Fusion Block) の構築と次元計算 ---
    fusion_block = FeatureFusionBlock(
        x_dims_dict=x_dims, strategy=FUSION_STRATEGY, branch_modules=BRANCH_MODULES
    )
    total_x_dim = fusion_block.output_dim

    print(f"入力ブランチ: {dict(x_dims)} -> Fusion後次元数: {total_x_dim}")
    print(f"出力次元数: {total_y_dim} {dict(y_dims)}")

    # --- Coreモデルと統合 ---
    core_model = _build_model(MODEL_TYPE, total_x_dim, total_y_dim, MODEL_PARAMS)
    model = UnifiedModel(fusion_block, core_model).to(device)

    print(f"メインモデル: {MODEL_TYPE}")
    print(f"結合戦略: {FUSION_STRATEGY}")
    print(f"総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # --- データセットの準備 ---
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, feature_info = create_datasets(
        DATA_DIR,
        SONGLIST_PATH,
        valid_ratio=VALID_RATIO,
        test_ratio=TEST_RATIO,
        split_mode=SPLIT_MODE,
    )

    print(
        f"Dataset Chunks -> Train: {len(train_dataset)} | Valid: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    if len(train_dataset) == 0:
        print("Error: Train dataset has 0 chunks.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- 損失関数とオプティマイザ ---
    criterion = build_loss(LOSS_TYPE).to(device)
    print(f"損失関数: {LOSS_TYPE}")
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=(device_type == "cuda"))

    steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=10.0,
    )

    early_stopper = EarlyStopping(patience=PATIENCE)

    # --- 学習ループ ---
    best_val_loss = float("inf")
    best_epoch = 0

    train_loss_history = []
    val_loss_history = []
    train_metrics_history = {}  # {"mse": [...], "diff": [...], ...}
    val_metrics_history = {}    # 同上

    hyperparams = {
        "MODEL": MODEL_TYPE,
        "LOSS": LOSS_TYPE,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "FUSION_STRATEGY": FUSION_STRATEGY,
    }
    for k, v in BRANCH_MODULES.items():
        hyperparams[f"Branch_{k}"] = v
    hyperparams.update(MODEL_PARAMS.get(MODEL_TYPE, {}))

    print("--- 学習開始 ---")
    total_start_time = time.time()

    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()

            # === 訓練モード ===
            model.train()
            train_loss_total = 0
            train_metrics_total = {}  # 動的にキーが追加される

            print(f"--- Epoch {epoch}/{EPOCHS} ---")
            train_batches = len(train_loader)

            for i, batch in enumerate(train_loader):
                x_dict, y_dict, mask_batch, len_batch = batch

                # x_dictはモデル内で処理するためデバイスに転送して辞書のまま保持
                x_dict_device = {k: v.to(device) for k, v in x_dict.items()}

                # y_dictは損失計算用なので結合する
                y_batch = _concat_branch_tensors(y_dict, device)
                mask_batch = mask_batch.to(device)
                len_batch = len_batch.to(device)

                max_len = list(x_dict.values())[0].size(1)
                padding_mask = ~create_mask(len_batch, max_len)

                with autocast(enabled=(device_type == "cuda")):
                    output = model(x_dict_device)
                    loss, metrics = criterion(
                        output, y_batch, len_batch, mask_batch
                    )

                for mk, mv in metrics.items():
                    train_metrics_total[mk] = train_metrics_total.get(mk, 0) + mv

                loss = loss / ACCUM_STEPS
                scaler.scale(loss).backward()

                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    torch.cuda.empty_cache()

                    current_loss = loss.item() * ACCUM_STEPS
                    if len(metrics) > 1:
                        metrics_str = ", ".join(f"{k}:{v:.6f}" for k, v in metrics.items())
                        log_msg = (
                            f"Epoch {epoch}/{EPOCHS} | Step {i+1}/{train_batches} | "
                            f"Loss: {current_loss:.6f} ({metrics_str})"
                        )
                    else:
                        log_msg = (
                            f"Epoch {epoch}/{EPOCHS} | Step {i+1}/{train_batches} | "
                            f"Loss: {current_loss:.6f}"
                        )
                    
                    with open(os.path.join(EXPERIMENT_DIR, "status.txt"), "w") as f:
                        f.write(log_msg + "\n")

                train_loss_total += loss.item() * ACCUM_STEPS

            avg_train_loss = train_loss_total / train_batches
            avg_train_metrics = {k: v / train_batches for k, v in train_metrics_total.items()}

            # === 検証モード ===
            model.eval()
            val_loss_total = 0
            val_metrics_total = {}

            print(f"  Validating...")
            with torch.no_grad():
                for batch in val_loader:
                    x_dict, y_dict, mask_batch, len_batch = batch

                    x_dict_device = {k: v.to(device) for k, v in x_dict.items()}
                    y_batch = _concat_branch_tensors(y_dict, device)
                    mask_batch = mask_batch.to(device)
                    len_batch = len_batch.to(device)

                    output = model(x_dict_device)
                    loss, metrics = criterion(
                        output, y_batch, len_batch, mask_batch
                    )
                    val_loss_total += loss.item()
                    for mk, mv in metrics.items():
                        val_metrics_total[mk] = val_metrics_total.get(mk, 0) + mv

            avg_val_loss = val_loss_total / len(val_loader)
            avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics_total.items()}

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            for k in avg_train_metrics:
                train_metrics_history.setdefault(k, []).append(avg_train_metrics[k])
            for k in avg_val_metrics:
                val_metrics_history.setdefault(k, []).append(avg_val_metrics[k])

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            if len(avg_train_metrics) > 1:
                train_metrics_str = ", ".join(f"{k}:{v:.6f}" for k, v in avg_train_metrics.items())
                val_metrics_str = ", ".join(f"{k}:{v:.6f}" for k, v in avg_val_metrics.items())
                print(
                    f"Epoch {epoch} DONE | Time: {epoch_duration:.2f}s\n"
                    f"  Train: {avg_train_loss:.6f} ({train_metrics_str})\n"
                    f"  Val:   {avg_val_loss:.6f} ({val_metrics_str})"
                )
            else:
                print(
                    f"Epoch {epoch} DONE | Time: {epoch_duration:.2f}s\n"
                    f"  Train: {avg_train_loss:.6f}\n"
                    f"  Val:   {avg_val_loss:.6f}"
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
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                if not is_tuning:
                    torch.save(model.state_dict(), model_save_path)
                    print(
                        f"  -> Model saved to {model_save_path} (Val Loss: {best_val_loss:.6f})"
                    )

            # --- Early Stopping ---
            if early_stopper(avg_val_loss):
                break

    except KeyboardInterrupt:
        print("\n\n!!! 学習が強制中断されました (KeyboardInterrupt) !!!")
        print("現在の進捗までのグラフとデータを保存します...")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"--- 学習完了 (Total Time: {total_duration / 3600:.2f} 時間) ---")

    if len(train_loss_history) > 0:
        print(f"Best Validation Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

        if not is_tuning:
            save_loss_plot(
                train_loss_history,
                val_loss_history,
                train_metrics_history,
                val_metrics_history,
                plot_save_path,
                best_epoch,
                best_val_loss,
                hyperparams,
            )
    else:
        print("学習データが記録されていないため、グラフは保存されませんでした。")

    # --- 軽量モード（Optuna用）---
    if is_tuning:
        builtins.print = _original_print
        print(
            f"[Trial完了] {best_epoch}エポックで終了 | Best Val Loss: {best_val_loss:.6f}"
        )
        history_dict = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "train_metrics": train_metrics_history,
            "val_metrics": val_metrics_history,
            "best_epoch": best_epoch,
            "hyperparams": hyperparams,
        }
        return best_val_loss, best_model_state, history_dict

    return best_val_loss


def evaluate_model(custom_config=None, model_load_path=None):
    if custom_config is None:
        with open(
            os.path.join(BASE_DIR, "train_settings.json"), "r", encoding="utf-8"
        ) as f:
            config = json.load(f)
    else:
        config = custom_config

    DATA_DIR = config["DATA_DIR"]
    OUTPUT_BASE_DIR = config.get("OUTPUT_BASE_DIR", "output")
    EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
    SONGLIST_PATH = config["SONGLIST_PATH"]
    SPLIT_MODE = config["SPLIT_MODE"]
    MODEL_TYPE = config["MODEL_TYPE"]
    FUSION_STRATEGY = config["FUSION_STRATEGY"]
    BRANCH_MODULES = config["BRANCH_MODULES"]
    VALID_RATIO = config["VALID_RATIO"]
    TEST_RATIO = config["TEST_RATIO"]
    MODEL_PARAMS = config.get("MODEL_PARAMS", {})

    EXPERIMENT_DIR = os.path.join(
        BASE_DIR, OUTPUT_BASE_DIR, EXPERIMENT_NAME, MODEL_TYPE
    )
    TEST_EVAL_DIR = os.path.join(EXPERIMENT_DIR, "test_evaluate")
    os.makedirs(TEST_EVAL_DIR, exist_ok=True)

    if model_load_path is None:
        model_load_path = os.path.join(EXPERIMENT_DIR, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- 共通評価モジュール(Evaluator)によるテストデータ詳細評価 ---")
    print(f"Evaluating Model: {model_load_path}")

    features_path = os.path.join(DATA_DIR, "features.json")
    if not os.path.exists(features_path):
        print(f"Error: {features_path} が見つかりません。")
        return

    with open(features_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    x_dims = feature_info.get("x_dims", {})
    y_dims = feature_info.get("y_dims", {})
    total_y_dim = sum(y_dims.values())

    fusion_block = FeatureFusionBlock(
        x_dims_dict=x_dims, strategy=FUSION_STRATEGY, branch_modules=BRANCH_MODULES
    )
    total_x_dim = fusion_block.output_dim

    base_model_type = (
        MODEL_TYPE.split("_trial_")[0] if "_trial_" in MODEL_TYPE else MODEL_TYPE
    )
    core_model = _build_model(base_model_type, total_x_dim, total_y_dim, MODEL_PARAMS)
    model = UnifiedModel(fusion_block, core_model).to(device)

    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    _, _, test_dataset, _ = create_datasets(
        DATA_DIR,
        SONGLIST_PATH,
        valid_ratio=VALID_RATIO,
        test_ratio=TEST_RATIO,
        split_mode=SPLIT_MODE,
    )

    if len(test_dataset) == 0:
        print(
            "\n※ テストデータ(TEST_RATIO=0)が存在しないため、詳細評価はスキップしました。"
        )
        return

    evaluator = Evaluator(device=device)
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
        y_param_names = [f"Dim_{i}" for i in range(total_y_dim)]

    test_files_unique = list(dict.fromkeys([c[0] for c in test_dataset.chunk_map]))

    for i, song_name in enumerate(test_files_unique):
        ds = VocalDataset(DATA_DIR, [song_name], feature_info, mode="eval_full")
        if len(ds) == 0:
            continue
        dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        all_y_true = []
        all_y_pred = []
        all_v_mask = []

        with torch.no_grad():
            for batch in dl:
                x_dict, y_dict, mask_batch, len_batch = batch
                x_dict_device = {k: v.to(device) for k, v in x_dict.items()}
                y_batch = _concat_branch_tensors(y_dict, device)
                mask_batch = mask_batch.to(device)
                len_batch = len_batch.to(device)

                with autocast(enabled=(device_type == "cuda")):
                    output = model(x_dict_device)

                for b_idx in range(len(len_batch)):
                    seq_len = len_batch[b_idx].item()
                    y_true_seq = y_batch[b_idx, :seq_len]
                    y_pred_seq = output[b_idx, :seq_len]
                    v_mask_seq = mask_batch[b_idx, :seq_len]
                    if v_mask_seq.dim() > 1:
                        v_mask_seq = v_mask_seq.squeeze(-1)

                    all_y_true.append(y_true_seq.cpu().numpy())
                    all_y_pred.append(y_pred_seq.cpu().numpy())
                    all_v_mask.append(v_mask_seq.cpu().numpy())

        if len(all_y_true) > 0:
            y_true_np = np.concatenate(all_y_true, axis=0)
            y_pred_np = np.concatenate(all_y_pred, axis=0)
            v_mask_np = np.concatenate(all_v_mask, axis=0)
            evaluator.add_song_result(
                song_name,
                y_true_np,
                y_pred_np,
                y_param_names,
                voicing_mask=v_mask_np,
            )

            if i < 5:
                wave_path = os.path.join(TEST_EVAL_DIR, f"waveform_{song_name}.png")
                evaluator.plot_waveform(
                    song_name,
                    y_true_np,
                    y_pred_np,
                    wave_path,
                    y_param_names,
                    voicing_mask=v_mask_np,
                )

    evaluator.plot_scatter(TEST_EVAL_DIR, EXPERIMENT_NAME, y_param_names)
    excel_path = os.path.join(TEST_EVAL_DIR, f"{EXPERIMENT_NAME}_eval.xlsx")
    evaluator.export_report(excel_path)


if __name__ == "__main__":
    with open(os.path.join(BASE_DIR, "train_settings.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    train_model(custom_config=config)
    evaluate_model(custom_config=config)
