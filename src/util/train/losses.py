"""
losses.py
=========
学習で使用するカスタム損失関数とマスク生成の共通ユーティリティ。

train_encoder.py の AdvancedLoss クラスと create_mask 関数を移植したもの。
各Stepの学習スクリプトから呼び出して使用する。
"""

import torch
import torch.nn as nn


def create_mask(lengths, max_len):
    """
    パディングされたバッチの有効フレーム範囲を示すブーリアンマスクを生成する。

    Args:
        lengths: 各サンプルの実際のフレーム数 (B,)
        max_len: バッチ内の最大フレーム数

    Returns:
        mask: (B, max_len) のブーリアンテンソル。有効フレームがTrue。
    """
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask


class AdvancedLoss(nn.Module):
    """
    高機能損失関数 (Masked MSE + Derivative Loss)。

    通常のMSE損失に加えて、時間方向の微分（変化量）の誤差も計算する。
    パディング部分と無音部分をマスクで除外して計算する。
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, lengths, voicing_mask):
        # --- 1. マスク作成 (パディング & 無音) ---
        max_len = target.size(1)
        mask_len = create_mask(lengths, max_len).unsqueeze(-1).float()  # (B, S, 1)

        # voicing_mask が存在する場合は乗算して無音部分も除外
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
        movement_weight = torch.abs(target_diff) * 25.0

        loss_diff = (loss_diff_element * mask_diff * movement_weight).sum() / (
            mask_diff.sum() + 1e-9
        )

        total_loss = loss_mse + loss_diff

        return total_loss, loss_mse.item(), loss_diff.item()
