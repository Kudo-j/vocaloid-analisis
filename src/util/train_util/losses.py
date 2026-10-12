"""
losses.py
=========
学習で使用するカスタム損失関数とマスク生成の共通ユーティリティ。

全てのLossクラスは統一インターフェースに従う:
    forward(pred, target, lengths, voicing_mask)
        -> (total_loss: Tensor, metrics: dict[str, float])

metrics辞書のキーは自由に定義可能で、train.pyとplot.pyが動的に処理する。
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


def _build_final_mask(target, lengths, voicing_mask):
    """
    パディングマスクと無音マスクを合成した最終マスクを生成する共通処理。

    Args:
        target: (B, S, D) のターゲットテンソル。
        lengths: (B,) の各サンプルの実際のフレーム数。
        voicing_mask: (B, S, 1) の無音マスク。Noneの場合はパディングのみ考慮。

    Returns:
        final_mask: (B, S, 1) のfloatテンソル。
    """
    max_len = target.size(1)
    mask_len = create_mask(lengths, max_len).unsqueeze(-1).float()  # (B, S, 1)

    if voicing_mask is not None:
        final_mask = mask_len * voicing_mask
    else:
        final_mask = mask_len

    return final_mask


class MaskedMSELoss(nn.Module):
    """
    マスク付きMSE損失関数。

    パディング部分と無音部分をマスクで除外した純粋なMSEのみを計算する。
    基礎的な実験や、Loss関数の影響を排除したい比較実験で使用する。
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, lengths, voicing_mask):
        final_mask = _build_final_mask(target, lengths, voicing_mask)

        loss_mse_element = self.mse(pred, target)
        loss_mse = (loss_mse_element * final_mask).sum() / (final_mask.sum() + 1e-9)

        return loss_mse, {"mse": loss_mse.item()}


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
        final_mask = _build_final_mask(target, lengths, voicing_mask)

        # --- 通常のMSE Loss ---
        loss_mse_element = self.mse(pred, target)
        loss_mse = (loss_mse_element * final_mask).sum() / (final_mask.sum() + 1e-9)

        # --- 微分Loss ---
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        mask_diff = final_mask[:, 1:, :] * final_mask[:, :-1, :]

        loss_diff_element = self.mse(pred_diff, target_diff)
        movement_weight = torch.abs(target_diff) * 25.0

        loss_diff = (loss_diff_element * mask_diff * movement_weight).sum() / (
            mask_diff.sum() + 1e-9
        )

        total_loss = loss_mse + loss_diff

        return total_loss, {"mse": loss_mse.item(), "diff": loss_diff.item()}


# --- Lossクラスのレジストリ ---
LOSS_REGISTRY = {
    "mse": MaskedMSELoss,
    "advanced": AdvancedLoss,
}


def build_loss(loss_type):
    """
    文字列からLossクラスのインスタンスを生成するファクトリ関数。

    Args:
        loss_type: "mse" または "advanced"。

    Returns:
        nn.Module: Lossクラスのインスタンス。

    Raises:
        ValueError: 未知のloss_typeが指定された場合。
    """
    if loss_type not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(
            f"未知のLoss種別: '{loss_type}' (選択肢: {available})"
        )
    return LOSS_REGISTRY[loss_type]()
