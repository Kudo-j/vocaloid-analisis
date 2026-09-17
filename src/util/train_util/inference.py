"""
inference.py
============
スライディングウィンドウ（重み付きOverlap-Add）推論のための共通ユーティリティ。
predict.py および train.py の評価ループで使用される推論処理を提供する。
"""

import numpy as np
import torch

__all__ = ["_get_window", "weighted_overlap_add", "run_sliding_window_inference"]


def _get_window(length: int, offset: float = 0.1) -> np.ndarray:
    """
    オフセット付きハニング窓を生成する。

    Args:
        length (int): 窓の長さ（フレーム数）。
        offset (float): 端点における最小重み（ゼロ除算防止および端情報の保持用）。

    Returns:
        np.ndarray: 形状 (length,) の1次元窓関数配列。
    """
    if length <= 2:
        return np.ones(length, dtype=np.float32)
    hann = np.hanning(length).astype(np.float32)
    return offset + (1.0 - offset) * hann


def weighted_overlap_add(
    pred_sum: np.ndarray,
    count_map: np.ndarray,
    start: int,
    end: int,
    values: np.ndarray,
    offset: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    窓関数による重み付けを行い、予測チャンクを累積配列に加算（Overlap-Add）する。

    Args:
        pred_sum (np.ndarray): 累積された予測値の合計配列 (T, D)。
        count_map (np.ndarray): 累積された重みの合計配列 (T, 1)。
        start (int): チャンクの開始インデックス（含む）。
        end (int): チャンクの終了インデックス（含まない）。
        values (np.ndarray): チャンクの予測値配列 (end - start, D)。
        offset (float): ハニング窓のオフセット値。

    Returns:
        tuple[np.ndarray, np.ndarray]: 更新された (pred_sum, count_map) のタプル。
    """
    length = end - start
    w = _get_window(length, offset)[:, np.newaxis]  # (length, 1)
    pred_sum[start:end] += values * w
    count_map[start:end] += w
    return pred_sum, count_map


def run_sliding_window_inference(
    model: torch.nn.Module,
    x_tensors: dict[str, torch.Tensor],
    total_frames: int,
    chunk_size: int,
    chunk_hop: int,
    device: torch.device,
    device_type: str,
    offset: float = 0.1,
) -> np.ndarray:
    """
    重み付きOverlap-Addを用いたスライディングウィンドウ推論を実行する。

    Args:
        model: ロード済みDNNモデル（指定デバイス配置済み）。
        x_tensors: 入力テンソルの辞書 {ブランチ名: Tensor (1, T, D)}。
        total_frames (int): 推論対象の全フレーム数。
        chunk_size (int): ウィンドウサイズ（チャンク長）。
        chunk_hop (int): ホップサイズ（移動量）。
        device (torch.device): 実行デバイス。
        device_type (str): "cuda" または "cpu" (autocast判定用)。
        offset (float): ハニング窓のオフセット値。

    Returns:
        np.ndarray: 全フレームの最終予測配列 (total_frames, 出力次元数)。
    """
    pred_sum = None
    count_map = None
    start = 0
    model.eval()
    with torch.no_grad():
        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            x_chunk = {k: v[:, start:end, :].to(device) for k, v in x_tensors.items()}
            with torch.cuda.amp.autocast(enabled=(device_type == "cuda")):
                output = model(x_chunk)
            chunk_len = end - start
            out_np = output[0, :chunk_len].cpu().numpy()
            if pred_sum is None:
                dim = out_np.shape[1] if out_np.ndim > 1 else 1
                pred_sum = np.zeros((total_frames, dim), dtype=np.float32)
                count_map = np.zeros((total_frames, 1), dtype=np.float32)
            pred_sum, count_map = weighted_overlap_add(
                pred_sum, count_map, start, end, out_np, offset
            )
            start += chunk_hop

    # オーバーラップ部分の重み付き平均を算出
    valid_idx = count_map[:, 0] > 0
    pred_sum[valid_idx] /= count_map[valid_idx]
    return pred_sum
