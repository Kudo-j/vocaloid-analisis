"""
inference.py
============
スライディングウィンドウ（重み付きOverlap-Add）推論のための共通ユーティリティ。
predict.py および train.py の評価ループで使用される推論処理を提供する。
"""

import numpy as np
import torch

__all__ = ["_get_window", "weighted_overlap_add", "run_sliding_window_inference", "eval_song"]


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


def eval_song(
    model,
    song_data,
    song_name="unknown",
    evaluator=None,
    y_param_names=None,
    save_wave_path=None,
    chunk_size=600,
    chunk_hop=300,
    offset=0.1,
    device=None,
    sr=44100,
    shift_length=512,
    remove_intro_silence=True,
    silence_margin_sec=1.0,
):
    """
    1曲分の特徴量データに対してスライディングウィンドウ推論・加重オーバーラップ加算を行い、
    必要に応じて Evaluator による評価メトリクス算出および波形プロット保存を行う共通関数。

    train.py の evaluate_model と predict.py の推論処理を共通化するために設計されている。
    入力データは build_dataset.py が生成する NPZ ファイルと同一の辞書構造を期待する。

    Args:
        model (torch.nn.Module): 学習済みモデル（eval() 状態であること）。
        song_data (dict): 1曲分のオンメモリデータ。以下のキーを含む:
            - "x" (dict[str, np.ndarray]): 入力ブランチ辞書 {"audio": (T, D1), ...}
            - "y" (dict[str, np.ndarray], optional): 正解ブランチ辞書 {"dynamic": (T, D), ...}
            - "voicing_mask" (np.ndarray, optional): 発声マスク (T, 1) または (T,)
            - "extra_features" (dict[str, np.ndarray], optional): 波形プロット用追加特徴量
        song_name (str): 曲名識別子。
        evaluator (Evaluator, optional): 評価器インスタンス。
            複数曲の集計を行う場合は呼び出し元から共有インスタンスを渡す。
            None かつ save_wave_path 指定時は内部で自動生成する。
        y_param_names (list[str], optional): 出力パラメータ名リスト (例: ["DYN", "PIT"])。
        save_wave_path (str, optional): 波形比較グラフの保存パス。None の場合はプロットをスキップ。
        chunk_size (int): スライディングウィンドウのチャンクサイズ (デフォルト: 600)。
        chunk_hop (int): スライディングウィンドウのホップサイズ (デフォルト: 300)。
        offset (float): ハニング窓のオフセット値 (デフォルト: 0.1)。
        device (torch.device, optional): 実行デバイス。None の場合はモデルから自動検出。
        sr (int): サンプリング周波数 (デフォルト: 44100)。
        shift_length (int): フレームシフト長 (デフォルト: 512)。
        remove_intro_silence (bool): 波形プロット時にイントロの無音区間をカットするか。
        silence_margin_sec (float): 波形プロット時に歌唱前後に残す無音の余白（秒、デフォルト: 1.0）。

    Returns:
        tuple[np.ndarray, np.ndarray | None]:
            - y_pred_full: 予測値配列 (total_frames, 出力次元数)
            - y_true_full: 正解値配列 (total_frames, 出力次元数)、正解データ未指定時は None
    """
    # デバイスの判定
    if device is None:
        device = next(model.parameters()).device
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # 入力データの取得
    x_dict = song_data["x"]
    y_dict = song_data.get("y")
    voicing_mask = song_data.get("voicing_mask")
    extra_features = song_data.get("extra_features")

    # x_dict の各配列を torch.Tensor (1, T, D) に整形
    x_tensors = {}
    total_frames = None
    for k, v in x_dict.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v).unsqueeze(0)
        elif isinstance(v, torch.Tensor):
            t = v.unsqueeze(0) if v.ndim == 2 else v
        else:
            raise TypeError(f"x_dict['{k}'] の型が不正: {type(v)}")
        x_tensors[k] = t
        if total_frames is None:
            total_frames = t.shape[1]

    # スライディングウィンドウ推論（加重オーバーラップ加算）
    y_pred_full = run_sliding_window_inference(
        model=model,
        x_tensors=x_tensors,
        total_frames=total_frames,
        chunk_size=chunk_size,
        chunk_hop=chunk_hop,
        device=device,
        device_type=device_type,
        offset=offset,
    )

    # 正解データの整形（ブランチ辞書 → 結合配列）
    y_true_full = None
    if y_dict is not None and len(y_dict) > 0:
        y_true_full = np.concatenate(
            list(y_dict.values()), axis=-1
        ).astype(np.float32)

    # voicing_mask の1次元化（Evaluator は (T,) 形式を期待する）
    v_mask = None
    if voicing_mask is not None:
        v_mask = voicing_mask.squeeze(-1) if voicing_mask.ndim > 1 else voicing_mask

    # Evaluator による評価・描画
    if y_true_full is not None and (evaluator is not None or save_wave_path is not None):
        if evaluator is None:
            # save_wave_path のみ指定された場合、Evaluator を内部生成
            try:
                from train_util.evaluate import Evaluator
            except ImportError:
                from evaluate import Evaluator
            evaluator = Evaluator(device=device)

        evaluator.add_song_result(
            song_name,
            y_true_full,
            y_pred_full,
            y_param_names or [],
            voicing_mask=v_mask,
        )

        if save_wave_path is not None:
            evaluator.plot_waveform(
                song_name,
                y_true_full,
                y_pred_full,
                save_wave_path,
                y_param_names,
                voicing_mask=v_mask,
                extra_features=extra_features,
                sr=sr,
                shift_length=shift_length,
                remove_intro_silence=remove_intro_silence,
                silence_margin_sec=silence_margin_sec,
            )

    return y_pred_full, y_true_full
