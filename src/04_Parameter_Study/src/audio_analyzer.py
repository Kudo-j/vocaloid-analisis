"""
音響特徴量の抽出・差分計算・相関分析を担当するモジュール。
feature_extractor.py を内部で利用し、比較解析に特化した処理を提供する。
"""

import numpy as np
from scipy import stats
import librosa

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "util")
)
from feature_extractor import AudioFeatureExtractor


# 特徴量の表示名マッピング
FEATURE_DISPLAY_NAMES = {
    "mel": "Mel Spectrogram",
    "rms": "RMS Energy",
    "f0_world": "F0 (World)",
    "f0_crepe": "F0 (CREPE)",
    "mfcc": "MFCC",
    "spectral_centroid": "Spectral Centroid",
    "spectral_bandwidth": "Spectral Bandwidth",
    "spectral_contrast": "Spectral Contrast",
    "hnr": "Harmonic-to-Noise Ratio",
    "zero_crossing_rate": "Zero Crossing Rate",
    "cqt": "CQT",
    "stft": "STFT",
    "aperiodicity": "Aperiodicity",
    "waveform": "Waveform",
    "waveform_diff": "Waveform Diff",
    "source_mel": "Source Mel (LPC Residual)",
    "source_rms": "Source RMS (LPC Residual)",
    "source_stft": "Source STFT (LPC Residual)",
    "formant_1_amp": "Formant 1 Amplitude",
    "formant_2_amp": "Formant 2 Amplitude",
    "formant_3_amp": "Formant 3 Amplitude",
    "formant_4_amp": "Formant 4 Amplitude",
    "formant_5_amp": "Formant 5 Amplitude",
    "formant_6_amp": "Formant 6 Amplitude",
    "formant_7_amp": "Formant 7 Amplitude",
    "formant_8_amp": "Formant 8 Amplitude",
    "formant_9_amp": "Formant 9 Amplitude",
    "formant_10_amp": "Formant 10 Amplitude",
    "air_freq_amp": "Air Frequency Amplitude",
}

# 特徴量のY軸ラベル（単位）
FEATURE_YLABELS = {
    "mel": "Frequency (Mel)",
    "rms": "Amplitude",
    "f0_world": "Frequency (Hz)",
    "f0_crepe": "Frequency (Hz)",
    "mfcc": "Value",
    "spectral_centroid": "Frequency (Hz)",
    "spectral_bandwidth": "Frequency (Hz)",
    "spectral_contrast": "Value",
    "hnr": "dB",
    "zero_crossing_rate": "Rate",
    "cqt": "Frequency (Hz)",
    "stft": "Frequency (Hz)",
    "aperiodicity": "Value",
    "waveform": "Amplitude",
    "waveform_diff": "Amplitude",
    "source_mel": "Frequency (Mel)",
    "source_rms": "Amplitude",
    "source_stft": "Frequency (Hz)",
    "formant_1_amp": "Amplitude (dB)",
    "formant_2_amp": "Amplitude (dB)",
    "formant_3_amp": "Amplitude (dB)",
    "formant_4_amp": "Amplitude (dB)",
    "formant_5_amp": "Amplitude (dB)",
    "formant_6_amp": "Amplitude (dB)",
    "formant_7_amp": "Amplitude (dB)",
    "air_freq_amp": "Amplitude (dB)",
}

# 特徴量の描画タイプ分類
FEATURE_PLOT_TYPES = {
    # 1次元時系列 → オーバーレイ折れ線グラフ
    "rms": "line_1d",
    "f0_world": "line_1d",
    "f0_crepe": "line_1d",
    "spectral_centroid": "line_1d",
    "spectral_bandwidth": "line_1d",
    "hnr": "line_1d",
    "zero_crossing_rate": "line_1d",
    "waveform": "line_1d",
    "waveform_diff": "line_1d",
    "source_rms": "line_1d",
    "formant_1_amp": "line_1d",
    "formant_2_amp": "line_1d",
    "formant_3_amp": "line_1d",
    "formant_4_amp": "line_1d",
    "formant_5_amp": "line_1d",
    "formant_6_amp": "line_1d",
    "formant_7_amp": "line_1d",
    "air_freq_amp": "line_1d",
    # 多次元 → スペクトログラム/ヒートマップ
    "mel": "spectrogram",
    "mfcc": "heatmap",
    "spectral_contrast": "heatmap",
    "cqt": "spectrogram",
    "stft": "spectrogram",
    "aperiodicity": "heatmap",
    "source_mel": "spectrogram",
    "source_stft": "spectrogram",
}


class AudioAnalyzer:
    """音響特徴量の抽出・差分計算・相関分析"""

    def __init__(self, sr=44100, shift_length=512):
        self.sr = sr
        self.shift_length = shift_length
        self.extractor = AudioFeatureExtractor(sr=sr, shift_length=shift_length)

    def extract_features(self, wav_path, feature_types):
        """
        WAVファイルから指定した特徴量を個別に抽出する。

        Args:
            wav_path: WAVファイルパス
            feature_types: 特徴量名のリスト (例: ["mel", "rms", "f0_world"])

        Returns:
            dict: {feature_name: np.ndarray (T, D), ...}
        """
        # 音声を一度だけ読み込む
        y = self.extractor._load_audio(wav_path)

        features = {}
        for f_type in feature_types:
            try:
                feat = self._extract_single(y, f_type)
                features[f_type] = feat
            except Exception as e:
                print(f"Warning: Failed to extract {f_type}: {e}")

        return features

    def extract_features_paired(self, wav_path_a, wav_path_b, feature_types):
        """
        2つのWAVファイルから特徴量を抽出する。
        MelやSTFTなどの対数化(デシベル計算)処理において、両ファイルの最大ピークを
        共通のリファレンスとして使用することで、無音部分などの計算ズレ（アーティファクト）を防ぐ。
        """
        # 音声を読み込む
        y_a = self.extractor._load_audio(wav_path_a)
        y_b = self.extractor._load_audio(wav_path_b)

        # --- Mel/MFCC 共通リファレンスパワー ---
        if (
            "mel" in feature_types
            or "mfcc" in feature_types
            or "source_mel" in feature_types
        ):
            self.extractor.config["global_mel_ref"] = 1.0
        else:
            self.extractor.config.pop("global_mel_ref", None)

        # --- STFT 共通リファレンス振幅 ---
        if "stft" in feature_types or "source_stft" in feature_types:
            self.extractor.config["global_stft_ref"] = 1.0
        else:
            self.extractor.config.pop("global_stft_ref", None)

        features_a = {}
        features_b = {}
        for f_type in feature_types:
            try:
                features_a[f_type] = self._extract_single(y_a, f_type)
                features_b[f_type] = self._extract_single(y_b, f_type)
            except Exception as e:
                print(f"Warning: Failed to extract {f_type}: {e}")

        # 使い終わったらリファレンス設定を消しておく
        self.extractor.config.pop("global_mel_ref", None)
        self.extractor.config.pop("global_stft_ref", None)

        return features_a, features_b

    def _extract_single(self, y, f_type):
        """単一の特徴量を抽出"""
        ext = self.extractor

        if f_type == "mel":
            return ext._compute_mel(y)
        elif f_type == "rms":
            return ext._compute_rms(y)
        elif f_type == "mfcc":
            return ext._compute_mfcc(y)
        elif f_type == "f0_world":
            ext.config["f0"]["method"] = "pyworld"
            return ext._compute_f0(y)
        elif f_type == "f0_crepe":
            ext.config["f0"]["method"] = "crepe"
            return ext._compute_f0(y)
        elif f_type == "spectral_centroid":
            return ext._compute_spectral_centroid(y)
        elif f_type == "spectral_bandwidth":
            return ext._compute_spectral_bandwidth(y)
        elif f_type == "spectral_contrast":
            return ext._compute_spectral_contrast(y)
        elif f_type == "hnr":
            return ext._compute_hnr(y)
        elif f_type == "zero_crossing_rate":
            return ext._compute_zero_crossing_rate(y)
        elif f_type == "cqt":
            return ext._compute_cqt(y)
        elif f_type == "stft":
            return ext._compute_stft(y)
        elif f_type == "aperiodicity":
            return ext._compute_aperiodicity(y)
        elif f_type == "source_mel":
            return ext._compute_source_mel(y)
        elif f_type == "source_rms":
            return ext._compute_source_rms(y)
        elif f_type == "source_stft":
            return ext._compute_source_stft(y)
        elif f_type in ext.config["envelope_bands"]:
            return ext._compute_formant_amp(y, f_type)
        elif f_type == "waveform":
            return y.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown feature type: {f_type}")

    def compute_diff(self, features_a, features_b):
        """
        2つの特徴量辞書の差分を計算する。

        Args:
            features_a: extract_features() の出力
            features_b: extract_features() の出力

        Returns:
            dict: {feature_name: np.ndarray (差分), ...}
        """
        diff = {}
        for f_type in features_a:
            if f_type in features_b:
                fa = features_a[f_type]
                fb = features_b[f_type]

                # 時間軸を短い方に合わせる
                min_len = min(fa.shape[0], fb.shape[0])
                diff[f_type] = fb[:min_len] - fa[:min_len]

        return diff

    def compute_correlations(self, param_curve, diff_features, features_a, features_b):
        """
        VSQXパラメータカーブと各音響特徴量の差分の相関係数を算出する。

        Args:
            param_curve: np.ndarray (T,) — VSQXパラメータの時系列
            diff_features: compute_diff() の出力
            features_a: 基準音声の特徴量
            features_b: パラメータ付き音声の特徴量

        Returns:
            dict: {
                feature_name: {
                    "r": float (相関係数),
                    "p": float (p値),
                    "r_per_band": list (多次元の場合、帯域ごとの相関)
                },
                ...
            }
        """
        correlations = {}

        for f_type, diff_signal in diff_features.items():
            if f_type == "waveform" or f_type == "waveform_diff":
                correlations[f_type] = {"r": 0.0, "p": 1.0, "r_per_band": None}
                continue

            min_len = min(len(param_curve), diff_signal.shape[0])
            pc = param_curve[:min_len]
            ds = diff_signal[:min_len]

            if ds.ndim == 1 or ds.shape[1] == 1:
                # 1次元特徴量
                ds_flat = ds.flatten()[:min_len]

                # 分散がゼロの場合は相関計算不可
                if np.std(pc) < 1e-10 or np.std(ds_flat) < 1e-10:
                    correlations[f_type] = {"r": 0.0, "p": 1.0, "r_per_band": None}
                else:
                    r, p = stats.pearsonr(pc, ds_flat)
                    correlations[f_type] = {
                        "r": float(r),
                        "p": float(p),
                        "r_per_band": None,
                    }
            else:
                # 多次元特徴量 → 帯域ごとの相関を計算
                n_bands = ds.shape[1]
                r_per_band = []

                for i in range(n_bands):
                    band_signal = ds[:, i]
                    if np.std(pc) < 1e-10 or np.std(band_signal) < 1e-10:
                        r_per_band.append(0.0)
                    else:
                        r, _ = stats.pearsonr(pc, band_signal)
                        r_per_band.append(float(r))

                # 代表値: 絶対値の平均
                mean_abs_r = float(np.mean(np.abs(r_per_band)))

                # 全体としてのp値は帯域平均のr値から概算
                # (簡略化: 全帯域の平均相関のp値)
                avg_r = float(np.mean(r_per_band))
                n = min_len
                if abs(avg_r) < 1.0 and n > 2:
                    t_stat = avg_r * np.sqrt(n - 2) / np.sqrt(1 - avg_r**2)
                    p_val = float(2 * stats.t.sf(abs(t_stat), n - 2))
                else:
                    p_val = 0.0

                correlations[f_type] = {
                    "r": mean_abs_r,
                    "p": p_val,
                    "r_per_band": r_per_band,
                }

        return correlations

    def sort_by_correlation(self, correlations):
        """
        相関の絶対値が大きい順にソートした特徴量名リストを返す。

        Returns:
            list: [(feature_name, corr_info), ...] 相関強い順
        """
        return sorted(correlations.items(), key=lambda x: abs(x[1]["r"]), reverse=True)

    def get_time_axis(self, total_frames):
        """フレーム数に対応する時間軸を返す"""
        return librosa.frames_to_time(
            np.arange(total_frames), sr=self.sr, hop_length=self.shift_length
        )
