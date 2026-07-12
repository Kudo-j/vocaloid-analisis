import librosa
import numpy as np
import pyworld as pw
import scipy.fft
import scipy.signal

# torchcrepe, torch は CREPE使用時のみ遅延インポート


class AudioFeatureExtractor:
    def __init__(self, sr=44100, hop_length=512, device="cuda"):
        self.sr = sr
        self.hop_length = hop_length
        # torch はCREPE使用時のみ必要なので遅延チェック
        try:
            import torch

            self.device = device if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

        # World用のフレーム周期 (ms) 計算
        self.frame_period = hop_length / sr * 1000

        # デフォルト設定
        self.config = {
            "cqt": {
                "n_bins": 168,
                "bins_per_octave": 24,
                "fmin": librosa.note_to_hz("C1"),
            },
            "mel": {"n_mels": 80, "n_fft": 2048, "fmax": None},
            "mfcc": {"n_mfcc": 20},
            "stft": {"n_fft": 2048, "win_length": None, "window": "hann"},
            "f0": {
                # "pyworld" (harvest) or "crepe" を選択可能にする
                "method": "pyworld",
                "fmin": librosa.note_to_hz("C1"),
                "fmax": librosa.note_to_hz("C7"),
                "model_capacity": "full",  # for crepe ("tiny", "small", "medium", "large", "full")
                "pyworld_algo": "dio",  # "harvest" は非常に遅いため "dio" をデフォルトにする
            },
            "cepstrum": {"n_fft": 2048, "n_coef": 40},
            "smoothed_cepstrum": {"n_fft": 2048, "n_coef": 40},
            "lpc": {"order": 32},  # LPC次数 (声道フィルタの推定精度)
        }

    def _load_audio(self, wav_path):
        # pyworldは倍精度(float64)推奨
        y, _ = librosa.load(wav_path, sr=self.sr)
        return y.astype(np.float64)

    # --- 既存の特徴量 (CQT, Mel, MFCC, RMS) ---
    def _compute_cqt(self, y):
        params = self.config["cqt"]
        cqt = librosa.cqt(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=params["n_bins"],
            bins_per_octave=params["bins_per_octave"],
            fmin=params["fmin"],
        )
        return librosa.amplitude_to_db(np.abs(cqt), ref=1.0, top_db=None).T

    def _compute_mel(self, y):
        params = self.config["mel"]
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            n_fft=params["n_fft"],
            n_mels=params["n_mels"],
            fmax=params["fmax"],
        )
        ref_val = self.config.get("global_mel_ref", np.max)
        # global_mel_ref が設定されている場合、top_db クランプを無効化する。
        # top_db はファイルごとの最大値に依存するため、共通基準使用時にはズレの原因になる。
        use_top_db = None if isinstance(ref_val, (int, float, np.floating)) else 80.0
        return librosa.power_to_db(mel, ref=ref_val, top_db=use_top_db).T

    def _compute_mfcc(self, y):
        params = self.config["mfcc"]
        mel_params = self.config["mel"]

        # mfccの基準値（無音時のdBフロア）を揃えるため、手動でMel Spectrogramを対数化して渡す
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            n_fft=mel_params["n_fft"],
            n_mels=mel_params["n_mels"],
            fmax=mel_params["fmax"],
        )
        ref_val = self.config.get("global_mel_ref", np.max)
        use_top_db = None if isinstance(ref_val, (int, float, np.floating)) else 80.0
        log_S = librosa.power_to_db(S, ref=ref_val, top_db=use_top_db)

        return librosa.feature.mfcc(S=log_S, sr=self.sr, n_mfcc=params["n_mfcc"]).T

    def _compute_rms(self, y):
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        # 振幅をdB表記に変換 (ref=1.0)
        rms_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)
        return rms_db.T

    # --- STFT ---
    def _compute_stft(self, y):
        params = self.config["stft"]
        D = librosa.stft(
            y=y,
            n_fft=params["n_fft"],
            hop_length=self.hop_length,
            win_length=params["win_length"],
            window=params["window"],
        )
        ref_val = self.config.get("global_stft_ref", np.max)
        use_top_db = None if isinstance(ref_val, (int, float, np.floating)) else 80.0
        D_db = librosa.amplitude_to_db(np.abs(D), ref=ref_val, top_db=use_top_db)
        return D_db.T

    # --- ★F0抽出 (World / CREPE 切り替え対応) ---
    def _get_pw_f0(self, y):
        """共通のF0計算処理 (挙動統一と高速化のため)"""
        params = self.config["f0"]
        # pyworldは明示的に float64 かつ C-contiguous である必要がある
        y_pw = np.ascontiguousarray(y, dtype=np.float64)
        algo = params.get("pyworld_algo", "harvest")

        if algo == "dio":
            _f0, t = pw.dio(
                y_pw,
                self.sr,
                frame_period=self.frame_period,
                f0_floor=params["fmin"],
                f0_ceil=params["fmax"],
            )
            f0 = pw.stonemask(y_pw, _f0, t, self.sr)
        else:
            f0, t = pw.harvest(
                y_pw,
                self.sr,
                frame_period=self.frame_period,
                f0_floor=params["fmin"],
                f0_ceil=params["fmax"],
            )
        return f0, t

    # --- ★F0抽出 (World / CREPE 切り替え対応) ---
    def _compute_f0(self, y):
        """基本周波数 (Time, 1)"""
        params = self.config["f0"]
        method = params["method"]

        if method == "crepe":
            # --- CREPE (Neural Network based) ---
            import torch
            import torchcrepe

            audio = torch.tensor(np.copy(y)).unsqueeze(0).float()
            if self.device == "cuda" and torch.cuda.is_available():
                audio = audio.to("cuda")

            f0_crepe, _, _ = torchcrepe.predict(
                audio,
                self.sr,
                hop_length=self.hop_length,
                fmin=params["fmin"],
                fmax=params["fmax"],
                model=params["model_capacity"],
                batch_size=2048,
                device=self.device,
                return_harmonicity=False,
            )
            f0 = f0_crepe.squeeze().cpu().numpy()
            f0 = np.nan_to_num(f0)
        elif method == "pyworld":
            f0, _ = self._get_pw_f0(y)
        else:
            raise ValueError(f"Unknown F0 method: {method}")

        return f0.reshape(-1, 1)

    # --- 非周期成分 (Aperiodicity) ---
    def _compute_aperiodicity(self, y):
        """非周期性指標 (Time, Freq_Bins)"""
        y_pw = np.ascontiguousarray(y, dtype=np.float64)
        _f0, t = self._get_pw_f0(y_pw)
        ap = pw.d4c(y_pw, _f0, t, self.sr)
        return ap

    # --- ケプストラム ---
    def _compute_cepstrum(self, y):
        params = self.config["cepstrum"]
        D = librosa.stft(y, n_fft=params["n_fft"], hop_length=self.hop_length)
        mag = np.abs(D)
        log_mag = np.log(mag + 1e-10)
        cepstrum = scipy.fft.ifft(log_mag, axis=0).real.T

        if params["n_coef"] is not None:
            cepstrum = cepstrum[:, : params["n_coef"]]
        return cepstrum

    # --- 平滑化ケプストラム (Smoothed Cepstrum) ---
    def _compute_smoothed_cepstrum(self, y):
        params = self.config["smoothed_cepstrum"]
        y_pw = np.ascontiguousarray(y, dtype=np.float64)

        # Worldによるスペクトル包絡抽出にはF0が必要
        _f0, t = self._get_pw_f0(y_pw)
        sp = pw.cheaptrick(y_pw, _f0, t, self.sr)

        log_sp = np.log(sp + 1e-10)
        cepstrum = scipy.fft.ifft(log_sp, axis=1).real

        if params["n_coef"] is not None:
            cepstrum = cepstrum[:, : params["n_coef"]]
        return cepstrum

    # --- 新規: スペクトル重心 ---
    def _compute_spectral_centroid(self, y):
        """スペクトル重心 (Time, 1) - 音の明るさの指標"""
        return librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=self.hop_length
        ).T

    def _compute_waveform(self, y):
        # 生の波形データをそのまま特徴量として返す(Shape: (Samples, 1))
        return y.reshape(-1, 1)

    # --- 新規: スペクトル帯域幅 ---
    def _compute_spectral_bandwidth(self, y):
        """スペクトル帯域幅 (Time, 1) - 音のスペクトル幅"""
        return librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, hop_length=self.hop_length
        ).T

    # --- 新規: スペクトルコントラスト ---
    def _compute_spectral_contrast(self, y):
        """スペクトルコントラスト (Time, 7) - 周波数帯域ごとのコントラスト"""
        return librosa.feature.spectral_contrast(
            y=y, sr=self.sr, hop_length=self.hop_length
        ).T

    # --- 新規: 調波対雑音比 (HNR) ---
    def _compute_hnr(self, y):
        """調波対雑音比 (Time, 1) - 息成分の指標"""
        y_pw = np.ascontiguousarray(y, dtype=np.float64)
        f0, t = self._get_pw_f0(y_pw)
        sp = pw.cheaptrick(y_pw, f0, t, self.sr)
        ap = pw.d4c(y_pw, f0, t, self.sr)

        # HNR = 10 * log10(harmonic_power / noise_power)
        # ap は非周期性比率 (0=完全調波, 1=完全ノイズ)
        ap_clipped = np.clip(ap, 1e-10, 1.0 - 1e-10)
        hnr_per_band = 10.0 * np.log10((1.0 - ap_clipped) / ap_clipped)
        # 帯域平均して1次元に
        hnr = np.mean(hnr_per_band, axis=1, keepdims=True)
        return hnr.astype(np.float32)

    # --- 新規: ゼロ交差率 ---
    def _compute_zero_crossing_rate(self, y):
        """ゼロ交差率 (Time, 1)"""
        return librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length).T

    # --- LPC残差（音源信号）抽出 ---
    def _compute_lpc_residual(self, y):
        """
        LPC逆フィルタリングにより声道フィルタ（フォルマント）を除去し、
        声帯音源信号（LPC残差）を返す。

        処理:
          1. 音声をフレームに分割（窓関数適用）
          2. 各フレームでLPC係数を推定
          3. LPC逆フィルタ（FIRフィルタ）を適用し残差を得る
          4. フレームを重畳加算（OLA）で再結合

        Returns:
            np.ndarray: 残差信号（元の音声と同じ長さ）
        """
        lpc_order = self.config["lpc"]["order"]
        frame_length = self.config["mel"]["n_fft"]  # 2048
        hop = self.hop_length

        y_f32 = y.astype(np.float32)

        # フレーム分割
        frames = librosa.util.frame(y_f32, frame_length=frame_length, hop_length=hop)
        # frames.shape = (frame_length, n_frames)
        n_frames = frames.shape[1]
        window = scipy.signal.windows.hann(frame_length, sym=False).astype(np.float32)

        # 出力バッファ（OLA用）
        residual = np.zeros(len(y_f32), dtype=np.float32)
        window_sum = np.zeros(len(y_f32), dtype=np.float32)

        for i in range(n_frames):
            frame = frames[:, i] * window

            # エネルギーが極小のフレーム（無音）はスキップ
            if np.sum(frame**2) < 1e-10:
                continue

            # LPC係数を推定 (librosa.lpcは1-indexed: [1, a1, a2, ...])
            try:
                lpc_coeffs = librosa.lpc(frame, order=lpc_order)
            except Exception:
                continue

            # LPC逆フィルタリング: 残差 = フィルタ出力
            # y[n] = x[n] - sum(a[k] * y[n-k]) → x[n] = y[n] + sum(a[k] * y[n-k])
            # scipy.signal.lfilter(lpc_coeffs, 1, frame) で残差が得られる
            frame_residual = scipy.signal.lfilter(lpc_coeffs, [1.0], frame)

            # OLA
            start = i * hop
            end = start + frame_length
            residual[start:end] += frame_residual
            window_sum[start:end] += window**2

        # 正規化（窓の重畳補正）
        nonzero = window_sum > 1e-8
        residual[nonzero] /= window_sum[nonzero]

        return residual.astype(np.float64)

    # --- 音源メルスペクトログラム (フォルマント除去済み) ---
    def _compute_source_mel(self, y):
        """
        LPC残差信号のメルスペクトログラム。
        声道フィルタ（フォルマント）が除去されているため、
        DYNなどのパラメータが「音源（声帯振動）」に与える純粋な影響を可視化できる。
        """
        residual = self._compute_lpc_residual(y)
        params = self.config["mel"]
        mel = librosa.feature.melspectrogram(
            y=residual,
            sr=self.sr,
            hop_length=self.hop_length,
            n_fft=params["n_fft"],
            n_mels=params["n_mels"],
            fmax=params["fmax"],
        )
        ref_val = self.config.get("global_mel_ref", np.max)
        use_top_db = None if isinstance(ref_val, (int, float, np.floating)) else 80.0
        return librosa.power_to_db(mel, ref=ref_val, top_db=use_top_db).T

    # --- 音源RMS (フォルマント除去済み) ---
    def _compute_source_rms(self, y):
        """
        LPC残差信号のRMSエネルギー。
        フォルマントの影響を受けない、純粋な音源エネルギーの変化を捉える。
        """
        residual = self._compute_lpc_residual(y)
        rms = librosa.feature.rms(y=residual, hop_length=self.hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)
        return rms_db.T

    # --- 音源STFT (フォルマント除去済み線形スペクトログラム) ---
    def _compute_source_stft(self, y):
        """
        LPC残差信号のSTFTスペクトログラム（線形周波数軸）。
        フォルマントの癖が除去された状態で、DYNなどのパラメータが
        各周波数帯域の倍音エネルギーにどう影響するかを可視化できる。
        """
        residual = self._compute_lpc_residual(y)
        params = self.config["stft"]
        D = librosa.stft(
            y=residual,
            n_fft=params["n_fft"],
            hop_length=self.hop_length,
            win_length=params["win_length"],
            window=params["window"],
        )
        ref_val = self.config.get("global_stft_ref", np.max)
        use_top_db = None if isinstance(ref_val, (int, float, np.floating)) else 80.0
        D_db = librosa.amplitude_to_db(np.abs(D), ref=ref_val, top_db=use_top_db)
        return D_db.T

    # --- メイン取得メソッド ---
    def get_features(self, wav_path, feature_types):
        if isinstance(feature_types, str):
            feature_types = [feature_types]

        y = self._load_audio(wav_path)
        feature_list = []

        for f_type in feature_types:
            if f_type == "cqt":
                feat = self._compute_cqt(y)
            elif f_type == "mel":
                feat = self._compute_mel(y)
            elif f_type == "mfcc":
                feat = self._compute_mfcc(y)
            elif f_type == "rms":
                feat = self._compute_rms(y)
            elif f_type == "stft":
                feat = self._compute_stft(y)
            elif f_type == "f0_world":  # ★エイリアス対応
                # 設定を一時的に上書きして実行しても良いが、シンプルにmethod指定で分岐
                self.config["f0"]["method"] = "pyworld"
                feat = self._compute_f0(y)
            elif f_type == "f0_crepe":  # ★エイリアス対応
                self.config["f0"]["method"] = "crepe"
                feat = self._compute_f0(y)
            elif f_type == "aperiodicity":
                feat = self._compute_aperiodicity(y)
            elif f_type == "cepstrum":
                feat = self._compute_cepstrum(y)
            elif f_type == "smoothed_cepstrum":
                feat = self._compute_smoothed_cepstrum(y)
            # --- 新規特徴量 ---
            elif f_type == "spectral_centroid":
                feat = self._compute_spectral_centroid(y)
            elif f_type == "spectral_bandwidth":
                feat = self._compute_spectral_bandwidth(y)
            elif f_type == "spectral_contrast":
                feat = self._compute_spectral_contrast(y)
            elif f_type == "hnr":
                feat = self._compute_hnr(y)
            elif f_type == "zero_crossing_rate":
                feat = self._compute_zero_crossing_rate(y)
            elif f_type == "source_mel":
                feat = self._compute_source_mel(y)
            elif f_type == "source_rms":
                feat = self._compute_source_rms(y)
            elif f_type == "source_stft":
                feat = self._compute_source_stft(y)
            else:
                raise ValueError(f"Unknown feature type: {f_type}")

            feature_list.append(feat)

        # 時間軸の長さを最短に合わせる
        min_len = min(f.shape[0] for f in feature_list)
        feature_list = [f[:min_len] for f in feature_list]

        # 結合
        combined_features = np.hstack(feature_list).astype(np.float32)

        return combined_features
