import librosa
import numpy as np
import pyworld as pw
import scipy.fft
import scipy.signal

# torchcrepe, torch は CREPE使用時のみ遅延インポート


class AudioFeatureExtractor:
    def __init__(self, sr=44100, shift_length=512, device="cuda"):
        self.sr = sr
        self.shift_length = shift_length
        # torch はCREPE使用時のみ必要なので遅延チェック
        try:
            import torch

            self.device = device if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

        # World用のフレーム周期 (ms) 計算
        self.frame_period = shift_length / sr * 1000

        # デフォルト設定
        self.config = {
            "cqt": {
                "n_bins": 168,
                "bins_per_octave": 24,
                "fmin": librosa.note_to_hz("C1"),
            },
            "mel": {"n_mels": 80, "n_fft": 1024, "fmax": None},
            "mfcc": {"n_mfcc": 20},
            "stft": {"n_fft": 1024, "win_length": None, "window": "hann"},
            "rms": {"frame_length": 1024},
            "f0": {
                # "pyworld" (harvest) or "crepe" を選択可能にする
                "method": "pyworld",
                "fmin": librosa.note_to_hz("C1"),
                "fmax": librosa.note_to_hz("C7"),
                "model_capacity": "full",  # for crepe ("tiny", "small", "medium", "large", "full")
                "pyworld_algo": "dio",  # "harvest" は非常に遅いため "dio" をデフォルトにする
            },
            "cepstrum": {"n_fft": 1024, "n_coef": 40},
            "smoothed_cepstrum": {"n_fft": 1024, "n_coef": 40},
            "lpc": {"order": 32},  # LPC次数 (声道フィルタの推定精度)
            "bark": {
                "n_barks": 24,
                "n_fft": 1024,
                "compression": "power",  # "power" (Stevensべき乗則 0.33) または "db"
                "power_exponent": 0.33,
                "use_equal_loudness": True,
            },
            "loudness": {
                "n_barks": 24,
                "n_fft": 1024,
                "power_exponent": 0.33,
                "use_equal_loudness": True,
            },
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
            hop_length=self.shift_length,
            n_bins=params["n_bins"],
            bins_per_octave=params["bins_per_octave"],
            fmin=params["fmin"],
        )
        return librosa.amplitude_to_db(np.abs(cqt), ref=1.0, top_db=None).T

    def _compute_cqt_framenorm(self, y):
        """
        フレーム単位で総エネルギーを1.0に規格化し、音量輪郭を除去したCQT特徴量。
        有音フレームのみ総エネルギーで除算し、ピークから-60dB以下の無音フレームはフロアリングする。
        """
        params = self.config["cqt"]
        cqt = librosa.cqt(
            y=y,
            sr=self.sr,
            hop_length=self.shift_length,
            n_bins=params["n_bins"],
            bins_per_octave=params["bins_per_octave"],
            fmin=params["fmin"],
        )
        # 線形パワー (|CQT|^2)
        cqt_power = np.abs(cqt) ** 2

        # フレームごとの総エネルギー (axis=0: 周波数軸の総和)
        total_energy = np.sum(cqt_power, axis=0, keepdims=True)

        # 相対閾値 (-60dB = ピークエネルギーの1e-6倍)
        max_energy = np.max(total_energy)
        energy_threshold = max(max_energy * 1e-6, 1e-10)

        # 有音フレームのみ総エネルギーで正規化（合計1.0化）
        is_voiced = total_energy >= energy_threshold
        safe_denom = np.where(is_voiced, total_energy, 1.0)
        cqt_norm = np.where(is_voiced, cqt_power / safe_denom, cqt_power)

        cqt_db = librosa.power_to_db(cqt_norm, ref=1.0, top_db=80.0)
        return cqt_db.T

    def _compute_mel(self, y):
        params = self.config["mel"]
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            hop_length=self.shift_length,
            n_fft=params["n_fft"],
            n_mels=params["n_mels"],
            fmax=params["fmax"],
        )
        ref_val = self.config.get("global_mel_ref", np.max)
        # global_mel_ref が設定されている場合、top_db クランプを無効化する。
        # top_db はファイルごとの最大値に依存するため、共通基準使用時にはズレの原因になる。
        use_top_db = None if isinstance(ref_val, (int, float, np.floating)) else 80.0
        return librosa.power_to_db(mel, ref=ref_val, top_db=use_top_db).T

    def _compute_mel_framenorm(self, y):
        """
        フレーム単位で総エネルギーを1.0に規格化し、音量輪郭を除去したメルスペクトログラム。
        有音フレームのみ総エネルギーで除算し、ピークから-60dB以下の無音フレームはフロアリングする。
        """
        params = self.config["mel"]
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            hop_length=self.shift_length,
            n_fft=params["n_fft"],
            n_mels=params["n_mels"],
            fmax=params["fmax"],
            power=2.0,
        )
        # フレームごとの総エネルギー (axis=0: 周波数軸の総和)
        total_energy = np.sum(mel, axis=0, keepdims=True)

        # 相対閾値 (-60dB = ピークエネルギーの1e-6倍)
        max_energy = np.max(total_energy)
        energy_threshold = max(max_energy * 1e-6, 1e-10)

        # 有音フレームのみ総エネルギーで正規化（合計1.0化）
        is_voiced = total_energy >= energy_threshold
        safe_denom = np.where(is_voiced, total_energy, 1.0)
        mel_norm = np.where(is_voiced, mel / safe_denom, mel)

        mel_db = librosa.power_to_db(mel_norm, ref=1.0, top_db=80.0)
        return mel_db.T

    def _compute_mfcc(self, y):
        params = self.config["mfcc"]
        mel_params = self.config["mel"]

        # mfccの基準値（無音時のdBフロア）を揃えるため、手動でMel Spectrogramを対数化して渡す
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            hop_length=self.shift_length,
            n_fft=mel_params["n_fft"],
            n_mels=mel_params["n_mels"],
            fmax=mel_params["fmax"],
        )
        ref_val = self.config.get("global_mel_ref", np.max)
        use_top_db = None if isinstance(ref_val, (int, float, np.floating)) else 80.0
        log_S = librosa.power_to_db(S, ref=ref_val, top_db=use_top_db)

        return librosa.feature.mfcc(S=log_S, sr=self.sr, n_mfcc=params["n_mfcc"]).T

    def _compute_rms(self, y):
        params = self.config["rms"]
        rms = librosa.feature.rms(
            y=y, frame_length=params["frame_length"], hop_length=self.shift_length
        )
        # # 振幅をdB表記に変換 (ref=1.0) (線形スケールで学習するために無効化)
        # rms_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)
        # return rms_db.T
        return rms.T

    # --- STFT ---
    def _compute_stft(self, y):
        params = self.config["stft"]
        D = librosa.stft(
            y=y,
            n_fft=params["n_fft"],
            hop_length=self.shift_length,
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
                shift_length=self.shift_length,
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
        D = librosa.stft(y, n_fft=params["n_fft"], hop_length=self.shift_length)
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
        params = self.config["stft"]
        return librosa.feature.spectral_centroid(
            y=y, sr=self.sr, n_fft=params["n_fft"], hop_length=self.shift_length
        ).T

    def _compute_waveform(self, y):
        # 生の波形データをそのまま特徴量として返す(Shape: (Samples, 1))
        return y.reshape(-1, 1)

    # --- 新規: スペクトル帯域幅 ---
    def _compute_spectral_bandwidth(self, y):
        """スペクトル帯域幅 (Time, 1) - 音のスペクトル幅"""
        params = self.config["stft"]
        return librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, n_fft=params["n_fft"], hop_length=self.shift_length
        ).T

    # --- 新規: スペクトルコントラスト ---
    def _compute_spectral_contrast(self, y):
        """スペクトルコントラスト (Time, 7) - 周波数帯域ごとのコントラスト"""
        params = self.config["stft"]
        return librosa.feature.spectral_contrast(
            y=y, sr=self.sr, n_fft=params["n_fft"], hop_length=self.shift_length
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
        params = self.config["rms"]
        return librosa.feature.zero_crossing_rate(
            y=y, frame_length=params["frame_length"], hop_length=self.shift_length
        ).T

    # --- バーク尺度および等ラウドネス補正ヘルパー ---
    def _get_bark_filterbank(self, n_fft, n_barks=24):
        """
        Traunmuller (1990) の式に基づく24バンド臨界帯域三角フィルタバンク行列を生成。
        Returns:
            fb (np.ndarray): (n_barks, n_fft // 2 + 1)
            freqs (np.ndarray): (n_fft // 2 + 1,)
        """
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
        # Traunmuller (1990): z = 26.81 * f / (1960 + f) - 0.53
        bark_freqs = np.maximum(0.0, 26.81 * freqs / (1960.0 + freqs) - 0.53)
        centers = np.linspace(1.0, 24.0, n_barks)
        fb = np.zeros((n_barks, len(freqs)), dtype=np.float32)
        for i, c in enumerate(centers):
            # 三角フィルタ（帯域幅 +-1 Bark）
            fb[i] = np.maximum(0.0, 1.0 - np.abs(bark_freqs - c))
        return fb, freqs

    def _get_equal_loudness_curve(self, freqs):
        """
        Terhardt (1979) / ISO 226 近似による等ラウドネス周波数重み付け曲線（40 phon基準）。
        1kHz〜4kHzを強調し、低域と超高域を減衰させる。
        """
        f_sq = freqs**2
        with np.errstate(divide="ignore", invalid="ignore"):
            eq = ((f_sq + 1.44e6) * (freqs**4)) / (
                ((f_sq + 1.6e5) ** 2) * (f_sq + 9.61e6)
            )
            eq[0] = 0.0
        max_val = np.max(eq)
        if max_val > 0:
            eq /= max_val
        return eq.astype(np.float32)

    # --- バークスペクトログラム (比ラウドネス分布: 24次元) ---
    def _compute_bark_spectrogram(self, y):
        """
        24次元のバークスペクトログラム（比ラウドネスまたはBarkパワースペクトル）。
        メルスペクトログラムの比較・代替用。
        """
        params = self.config["bark"]
        n_fft = params["n_fft"]
        n_barks = params["n_barks"]
        use_eq = params["use_equal_loudness"]
        compression = params["compression"]
        exponent = params["power_exponent"]

        # STFT パワースペクトル (|D|^2)
        D = librosa.stft(y, n_fft=n_fft, hop_length=self.shift_length)
        power_spec = np.abs(D) ** 2

        # フィルタバンクと等ラウドネス曲線
        fb, freqs = self._get_bark_filterbank(n_fft, n_barks)
        if use_eq:
            eq = self._get_equal_loudness_curve(freqs)
            power_spec = power_spec * eq[:, None]

        # Bark帯域エネルギー集約 (n_barks, frames)
        bark_energy = np.dot(fb, power_spec)

        if compression == "power":
            # Stevensのべき法則による比ラウドネス (Sone/Bark相当)
            spec_loudness = np.maximum(bark_energy, 1e-10) ** exponent
            return spec_loudness.T
        elif compression == "db":
            # 対数dB表現
            ref_val = self.config.get("global_stft_ref", np.max)
            use_top_db = (
                None if isinstance(ref_val, (int, float, np.floating)) else 80.0
            )
            bark_db = librosa.power_to_db(bark_energy, ref=ref_val, top_db=use_top_db)
            return bark_db.T
        else:
            return bark_energy.T

    # --- 全知覚ラウドネス (Total Loudness: 1次元) ---
    def _compute_loudness(self, y):
        """
        1次元の全知覚ラウドネス（Total Loudness, 単位: Sone相当）。
        全24バーク帯域の比ラウドネスの総和。RMSの比較・代替用。
        """
        params = self.config["loudness"]
        n_fft = params["n_fft"]
        n_barks = params["n_barks"]
        use_eq = params["use_equal_loudness"]
        exponent = params["power_exponent"]

        # STFT パワースペクトル (|D|^2)
        D = librosa.stft(y, n_fft=n_fft, hop_length=self.shift_length)
        power_spec = np.abs(D) ** 2

        fb, freqs = self._get_bark_filterbank(n_fft, n_barks)
        if use_eq:
            eq = self._get_equal_loudness_curve(freqs)
            power_spec = power_spec * eq[:, None]

        bark_energy = np.dot(fb, power_spec)
        # 各臨界帯域の比ラウドネス (Sone/Bark)
        spec_loudness = np.maximum(bark_energy, 1e-10) ** exponent
        # 全臨界帯域の積分（総和）= 全ラウドネス
        total_loudness = np.sum(spec_loudness, axis=0, keepdims=True)
        return total_loudness.T

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
        hop = self.shift_length

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
            hop_length=self.shift_length,
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
        params = self.config["rms"]
        rms = librosa.feature.rms(
            y=residual,
            frame_length=params["frame_length"],
            hop_length=self.shift_length,
        )
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
            hop_length=self.shift_length,
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
            elif f_type == "cqt_framenorm":
                feat = self._compute_cqt_framenorm(y)
            elif f_type == "mel":
                feat = self._compute_mel(y)
            elif f_type == "mel_framenorm":
                feat = self._compute_mel_framenorm(y)
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
            elif f_type == "bark":
                feat = self._compute_bark_spectrogram(y)
            elif f_type == "loudness":
                feat = self._compute_loudness(y)
            else:
                raise ValueError(f"Unknown feature type: {f_type}")

            feature_list.append(feat)

        # 時間軸の長さを最短に合わせる
        min_len = min(f.shape[0] for f in feature_list)
        feature_list = [f[:min_len] for f in feature_list]

        # 結合
        combined_features = np.hstack(feature_list).astype(np.float32)

        return combined_features

    def get_feature_dim(self, f_type):
        """指定された特徴量の次元数を返す"""
        if f_type in ["mel", "mel_framenorm", "source_mel"]:
            return self.config["mel"]["n_mels"]
        elif f_type == "mfcc":
            return self.config["mfcc"]["n_mfcc"]
        elif f_type in ["cqt", "cqt_framenorm"]:
            return self.config["cqt"]["n_bins"]
        elif f_type in [
            "rms",
            "source_rms",
            "f0_world",
            "f0_crepe",
            "spectral_centroid",
            "spectral_bandwidth",
            "hnr",
            "zero_crossing_rate",
            "waveform",
            "loudness",
        ]:
            return 1
        elif f_type == "bark":
            return self.config["bark"]["n_barks"]
        elif f_type == "spectral_contrast":
            return 7
        elif f_type in ["stft", "source_stft"]:
            return self.config["stft"]["n_fft"] // 2 + 1
        elif f_type in ["cepstrum", "smoothed_cepstrum"]:
            return self.config["cepstrum"]["n_coef"]
        else:
            raise ValueError(f"Unknown feature type for dim: {f_type}")
