import numpy as np
import librosa
from vsqx_parser import VSQXParser


class VSQXDNNFormatter(VSQXParser):
    def parse_vsqx(self, file_path, total_frames):
        """
        VSQXファイルからDNN入力用の密なNumPy配列を構築する。
        内部で read_sparse() を呼び出し、疎なデータを配列に変換する。
        """
        # --- Layer 1 呼び出し: 疎なデータの取得 ---
        sparse_data = self.read_sparse(file_path)
        if sparse_data is None:
            raise ValueError(f"Failed to parse VSQX file: {file_path}")

        tempo_map = sparse_data["tempo_map"]
        resolution = sparse_data["resolution"]
        static_params = sparse_data["static_params"]
        voice_libraries = sparse_data["voice_libraries"]

        # --- 利用するパラメータの特定 (オプトイン方式) ---
        has_input = "INPUT_VSQX_PARAMS" in self.vsqx_config
        has_target = "TARGET_PARAM_CONFIG" in self.vsqx_config

        if has_input or has_target:
            input_params = []
            for inputs in self.vsqx_config.get("INPUT_VSQX_PARAMS", {}).values():
                input_params.extend(inputs)
            target_params = []
            for targets in self.vsqx_config.get("TARGET_PARAM_CONFIG", {}).values():
                target_params.extend(targets)
            # 実際に使うパラメータの集合 (O(1)で検索できるようにset化)
            active_params = set(input_params + target_params)
        else:
            # 外部から設定が注入されていない場合は、安全のため従来通り全パラメータを有効化(フォールバック)
            print(
                "[Warning] VSQX入力/ターゲット設定が見つかりません。安全のため全パラメータをフォールバックします。"
            )
            all_params = (
                self.vsqx_config.get("DYNAMIC_PARAM_IDS", [])
                + self.vsqx_config.get("NOTE_PARAM_IDS", [])
                + self.vsqx_config.get("STATIC_PARAM_IDS", [])
            )
            active_params = set(all_params)

        # --- 配列の初期化 ---
        # 正規化後に完全に0.0となるような中央値で初期化する
        dynamic_params_ts = {}
        for pid in self.vsqx_config["DYNAMIC_PARAM_IDS"]:
            if pid not in active_params:
                continue
            if pid == "PIT":
                dynamic_params_ts[pid] = np.zeros(total_frames, dtype=np.float32)
            elif pid == "PBS":
                dynamic_params_ts[pid] = np.full(total_frames, 12.0, dtype=np.float32)
            else:
                dynamic_params_ts[pid] = np.full(total_frames, 63.5, dtype=np.float32)

        note_params_ts = {}
        for pid in self.vsqx_config["NOTE_PARAM_IDS"]:
            if pid not in active_params:
                continue
            max_v = self.vsqx_config["NOTE_PARAM_MAX_VALUES"].get(pid, 1.0)
            if pid in ["fallPort", "risePort", "vibType"]:
                note_params_ts[pid] = np.zeros(total_frames, dtype=np.float32)
            else:
                note_params_ts[pid] = np.full(
                    total_frames, max_v / 2.0, dtype=np.float32
                )
        melody_params_ts = {
            "NOTE_NUMBER": np.zeros(total_frames, dtype=np.float32),
            "DURATION": np.zeros(total_frames, dtype=np.float32),
        }
        phoneme_ts = np.full(total_frames, "unknown", dtype=object)
        voicing_mask_ts = np.zeros((total_frames, 1), dtype=np.float32)
        evec_id_ts = np.zeros(total_frames, dtype=np.int32)
        repeat_ts = np.zeros(total_frames, dtype=np.float32)
        static_params_ts = {
            pid: np.full(
                total_frames, float(static_params.get(pid, 0)), dtype=np.float32
            )
            for pid in self.vsqx_config["STATIC_PARAM_IDS"]
            if pid in active_params
        }

        voice_lib_ts = np.zeros((total_frames, 2), dtype=np.int32)
        voice_lib_ts[:, 0].fill(
            voice_libraries.get(
                "1st_vocal", self.vsqx_config["VOICE_LIB_NAME_TO_ID"]["unknown"]
            )
        )
        voice_lib_ts[:, 1].fill(
            voice_libraries.get(
                "2nd_vocal", self.vsqx_config["VOICE_LIB_NAME_TO_ID"]["unknown"]
            )
        )

        # --- ノート情報の配列化 ---
        for note_info in sparse_data["notes"]:
            abs_start_tick = note_info["abs_start_tick"]
            abs_end_tick = note_info["abs_end_tick"]
            dur_tick = note_info["dur_tick"]

            # Tick → 秒 → フレームインデックス変換
            start_sec = self.tick_to_second(abs_start_tick, tempo_map, resolution)
            end_sec = self.tick_to_second(abs_end_tick, tempo_map, resolution)

            s_idx = np.clip(
                librosa.time_to_frames(
                    start_sec, sr=self.sr, hop_length=self.hop_length
                ),
                0,
                total_frames,
            )
            e_idx = np.clip(
                librosa.time_to_frames(end_sec, sr=self.sr, hop_length=self.hop_length),
                0,
                total_frames,
            )

            # 1. スカラーノートパラメータの適用
            n_params = note_info["params"]
            for pid, val in n_params.items():
                if pid in note_params_ts:
                    note_params_ts[pid][s_idx:e_idx] = val

            # 2. シーケンスデータの補間 (vibDep, vibRate)
            #    interpolate_to_frames を使い、ノート区間のみに適用する
            note_frames = e_idx - s_idx
            if note_frames > 0:
                for seq_id, seq_points in note_info["seqs"].items():
                    if seq_id in note_params_ts and seq_points:
                        # ノート区間だけの密な配列を生成して上書き
                        interp_vals = self.interpolate_to_frames(
                            seq_points,
                            tempo_map,
                            resolution,
                            total_frames,
                            default_value=n_params.get(seq_id, 0),
                        )
                        note_params_ts[seq_id][s_idx:e_idx] = interp_vals[s_idx:e_idx]

            # 3. vibLen によるマスク処理 (シーケンス有無に関わらず実行)
            vib_len_val = n_params.get("vibLen", 0)

            if vib_len_val < 100:
                vib_start_ratio = (100 - vib_len_val) / 100.0
                vib_start_tick = abs_start_tick + (dur_tick * vib_start_ratio)
                vib_start_sec = self.tick_to_second(
                    vib_start_tick, tempo_map, resolution
                )

                current_frames = np.arange(s_idx, e_idx)
                current_times_sec = librosa.frames_to_time(
                    current_frames, sr=self.sr, hop_length=self.hop_length
                )
                mask_indices = current_times_sec < vib_start_sec

                if "vibDep" in note_params_ts:
                    current_vals = note_params_ts["vibDep"][s_idx:e_idx]
                    current_vals[mask_indices] = 0
                    note_params_ts["vibDep"][s_idx:e_idx] = current_vals
                if "vibRate" in note_params_ts:
                    current_vals = note_params_ts["vibRate"][s_idx:e_idx]
                    current_vals[mask_indices] = 0
                    note_params_ts["vibRate"][s_idx:e_idx] = current_vals

            # 4. メロディ・音素・EVEC・連打・発声マスクの埋め込み
            melody_params_ts["NOTE_NUMBER"][s_idx:e_idx] = note_info["note_num"]
            melody_params_ts["DURATION"][s_idx:e_idx] = dur_tick
            voicing_mask_ts[s_idx:e_idx] = 1.0

            phoneme_ts[s_idx:e_idx] = note_info["phoneme"]
            evec_id_ts[s_idx:e_idx] = note_info["evec_id"]
            repeat_ts[s_idx:e_idx] = note_info["repeat"]

        # --- 動的パラメータの補間 (interpolate_to_frames に統一) ---
        for param_id in self.vsqx_config["DYNAMIC_PARAM_IDS"]:
            if param_id not in dynamic_params_ts:
                continue  # 配列が確保されていない(使わない)パラメータは補間をスキップ
            tick_points = sparse_data["dynamic_params"].get(param_id, [])
            if tick_points:
                # デフォルト値は配列初期化時と同じ値を使用
                if param_id == "PIT":
                    default_val = 0
                elif param_id == "PBS":
                    default_val = 12.0
                else:
                    default_val = 63.5
                dynamic_params_ts[param_id] = self.interpolate_to_frames(
                    tick_points, tempo_map, resolution, total_frames, default_val
                )

        return {
            "dynamic": dynamic_params_ts,
            "note": note_params_ts,
            "melody": melody_params_ts,
            "phoneme": phoneme_ts,
            "static": static_params_ts,
            "voice_lib": voice_lib_ts,
            "evec_id": evec_id_ts,
            "repeat": repeat_ts,
            "voicing_mask": voicing_mask_ts,
        }

    def normalize_features(self, raw_data):
        """
        parse_vsqxで得られた生データを、モデル入力用に正規化・One-hot化する。
        推論時にもこのメソッドを通すことで学習時と同じ尺度を保証する。
        """
        if raw_data is None:
            return None

        normalized = {}

        # 1. Dynamic
        norm_dyn = {}
        for pid, val in raw_data["dynamic"].items():
            if pid == "PIT":
                norm_val = (val / 8192.0)[:, np.newaxis]
            elif pid == "PBS":
                norm_val = ((val / 24.0) * 2.0 - 1.0)[:, np.newaxis]
            else:
                norm_val = ((val / 127.0) * 2.0 - 1.0)[:, np.newaxis]
            norm_dyn[pid] = norm_val
        normalized["dynamic"] = norm_dyn

        # 2. Note
        norm_note = {}
        for pid, val in raw_data["note"].items():
            if pid == "vibType":
                idx = np.clip(val.astype(int) - 1, 0, 15)
                norm_val = np.eye(16, dtype=np.float32)[idx]
            else:
                max_v = self.vsqx_config["NOTE_PARAM_MAX_VALUES"].get(pid, 1.0)
                if pid in ["fallPort", "risePort"]:
                    norm_val = (val / max_v)[:, np.newaxis]
                else:
                    norm_val = ((val / max_v) * 2.0 - 1.0)[:, np.newaxis]
            norm_note[pid] = norm_val
        normalized["note"] = norm_note

        # 3. Melody
        norm_melody = {}
        for pid, val in raw_data["melody"].items():
            max_v = self.vsqx_config["NOTE_PARAM_MAX_VALUES"].get(pid, 1.0)
            if pid == "NOTE_NUMBER":
                norm_val = ((val / max_v) * 2.0 - 1.0)[:, np.newaxis]
            else:
                norm_val = (val / max_v)[:, np.newaxis]
            norm_melody[pid] = norm_val
        normalized["melody"] = norm_melody

        # 4. Static
        norm_static = {}
        for pid, val in raw_data["static"].items():
            norm_val = (val / 127.0)[:, np.newaxis]
            norm_static[pid] = norm_val
        normalized["static"] = norm_static

        # 5. Voice Lib (Already numeric, just ensure float32)
        normalized["voice_lib"] = raw_data["voice_lib"].astype(np.float32)

        # 6. Phoneme One-hot
        # ▼▼▼ 変更: 音素マップが存在する場合のみOne-hot化を実行 ▼▼▼
        if self.num_phoneme_classes > 0:
            ph_indices = np.array(
                [
                    self.phoneme_map.get(p, self.phoneme_map.get("unknown", 0))
                    for p in raw_data["phoneme"]
                ],
                dtype=int,
            )
            normalized["phoneme"] = np.eye(self.num_phoneme_classes, dtype=np.float32)[
                ph_indices
            ]

        # 7. EVEC One-hot
        evec_indices = np.clip(raw_data["evec_id"], 0, self.num_evec_classes - 1)
        normalized["evec"] = np.eye(self.num_evec_classes, dtype=np.float32)[
            evec_indices
        ]

        # 8. Repeat Count
        normalized["repeat"] = raw_data["repeat"][:, np.newaxis].astype(np.float32)
        # ★追加: 発声マスクをそのまま引き継ぐ
        normalized["voicing_mask"] = raw_data["voicing_mask"]

        return normalized



    def _get_feature_data(self, param_id, normalized_data):
        """
        ID文字列(例: "PHONEME_P")を受け取り、
        正規化済みの実データ(例: normalized_data["phoneme"])とその次元数を返す。
        カテゴリを横断して検索を行う。
        """
        # 1. 特殊マッピング (ID名 != 内部キー名 の場合)
        if param_id == "PHONEME_P":
            return normalized_data["phoneme"], self.num_phoneme_classes
        if param_id == "EVEC":
            return normalized_data["evec"], self.num_evec_classes
        if param_id == "REPEAT":
            return normalized_data["repeat"], 1

        # 2. カテゴリ内検索 (ID名 == 内部キー名 の場合)
        # 検索対象のカテゴリカテゴリ
        categories = ["dynamic", "note", "melody", "static"]

        for cat in categories:
            if cat in normalized_data and param_id in normalized_data[cat]:
                data = normalized_data[cat][param_id]
                return data, data.shape[1]

        # 3. その他 (Voice Libなど)
        # 必要に応じてマッピングを追加
        if param_id == "VOICE_LIB":  # もしconfigで定義があれば
            return normalized_data["voice_lib"], 2

        print(f"Warning: Parameter ID '{param_id}' not found in parsed data.")
        return None, 0
