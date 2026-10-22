import os
import numpy as np
import librosa
from vsqx_parser import VSQXParser


class VSQXDNNFormatter(VSQXParser):
    def scan_max_note_duration(self, vsqx_dir):
        """
        全VSQXファイルを走査し、データセット内の最大ノート継続フレーム数を算出する。
        算出値は self.max_note_frames に格納される。
        parse_vsqx() の前に呼び出す必要がある。
        """
        print("Scanning all VSQX files for max note duration...")
        max_frames = 0

        for root_dir, _, files in os.walk(vsqx_dir):
            for file in files:
                if file.endswith(".vsqx"):
                    vsqx_path = os.path.join(root_dir, file)
                    try:
                        sparse_data = self.read_sparse(vsqx_path)
                        if sparse_data is None:
                            continue

                        tempo_map = sparse_data["tempo_map"]
                        resolution = sparse_data["resolution"]

                        for note_info in sparse_data["notes"]:
                            start_sec = self.tick_to_second(
                                note_info["abs_start_tick"], tempo_map, resolution
                            )
                            end_sec = self.tick_to_second(
                                note_info["abs_end_tick"], tempo_map, resolution
                            )
                            s_idx = librosa.time_to_frames(
                                start_sec, sr=self.sr, hop_length=self.shift_length
                            )
                            e_idx = librosa.time_to_frames(
                                end_sec, sr=self.sr, hop_length=self.shift_length
                            )
                            n_dur = e_idx - s_idx
                            if n_dur > max_frames:
                                max_frames = n_dur

                    except Exception as e:
                        print(f"Warning: Could not parse {vsqx_path}. Error: {e}")

        if max_frames == 0:
            raise ValueError(
                "max_note_frames が 0 です。"
                "VSQXディレクトリ内にノートが1つも見つかりませんでした。"
            )

        self.max_note_frames = int(max_frames)
        print(f"Max note duration: {self.max_note_frames} frames")

    def parse_vsqx(self, file_path, total_frames):
        """
        VSQXファイルからDNN入力用の密なNumPy配列を構築する。
        内部で read_sparse() を呼び出し、疎なデータを配列に変換する。
        """
        # --- max_note_frames の事前設定チェック ---
        if not hasattr(self, "max_note_frames") or self.max_note_frames is None:
            raise ValueError(
                "max_note_frames が設定されていません。"
                "parse_vsqx() の前に scan_max_note_duration() を実行してください。"
            )

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

        # 学習に使用するパラメータだけ処理を行うため、入力とターゲットのパラメータを特定する
        if has_input or has_target:
            input_params = []
            # パラメータは辞書型なのでvalueを取り出し、リストに追加
            for inputs in self.vsqx_config.get("INPUT_VSQX_PARAMS", {}).values():
                input_params.extend(inputs)
            target_params = []
            for targets in self.vsqx_config.get("TARGET_PARAM_CONFIG", {}).values():
                target_params.extend(targets)
            # 実際に使うパラメータの集合 (検索高速化のためset化)
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
        # 正規化後にデフォルト値が0.0となるように中央値で初期化する
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
        note_elapse_ts = np.zeros(total_frames, dtype=np.float32)
        note_remain_ts = np.zeros(total_frames, dtype=np.float32)
        note_progress_ts = np.zeros(total_frames, dtype=np.float32)
        note_duration_norm_ts = np.zeros(total_frames, dtype=np.float32)
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
            # ここで求めた時間内に値を埋める
            start_sec = self.tick_to_second(abs_start_tick, tempo_map, resolution)
            end_sec = self.tick_to_second(abs_end_tick, tempo_map, resolution)

            s_idx = np.clip(
                librosa.time_to_frames(
                    start_sec, sr=self.sr, hop_length=self.shift_length
                ),
                0,
                total_frames,
            )
            e_idx = np.clip(
                librosa.time_to_frames(end_sec, sr=self.sr, hop_length=self.shift_length),
                0,
                total_frames,
            )

            # 1. スカラーノートパラメータ（Velocity, Accentなど）の入力
            #    固定値でノート区間を埋める
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
                    current_frames, sr=self.sr, hop_length=self.shift_length
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

            # 5. ノート位置・継続時間特徴量の計算
            if note_frames > 0:
                f_elapsed = np.arange(note_frames, dtype=np.float32)
                note_elapse_ts[s_idx:e_idx] = (f_elapsed + 1.0) / self.max_note_frames
                note_remain_ts[s_idx:e_idx] = (note_frames - f_elapsed) / self.max_note_frames
                note_progress_ts[s_idx:e_idx] = (f_elapsed + 1.0) / note_frames
                note_duration_norm_ts[s_idx:e_idx] = note_frames / self.max_note_frames

        # --- 動的パラメータの補間 ---
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
            "note_elapse": note_elapse_ts,
            "note_remain": note_remain_ts,
            "note_progress": note_progress_ts,
            "note_duration_norm": note_duration_norm_ts,
        }

    def normalize_features(self, raw_data):
        """
        parse_vsqxで得られた生データを、モデル入力用に正規化・One-hot化する。
        推論時にもこのメソッドを通すことで学習時と同じ尺度を保証する。
        戻り値は、configで指定される名前（"PIT", "PHONEME_P"など）をキーとした
        フラットな辞書（Flat Dictionary）を返す。
        """
        if raw_data is None:
            raise ValueError("raw_data is None. Cannot normalize empty VSQX data.")

        flat_normalized = {}

        # 1. Dynamic
        for pid, val in raw_data["dynamic"].items():
            if pid == "PIT":
                norm_val = (val / 8192.0)[:, np.newaxis]
            elif pid == "PBS":
                norm_val = ((val / 24.0) * 2.0 - 1.0)[:, np.newaxis]
            else:
                norm_val = ((val / 127.0) * 2.0 - 1.0)[:, np.newaxis]
            flat_normalized[pid] = norm_val

        # 2. Note
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
            flat_normalized[pid] = norm_val

        # 3. Melody
        voicing_mask = raw_data.get("voicing_mask", None)
        if voicing_mask is not None:
            is_voiced = voicing_mask.reshape(-1) > 0.0
        else:
            is_voiced = raw_data["melody"].get("NOTE_NUMBER", np.array([])) > 0.0

        for pid, val in raw_data["melody"].items():
            max_v = self.vsqx_config["NOTE_PARAM_MAX_VALUES"].get(pid, 1.0)
            if pid == "NOTE_NUMBER":
                # 1. 旧デザイン (1次元スカラー: 正規化音高のみ)
                # 有声区間は (val / max_v) * 2.0 - 1.0、休符区間は -1.0
                height_z = np.where(
                    is_voiced, (val / max_v) * 2.0 - 1.0, -1.0
                ).astype(np.float32)
                height_z_1d = height_z[:, np.newaxis]
                flat_normalized["NOTE_NUMBER_SCALAR"] = height_z_1d

                # 2. 新デザイン (3次元滑らか音高螺旋: Pitch Helix)
                # 成分1, 2 (Chroma X, Chroma Y): 12平均律の周期性 (単位円座標)
                # 成分3 (Pitch Height Z): 音高の絶対的な高さ (-1.0 〜 1.0)
                # 休符区間 (is_voiced == False) では Chroma は (0.0, 0.0)（原点＝無音）、Z は -1.0 とする
                theta = 2.0 * np.pi * (val % 12.0) / 12.0
                chroma_x = np.where(is_voiced, np.cos(theta), 0.0).astype(np.float32)
                chroma_y = np.where(is_voiced, np.sin(theta), 0.0).astype(np.float32)
                norm_val_helix = np.stack([chroma_x, chroma_y, height_z], axis=-1)
                flat_normalized[pid] = norm_val_helix
                flat_normalized["NOTE_NUMBER_HELIX"] = norm_val_helix

                # 3. ワンホットデザイン (13次元ハイブリッド音高表現: 音名ワンホット 12次元 + 音高スカラー 1次元)
                # 第0〜11成分 (Pitch Class One-Hot): C=0, C#=1, ..., B=11 の12次元ワンホット
                # 第12成分 (Pitch Height Z): 音高の絶対的な高さ (-1.0 〜 1.0)
                # 休符区間 (is_voiced == False) では全音名 0.0、Z は -1.0 とする
                pitch_class = np.mod(val.astype(int), 12)
                one_hot_12 = np.eye(12, dtype=np.float32)[pitch_class]
                one_hot_12 = np.where(
                    is_voiced[:, np.newaxis], one_hot_12, 0.0
                ).astype(np.float32)
                norm_val_onehot = np.concatenate([one_hot_12, height_z_1d], axis=-1)
                flat_normalized["NOTE_NUMBER_ONEHOT"] = norm_val_onehot

                # 4. 音名ワンホット (13次元純粋One-Hot表現: 音名12次元 + 休符1次元)
                # 第0〜11成分: C=0, C#=1, ..., B=11 の12音名 (有声区間)
                # 第12成分: 休符・無音 (休符区間 is_voiced == False では第12成分が1.0、他は0.0)
                # 全フレームが「1成分のみ1.0、他0.0」の完全なOne-hotとなり、BranchEmbedding (torch.argmax) での休符とC音の衝突を防止する
                pitch_class_idx = np.where(is_voiced, np.mod(val.astype(int), 12), 12)
                one_hot_13 = np.eye(13, dtype=np.float32)[pitch_class_idx]
                flat_normalized["NOTE_CLASS_ONEHOT"] = one_hot_13
            else:
                norm_val = (val / max_v)[:, np.newaxis]
                flat_normalized[pid] = norm_val

        # 4. Static
        for pid, val in raw_data["static"].items():
            norm_val = (val / 127.0)[:, np.newaxis]
            flat_normalized[pid] = norm_val

        # 5. Voice Lib (Already numeric, just ensure float32)
        flat_normalized["VOICE_LIB"] = raw_data["voice_lib"].astype(np.float32)

        # 6. Phoneme One-hot
        if self.num_phoneme_classes > 0:
            ph_indices = np.array(
                [
                    self.phoneme_map.get(p, self.phoneme_map.get("unknown", 0))
                    for p in raw_data["phoneme"]
                ],
                dtype=int,
            )
            flat_normalized["PHONEME_P"] = np.eye(self.num_phoneme_classes, dtype=np.float32)[
                ph_indices
            ]

        # 7. EVEC One-hot
        evec_indices = np.clip(raw_data["evec_id"], 0, self.num_evec_classes - 1)
        flat_normalized["EVEC"] = np.eye(self.num_evec_classes, dtype=np.float32)[
            evec_indices
        ]

        # 8. Repeat Count
        flat_normalized["REPEAT"] = raw_data["repeat"][:, np.newaxis].astype(np.float32)

        # 9. Voicing Mask（そのまま）
        flat_normalized["voicing_mask"] = raw_data["voicing_mask"]

        # 10. ノート位置・継続時間特徴量（既に0.0〜1.0正規化済み、2次元化のみ）
        _NOTE_POS_KEYS = {
            "note_elapse": "NOTE_ELAPSE",
            "note_remain": "NOTE_REMAIN",
            "note_progress": "NOTE_PROGRESS",
            "note_duration_norm": "NOTE_DURATION_NORM",
        }
        for raw_key, norm_key in _NOTE_POS_KEYS.items():
            if raw_key in raw_data:
                flat_normalized[norm_key] = raw_data[raw_key][:, np.newaxis].astype(
                    np.float32
                )

        return flat_normalized
