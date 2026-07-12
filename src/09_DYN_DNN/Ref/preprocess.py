# 固定値のtick調整
import os
import xml.etree.ElementTree as ET
import json
import pickle
import numpy as np
import librosa
import gc
from collections import defaultdict
from scipy import stats
import config as config
from feature_extractor import AudioFeatureExtractor


class VSQXPreprocessor:
    """
    VSQXとWAVの前処理、特徴量抽出、データセット作成を一元管理するクラス。
    学習データ作成時だけでなく、推論時にVSQXをモデル入力形式に変換する際にも使用可能。
    """

    def __init__(self):
        # Configから設定をロード
        self.sr = config.SR
        self.hop_length = config.HOP_LENGTH

        # 特徴量抽出器の初期化
        self.extractor = AudioFeatureExtractor(sr=self.sr, hop_length=self.hop_length)

        # 状態変数
        self.phoneme_map = {}
        self.num_phoneme_classes = 0
        self.evec_map = getattr(config, "EVEC_ID_MAP", {"None": 0})
        self.num_evec_classes = max(self.evec_map.values()) + 1 if self.evec_map else 1

        # VSQX解析用の名前空間
        self.ns = {"vsq4": "http://www.yamaha.co.jp/vocaloid/schema/vsq4/"}

    def normalize_p_tag_for_scan(self, p_text):
        """
        音素スキャン用: <p>タグからEVEC情報(#数値や連打)を除去し、
        辞書登録用の正規化された音素名文字列を返す。
        """
        if p_text is None:
            return None
        text = p_text.strip()
        tokens = text.split()

        # パターンA: ボイスリリース
        if text.startswith("*#"):
            return "br"

        # #が含まれない場合
        if "#" not in text:
            return " ".join(tokens)

        # パターンB & C: 音素の正規化
        clean_tokens = []
        prev_base = None
        for t in tokens:
            base = t.split("#")[0]
            if base != prev_base:
                clean_tokens.append(base)
            prev_base = base

        return " ".join(clean_tokens)

    def parse_p_tag_final(self, p_text):
        # データ生成用: <p>タグを解析し、音素・EVEC ID・連打数を返す。
        result = {"phoneme": "", "evec_id": 0, "repeat": 0}

        if p_text is None:
            return result

        text = p_text.strip()
        tokens = text.split()

        # 1. 音素名の正規化 (self. をつけて呼び出す)
        result["phoneme"] = self.normalize_p_tag_for_scan(text)

        # 2. EVEC情報の抽出
        # パターンA: リリース
        if text.startswith("*#"):
            suffix = text.split("#")[1]
            evec_key = (
                "Release_Short"
                if suffix == "1"
                else "Release_Long" if suffix == "2" else "None"
            )
            # ★ self.evec_map を使用
            result["evec_id"] = self.evec_map.get(evec_key, 0)
            return result

        # '#'がない場合 -> Normal
        if "#" not in text:
            return result

        # '#'を含むトークンを探す
        target_token = next((t for t in tokens if "#" in t), None)
        if target_token:
            base, suffix = target_token.split("#")

            suffix_to_name = {"2": "Soft", "6": "Power"}
            strength = suffix_to_name.get(suffix, None)

            if strength:
                vowels = {"a", "i", "M", "e", "o"}
                is_vowel_color = (
                    len(result["phoneme"].split()) == 1 and result["phoneme"] in vowels
                )

                if is_vowel_color:
                    # ★ self.evec_map を使用
                    result["evec_id"] = self.evec_map.get(f"Color_{strength}", 0)
                else:
                    # ★ self.evec_map を使用
                    result["evec_id"] = self.evec_map.get(f"Exp_{strength}", 0)

                    count = 0
                    for t in tokens:
                        if t.split("#")[0] == base:
                            count += 1
                    result["repeat"] = max(0, count - 1)

        return result

    def set_phoneme_map(self, phoneme_map):
        self.phoneme_map = phoneme_map
        self.num_phoneme_classes = len(phoneme_map)

    def scan_all_phonemes(self, vsqx_dir):
        print("Scanning all VSQX files for unique phonemes (recursive)...")
        all_phonemes = set()

        for root, dirs, files in os.walk(vsqx_dir):
            for file in files:
                if file.endswith(".vsqx"):
                    vsqx_path = os.path.join(root, file)
                    try:
                        tree = ET.parse(
                            vsqx_path, parser=ET.XMLParser(encoding="utf-8")
                        )
                        root_xml = tree.getroot()

                        # トラック検索の堅牢化
                        target_track = None
                        for track in root_xml.findall(".//vsq4:vsTrack", self.ns):
                            t_no = track.find("vsq4:tNo", self.ns)
                            if t_no is not None and int(t_no.text) == 0:
                                target_track = track
                                break

                        if target_track is None:
                            all_tracks = root_xml.findall(".//vsq4:vsTrack", self.ns)
                            if all_tracks:
                                target_track = all_tracks[0]

                        if target_track is not None:
                            # 複数のパート(vsPart)全てから音素を収集
                            for vs_part in target_track.findall("vsq4:vsPart", self.ns):
                                for note in vs_part.findall("vsq4:note", self.ns):
                                    p_elem = note.find("vsq4:p", self.ns)
                                    if p_elem is not None and p_elem.text is not None:
                                        normalized_ph = self.normalize_p_tag_for_scan(
                                            p_elem.text
                                        )
                                        if normalized_ph:
                                            all_phonemes.add(normalized_ph)
                    except Exception as e:
                        print(f"Warning: Could not parse {vsqx_path}. Error: {e}")

        # ID割り当て: silenceを0に固定
        if "silence" in all_phonemes:
            all_phonemes.remove("silence")
        if "unknown" in all_phonemes:
            all_phonemes.remove("unknown")

        sorted_phonemes = sorted(list(all_phonemes))

        self.phoneme_map = {"silence": 0}
        current_id = 1
        for ph in sorted_phonemes:
            self.phoneme_map[ph] = current_id
            current_id += 1

        # unknownを最後に追加
        self.phoneme_map["unknown"] = current_id
        self.num_phoneme_classes = len(self.phoneme_map)

        print(
            f"Found {self.num_phoneme_classes} unique phonemes. (silence ID is fixed to 0)"
        )
        print(f"EVEC Map Loaded: {len(self.evec_map)} classes defined in config.")

    def _get_tempo_map(self, root_xml):
        """
        VSQX内の全テンポ変更イベントを取得し、
        (開始Tick, その区間のBPM) のリストを作成する
        """
        resolution_elem = root_xml.find(".//vsq4:resolution", self.ns)
        resolution = int(resolution_elem.text) if resolution_elem is not None else 480

        # マスタートラックまたはテンポトラックからテンポ情報を収集
        tempos = []

        # vsq4では通常 masterTrack -> tempo にある
        master_track = root_xml.find(".//vsq4:masterTrack", self.ns)
        if master_track is not None:
            for tempo_tag in master_track.findall("vsq4:tempo", self.ns):
                t = int(tempo_tag.find("vsq4:t", self.ns).text)
                v = int(tempo_tag.find("vsq4:v", self.ns).text)
                bpm = v / 100.0
                tempos.append((t, bpm))

        # フォールバック: データがない場合は初期値
        if not tempos:
            tempos.append((0, 120.0))

        # Tick順にソート
        tempos.sort(key=lambda x: x[0])

        # 最初のTickが0でない場合、0からの初期BPM(120)を追加
        if tempos[0][0] > 0:
            tempos.insert(0, (0, 120.0))

        return tempos, resolution

    def _tick_to_second(self, target_tick, tempo_map, resolution):
        """
        テンポマップに基づいて Tick を絶対時間(秒)に変換する
        """
        total_time = 0.0
        current_tick = 0

        for i in range(len(tempo_map)):
            start_tick, bpm = tempo_map[i]

            # 次のテンポ変更までの区間（またはターゲットまで）
            if i + 1 < len(tempo_map):
                next_tick = tempo_map[i + 1][0]
                end_tick = min(target_tick, next_tick)
            else:
                end_tick = target_tick

            if end_tick <= start_tick:
                continue

            # この区間のTick数
            duration_tick = end_tick - max(start_tick, current_tick)
            if duration_tick > 0:
                # 秒 = Tick / Resolution * (60 / BPM)
                sec_per_tick = 60.0 / (bpm * resolution)
                total_time += duration_tick * sec_per_tick
                current_tick = end_tick

            if current_tick >= target_tick:
                break

        return total_time

    def parse_vsqx(self, file_path, total_frames, pre_measure_ticks=1920 * 4):
        try:
            tree = ET.parse(file_path, parser=ET.XMLParser(encoding="utf-8"))
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing VSQX {file_path}: {e}")
            return None

        # --- 1. Tempo Map & Resolution ---
        tempo_map, resolution = self._get_tempo_map(root)

        # VSQXから preMeasure を動的に取得して上書き
        pre_measure = 4
        master_track = root.find(".//vsq4:masterTrack", self.ns)
        if master_track is not None:
            pre_measure_elem = master_track.find("vsq4:preMeasure", self.ns)
            if pre_measure_elem is not None and pre_measure_elem.text:
                pre_measure = int(pre_measure_elem.text)
        pre_measure_ticks = 1920 * pre_measure

        # --- 2. Static Params & Voice Library ---
        static_params = {}
        voice_libraries = {"1st_vocal": None, "2nd_vocal": None}
        v_voice_table = root.find(".//vsq4:vVoiceTable", self.ns)

        if v_voice_table is not None:
            vvoice = v_voice_table.find("vsq4:vVoice", self.ns)
            if vvoice is not None:
                self._extract_voice_info(
                    vvoice, "id", "vPrm", "1st_vocal", voice_libraries, static_params
                )
                self._extract_voice_info(
                    vvoice, "id2", "vPrm2", "2nd_vocal", voice_libraries, static_params
                )

        # --- 3. Initialize Arrays ---
        dynamic_params_ts = {
            pid: np.zeros(total_frames, dtype=np.float32)
            for pid in config.DYNAMIC_PARAM_IDS
        }
        note_params_ts = {
            pid: np.zeros(total_frames, dtype=np.float32)
            for pid in config.NOTE_PARAM_IDS
        }
        melody_params_ts = {
            "NOTE_NUMBER": np.zeros(total_frames, dtype=np.float32),
            "DURATION": np.zeros(total_frames, dtype=np.float32),
        }
        phoneme_ts = np.full(total_frames, "silence", dtype=object)
        voicing_mask_ts = np.zeros(
            (total_frames, 1), dtype=np.float32
        )  # ★追加 (発声マスク)
        evec_id_ts = np.zeros(total_frames, dtype=np.int32)
        repeat_ts = np.zeros(total_frames, dtype=np.float32)
        static_params_ts = {
            pid: np.full(
                total_frames, float(static_params.get(pid, 0)), dtype=np.float32
            )
            for pid in config.STATIC_PARAM_IDS
        }

        voice_lib_ts = np.zeros((total_frames, 2), dtype=np.int32)
        voice_lib_ts[:, 0].fill(
            voice_libraries.get("1st_vocal", config.VOICE_LIB_NAME_TO_ID["unknown"])
        )
        voice_lib_ts[:, 1].fill(
            voice_libraries.get("2nd_vocal", config.VOICE_LIB_NAME_TO_ID["unknown"])
        )

        # --- 4. Track Parsing (修正: 堅牢化) ---
        target_track = None
        # 全トラックを走査して tNo=0 を探す
        for track in root.findall(".//vsq4:vsTrack", self.ns):
            t_no = track.find("vsq4:tNo", self.ns)
            if t_no is not None and int(t_no.text) == 0:
                target_track = track
                break

        if target_track is None:
            print(f"Skip: No track 0 found in {file_path}")
            return None

        # --- 5. Part Parsing (修正: マルチパート対応) ---
        # 1つのトラック内に複数の vsPart がある場合に対応するため、findallで全て取得する
        vs_parts = target_track.findall("vsq4:vsPart", self.ns)
        if not vs_parts:
            print(f"Skip: No vsPart found in Track 0 of {file_path}")
            return None

        # 全パートのCCデータを統合するための辞書
        all_dynamic_data = defaultdict(list)
        vsqx_durations = []

        # --- 各パートごとのループ処理 ---
        for vs_part in vs_parts:
            # Part Start (相対位置の基準)
            part_start_tick = 0
            t_elem = vs_part.find("vsq4:t", self.ns)
            if t_elem is not None:
                part_start_tick = int(t_elem.text)

            # プリメジャー補正 (VSQXの開始位置オフセット)
            part_offset = part_start_tick - pre_measure_ticks

            # --- A. Dynamic Params Extraction (CC) ---
            for cc in vs_part.findall("vsq4:cc", self.ns):
                v_elem = cc.find("vsq4:v", self.ns)
                t_cc = cc.find("vsq4:t", self.ns)
                if v_elem is not None and t_cc is not None:
                    tag_id = v_elem.get("id")
                    param_id = config.ID_MAPPING.get(tag_id)
                    if param_id:
                        # パート内Tick + オフセット = 曲全体の絶対Tick
                        abs_tick = int(t_cc.text) + part_offset
                        all_dynamic_data[param_id].append(
                            {"tick": abs_tick, "value": int(v_elem.text)}
                        )

            # --- B. Note Params Extraction ---
            for note in vs_part.findall("vsq4:note", self.ns):
                note_t = note.find("vsq4:t", self.ns)
                note_dur = note.find("vsq4:dur", self.ns)

                if note_t is not None and note_dur is not None:
                    rel_start_tick = int(note_t.text)
                    dur_tick = int(note_dur.text)
                    vsqx_durations.append(dur_tick)

                    # 絶対Tickの計算
                    abs_start_tick = rel_start_tick + part_offset
                    abs_end_tick = abs_start_tick + dur_tick

                    # Tick -> 秒 変換 (テンポマップ使用)
                    start_sec = self._tick_to_second(
                        abs_start_tick, tempo_map, resolution
                    )
                    end_sec = self._tick_to_second(abs_end_tick, tempo_map, resolution)

                    # フレームインデックス変換
                    s_idx = np.clip(
                        librosa.time_to_frames(
                            start_sec, sr=self.sr, hop_length=self.hop_length
                        ),
                        0,
                        total_frames,
                    )
                    e_idx = np.clip(
                        librosa.time_to_frames(
                            end_sec, sr=self.sr, hop_length=self.hop_length
                        ),
                        0,
                        total_frames,
                    )

                    # パラメータ取得
                    n_params = {
                        pid: config.DEFAULT_NOTE_VALUES.get(pid)
                        for pid in config.NOTE_PARAM_IDS
                    }
                    v_tag = note.find("vsq4:v", self.ns)
                    if v_tag is not None:
                        n_params["VEL"] = int(v_tag.text)

                    nstyle = note.find("vsq4:nStyle", self.ns)
                    if nstyle is not None:
                        # まずスカラー値(seq以外)を取得
                        for v_style in nstyle.findall("vsq4:v", self.ns):
                            pid = v_style.get("id")
                            if pid in n_params and v_style.text:
                                n_params[pid] = int(v_style.text)

                    m_params = {
                        "NOTE_NUMBER": int(note.find("vsq4:n", self.ns).text),
                        "DURATION": dur_tick,
                        "PHONEME_P": note.find("vsq4:p", self.ns).text,
                    }

                    # ========================================================
                    # ★修正箇所: ビブラートシーケンス (vibDep, vibRate) の処理
                    # ========================================================

                    # 1. まず n_params (スカラー/デフォルト値) を適用
                    #    (シーケンスがない場合はこの値がベースになります)
                    for pid, val in n_params.items():
                        if pid in note_params_ts:
                            note_params_ts[pid][s_idx:e_idx] = val

                    # 2. シーケンスデータの処理 (vibDep, vibRate)
                    seq_target_ids = ["vibDep", "vibRate"]

                    # ノート区間の時間軸を作成
                    current_frames = np.arange(s_idx, e_idx)
                    current_times_sec = librosa.frames_to_time(
                        current_frames, sr=self.sr, hop_length=self.hop_length
                    )

                    if nstyle is not None:
                        # 【重要】 findall("vsq4:seq") をやめ、子要素を直接ループする
                        # これにより名前空間(vsq4:)の問題を完全に回避します
                        for child in nstyle:
                            seq_id = child.get("id")

                            # IDが対象(vibDep/vibRate)なら解析
                            if seq_id in seq_target_ids and seq_id in note_params_ts:
                                seq_data_points = []
                                # 子要素(cc)をループ探索
                                for cc in child:
                                    # タグ名に 'cc' が含まれているか判定
                                    if "cc" in cc.tag:
                                        p_val = None
                                        v_val = None
                                        # ccの中身(p, v)を探索
                                        for sub in cc:
                                            # タグ名の末尾判定で p と v を特定
                                            if sub.tag.endswith("p"):
                                                try:
                                                    p_val = int(sub.text)
                                                except:
                                                    pass
                                            elif sub.tag.endswith("v"):
                                                try:
                                                    v_val = int(sub.text)
                                                except:
                                                    pass

                                        # p, v が両方あれば採用
                                        if p_val is not None and v_val is not None:
                                            # 座標変換: 相対(0-65536) -> 絶対時刻
                                            # test_EVEC.vsqx の p=15729 はノートの約24%地点
                                            rel_pos_ratio = p_val / 65536.0
                                            point_abs_tick = abs_start_tick + (
                                                dur_tick * rel_pos_ratio
                                            )
                                            point_time_sec = self._tick_to_second(
                                                point_abs_tick, tempo_map, resolution
                                            )

                                            seq_data_points.append(
                                                (point_time_sec, v_val)
                                            )

                                # データがあれば補間して上書き
                                if seq_data_points:
                                    seq_data_points.sort(key=lambda x: x[0])

                                    xp = [p[0] for p in seq_data_points]
                                    yp = [p[1] for p in seq_data_points]

                                    # 線形補間
                                    interp_vals = np.interp(
                                        current_times_sec,
                                        xp,
                                        yp,
                                        left=yp[0],
                                        right=yp[-1],
                                    )

                                    note_params_ts[seq_id][s_idx:e_idx] = interp_vals
                    # 3. vibLen によるマスク処理 (シーケンス有無に関わらず実行)
                    #    ここを通さないと、デフォルト値や1点データの場合に最初から値が入ってしまう
                    vib_len_val = n_params.get("vibLen", 0)

                    # vibLen < 100 なら、ビブラート開始前を 0 にする
                    if vib_len_val < 100:
                        vib_start_ratio = (100 - vib_len_val) / 100.0
                        vib_start_tick = abs_start_tick + (dur_tick * vib_start_ratio)
                        vib_start_sec = self._tick_to_second(
                            vib_start_tick, tempo_map, resolution
                        )

                        mask_indices = current_times_sec < vib_start_sec

                        # vibDep (深さ) を強制的に 0 にする
                        if "vibDep" in note_params_ts:
                            current_vals = note_params_ts["vibDep"][s_idx:e_idx]
                            current_vals[mask_indices] = 0
                            note_params_ts["vibDep"][s_idx:e_idx] = current_vals
                        if "vibRate" in note_params_ts:
                            current_vals = note_params_ts["vibRate"][s_idx:e_idx]
                            current_vals[mask_indices] = 0
                            note_params_ts["vibRate"][s_idx:e_idx] = current_vals

                    # ▼▼▼ 修正: parse_p_tag_final を使用して3情報を取得 ▼▼▼
                    p_text = note.find("vsq4:p", self.ns).text
                    # クラス内メソッドとして呼び出す (self.parse_p_tag_final)
                    parsed_info = self.parse_p_tag_final(p_text)

                    # 配列への埋め込み: Melody Params & 3 Streams
                    melody_params_ts["NOTE_NUMBER"][s_idx:e_idx] = int(
                        note.find("vsq4:n", self.ns).text
                    )
                    melody_params_ts["DURATION"][s_idx:e_idx] = dur_tick
                    # ★追加: ノートが存在する区間を 1.0 (発声区間) とする
                    voicing_mask_ts[s_idx:e_idx] = 1.0

                    phoneme_ts[s_idx:e_idx] = parsed_info["phoneme"]  # 正規化音素
                    evec_id_ts[s_idx:e_idx] = parsed_info["evec_id"]  # EVEC ID
                    repeat_ts[s_idx:e_idx] = parsed_info["repeat"]  # 連打数

        # --- 6. Dynamic Params Interpolation (全パート統合後) ---
        frame_times_sec = librosa.frames_to_time(
            np.arange(total_frames), sr=self.sr, hop_length=self.hop_length
        )

        for param_id in config.DYNAMIC_PARAM_IDS:
            data = all_dynamic_data.get(param_id, [])
            if len(data) > 0:
                # Tick順にソート（重要：パートまたぎで順序が崩れている可能性があるため）
                data.sort(key=lambda x: x["tick"])

                times_sec = []
                values = []
                for d in data:
                    sec = self._tick_to_second(d["tick"], tempo_map, resolution)
                    times_sec.append(sec)
                    values.append(d["value"])

                # 補間
                interpolated_values = np.interp(frame_times_sec, times_sec, values)
                dynamic_params_ts[param_id] = interpolated_values

        # Max Duration Update
        if vsqx_durations:
            config.NOTE_PARAM_MAX_VALUES["DURATION"] = float(max(vsqx_durations))

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

    def _extract_voice_info(
        self, vvoice_elem, id_tag, prm_tag, label, voice_libs, static_params
    ):
        """Helper to extract voice library info"""
        id_elem = vvoice_elem.find(f"vsq4:{id_tag}", self.ns)
        if id_elem is not None and id_elem.text:
            v_name = config.VOICE_LIB_ID_TO_NAME.get(id_elem.text, "unknown")
            voice_libs[label] = config.VOICE_LIB_NAME_TO_ID.get(
                v_name, config.VOICE_LIB_NAME_TO_ID["unknown"]
            )

            vprm = vvoice_elem.find(f"vsq4:{prm_tag}", self.ns)
            if vprm is not None:
                for param in vprm:
                    pid = param.tag.split("}")[-1].upper()
                    if param.text:
                        static_params[f"{pid}_{label}"] = int(param.text)

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
                max_v = config.NOTE_PARAM_MAX_VALUES.get(pid, 1.0)
                if pid in ["fallPort", "risePort"]:
                    norm_val = (val / max_v)[:, np.newaxis]
                else:
                    norm_val = ((val / max_v) * 2.0 - 1.0)[:, np.newaxis]
            norm_note[pid] = norm_val
        normalized["note"] = norm_note

        # 3. Melody
        norm_melody = {}
        for pid, val in raw_data["melody"].items():
            max_v = config.NOTE_PARAM_MAX_VALUES.get(pid, 1.0)
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

    def _flatten_input_config(self, config_item):
        """
        リストと文字列が混在する設定(INPUT_VSQX_PARAMS)を
        フラットな文字列リストに展開する
        """
        flat_list = []
        if isinstance(config_item, list):
            for item in config_item:
                flat_list.extend(self._flatten_input_config(item))
        else:
            flat_list.append(config_item)
        return flat_list

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

    def build_dataset(self, wav_dir, vsqx_dir, output_dir):
        """無音削除を廃止したデータセット作成処理"""
        os.makedirs(output_dir, exist_ok=True)
        # ▼▼▼ 変更: configの解析を先に行う ▼▼▼
        input_param_ids = self._flatten_input_config(config.INPUT_VSQX_PARAMS)
        target_param_ids = []
        for cat, p_list in config.TARGET_PARAM_CONFIG.items():
            if isinstance(p_list, list):
                target_param_ids.extend(p_list)
            else:
                target_param_ids.append(p_list)

        # 音素("PHONEME_P")が入力・出力のいずれかに含まれているかチェック
        use_phoneme = ("PHONEME_P" in input_param_ids) or (
            "PHONEME_P" in target_param_ids
        )

        if use_phoneme:
            self.scan_all_phonemes(vsqx_dir)
        else:
            print("Skipping phoneme scan (PHONEME_P not found in config).")

        wav_files = {
            os.path.splitext(f)[0]: os.path.join(r, f)
            for r, _, fs in os.walk(wav_dir)
            for f in fs
            if f.endswith(".wav")
        }
        vsqx_files = {
            os.path.splitext(f)[0]: os.path.join(r, f)
            for r, _, fs in os.walk(vsqx_dir)
            for f in fs
            if f.endswith(".vsqx")
        }

        file_pairs = sorted(
            [(k, wav_files[k], vsqx_files[k]) for k in wav_files if k in vsqx_files]
        )
        print(f"Processing {len(file_pairs)} pairs...")

        # --- 1. Global Statistics Calculation (First Pass) ---
        # 全ファイルの音声特徴量を走査して、全体の平均と標準偏差を計算する
        print("Calculating global statistics for audio features...")
        sum_x = None
        sum_sq_x = None
        total_frames_all = 0

        for idx, (name_stem, wav_path, _) in enumerate(file_pairs):
            try:
                # 特徴量抽出（正規化なし）
                features = self.extractor.get_features(wav_path, config.FEATURE_TYPES)

                if sum_x is None:
                    dim = features.shape[1]
                    sum_x = np.zeros(dim, dtype=np.float64)
                    sum_sq_x = np.zeros(dim, dtype=np.float64)

                sum_x += np.sum(features, axis=0)
                sum_sq_x += np.sum(features**2, axis=0)
                total_frames_all += features.shape[0]

                if (idx + 1) % 100 == 0:
                    print(f"Stats scanning: {idx + 1}/{len(file_pairs)}")
            except Exception as e:
                print(f"Error reading {name_stem} for stats: {e}")

        # 平均・標準偏差の算出
        if total_frames_all > 0:
            global_mean = sum_x / total_frames_all
            global_var = (sum_sq_x / total_frames_all) - (global_mean**2)
            # 負の値によるエラー防止とゼロ除算防止
            global_std = np.sqrt(np.maximum(global_var, 1e-7))
        else:
            print("Warning: No valid audio frames found. Using default normalization.")
            global_mean = 0.0
            global_std = 1.0

        print("Global statistics calculated.")

        feature_info = {
            "num_x_features": 0,
            "num_y_features": 0,
            "x_feature_map": [],
            "y_feature_map": [],
            "phoneme_map": self.phoneme_map,
            "num_phoneme_classes": self.num_phoneme_classes,
            "num_evec_classes": self.num_evec_classes,
            "file_lengths": {},
            "feature_types": config.FEATURE_TYPES,
            "input_vsqx_params": input_param_ids,
            "target_vsqx_params": target_param_ids,
            "sr": self.sr,
            "hop_length": self.hop_length,
            # ★ 統計データを追加 (JSON保存用にリスト変換)
            "audio_global_mean": (
                global_mean.tolist()
                if isinstance(global_mean, np.ndarray)
                else global_mean
            ),
            "audio_global_std": (
                global_std.tolist()
                if isinstance(global_std, np.ndarray)
                else global_std
            ),
        }
        is_first = True

        # --- 2. Dataset Generation (Second Pass) ---
        # 本番のデータセット作成処理
        for name_stem, wav_path, vsqx_path in file_pairs:
            try:
                # Audio Processing
                audio_features = self.extractor.get_features(
                    wav_path, config.FEATURE_TYPES
                )
                
                # ★追加: check_timelag.py で特徴量の種類に依存せず確実に同期チェックできるよう、
                # 常に純粋なRMS（音量エンベロープ）を計算して保存する
                y_raw = self.extractor._load_audio(wav_path)
                audio_env_raw = librosa.feature.rms(y=y_raw, hop_length=self.hop_length)[0]
                audio_env_raw = audio_env_raw[:audio_features.shape[0]]  # フレーム長を合わせる

                # ★ Apply Global Normalization ★
                # ファイル単体(stats.zscore)ではなく、全体統計を使って正規化する
                audio_features = (audio_features - global_mean) / (global_std + 1e-7)
                audio_features = np.nan_to_num(audio_features)

                total_frames = audio_features.shape[0]

                # VSQX Processing
                raw_vsqx = self.parse_vsqx(vsqx_path, total_frames)
                if raw_vsqx is None:
                    continue
                norm_vsqx = self.normalize_features(raw_vsqx)

                # Concatenate Features
                X_list = [audio_features]
                if is_first:
                    feature_info["x_feature_map"].append(
                        {"name": "Audio_Features", "dim": audio_features.shape[1]}
                    )

                for pid in input_param_ids:
                    data, dim = self._get_feature_data(pid, norm_vsqx)
                    if data is not None:
                        X_list.append(data)
                        if is_first:
                            feature_info["x_feature_map"].append(
                                {"name": pid, "dim": dim}
                            )

                Y_list = []
                for pid in target_param_ids:
                    data, dim = self._get_feature_data(pid, norm_vsqx)
                    if data is not None:
                        Y_list.append(data)
                        if is_first:
                            feature_info["y_feature_map"].append(
                                {"name": pid, "dim": dim}
                            )

                X_arr = np.concatenate(X_list, axis=-1).astype(np.float32)
                Y_arr = (
                    np.concatenate(Y_list, axis=-1).astype(np.float32)
                    if Y_list
                    else np.empty((total_frames, 0), dtype=np.float32)
                )
                # 名残、整理した方が良いが面倒
                X_final, Y_final = X_arr, Y_arr

                if is_first:
                    feature_info["num_x_features"] = X_final.shape[1]
                    feature_info["num_y_features"] = Y_final.shape[1]
                    is_first = False

                voicing_mask_data = norm_vsqx["voicing_mask"]

                np.savez_compressed(
                    os.path.join(output_dir, f"{name_stem}.npz"),
                    x=X_final,
                    y=Y_final,
                    voicing_mask=voicing_mask_data,
                    audio_env=audio_env_raw,  # ★追加: 生のエンベロープを保存
                )
                feature_info["file_lengths"][name_stem] = X_final.shape[0]
                print(f"Saved: {name_stem} ({X_final.shape})")

                # メモリを解放
                del audio_features, raw_vsqx, norm_vsqx, X_arr, Y_arr, X_final, Y_final
                gc.collect()

            except Exception as e:
                print(f"CRITICAL ERROR processing {name_stem}: {e}")
                import traceback

                traceback.print_exc()

        with open(
            os.path.join(output_dir, "features.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(feature_info, f, indent=4, ensure_ascii=False)
        print("Dataset generation completed.")


if __name__ == "__main__":
    # 実行例
    builder = VSQXPreprocessor()
    builder.build_dataset(
        wav_dir=config.WAV_DIR, vsqx_dir=config.VSQX_DIR, output_dir=config.OUTPUT_DIR
    )
