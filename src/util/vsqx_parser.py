import os
import xml.etree.ElementTree as ET
import json
import pickle
import numpy as np
import librosa
import gc
from collections import defaultdict
from scipy import stats
from feature_extractor import AudioFeatureExtractor


class VSQXParser:
    """
    VSQXの前処理、特徴量抽出を管理するクラス。
    """

    def __init__(self, sr, hop_length):
        # 外部からsrとhop_lengthを受け取る設計に変更
        self.sr = sr
        self.hop_length = hop_length

        self.vsqx_config = {
            "DYNAMIC_PARAM_IDS": [
                "DYN",
                "BRE",
                "BRI",
                "CLE",
                "GEN",
                "POR",
                "PIT",
                "PBS",
                "GWL",
                "XSY",
            ],
            "NOTE_PARAM_IDS": [
                "VEL",
                "accent",
                "bendDep",
                "bendLen",
                "decay",
                "fallPort",
                "opening",
                "risePort",
                "vibLen",
                "vibType",
                "vibDep",
                "vibRate",
            ],
            "MELODY_PARAM_IDS": [
                "NOTE_NUMBER",
                "DURATION",
                "PHONEME_P",
                "EVEC",
                "REPEAT",
            ],
            "STATIC_PARAM_IDS": [
                "BRE_1st_vocal",
                "BRI_1st_vocal",
                "CLE_1st_vocal",
                "GEN_1st_vocal",
                "OPE_1st_vocal",
                "BRE_2nd_vocal",
                "BRI_2nd_vocal",
                "CLE_2nd_vocal",
                "GEN_2nd_vocal",
                "OPE_2nd_vocal",
                "VOL_2nd_vocal",
            ],
            "ID_MAPPING": {
                "D": "DYN",
                "B": "BRE",
                "R": "BRI",
                "C": "CLE",
                "G": "GEN",
                "T": "POR",
                "P": "PIT",
                "S": "PBS",
                "W": "GWL",
                "X": "XSY",
            },
            "DEFAULT_DYNAMIC_VALUES": {
                "DYN": 64,
                "BRE": 0,
                "BRI": 64,
                "CLE": 0,
                "GEN": 64,
                "POR": 64,
                "PIT": 0,
                "PBS": 1,
                "GWL": 0,
                "XSY": 0,
            },
            "DEFAULT_NOTE_VALUES": {
                "VEL": 64,
                "accent": 50,
                "bendDep": 8,
                "bendLen": 0,
                "decay": 50,
                "fallPort": 0,
                "opening": 127,
                "risePort": 0,
                "vibLen": 66,
                "vibType": 1,
                "vibDep": 120,
                "vibRate": 30,
            },
            "NOTE_PARAM_MAX_VALUES": {
                "VEL": 127.0,
                "accent": 100.0,
                "bendDep": 100.0,
                "bendLen": 100.0,
                "decay": 100.0,
                "fallPort": 1.0,
                "opening": 127.0,
                "risePort": 1.0,
                "vibLen": 100.0,
                "vibType": 16.0,
                "vibDep": 127,
                "vibRate": 127,
                "NOTE_NUMBER": 127.0,
                "DURATION": 7680.0,
            },
            "VOICE_LIB_ID_TO_NAME": {
                "BCNFCY43LB2LZCD4": "MIKU_V4X_Original_EVEC",
                "BFPL93T7GE3RWFC9": "MIKU_V4X_Soft_EVEC",
                "BKGHF6Y5PHTN4KD6": "MIKU_V4X_Solid_EVEC",
                "BHNME578NG2S3HD8": "MIKU_V4X_Dark",
                "BDGE8ZB2FCTKTDBA": "MIKU_V4X_Sweet",
            },
            "VOICE_LIB_NAME_TO_ID": {
                "MIKU_V4X_Original_EVEC": 0,
                "MIKU_V4X_Soft_EVEC": 1,
                "MIKU_V4X_Solid_EVEC": 2,
                "MIKU_V4X_Dark": 3,
                "MIKU_V4X_Sweet": 4,
                "unknown": 5,
            },
            "EVEC_ID_MAP": {
                "None": 0,
                "Release_Short": 1,
                "Release_Long": 2,
                "Color_Soft": 3,
                "Color_Power": 4,
                "Exp_Soft": 5,
                "Exp_Power": 6,
            },
            "VOWELS": {"a", "i", "M", "e", "o"},
        }

        # 状態変数
        self.phoneme_map = {}
        self.num_phoneme_classes = 0
        self.evec_map = self.vsqx_config.get("EVEC_ID_MAP", {"None": 0})
        self.num_evec_classes = max(self.evec_map.values()) + 1 if self.evec_map else 1

        # VSQX解析用の名前空間
        self.ns = {"vsq4": "http://www.yamaha.co.jp/vocaloid/schema/vsq4/"}

    def normalize_p_tag_for_scan(self, p_text):
        """
        音素スキャン用: <p>タグからEVEC情報(#数値や連打などの特殊な音素)を除去し、
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

        # パターンB & C: #の除去＆連続する音素の結合（音素の正規化）
        clean_tokens = []
        prev_base = None
        for t in tokens:
            base = t.split("#")[0]
            if base != prev_base:
                clean_tokens.append(base)
            prev_base = base

        return " ".join(clean_tokens)

    def parse_p_tag_final(self, p_text):
        """
        最終的な歌詞データ生成用
        <p>タグを解析し、音素(上記正規化関数で返す音素)・EVEC ID・連打数を返す。
        （下記resultの形式）
        """
        result = {"phoneme": "", "evec_id": 0, "repeat": 0}

        if p_text is None:
            return result

        text = p_text.strip()
        tokens = text.split()

        # 1. 音素名の正規化
        result["phoneme"] = self.normalize_p_tag_for_scan(text)

        # 2. EVEC情報の抽出
        # パターンA: brリリース
        if text.startswith("*#"):
            suffix = text.split("#")[1]
            evec_key = (
                "Release_Short"
                if suffix == "1"
                else "Release_Long" if suffix == "2" else "None"
            )
            # self.evec_map を使用
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
                vowels = self.vsqx_config["VOWELS"]
                # 母音判定
                is_vowel_color = (
                    len(result["phoneme"].split()) == 1 and result["phoneme"] in vowels
                )

                # EVEC ID付与
                if is_vowel_color:
                    # 母音ならColor
                    result["evec_id"] = self.evec_map.get(f"Color_{strength}", 0)
                else:
                    # 子音ならExp＆連打数カウント
                    result["evec_id"] = self.evec_map.get(f"Exp_{strength}", 0)
                    count = 0
                    for t in tokens:
                        if t.split("#")[0] == base:
                            count += 1
                    result["repeat"] = max(0, count - 1)

        return result

    def set_phoneme_map(self, phoneme_map):
        """
        動的な音素マップを作成するための関数
        """
        self.phoneme_map = phoneme_map
        self.num_phoneme_classes = len(phoneme_map)

    def scan_all_phonemes(self, vsqx_dir):
        """
        再帰的にVSQXファイルをスキャンし、音素辞書を作成する
        """
        print("Scanning all VSQX files for unique phonemes...")
        all_phonemes = set()

        for root, dirs, files in os.walk(vsqx_dir):
            for file in files:
                if file.endswith(".vsqx"):
                    vsqx_path = os.path.join(root, file)
                    try:
                        # VSQXファイルをtree化
                        tree = ET.parse(
                            vsqx_path, parser=ET.XMLParser(encoding="utf-8")
                        )
                        root_xml = tree.getroot()

                        # トラック検索
                        target_track = None
                        for track in root_xml.findall(".//vsq4:vsTrack", self.ns):
                            t_no = track.find("vsq4:tNo", self.ns)
                            if t_no is not None and int(t_no.text) == 0:
                                target_track = track
                                break

                        if target_track is None:
                            raise ValueError(
                                f"{vsqx_path} に tNo=0 のメイントラックが存在しません。ファイルが破損しているか規格外です。"
                            )

                        if target_track is not None:
                            # 複数の歌唱領域(vsPart)全てから音素を収集
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

        # 例外音素（silence, unknown）を除外してソート
        if "silence" in all_phonemes:
            all_phonemes.remove("silence")
        if "unknown" in all_phonemes:
            all_phonemes.remove("unknown")

        sorted_phonemes = sorted(list(all_phonemes))

        self.phoneme_map = {}
        current_id = 0

        # 1. 通常の音素に0から順にIDを割り当て
        for ph in sorted_phonemes:
            self.phoneme_map[ph] = current_id
            current_id += 1

        # 2. 例外音素（メタキーワード）を辞書の最後に追加
        self.phoneme_map["silence"] = current_id
        current_id += 1
        self.phoneme_map["unknown"] = current_id

        self.num_phoneme_classes = len(self.phoneme_map)

        print(f"Found {self.num_phoneme_classes} unique phonemes.")
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
                bpm = v / 100.0  # vsqxファイルではBPMが100倍の値で記録されているため
                tempos.append((t, bpm))

        # エラー処理
        if not tempos:
            raise ValueError(
                "No tempo information found in the VSQX file. The file might be corrupted or missing a masterTrack."
            )

        # Tick順にソート
        tempos.sort(key=lambda x: x[0])

        # 最初のTickが0でない場合、エラーで止めるのではなく、最初のBPMをコピーしてTick 0に挿入する
        if tempos[0][0] > 0:
            tempos.insert(0, (0, tempos[0][1]))

        return tempos, resolution

    def _shift_tempo_map(self, tempo_map, offset_ticks):
        """
        テンポマップのTickを offset_ticks 分だけマイナス方向にシフトし、
        Tick 0 以降のマップとして再構築(クロップ)する。
        これにより浮動小数点の丸め誤差を回避しつつ、正しいBPM区間を適用する。
        """
        if offset_ticks == 0:
            return tempo_map

        shifted = [(t - offset_ticks, bpm) for t, bpm in tempo_map]

        # Tick 0 時点で有効なBPMを特定する
        bpm_at_zero = shifted[0][1]
        for t, bpm in shifted:
            if t <= 0:
                bpm_at_zero = bpm

        final_map = [(0, bpm_at_zero)]
        for t, bpm in shifted:
            if t > 0:
                final_map.append((t, bpm))

        return final_map

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

    def _interpolate_to_frames(
        self, tick_points, tempo_map, resolution, total_frames, default_value=64
    ):
        """
        {"tick": int, "value": int} 形式の点データリストを
        フレーム単位の密な1次元配列に線形補間する。

        既存の parse_vsqx 内 L702-L721 の直書き処理を共通メソッドとして切り出したもの。
        アルゴリズムは完全に既存のものを踏襲する。

        Args:
            tick_points: [{"tick": int, "value": int}, ...] 形式の点データリスト
            tempo_map: _get_tempo_map() の戻り値 [(tick, bpm), ...]
            resolution: VSQXの分解能 (通常480)
            total_frames: 出力フレーム数
            default_value: tick_points が空の場合のフォールバック値

        Returns:
            np.ndarray: shape (total_frames,) の補間済み配列
        """
        if not tick_points:
            return np.full(total_frames, default_value, dtype=np.float32)

        # Tick順にソート（パートまたぎで順序が崩れている可能性があるため）
        sorted_points = sorted(tick_points, key=lambda x: x["tick"])

        # 各点の tick を秒に変換
        times_sec = []
        values = []
        for d in sorted_points:
            sec = self._tick_to_second(d["tick"], tempo_map, resolution)
            times_sec.append(sec)
            values.append(d["value"])

        # フレーム時間軸を生成し線形補間
        frame_times_sec = librosa.frames_to_time(
            np.arange(total_frames), sr=self.sr, hop_length=self.hop_length
        )
        interpolated_values = np.interp(frame_times_sec, times_sec, values)

        return interpolated_values.astype(np.float32)

    # =====================================================================
    # Layer 1: 疎なデータ抽出層
    # XMLへのアクセスはこのメソッドとそのヘルパーのみに限定する。
    # =====================================================================

    def read_sparse(self, file_path, use_part_offset=False):
        """
        VSQXをパースし、全パラメータを tick 形式の疎な点データ辞書として返す。

        Args:
            file_path: VSQXファイルのパス
            use_part_offset: True の場合、パート開始Tickからプリメジャーを
                差し引いたオフセットを適用する（DNN用）。
                False の場合、オフセットを適用しない（可視化用）。

        Returns:
            dict: 疎なデータ辞書。構造は以下の通り:
                {
                    "tempo_map": [(tick, bpm), ...],
                    "resolution": int,
                    "pre_measure_ticks": int,
                    "static_params": {"BRE_1st_vocal": int, ...},
                    "voice_info": {"1st_vocal": str, "2nd_vocal": str},
                    "voice_libraries": {"1st_vocal": int, "2nd_vocal": int},
                    "dynamic_params": {"DYN": [{"tick": int, "value": int}, ...], ...},
                    "notes": [{...}, ...],
                }
            パース失敗時は None を返す。
        """
        try:
            tree = ET.parse(file_path, parser=ET.XMLParser(encoding="utf-8"))
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing VSQX {file_path}: {e}")
            return None

        # --- 1. Tempo Map & Resolution ---
        tempo_map, resolution = self._get_tempo_map(root)

        # --- 2. PreMeasure ---
        pre_measure = 4
        master_track = root.find(".//vsq4:masterTrack", self.ns)
        if master_track is not None:
            pre_measure_elem = master_track.find("vsq4:preMeasure", self.ns)
            if pre_measure_elem is not None and pre_measure_elem.text:
                pre_measure = int(pre_measure_elem.text)
        pre_measure_ticks = 1920 * pre_measure

        # --- 3. Static Params & Voice Library ---
        static_params = {}
        voice_libraries = {"1st_vocal": None, "2nd_vocal": None}
        voice_info = {"1st_vocal": "unknown", "2nd_vocal": "unknown"}
        v_voice_table = root.find(".//vsq4:vVoiceTable", self.ns)

        if v_voice_table is not None:
            vvoice = v_voice_table.find("vsq4:vVoice", self.ns)
            if vvoice is not None:
                # 数値ID（DNN用）
                self._extract_voice_info(
                    vvoice, "id", "vPrm", "1st_vocal", voice_libraries, static_params
                )
                self._extract_voice_info(
                    vvoice, "id2", "vPrm2", "2nd_vocal", voice_libraries, static_params
                )
                # 文字列名（可視化用）
                name_elem = vvoice.find("vsq4:name", self.ns)
                if name_elem is not None and name_elem.text:
                    voice_info["1st_vocal"] = name_elem.text

        # --- 4. Track Parsing ---
        target_track = None
        for track in root.findall(".//vsq4:vsTrack", self.ns):
            t_no = track.find("vsq4:tNo", self.ns)
            if t_no is not None and int(t_no.text) == 0:
                target_track = track
                break

        if target_track is None:
            print(f"Skip: No track 0 found in {file_path}")
            return None

        # --- 5. Part Parsing ---
        vs_parts = target_track.findall("vsq4:vsPart", self.ns)
        if not vs_parts:
            print(f"Skip: No vsPart found in Track 0 of {file_path}")
            return None

        all_dynamic_data = defaultdict(list)
        all_notes = []

        for vs_part in vs_parts:
            # パート開始Tick
            t_elem = vs_part.find("vsq4:t", self.ns)
            part_start_tick = int(t_elem.text) if t_elem is not None else 0

            # オフセット計算
            if use_part_offset:
                part_offset = part_start_tick - pre_measure_ticks
            else:
                part_offset = 0

            # --- A. Dynamic Params (CC) ---
            for cc in vs_part.findall("vsq4:cc", self.ns):
                v_elem = cc.find("vsq4:v", self.ns)
                t_cc = cc.find("vsq4:t", self.ns)
                if v_elem is not None and t_cc is not None:
                    tag_id = v_elem.get("id")
                    param_id = self.vsqx_config["ID_MAPPING"].get(tag_id)
                    if param_id:
                        abs_tick = int(t_cc.text) + part_offset
                        all_dynamic_data[param_id].append(
                            {"tick": abs_tick, "value": int(v_elem.text)}
                        )

            # --- B. Note Params ---
            for note in vs_part.findall("vsq4:note", self.ns):
                note_t = note.find("vsq4:t", self.ns)
                note_dur = note.find("vsq4:dur", self.ns)

                if note_t is not None and note_dur is not None:
                    rel_start_tick = int(note_t.text)
                    dur_tick = int(note_dur.text)

                    abs_start_tick = rel_start_tick + part_offset
                    abs_end_tick = abs_start_tick + dur_tick

                    # スカラーノートパラメータ（デフォルト値で初期化）
                    n_params = {
                        pid: self.vsqx_config["DEFAULT_NOTE_VALUES"].get(pid)
                        for pid in self.vsqx_config["NOTE_PARAM_IDS"]
                    }
                    v_tag = note.find("vsq4:v", self.ns)
                    if v_tag is not None:
                        n_params["VEL"] = int(v_tag.text)

                    nstyle = note.find("vsq4:nStyle", self.ns)
                    if nstyle is not None:
                        for v_style in nstyle.findall("vsq4:v", self.ns):
                            pid = v_style.get("id")
                            if pid in n_params and v_style.text:
                                n_params[pid] = int(v_style.text)

                    # シーケンスデータ (vibDep, vibRate)
                    seqs = {}
                    seq_target_ids = ["vibDep", "vibRate"]

                    if nstyle is not None:
                        for child in nstyle:
                            seq_id = child.get("id")
                            if seq_id in seq_target_ids:
                                seq_data_points = []
                                for cc_elem in child:
                                    if "cc" in cc_elem.tag:
                                        p_val = None
                                        v_val = None
                                        for sub in cc_elem:
                                            if sub.tag.endswith("p"):
                                                try:
                                                    p_val = int(sub.text)
                                                except Exception:
                                                    pass
                                            elif sub.tag.endswith("v"):
                                                try:
                                                    v_val = int(sub.text)
                                                except Exception:
                                                    pass
                                        if p_val is not None and v_val is not None:
                                            # 相対位置(0-65536) → 絶対tickに変換
                                            rel_pos_ratio = p_val / 65536.0
                                            point_abs_tick = int(
                                                abs_start_tick
                                                + (dur_tick * rel_pos_ratio)
                                            )
                                            seq_data_points.append(
                                                {"tick": point_abs_tick, "value": v_val}
                                            )
                                if seq_data_points:
                                    seqs[seq_id] = sorted(
                                        seq_data_points, key=lambda x: x["tick"]
                                    )

                    # 音素・EVEC・連打の解析
                    p_elem = note.find("vsq4:p", self.ns)
                    p_text = p_elem.text if p_elem is not None else None
                    parsed_info = self.parse_p_tag_final(p_text)

                    # 歌詞
                    lyric_elem = note.find("vsq4:y", self.ns)
                    lyric = lyric_elem.text if lyric_elem is not None else ""

                    # ノート番号
                    note_n = note.find("vsq4:n", self.ns)
                    note_num = int(note_n.text) if note_n is not None else 0

                    all_notes.append(
                        {
                            "abs_start_tick": abs_start_tick,
                            "abs_end_tick": abs_end_tick,
                            "dur_tick": dur_tick,
                            "note_num": note_num,
                            "lyric": lyric,
                            "phoneme": parsed_info["phoneme"],
                            "evec_id": parsed_info["evec_id"],
                            "repeat": parsed_info["repeat"],
                            "params": n_params,
                            "seqs": seqs,
                        }
                    )

        # 動的パラメータをtick順にソート
        for pid in all_dynamic_data:
            all_dynamic_data[pid].sort(key=lambda x: x["tick"])

        # テンポマップのシフト処理 (DNN構築用のオフセット適用時)
        if use_part_offset:
            tempo_map = self._shift_tempo_map(tempo_map, pre_measure_ticks)

        return {
            "tempo_map": tempo_map,
            "resolution": resolution,
            "pre_measure_ticks": pre_measure_ticks,
            "static_params": static_params,
            "voice_info": voice_info,
            "voice_libraries": voice_libraries,
            "dynamic_params": dict(all_dynamic_data),
            "notes": sorted(all_notes, key=lambda n: n["abs_start_tick"]),
        }

    # =====================================================================
    # Layer 2: 密なデータ変換層
    # read_sparse() の結果をDNN入力用の密なNumPy配列に変換する。
    # =====================================================================

    def _extract_voice_info(
        self, vvoice_elem, id_tag, prm_tag, label, voice_libs, static_params
    ):
        """Helper to extract voice library info"""
        id_elem = vvoice_elem.find(f"vsq4:{id_tag}", self.ns)
        if id_elem is not None and id_elem.text:
            v_name = self.vsqx_config["VOICE_LIB_ID_TO_NAME"].get(
                id_elem.text, "unknown"
            )
            voice_libs[label] = self.vsqx_config["VOICE_LIB_NAME_TO_ID"].get(
                v_name, self.vsqx_config["VOICE_LIB_NAME_TO_ID"]["unknown"]
            )

            vprm = vvoice_elem.find(f"vsq4:{prm_tag}", self.ns)
            if vprm is not None:
                for param in vprm:
                    pid = param.tag.split("}")[-1].upper()
                    if param.text:
                        static_params[f"{pid}_{label}"] = int(param.text)

    def tick_to_second(self, target_tick, tempo_map, resolution):
        return self._tick_to_second(target_tick, tempo_map, resolution)
