"""
VSQXファイルから動的パラメータ・ノート情報・静的パラメータを
軽量に読み取る解析ツール専用モジュール。

preprocess.py の解析ロジックをベースに、
比較表示に必要な情報のみを抽出する簡略版。
"""

import xml.etree.ElementTree as ET
import numpy as np
import librosa


class VSQXReader:
    """VSQXファイルの軽量リーダー"""

    def __init__(self, sr=44100, hop_length=256, use_part_offset=False):
        self.sr = sr
        self.hop_length = hop_length
        self.use_part_offset = use_part_offset
        self.ns = {"vsq4": "http://www.yamaha.co.jp/vocaloid/schema/vsq4/"}

        # config.py と同一のマッピング
        self.id_mapping = {
            "D": "DYN", "B": "BRE", "R": "BRI", "C": "CLE",
            "G": "GEN", "T": "POR", "P": "PIT", "S": "PBS",
            "W": "GWL", "X": "XSY",
        }

    def read(self, vsqx_path):
        """
        VSQXファイルを解析し、パラメータ情報を辞書で返す。

        Returns:
            dict: {
                "dynamic_params": {param_id: [(time_sec, value), ...], ...},
                "notes": [{"start_sec", "end_sec", "note_num", "lyric", "phoneme"}, ...],
                "static_params": {"BRE_1st_vocal": val, ...},
                "voice_info": {"1st_vocal": name, "2nd_vocal": name},
                "tempo": float,
                "resolution": int,
            }
        """
        try:
            tree = ET.parse(vsqx_path, parser=ET.XMLParser(encoding="utf-8"))
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing VSQX {vsqx_path}: {e}")
            return None

        # --- テンポマップ・解像度・プリメジャー ---
        tempo_map, resolution = self._get_tempo_map(root)
        primary_tempo = tempo_map[0][1] if tempo_map else 120.0
        
        pre_measure = 4
        master_track = root.find("vsq4:masterTrack", self.ns)
        if master_track is not None:
            pre_measure_elem = master_track.find("vsq4:preMeasure", self.ns)
            if pre_measure_elem is not None and pre_measure_elem.text:
                pre_measure = int(pre_measure_elem.text)
        pre_measure_ticks = 1920 * pre_measure

        # --- 静的パラメータ・ボイスライブラリ ---
        static_params = {}
        voice_info = {"1st_vocal": "unknown", "2nd_vocal": "unknown"}
        v_voice_table = root.find(".//vsq4:vVoiceTable", self.ns)

        if v_voice_table is not None:
            vvoice = v_voice_table.find("vsq4:vVoice", self.ns)
            if vvoice is not None:
                # 1st vocal
                id_elem = vvoice.find("vsq4:id", self.ns)
                name_elem = vvoice.find("vsq4:name", self.ns)
                if name_elem is not None and name_elem.text:
                    voice_info["1st_vocal"] = name_elem.text

                vprm = vvoice.find("vsq4:vPrm", self.ns)
                if vprm is not None:
                    for param in vprm:
                        pid = param.tag.split("}")[-1].upper()
                        if param.text:
                            static_params[f"{pid}_1st_vocal"] = int(param.text)

                # 2nd vocal
                name2_elem = vvoice.find("vsq4:id2", self.ns)
                vprm2 = vvoice.find("vsq4:vPrm2", self.ns)
                if vprm2 is not None:
                    for param in vprm2:
                        pid = param.tag.split("}")[-1].upper()
                        if param.text:
                            static_params[f"{pid}_2nd_vocal"] = int(param.text)

        # --- トラック解析 ---
        target_track = None
        for track in root.findall(".//vsq4:vsTrack", self.ns):
            t_no = track.find("vsq4:tNo", self.ns)
            if t_no is not None and int(t_no.text) == 0:
                target_track = track
                break

        if target_track is None:
            all_tracks = root.findall(".//vsq4:vsTrack", self.ns)
            if all_tracks:
                target_track = all_tracks[0]

        dynamic_params = {}
        notes = []

        if target_track is not None:
            vs_parts = target_track.findall("vsq4:vsPart", self.ns)

            for vs_part in vs_parts:
                # パート開始Tick
                t_elem = vs_part.find("vsq4:t", self.ns)
                part_start_tick = int(t_elem.text) if t_elem is not None else 0

                # DNNパイプライン等でWAVが絶対時間で書き出されている場合は part_offset を適用
                part_offset = 0
                if self.use_part_offset:
                    # 動的に取得したプリメジャー補正を差し引いたオフセット
                    part_offset = part_start_tick - pre_measure_ticks

                # --- CC (動的パラメータ) ---
                for cc in vs_part.findall("vsq4:cc", self.ns):
                    v_elem = cc.find("vsq4:v", self.ns)
                    t_cc = cc.find("vsq4:t", self.ns)
                    if v_elem is not None and t_cc is not None:
                        tag_id = v_elem.get("id")
                        param_id = self.id_mapping.get(tag_id)
                        if param_id:
                            rel_tick = int(t_cc.text)
                            abs_tick = rel_tick + part_offset
                            time_sec = self._tick_to_second(
                                abs_tick, tempo_map, resolution
                            )
                            value = int(v_elem.text)

                            if param_id not in dynamic_params:
                                dynamic_params[param_id] = []
                            dynamic_params[param_id].append((time_sec, value))

                # --- ノート情報 ---
                for note in vs_part.findall("vsq4:note", self.ns):
                    note_t = note.find("vsq4:t", self.ns)
                    note_dur = note.find("vsq4:dur", self.ns)
                    note_n = note.find("vsq4:n", self.ns)

                    if note_t is not None and note_dur is not None:
                        rel_start_tick = int(note_t.text)
                        dur_tick = int(note_dur.text)
                        
                        abs_start_tick = rel_start_tick + part_offset
                        abs_end_tick = abs_start_tick + dur_tick

                        start_sec = self._tick_to_second(
                            abs_start_tick, tempo_map, resolution
                        )
                        end_sec = self._tick_to_second(
                            abs_end_tick, tempo_map, resolution
                        )

                        lyric_elem = note.find("vsq4:y", self.ns)
                        phoneme_elem = note.find("vsq4:p", self.ns)

                        notes.append({
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "note_num": int(note_n.text) if note_n is not None else 0,
                            "lyric": lyric_elem.text if lyric_elem is not None else "",
                            "phoneme": phoneme_elem.text if phoneme_elem is not None else "",
                        })

        # 時間順にソート
        for pid in dynamic_params:
            dynamic_params[pid].sort(key=lambda x: x[0])

        return {
            "dynamic_params": dynamic_params,
            "notes": sorted(notes, key=lambda n: n["start_sec"]),
            "static_params": static_params,
            "voice_info": voice_info,
            "tempo": primary_tempo,
            "resolution": resolution,
        }

    def interpolate_param_to_frames(self, control_points, total_frames,
                                     default_value=64):
        """
        CC制御点リストをフレーム単位の時系列に線形補間する。

        Args:
            control_points: [(time_sec, value), ...] のリスト
            total_frames: 出力フレーム数
            default_value: 制御点がない場合のデフォルト値

        Returns:
            np.ndarray: shape (total_frames,)
        """
        if not control_points:
            return np.full(total_frames, default_value, dtype=np.float32)

        frame_times = librosa.frames_to_time(
            np.arange(total_frames), sr=self.sr, hop_length=self.hop_length
        )

        times = [cp[0] for cp in control_points]
        values = [cp[1] for cp in control_points]

        interpolated = np.interp(frame_times, times, values)
        return interpolated.astype(np.float32)

    def find_changed_params(self, vsqx_data_a, vsqx_data_b):
        """
        2つのVSQXデータを比較し、CC制御点に差分があるパラメータを検出する。

        Returns:
            list: 変更があったパラメータIDのリスト (例: ["DYN", "BRE"])
        """
        changed = []
        all_param_ids = set(
            list(vsqx_data_a["dynamic_params"].keys()) +
            list(vsqx_data_b["dynamic_params"].keys())
        )

        for pid in all_param_ids:
            points_a = vsqx_data_a["dynamic_params"].get(pid, [])
            points_b = vsqx_data_b["dynamic_params"].get(pid, [])

            # 片方にだけ存在する、または制御点が異なる
            if len(points_a) != len(points_b):
                changed.append(pid)
            elif points_a != points_b:
                changed.append(pid)

        return sorted(changed)

    def find_static_param_diffs(self, vsqx_data_a, vsqx_data_b):
        """
        2つのVSQXデータの静的パラメータの差分を検出する。

        Returns:
            dict: {param_id: (value_a, value_b), ...} 差分があるもののみ
        """
        diffs = {}
        all_keys = set(
            list(vsqx_data_a["static_params"].keys()) +
            list(vsqx_data_b["static_params"].keys())
        )

        for key in sorted(all_keys):
            val_a = vsqx_data_a["static_params"].get(key, 0)
            val_b = vsqx_data_b["static_params"].get(key, 0)
            if val_a != val_b:
                diffs[key] = (val_a, val_b)

        return diffs

    # ---- 内部メソッド (preprocess.py 由来) ----

    def _get_tempo_map(self, root_xml):
        """テンポマップを取得"""
        resolution_elem = root_xml.find(".//vsq4:resolution", self.ns)
        resolution = int(resolution_elem.text) if resolution_elem is not None else 480

        tempos = []
        master_track = root_xml.find(".//vsq4:masterTrack", self.ns)
        if master_track is not None:
            for tempo_tag in master_track.findall("vsq4:tempo", self.ns):
                t = int(tempo_tag.find("vsq4:t", self.ns).text)
                v = int(tempo_tag.find("vsq4:v", self.ns).text)
                bpm = v / 100.0
                tempos.append((t, bpm))

        if not tempos:
            tempos.append((0, 120.0))

        tempos.sort(key=lambda x: x[0])

        if tempos[0][0] > 0:
            tempos.insert(0, (0, 120.0))

        return tempos, resolution

    def _tick_to_second(self, target_tick, tempo_map, resolution):
        """Tickを絶対時間(秒)に変換"""
        total_time = 0.0
        current_tick = 0

        for i in range(len(tempo_map)):
            start_tick, bpm = tempo_map[i]

            if i + 1 < len(tempo_map):
                next_tick = tempo_map[i + 1][0]
                end_tick = min(target_tick, next_tick)
            else:
                end_tick = target_tick

            if end_tick <= start_tick:
                continue

            duration_tick = end_tick - max(start_tick, current_tick)
            if duration_tick > 0:
                sec_per_tick = 60.0 / (bpm * resolution)
                total_time += duration_tick * sec_per_tick
                current_tick = end_tick

            if current_tick >= target_tick:
                break

        return total_time
