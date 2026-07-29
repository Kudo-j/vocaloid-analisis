import numpy as np
import librosa
import xml.etree.ElementTree as ET
from vsqx_parser import VSQXParser

class VSQXReader(VSQXParser):
    def __init__(self, sr=44100, hop_length=256, use_part_offset=False):
        super().__init__(sr, hop_length)
        self.use_part_offset = use_part_offset

    def read(self, vsqx_path):
        sparse_data = self.read_sparse(vsqx_path, use_part_offset=self.use_part_offset)
        if sparse_data is None:
            return None
        
        tempo_map = sparse_data["tempo_map"]
        resolution = sparse_data["resolution"]
        primary_tempo = tempo_map[0][1] if tempo_map else 120.0
        
        dynamic_params = {}
        for pid, points in sparse_data["dynamic_params"].items():
            dynamic_params[pid] = [
                (self._tick_to_second(pt["tick"], tempo_map, resolution), pt["value"])
                for pt in points
            ]
            
        notes = []
        for note in sparse_data["notes"]:
            notes.append({
                "start_sec": self._tick_to_second(note["abs_start_tick"], tempo_map, resolution),
                "end_sec": self._tick_to_second(note["abs_end_tick"], tempo_map, resolution),
                "note_num": note["note_num"],
                "lyric": note["lyric"],
                "phoneme": note["phoneme"],
            })
            
        return {
            "dynamic_params": dynamic_params,
            "notes": notes,
            "static_params": sparse_data["static_params"],
            "voice_info": sparse_data["voice_info"],
            "tempo": primary_tempo,
            "resolution": resolution,
        }

    # =====================================================================

    def interpolate_param_to_frames(self, tick_points, tempo_map, resolution,
                                     total_frames, default_value=64):
        """
        可視化ツール向けの公開補間API。
        内部で _interpolate_to_frames を呼ぶラッパー。

        Args:
            tick_points: [{"tick": int, "value": int}, ...] 形式の点データリスト
            tempo_map: read_sparse() の戻り値に含まれる tempo_map
            resolution: read_sparse() の戻り値に含まれる resolution
            total_frames: 出力フレーム数
            default_value: 制御点がない場合のデフォルト値

        Returns:
            np.ndarray: shape (total_frames,)
        """
        return self._interpolate_to_frames(
            tick_points, tempo_map, resolution, total_frames, default_value
        )

    def find_changed_params(self, vsqx_data_a, vsqx_data_b):
        """
        2つの read_sparse() 結果を比較し、CC制御点に差分がある動的パラメータを検出する。

        Args:
            vsqx_data_a: read_sparse() の戻り値
            vsqx_data_b: read_sparse() の戻り値

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

            if len(points_a) != len(points_b):
                changed.append(pid)
            elif points_a != points_b:
                changed.append(pid)

        return sorted(changed)

    def find_static_param_diffs(self, vsqx_data_a, vsqx_data_b):
        """
        2つの read_sparse() 結果の静的パラメータの差分を検出する。

        Args:
            vsqx_data_a: read_sparse() の戻り値
            vsqx_data_b: read_sparse() の戻り値

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
