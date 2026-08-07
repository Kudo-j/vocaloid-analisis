"""
dataset.py
==========
学習用データセットの読み込みとバッチ処理を行う共通ユーティリティ。

03_DNN_System/dataset.py を移植し、新しい辞書型NPZ形式
(x_audio, x_branch名, y_branch名) に対応したもの。
各Stepの学習スクリプトから呼び出して使用する。
"""

import os
import json
import torch
import numpy as np
import random
import shutil
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 定数
CHUNK_SIZE = 600
CHUNK_HOP = 600


# 発声マスクを使って無音判定する関数
def _is_silence_chunk(mask_chunk):
    """
    mask_chunk: (Seq, 1) の発声マスク
    1.0(有音) と 0.0(無音) で構成されている。合計が0なら完全な無音チャンク。
    """
    try:
        if mask_chunk.shape[0] > 0:
            # 区間内の合計が0なら無音と判定
            if np.sum(mask_chunk) == 0.0:
                return True
    except Exception:
        pass
    return False


def _load_npz_safe(path):
    """
    NPZファイルを安全に読み込み、辞書型のブランチデータを返す。

    新しい辞書型NPZ形式 (x_audio, x_branch名, y_branch名) に対応。
    従来形式 (x, y) との後方互換性も維持する。

    Returns:
        x_dict: 入力ブランチの辞書 {"audio": array, "branch名": array, ...}
        y_dict: ターゲットブランチの辞書 {"dynamic": array, ...}
        voicing_mask: 発声マスク配列
    """
    try:
        with np.load(path) as data:
            keys = list(data.keys())

            # --- 新しい辞書型NPZ形式 ---
            x_keys = [k for k in keys if k.startswith("x_")]
            y_keys = [k for k in keys if k.startswith("y_")]

            if x_keys:
                # 新形式: "x_audio", "x_branch名" などのキーから辞書を構築
                x_dict = {}
                for k in x_keys:
                    branch_name = k[2:]  # "x_" を除去
                    x_dict[branch_name] = data[k]

                y_dict = {}
                for k in y_keys:
                    branch_name = k[2:]  # "y_" を除去
                    y_dict[branch_name] = data[k]
            else:
                # --- 従来形式との後方互換 ---
                x_dict = {"all": data["x"]}
                y_dict = {"all": data["y"]}

            # voicing_maskの読み込み
            if "voicing_mask" in data:
                voicing_mask = data["voicing_mask"]
            else:
                # 最初のxブランチの長さからダミーマスクを作成
                first_x = list(x_dict.values())[0]
                voicing_mask = np.ones((len(first_x), 1), dtype=np.float32)

            return x_dict, y_dict, voicing_mask
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None, None


class VocalDataset(Dataset):
    """
    NPZファイルからチャンク単位でデータを読み込むDatasetクラス。
    辞書型ブランチ形式に対応。
    """

    def __init__(self, data_dir, file_list, feature_info, mode="eval"):
        self.data_dir = data_dir
        self.feature_info = feature_info
        self.mode = mode  # 'train' or 'eval'
        self.chunk_map = []  # (name_stem, start_frame, end_frame)

        if not file_list:
            return
        # ファイルをスキャンしてリストを作成
        for name in file_list:
            self._process_file(name)

    def _process_file(self, name_stem):
        npz_path = os.path.join(self.data_dir, f"{name_stem}.npz")
        if not os.path.exists(npz_path):
            return
        x_dict, y_dict, voicing_mask = _load_npz_safe(npz_path)
        if x_dict is None:
            return

        # 最初のxブランチからフレーム数を取得
        first_x = list(x_dict.values())[0]
        total_frames = len(first_x)

        # --- Trainモード ---
        if self.mode == "train":
            valid_count = 0
            start = 0
            while start + CHUNK_SIZE <= total_frames:
                end = start + CHUNK_SIZE
                if not _is_silence_chunk(voicing_mask[start:end]):
                    valid_count += 1
                start += CHUNK_HOP

            for _ in range(valid_count):
                self.chunk_map.append((name_stem, -1, total_frames))

        # --- Evalモード ---
        elif self.mode == "eval" or self.mode == "eval_full":
            start = 0
            while start < total_frames:
                end = start + CHUNK_SIZE
                if end > total_frames:
                    break
                if self.mode == "eval_full" or not _is_silence_chunk(voicing_mask[start:end]):
                    self.chunk_map.append((name_stem, start, end))
                if end == total_frames:
                    break
                start += CHUNK_HOP

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        name_stem, start_frame, end_frame_info = self.chunk_map[idx]
        npz_path = os.path.join(self.data_dir, f"{name_stem}.npz")

        x_dict, y_dict, full_mask = _load_npz_safe(npz_path)
        first_x = list(x_dict.values())[0]
        total_len = len(first_x)

        # --- Trainモード: ランダム切り出し (リトライ付き) ---
        if self.mode == "train" and start_frame == -1:
            max_start = total_len - CHUNK_SIZE
            if max_start < 0:
                max_start = 0

            # 無音じゃない場所を引くまで最大5回トライ
            for _ in range(5):
                rnd_start = random.randint(0, max_start)
                rnd_end = rnd_start + CHUNK_SIZE

                if not _is_silence_chunk(full_mask[rnd_start:rnd_end]):
                    start_frame = rnd_start
                    end_frame = rnd_end
                    break
            else:
                # 5回失敗したら最後の乱数値で妥協する（エラー防止）
                start_frame = rnd_start
                end_frame = rnd_start + CHUNK_SIZE

        # --- Evalモード: 固定位置 ---
        else:
            end_frame = end_frame_info

        # 各ブランチをスライスしてテンソル化
        x_tensors = {}
        for k, v in x_dict.items():
            x_tensors[k] = torch.FloatTensor(v[start_frame:end_frame])

        y_tensors = {}
        for k, v in y_dict.items():
            y_tensors[k] = torch.FloatTensor(v[start_frame:end_frame])

        mask_tensor = torch.FloatTensor(full_mask[start_frame:end_frame])

        return x_tensors, y_tensors, mask_tensor


def collate_fn(batch):
    """
    辞書型ブランチに対応したcollate関数。
    各ブランチごとにパディングを行う。
    """
    x_dicts, y_dicts, masks = zip(*batch)
    lengths = torch.LongTensor([len(list(x.values())[0]) for x in x_dicts])

    # 各ブランチごとにパディング
    x_padded = {}
    for key in x_dicts[0].keys():
        x_padded[key] = pad_sequence(
            [x[key] for x in x_dicts], batch_first=True, padding_value=0.0
        )

    y_padded = {}
    for key in y_dicts[0].keys():
        y_padded[key] = pad_sequence(
            [y[key] for y in y_dicts], batch_first=True, padding_value=0.0
        )

    mask_padded = pad_sequence(masks, batch_first=True, padding_value=0.0)

    return x_padded, y_padded, mask_padded, lengths


def _count_valid_chunks_in_file(filepath):
    """
    ファイルを開き、無音除去後の有効チャンク数をカウントする。
    _load_npz_safe を使用してデータを取得する。
    """
    valid_count = 0

    x_dict, _, voicing_mask = _load_npz_safe(filepath)

    if x_dict is None:
        return 0

    first_x = list(x_dict.values())[0]
    total_frames = len(first_x)

    start = 0
    while start + CHUNK_SIZE <= total_frames:
        end = start + CHUNK_SIZE
        if end > total_frames:
            break

        if not _is_silence_chunk(voicing_mask[start:end]):
            valid_count += 1

        if end == total_frames:
            break
        start += CHUNK_HOP

    return valid_count


def create_datasets(data_dir, songlist_path, valid_ratio=0.10, test_ratio=0.10, split_mode="random"):
    """
    1. 特徴量ロード
    2. 全ファイルスキャン & 有効チャンク数カウント (無音除去)
    3. ソング単位でTest/Valid/Trainに分割
    4. Datasetインスタンス生成
    """
    print(f"Preparing datasets from {data_dir}...")

    features_path = os.path.join(data_dir, "features.json")
    if not os.path.exists(features_path):
        raise FileNotFoundError("features.json not found.")

    with open(features_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    all_files = sorted(
        [f.replace(".npz", "") for f in os.listdir(data_dir) if f.endswith(".npz")]
    )

    if split_mode == "random":
        song_names = all_files  # 各ファイルを1つの独立したグループとして扱う
    else:
        if songlist_path and os.path.exists(songlist_path):
            with open(songlist_path, "r", encoding="utf-8") as f:
                raw_names = [line.strip() for line in f if line.strip()]
                song_names = sorted(list(set(raw_names)), key=len, reverse=True)
        else:
            # 曲名リストがない場合は、ファイル名の最初の '_' の前までを自動的に曲名とみなす
            raw_names = [f.split("_")[0] for f in all_files]
            song_names = sorted(list(set(raw_names)), key=len, reverse=True)
            print("Warning: Songlist not found. Grouping files by prefix for splitting.")

    # --- スキャンとカウント ---
    print("Scanning files for valid chunks (removing silence)...")
    song_groups = {}
    matched_files = set()
    total_valid_chunks = 0
    total_files = len(all_files)
    scanned_cnt = 0

    for song in song_names:
        group_files = []
        group_chunks = 0
        for f in all_files:
            if f in matched_files:
                continue
            if song in f:
                f_path = os.path.join(data_dir, f"{f}.npz")
                c = _count_valid_chunks_in_file(f_path)
                group_files.append(f)
                group_chunks += c
                matched_files.add(f)
                scanned_cnt += 1
                if scanned_cnt % 50 == 0:
                    print(f"  Scanned {scanned_cnt}/{total_files}...", end="\r")

        if group_files:
            song_groups[song] = {"files": group_files, "chunks": group_chunks}
            total_valid_chunks += group_chunks

    # Unmatched
    unmatched_files = [f for f in all_files if f not in matched_files]
    if unmatched_files:
        u_chunks = 0
        for f in unmatched_files:
            f_path = os.path.join(data_dir, f"{f}.npz")
            c = _count_valid_chunks_in_file(f_path)
            u_chunks += c
            scanned_cnt += 1
        song_groups["__Unmatched__"] = {"files": unmatched_files, "chunks": u_chunks}
        total_valid_chunks += u_chunks

    print(f"\nTotal Valid Chunks: {total_valid_chunks}")

    # --- 分割ロジック ---
    random.seed(42)
    keys = sorted(list(song_groups.keys()))
    random.shuffle(keys)

    target_test = int(total_valid_chunks * test_ratio)
    target_valid = int(total_valid_chunks * valid_ratio)

    current_test = 0
    current_valid = 0

    train_files, valid_files, test_files = [], [], []
    split_log = {"Train": [], "Valid": [], "Test": []}

    for song in keys:
        data = song_groups[song]
        files = data["files"]
        chunks = data["chunks"]

        if song == "__Unmatched__":
            train_files.extend(files)
            split_log["Train"].append((song, files, chunks))
            continue

        if target_test > 0 and current_test < target_test:
            test_files.extend(files)
            current_test += chunks
            split_log["Test"].append((song, files, chunks))
        elif target_valid > 0 and current_valid < target_valid:
            valid_files.extend(files)
            current_valid += chunks
            split_log["Valid"].append((song, files, chunks))
        else:
            train_files.extend(files)
            split_log["Train"].append((song, files, chunks))

    # --- ログ出力 (名前順ソート + Unmatched最後) ---
    log_path = os.path.join(data_dir, "dataset_split_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Total Valid Chunks: {total_valid_chunks}\n")
        f.write(f"Targets -> Test: {target_test} | Valid: {target_valid}\n\n")
        for cat in ["Test", "Valid", "Train"]:
            entries = split_log[cat]
            cat_songs = len(entries)
            cat_chunks = sum([e[2] for e in entries])
            f.write(f"=== {cat} Set ===\n")
            f.write(f"Summary: {cat_songs} Songs, {cat_chunks} Chunks\n")

            normal = sorted(
                [e for e in entries if e[0] != "__Unmatched__"], key=lambda x: x[0]
            )
            unmatched = [e for e in entries if e[0] == "__Unmatched__"]

            for i, (s, fs, c) in enumerate(normal + unmatched, 1):
                f.write(f"{i}. {s} (Chunks: {c})\n")
                for file in sorted(fs):
                    f.write(f"    - {file}\n")
            f.write("\n")

    print(f"Split log saved to: {log_path}")

    # --- Test データの隔離 (コピー) ---
    if test_files:
        test_dir = os.path.join(data_dir, "Test")
        # 既に Test フォルダが存在する場合は、一旦削除して中身をリセットする
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)
        print(f"Copying {len(test_files)} Test files to {test_dir} ...")
        for file_stem in test_files:
            src_path = os.path.join(data_dir, f"{file_stem}.npz")
            dst_path = os.path.join(test_dir, f"{file_stem}.npz")
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

    # --- Datasetインスタンス生成 ---
    train_ds = VocalDataset(data_dir, train_files, feature_info, mode="train")
    valid_ds = VocalDataset(data_dir, valid_files, feature_info, mode="eval")
    test_ds = VocalDataset(data_dir, test_files, feature_info, mode="eval")

    return train_ds, valid_ds, test_ds, feature_info
