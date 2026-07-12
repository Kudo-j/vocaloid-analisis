"""
dataset_09.py — 09_DYN_DNN 共通データセットクラス
08_DYN_DNN_Estimation/dataset.py を完全コピーし、以下を変更:
  - configのインポートをスクリプト先頭で切り替え可能にした
  - tuned/untunedの両方に対応（untunedが無い場合は空dictを返す）
  - 差分計算（mel_diff等）をDataset内で実施可能
既存の09_DYN_DNN/dataset.pyとの衝突を避けるためファイル名を変更。
"""
import os
import torch
import numpy as np
import random
import shutil
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import lru_cache


def _is_silence_chunk(mask_chunk):
    """無音チャンク判定（08完全再現）"""
    try:
        if mask_chunk.shape[0] > 0:
            if np.sum(mask_chunk) == 0.0:
                return True
    except Exception:
        pass
    return False


@lru_cache(maxsize=16)
def _load_npz_safe(path, input_features_tuple, vsqx_params_tuple=()):
    """NPZ読み込みヘルパー（08完全再現 + dict I/F対応 + VSQX対応）"""
    try:
        with np.load(path) as data:
            features_t = {}
            features_ut = {}
            for feat in input_features_tuple:
                # tuned/untuned の命名規則に対応
                feat_key_t = f"{feat}_tuned" if f"{feat}_tuned" in data else feat
                feat_key_ut = f"{feat}_untuned" if f"{feat}_untuned" in data else None

                if feat_key_t in data:
                    features_t[feat] = data[feat_key_t]
                if feat_key_ut is not None and feat_key_ut in data:
                    features_ut[feat] = data[feat_key_ut]

            # VSQX パラメータの読み込み（複数パラメータをconcatして1つのarrayにする）
            vsqx_data = None
            if vsqx_params_tuple:
                vsqx_parts = []
                for pid in vsqx_params_tuple:
                    key = f"vsqx_{pid}"
                    if key in data:
                        arr = data[key]
                        if arr.ndim == 1:
                            arr = arr[:, np.newaxis]
                        vsqx_parts.append(arr)
                if vsqx_parts:
                    vsqx_data = np.concatenate(vsqx_parts, axis=-1).astype(np.float32)

            dyn = data["dyn"]
            mask = data["voicing_mask"]
            return features_t, features_ut, dyn, mask, vsqx_data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None, None, None, None


def _count_valid_chunks_in_file(filepath, chunk_size, chunk_hop):
    """メモリ爆発対策: maskだけをピンポイントで読み込む（08完全再現）"""
    try:
        with np.load(filepath) as data:
            mask = data["voicing_mask"]
            valid_count = 0
            total_frames = len(mask)
            start = 0
            while start + chunk_size <= total_frames:
                end = start + chunk_size
                if end > total_frames:
                    break
                if not _is_silence_chunk(mask[start:end]):
                    valid_count += 1
                if end == total_frames:
                    break
                start += chunk_hop
            return valid_count
    except Exception:
        return 0


class DYNDataset(Dataset):
    """
    08_DYN_DNN_Estimation/dataset.py の DisentangleDataset を完全コピーし、
    クラス名とconfig参照方式を変更したもの。
    """
    def __init__(self, data_dir, file_list, config, mode="eval"):
        self.data_dir = data_dir
        self.mode = mode
        self.chunk_map = []
        self.cfg = config
        # lru_cache用にtupleに変換
        self.input_features_tuple = tuple(config.INPUT_FEATURES)
        self.vsqx_params_tuple = tuple(getattr(config, "INPUT_VSQX_PARAMS", []))

        if not file_list:
            return

        for name in file_list:
            self._process_file(name)

    def _process_file(self, name_stem):
        npz_path = os.path.join(self.data_dir, f"{name_stem}.npz")
        if not os.path.exists(npz_path):
            return

        # メモリ爆発対策: maskだけをピンポイントで読み込む
        try:
            with np.load(npz_path) as data:
                mask = data["voicing_mask"]
                total_frames = len(mask)

                if self.mode == "train":
                    valid_count = 0
                    start = 0
                    while start + self.cfg.CHUNK_SIZE <= total_frames:
                        end = start + self.cfg.CHUNK_SIZE
                        if not _is_silence_chunk(mask[start:end]):
                            valid_count += 1
                        start += self.cfg.CHUNK_HOP
                    for _ in range(valid_count):
                        self.chunk_map.append((name_stem, -1, total_frames))
                else:
                    # evalモード時、はみ出たチャンクは切り捨てる（08完全再現）
                    start = 0
                    while start < total_frames:
                        end = start + self.cfg.CHUNK_SIZE
                        if end > total_frames:
                            break
                        if not _is_silence_chunk(mask[start:end]):
                            self.chunk_map.append((name_stem, start, end))
                        if end == total_frames:
                            break
                        start += self.cfg.CHUNK_HOP
        except Exception:
            pass

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        name_stem, start_frame, end_frame_info = self.chunk_map[idx]
        npz_path = os.path.join(self.data_dir, f"{name_stem}.npz")

        full_features_t, full_features_ut, full_dyn, full_mask, full_vsqx = _load_npz_safe(
            npz_path, self.input_features_tuple, self.vsqx_params_tuple
        )

        if full_features_t is None or len(full_features_t) == 0:
            raise ValueError(f"Failed to load features from {npz_path}")

        # 長さは最初の特徴量から取得
        first_feat_name = list(full_features_t.keys())[0]
        total_len = len(full_features_t[first_feat_name])

        if self.mode == "train" and start_frame == -1:
            max_start = total_len - self.cfg.CHUNK_SIZE
            if max_start < 0:
                max_start = 0

            for _ in range(5):
                rnd_start = random.randint(0, max_start)
                rnd_end = rnd_start + self.cfg.CHUNK_SIZE

                if not _is_silence_chunk(full_mask[rnd_start:rnd_end]):
                    start_frame = rnd_start
                    end_frame = rnd_end
                    break
            else:
                start_frame = rnd_start
                end_frame = rnd_start + self.cfg.CHUNK_SIZE
        else:
            end_frame = end_frame_info

        # 辞書内のすべての特徴量をスライスしてTensor化
        chunk_features_t = {k: torch.FloatTensor(v[start_frame:end_frame]) for k, v in full_features_t.items()}
        chunk_features_ut = {k: torch.FloatTensor(v[start_frame:end_frame]) for k, v in full_features_ut.items()}

        # VSQX特徴量（存在する場合のみ）
        chunk_vsqx = None
        if full_vsqx is not None:
            chunk_vsqx = torch.FloatTensor(full_vsqx[start_frame:end_frame])

        return (
            chunk_features_t,
            chunk_features_ut,
            torch.FloatTensor(full_dyn[start_frame:end_frame]),
            torch.FloatTensor(full_mask[start_frame:end_frame]),
            chunk_vsqx,
        )


def collate_fn(batch):
    """08_DYN_DNN_Estimation/dataset.py の collate_fn を完全コピー + VSQX対応"""
    features_t_list, features_ut_list, dyns, masks, vsqx_list = zip(*batch)

    first_feat_name = list(features_t_list[0].keys())[0]
    lengths = torch.LongTensor([len(x[first_feat_name]) for x in features_t_list])

    # それぞれの特徴量をパディングして辞書にまとめる
    pad_features_t = {}
    pad_features_ut = {}

    # features_tの全キーを処理
    for feat in features_t_list[0].keys():
        feat_t = [x[feat] for x in features_t_list]
        pad_features_t[feat] = pad_sequence(feat_t, batch_first=True, padding_value=0.0)

    # features_utの全キーを処理（空でない場合のみ）
    if features_ut_list[0]:
        for feat in features_ut_list[0].keys():
            feat_ut = [x[feat] for x in features_ut_list]
            pad_features_ut[feat] = pad_sequence(feat_ut, batch_first=True, padding_value=0.0)

    # VSQX特徴量のパディング（存在する場合のみ）
    pad_vsqx = None
    if vsqx_list[0] is not None:
        pad_vsqx = pad_sequence(list(vsqx_list), batch_first=True, padding_value=0.0)

    pad_dyn = pad_sequence(dyns, batch_first=True, padding_value=0.0)
    pad_mask = pad_sequence(masks, batch_first=True, padding_value=0.0)

    return pad_features_t, pad_features_ut, pad_dyn, pad_mask, lengths, pad_vsqx


def create_datasets(config):
    """
    08_DYN_DNN_Estimation/dataset.py の create_datasets を完全コピーし、
    config引数で設定を受け取るように変更。
    """
    data_dir = config.PREPROCESSED_DIR
    songlist_path = getattr(config, "SONGLIST_PATH", None)
    valid_ratio = 0.10
    test_ratio = 0.10

    all_files = sorted([f.replace(".npz", "") for f in os.listdir(data_dir) if f.endswith(".npz")])

    song_names = []
    if songlist_path and os.path.exists(songlist_path):
        with open(songlist_path, "r", encoding="utf-8") as f:
            raw_names = [line.strip() for line in f if line.strip()]
            song_names = sorted(list(set(raw_names)), key=len, reverse=True)

    if not song_names:
        print("Warning: songlist.txt not found. Splitting randomly by file.")
        random.seed(42)
        random.shuffle(all_files)
        num_test = max(1, int(len(all_files) * test_ratio))
        num_valid = max(1, int(len(all_files) * valid_ratio))

        test_files = all_files[:num_test]
        valid_files = all_files[num_test:num_test+num_valid]
        train_files = all_files[num_test+num_valid:]

        train_ds = DYNDataset(data_dir, train_files, config, mode="train")
        valid_ds = DYNDataset(data_dir, valid_files, config, mode="eval")
        return train_ds, valid_ds

    # ---------------------------------------------------------
    # メモリ爆発回避と爆速化のためのキャッシュ機能（08完全再現）
    # ---------------------------------------------------------
    cache_path = os.path.join(data_dir, "chunk_cache.json")
    cache_data = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            print("Loaded chunk cache (scan skipped!)")
        except Exception:
            cache_data = {}

    print(f"Scanning files for valid chunks (removing silence)...")
    song_groups = {}
    matched_files = set()
    total_valid_chunks = 0
    scanned_cnt = 0
    total_files = len(all_files)

    for song in song_names:
        group_files = []
        group_chunks = 0
        for f in all_files:
            if f in matched_files:
                continue
            if song in f:
                if f in cache_data:
                    c = cache_data[f]
                else:
                    c = _count_valid_chunks_in_file(
                        os.path.join(data_dir, f"{f}.npz"),
                        config.CHUNK_SIZE, config.CHUNK_HOP
                    )
                    cache_data[f] = c

                group_files.append(f)
                group_chunks += c
                matched_files.add(f)
                scanned_cnt += 1
                if scanned_cnt % 50 == 0:
                    print(f"  Scanned {scanned_cnt}/{total_files}...", end="\r")

        if group_files:
            song_groups[song] = {"files": group_files, "chunks": group_chunks}
            total_valid_chunks += group_chunks

    unmatched_files = [f for f in all_files if f not in matched_files]
    if unmatched_files:
        u_chunks = 0
        for f in unmatched_files:
            if f in cache_data:
                c = cache_data[f]
            else:
                c = _count_valid_chunks_in_file(
                    os.path.join(data_dir, f"{f}.npz"),
                    config.CHUNK_SIZE, config.CHUNK_HOP
                )
                cache_data[f] = c
            u_chunks += c
            scanned_cnt += 1
        song_groups["__Unmatched__"] = {"files": unmatched_files, "chunks": u_chunks}
        total_valid_chunks += u_chunks

    print(f"\nTotal Valid Chunks: {total_valid_chunks}")

    # キャッシュ保存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)

    keys = sorted(list(song_groups.keys()))
    random.seed(42)
    random.shuffle(keys)

    target_test = int(total_valid_chunks * test_ratio)
    target_valid = int(total_valid_chunks * valid_ratio)

    train_files, valid_files, test_files = [], [], []
    current_test, current_valid = 0, 0
    split_log = {"Train": [], "Valid": [], "Test": []}

    for song in keys:
        files = song_groups[song]["files"]
        chunks = song_groups[song]["chunks"]

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

    # データ分割ログ保存
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
            normal = sorted([e for e in entries if e[0] != "__Unmatched__"], key=lambda x: x[0])
            unmatched = [e for e in entries if e[0] == "__Unmatched__"]
            for i, (s, fs, c) in enumerate(normal + unmatched, 1):
                f.write(f"{i}. {s} (Chunks: {c})\n")
                for file in sorted(fs):
                    f.write(f"    - {file}\n")
            f.write("\n")

    # テストファイルを別ディレクトリにコピー
    if test_files:
        test_dir = os.path.join(data_dir, "Test")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)
        print(f"\nCopying {len(test_files)} Test files to {test_dir} ...")
        for file_stem in test_files:
            src_path = os.path.join(data_dir, f"{file_stem}.npz")
            dst_path = os.path.join(test_dir, f"{file_stem}.npz")
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

    train_ds = DYNDataset(data_dir, train_files, config, mode="train")
    valid_ds = DYNDataset(data_dir, valid_files, config, mode="eval")

    print(f"\nDataset Split Result:")
    print(f"  Train : {len(train_files)} files ({len(train_ds)} chunks)")
    print(f"  Valid : {len(valid_files)} files ({len(valid_ds)} chunks)")
    print(f"  Test  : {len(test_files)} files (Isolated in {os.path.join(data_dir, 'Test')})")

    return train_ds, valid_ds
