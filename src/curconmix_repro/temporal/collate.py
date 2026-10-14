import torch
import numpy as np
from typing import List, Tuple

def pad_collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, str]]):
    """
    입력 batch: [(video, label, video_name), ...]
      - video: [T_i, C]  (np.float32)
      - label: [T_i, K]  (np.float32 or int)
      - video_name: str

    출력:
      - videos:  [B, T_max, C]  (torch.float32)
      - labels:  [B, T_max, K]  (torch.float32)
      - video_names: List[str]  길이 B
      - valid_len: [B]          (torch.long) 각 시퀀스의 실제 길이
    """
    videos_np, labels_np, names = [], [], []
    lens = []
    for v, y, n in batch:
        assert v.ndim == 2, f"video must be [T,C], got {v.shape}"
        assert y.ndim == 2, f"label must be [T,K], got {y.shape}"
        T_v, C = v.shape
        T_y, K = y.shape
        assert T_v == T_y, f"video T ({T_v}) and label T ({T_y}) must match"
        videos_np.append(v)
        labels_np.append(y)
        names.append(n)
        lens.append(T_v)

    B = len(batch)
    T_max = int(max(lens))
    C = int(videos_np[0].shape[1])
    K = int(labels_np[0].shape[1])

    videos = torch.zeros((B, T_max, C), dtype=torch.float32)
    labels = torch.zeros((B, T_max, K), dtype=torch.float32)
    valid_len = torch.tensor(lens, dtype=torch.long)

    for i, (v, y) in enumerate(zip(videos_np, labels_np)):
        t = v.shape[0]
        videos[i, :t, :] = torch.from_numpy(v.astype(np.float32))
        labels[i, :t, :] = torch.from_numpy(y.astype(np.float32))

    return videos, labels, names, valid_len
