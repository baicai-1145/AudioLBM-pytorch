#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tools.audio_utils import read_audio


def _read_and_fix(path: str, target_sr: int, segment_len: int, normalize: bool) -> np.ndarray:
    y, sr = read_audio(path, target_sr=target_sr, mono=True)
    if y.shape[0] != segment_len:
        if y.shape[0] < segment_len:
            pad = segment_len - y.shape[0]
            y = np.pad(y, (0, pad), mode="constant")
        else:
            y = y[:segment_len]
    if normalize:
        peak = float(np.max(np.abs(y)) + 1e-9)
        if peak > 1.0:
            y = (y / peak).astype(np.float32, copy=False)
    return y.astype(np.float32, copy=False)


class WavHRInMemoryDataset(Dataset):
    """
    Preload all HR segments into RAM for fastest iteration.
    Suitable when N * T * 4 bytes fits comfortably into system memory.
    """
    def __init__(self, metadata_path: str, target_sr: int = 48000, segment_sec: float = 5.12,
                 normalize: bool = True, preload_workers: int = 8):
        super().__init__()
        self.target_sr = int(target_sr)
        self.segment_len = int(round(float(segment_sec) * self.target_sr))
        # Gather paths
        paths: List[str] = []
        base_dir = os.path.dirname(os.path.abspath(metadata_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    p = r.get("hr_path", "")
                    if p and os.path.exists(p):
                        paths.append(p)
                        continue
                    # Fallback: reconstruct path relative to metadata directory (handles WSL→Windows迁移)
                    spk = r.get("speaker", "")
                    utt = r.get("utt", "")
                    seg = r.get("seg_idx", None)
                    if spk and utt and seg is not None:
                        guess = os.path.join(base_dir, spk, f"{utt}_seg{int(seg):04d}.wav")
                        if os.path.exists(guess):
                            paths.append(guess)
                            continue
                except Exception:
                    continue
        if not paths:
            raise RuntimeError(f"No valid records loaded from {metadata_path}")
        self.paths = paths
        # Allocate and preload
        N = len(paths)
        T = self.segment_len
        self.hr = np.empty((N, T), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=max(1, preload_workers)) as ex:
            futs = {ex.submit(_read_and_fix, p, self.target_sr, self.segment_len, normalize): i
                    for i, p in enumerate(paths)}
            for fut in tqdm(as_completed(futs), total=N, desc="preload_hr"):
                i = futs[fut]
                try:
                    self.hr[i] = fut.result()
                except Exception:
                    # Fill with silence on failure
                    self.hr[i].fill(0.0)

    def __len__(self) -> int:
        return self.hr.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        y = torch.from_numpy(self.hr[idx]).unsqueeze(0)  # [1,T]
        return {"hr": y, "sr": self.target_sr}
