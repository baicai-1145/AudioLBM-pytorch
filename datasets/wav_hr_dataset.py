#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.audio_utils import read_audio


class WavHRDataset(Dataset):
    """
    Loads HR segments from metadata.jsonl created by preprocess_vctk.py
    """
    def __init__(self, metadata_path: str, target_sr: int = 48000, segment_sec: float = 5.12, normalize: bool = True):
        super().__init__()
        self.target_sr = int(target_sr)
        self.segment_len = int(round(float(segment_sec) * self.target_sr))
        self.normalize = bool(normalize)
        self.recs: List[Dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if os.path.exists(r.get("hr_path", "")):
                        self.recs.append(r)
                except Exception:
                    continue
        if not self.recs:
            raise RuntimeError(f"No valid records loaded from {metadata_path}")

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        r = self.recs[idx]
        y, sr = read_audio(r["hr_path"], target_sr=self.target_sr, mono=True)
        if y.shape[0] != self.segment_len:
            if y.shape[0] < self.segment_len:
                pad = self.segment_len - y.shape[0]
                y = np.pad(y, (0, pad), mode="constant")
            else:
                y = y[: self.segment_len]
        if self.normalize:
            peak = float(np.max(np.abs(y)) + 1e-9)
            if peak > 1.0:
                y = (y / peak).astype(np.float32, copy=False)
        y_t = torch.from_numpy(y.astype(np.float32, copy=False)).unsqueeze(0)  # [1, T]
        return {"hr": y_t, "sr": self.target_sr}

