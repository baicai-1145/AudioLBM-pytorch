#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Any-to-48 kHz Dataset (online degradation) for AudioLBM
- Reads HR segments and metadata.jsonl from preprocessing
- Constructs LR on-the-fly via random low-pass filtering (paper Sec. 4.1)
- Emits frequency conditions: f_prior (continuous), f_target (quantized)
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from tools.audio_utils import lowpass, read_audio


@dataclass
class DegradeCfg:
    cutoff_min_hz: float = 1_000.0
    cutoff_max_hz: float = 20_000.0
    filter_types: Tuple[str, ...] = ("cheby1", "butter", "bessel", "elliptic")
    order_min: int = 2
    order_max: int = 10
    rp_db: float = 0.5
    rs_db: float = 60.0


def _quantize_f_target(f_target_hz: float, bins_hz: List[float]) -> float:
    """Quantize f_target to nearest bin."""
    if not bins_hz:
        return f_target_hz
    arr = np.asarray(bins_hz, dtype=np.float32)
    idx = int(np.argmin(np.abs(arr - f_target_hz)))
    return float(arr[idx])


class AnyToAnyDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        target_sr: int = 48_000,
        segment_sec: float = 5.12,
        degrade_cfg: Optional[DegradeCfg] = None,
        ftarget_bins_hz: Optional[List[float]] = None,
        normalize: bool = True,
        seed: int = 1337,
    ):
        super().__init__()
        self.target_sr = int(target_sr)
        self.segment_len = int(round(float(segment_sec) * self.target_sr))
        self.deg = degrade_cfg or DegradeCfg()
        self.ftarget_bins_hz = ftarget_bins_hz or [24_000.0]  # for any→48k, default 24k
        self.normalize = bool(normalize)
        self.rng = random.Random(seed)
        # load metadata
        self.recs: List[Dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    # sanity check: must exist
                    if os.path.exists(r.get("hr_path", "")):
                        self.recs.append(r)
                except Exception:
                    continue
        if not self.recs:
            raise RuntimeError(f"No valid records loaded from {metadata_path}")

    def __len__(self) -> int:
        return len(self.recs)

    def _sample_cutoff(self) -> float:
        c = self.rng.uniform(self.deg.cutoff_min_hz, self.deg.cutoff_max_hz)
        # ensure within Nyquist
        return float(max(10.0, min(c, self.target_sr * 0.5 - 100.0)))

    def _sample_order(self) -> int:
        return int(self.rng.randint(self.deg.order_min, self.deg.order_max))

    def _sample_filt(self) -> str:
        return self.rng.choice(self.deg.filter_types)

    def _read_hr(self, path: str) -> np.ndarray:
        # Preprocessed HR is already at target_sr and duration ~= segment_sec
        y, sr = read_audio(path, target_sr=self.target_sr, mono=True)
        if y.shape[0] != self.segment_len:
            # pad or crop conservatively
            if y.shape[0] < self.segment_len:
                pad = self.segment_len - y.shape[0]
                y = np.pad(y, (0, pad), mode="constant")
            else:
                y = y[: self.segment_len]
        if self.normalize:
            peak = float(np.max(np.abs(y)) + 1e-9)
            if peak > 1.0:
                y = (y / peak).astype(np.float32, copy=False)
        return y.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | float | int | str]:
        rec = self.recs[idx]
        path = rec["hr_path"]
        # HR
        hr = self._read_hr(path)
        # LR via random low-pass
        cutoff = self._sample_cutoff()
        order = self._sample_order()
        ftype = self._sample_filt()
        lr = lowpass(hr, self.target_sr, cutoff_hz=cutoff,
                     filter_type=ftype, order=order,
                     rp_db=self.deg.rp_db, rs_db=self.deg.rs_db)
        # Frequency conditions
        f_prior_hz = cutoff  # training-time LR effective band ≈ cutoff
        f_target_hz = self.target_sr / 2.0
        f_target_q_hz = _quantize_f_target(f_target_hz, self.ftarget_bins_hz)
        # Tensors
        hr_t = torch.from_numpy(hr.copy())  # [T]
        lr_t = torch.from_numpy(lr.copy())  # [T]
        cond = torch.tensor([f_prior_hz, f_target_q_hz], dtype=torch.float32)  # simple 2-token vector
        item = {
            "lr": lr_t,
            "hr": hr_t,
            "sr": int(self.target_sr),
            "f_prior_hz": float(f_prior_hz),
            "f_target_hz": float(f_target_hz),
            "f_target_q_hz": float(f_target_q_hz),
            "cond": cond,  # shape [2]
            "hr_path": path,
        }
        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # All segments should have the same length, so direct stack
    lr = torch.stack([b["lr"] for b in batch], dim=0)  # [B, T]
    hr = torch.stack([b["hr"] for b in batch], dim=0)  # [B, T]
    cond = torch.stack([b["cond"] for b in batch], dim=0)  # [B, 2]
    f_prior = torch.tensor([b["f_prior_hz"] for b in batch], dtype=torch.float32)
    f_target_q = torch.tensor([b["f_target_q_hz"] for b in batch], dtype=torch.float32)
    return {
        "lr": lr,
        "hr": hr,
        "cond": cond,
        "f_prior_hz": f_prior,
        "f_target_q_hz": f_target_q,
        "sr": torch.tensor(batch[0]["sr"], dtype=torch.int32),
        # paths left out to keep DataLoader light; add if needed
    }

