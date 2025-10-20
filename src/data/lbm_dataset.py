from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import AudioDatasetConfig, AudioSegmentDataset
from src.preprocess.bandwidth import BandwidthConfig, apply_lowpass


@dataclass
class LBMAny48Config:
    data: AudioDatasetConfig
    filter: BandwidthConfig = field(default_factory=BandwidthConfig)
    prior_min_hz: float = 1_000.0
    target_freqs_hz: Sequence[float] = (12_000.0, 16_000.0, 20_000.0, 24_000.0)
    normalize_lr: bool = True


class LBMDataset(Dataset):
    """
    Return HR/LR waveform pairs with frequency annotations for LBM training.
    """

    def __init__(self, cfg: LBMAny48Config) -> None:
        self.cfg = cfg
        self.base = AudioSegmentDataset(cfg.data)
        self.sample_rate = cfg.data.sample_rate

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        hr_tensor = self.base[idx]  # shape [1, L]
        hr_np = hr_tensor.squeeze(0).numpy()

        f_target = random.choice(self.cfg.target_freqs_hz)
        f_prior = random.uniform(self.cfg.prior_min_hz, f_target)

        lr_np = apply_lowpass(hr_np, self.sample_rate, cutoff_hz=f_prior, cfg=self.cfg.filter)
        if self.cfg.normalize_lr:
            max_val = np.max(np.abs(lr_np))
            if max_val > 0:
                lr_np = lr_np / max(max_val, 1e-4)

        lr_tensor = torch.from_numpy(lr_np).unsqueeze(0)

        return {
            "wave_hr": hr_tensor,
            "wave_lr": lr_tensor,
            "f_target": torch.tensor(f_target, dtype=torch.float32),
            "f_prior": torch.tensor(f_prior, dtype=torch.float32),
        }
