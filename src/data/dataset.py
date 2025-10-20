from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _default_file_filter(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".flac"}


@dataclass
class AudioDatasetConfig:
    root: Path
    segment_seconds: float = 5.12
    sample_rate: int = 48_000
    file_filter: Callable[[Path], bool] = _default_file_filter
    random_crop: bool = True
    normalize: bool = True


class AudioSegmentDataset(Dataset):
    def __init__(self, cfg: AudioDatasetConfig) -> None:
        self.cfg = cfg
        self.files: List[Path] = sorted([p for p in cfg.root.rglob("*") if cfg.file_filter(p)])
        if not self.files:
            raise RuntimeError(f"No audio files found under {cfg.root}")
        self.segment_length = int(cfg.segment_seconds * cfg.sample_rate)

    def __len__(self) -> int:
        return len(self.files)

    def _load_wave(self, path: Path) -> np.ndarray:
        wave, sr = sf.read(path)
        if wave.ndim > 1:
            wave = np.mean(wave, axis=1)
        if sr != self.cfg.sample_rate:
            raise ValueError(f"Unexpected sample rate {sr} for {path}, expected {self.cfg.sample_rate}")
        return wave.astype(np.float32)

    def _sample_segment(self, wave: np.ndarray) -> np.ndarray:
        if len(wave) >= self.segment_length:
            max_start = len(wave) - self.segment_length
            start = random.randint(0, max_start) if self.cfg.random_crop else 0
            segment = wave[start : start + self.segment_length]
        else:
            pad = self.segment_length - len(wave)
            segment = np.pad(wave, (0, pad), mode="constant")
        return segment

    def __getitem__(self, idx: int) -> Tensor:
        path = self.files[idx]
        wave = self._load_wave(path)
        segment = self._sample_segment(wave)
        if self.cfg.normalize:
            max_val = np.max(np.abs(segment))
            if max_val > 0:
                segment = segment / max(max_val, 1e-4)
        tensor = torch.from_numpy(segment).unsqueeze(0)
        return tensor
