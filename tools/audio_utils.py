#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio utilities for preprocessing:
- robust wav read/write (soundfile preferred)
- high-quality resampling (scipy.signal.resample_poly)
- low-pass filtering (Chebyshev I, Butterworth, Bessel, Elliptic)
"""
from __future__ import annotations

import os
import math
from typing import Tuple, Optional, Literal, Dict

import numpy as np
import soundfile as sf
from scipy import signal

FilterType = Literal["cheby1", "butter", "bessel", "elliptic"]


def read_audio(path: str, target_sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Read audio as float32 in [-1, 1]; optionally resample and mono-ize."""
    y, sr = sf.read(path, always_2d=False)
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)
    # Ensure mono
    if y.ndim == 2:
        if mono:
            y = y.mean(axis=1)
        else:
            # pick first channel
            y = y[:, 0]
    # Resample if needed
    if target_sr is not None and target_sr > 0 and sr != target_sr:
        y = resample_poly(y, sr, target_sr)
        sr = target_sr
    # Remove DC
    y = y - np.mean(y)
    # Peak protection (optional light clip)
    peak = np.max(np.abs(y) + 1e-8)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32, copy=False), sr


def write_audio(path: str, y: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")


def resample_poly(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling with reasonable default."""
    # Compute up/down factors
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return signal.resample_poly(y, up, down).astype(np.float32, copy=False)


def design_lowpass(sr: int,
                   cutoff_hz: float,
                   filter_type: FilterType = "cheby1",
                   order: int = 8,
                   rp_db: float = 0.5,
                   rs_db: float = 60.0) -> np.ndarray:
    """Design low-pass filter and return SOS coefficients."""
    nyq = 0.5 * sr
    wc = min(max(cutoff_hz / nyq, 1e-6), 0.999999)  # normalized
    if filter_type == "cheby1":
        z, p, k = signal.cheby1(order, rp_db, wc, btype="low", analog=False, output="zpk")
    elif filter_type == "elliptic":
        z, p, k = signal.ellip(order, rp_db, rs_db, wc, btype="low", output="zpk")
    elif filter_type == "butter":
        z, p, k = signal.butter(order, wc, btype="low", output="zpk")
    elif filter_type == "bessel":
        # Note: digital Bessel via bilinear transform; low roll-off as in paper table.
        z, p, k = signal.bessel(order, wc, btype="low", norm="phase", output="zpk")
    else:
        raise ValueError(f"Unsupported filter_type={filter_type}")
    sos = signal.zpk2sos(z, p, k)
    return sos


def lowpass(y: np.ndarray, sr: int, cutoff_hz: float,
            filter_type: FilterType = "cheby1",
            order: int = 8,
            rp_db: float = 0.5,
            rs_db: float = 60.0) -> np.ndarray:
    """Zero-phase low-pass filter via SOS filtfilt."""
    sos = design_lowpass(sr, cutoff_hz, filter_type=filter_type, order=order, rp_db=rp_db, rs_db=rs_db)
    return signal.sosfiltfilt(sos, y).astype(np.float32, copy=False)

