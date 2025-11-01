#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curvature-aware effective bandwidth estimation (Appendix A, Eq. 6–7):
- FFT magnitude -> Savitzky–Golay smoothing -> local downsampling
- Second-order curvature on log-magnitude spectrum
- Find smallest index i* where max curvature in a window < eps_sr and magnitude < tau
Returns f_eff in Hz.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict
from scipy.signal import savgol_filter


def _next_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())


def compute_f_eff(
    y: np.ndarray,
    sr: int,
    fft_size: int | None = None,
    sg_window: int = 101,
    sg_poly: int = 3,
    downsample: int = 4,
    eps_sr: float = 1e-3,
    energy_quantile: float = 0.97,
    window_k: int = 5,
) -> Tuple[float, Dict]:
    """
    Estimate effective bandwidth f_eff of waveform y (Hz).
    Returns:
      f_eff_hz, debug_info
    """
    x = y.astype(np.float32, copy=False)
    n = x.shape[0]
    n_fft = fft_size or _next_pow2(n)
    # rfft magnitude
    spec = np.fft.rfft(x, n=n_fft)
    mag = np.abs(spec).astype(np.float32, copy=False)
    # Avoid log(0)
    mag = np.maximum(mag, 1e-12)
    # Smooth via Savitzky–Golay on log magnitude
    log_mag = np.log(mag + 1e-12)
    w = sg_window if sg_window % 2 == 1 else sg_window + 1
    if w >= log_mag.size:
        w = max(5, (log_mag.size // 2) * 2 + 1)
    try:
        log_mag_s = savgol_filter(log_mag, window_length=w, polyorder=min(sg_poly, w - 1))
    except Exception:
        log_mag_s = log_mag
    # Local downsample in frequency bins to reduce noise
    if downsample > 1:
        ds = log_mag_s[::downsample]
    else:
        ds = log_mag_s
    # Second derivative (curvature proxy) over index
    g1 = np.gradient(ds)
    g2 = np.gradient(g1)
    g2_abs = np.abs(g2)
    # Energy threshold tau on smoothed magnitude (use original mag to set tau is also OK)
    mag_s = np.exp(ds)
    tau = np.quantile(mag_s, energy_quantile)
    # Sliding window to find smallest i* such that:
    # max_{j in [i, i+k]} |curvature| < eps_sr and magnitude < tau
    k = max(1, int(window_k))
    N = len(ds)
    i_star = N - 1
    for i in range(0, N - k):
        if np.max(g2_abs[i:i + k + 1]) < eps_sr and mag_s[i] < tau:
            i_star = i
            break
    # Map index to frequency: i* -> FFT bin index
    # rfft bins: 0..n_fft/2 -> Nyquist = sr/2
    # After downsample, bin width is: (sr/2) / (n_fft/2) * downsample = sr / n_fft * downsample
    bin_hz = (sr / n_fft) * downsample
    f_eff_hz = float(i_star * bin_hz)
    debug = dict(
        n_fft=int(n_fft),
        sg_window=int(w),
        sg_poly=int(sg_poly),
        downsample=int(downsample),
        eps_sr=float(eps_sr),
        energy_quantile=float(energy_quantile),
        window_k=int(window_k),
        tau=float(tau),
        i_star=int(i_star),
        bin_hz=float(bin_hz),
    )
    # Clamp to [0, nyquist]
    f_eff_hz = max(0.0, min(f_eff_hz, sr * 0.5 - 1.0))
    return f_eff_hz, debug

