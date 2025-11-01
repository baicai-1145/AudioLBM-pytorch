#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F


_WINDOW_CACHE: Dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_FREQ_WEIGHT_CACHE: Dict[Tuple[torch.device, torch.dtype, int, int, float, float], torch.Tensor] = {}


def _get_hann_window(device: torch.device, dtype: torch.dtype, win_length: int) -> torch.Tensor:
    key = (device, dtype, win_length)
    w = _WINDOW_CACHE.get(key)
    if w is None:
        w = torch.hann_window(win_length, device=device, dtype=dtype)
        _WINDOW_CACHE[key] = w
    return w


def stft_mag(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    """
    x: [B, T]
    returns magnitude [B, F, TT]
    """
    window = _get_hann_window(x.device, x.dtype, win_length)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=window, center=True, return_complex=True)
    mag = torch.abs(X)
    return mag


def mrstft_loss(x: torch.Tensor,
                x_hat: torch.Tensor,
                configs: List[Tuple[int, int, int]] | None = None,
                *,
                sr: int = 48000,
                hf_freq: int | None = None,
                hf_boost: float = 0.0,
                hf_power: float = 2.0) -> torch.Tensor:
    """
    Multi-resolution STFT loss (sum over resolutions of spectral convergence and log-mag L1):
      L = sum_i ( ||Si(x)-Si(x_hat)||_F / ||Si(x)||_F + mean(|log(Si(x)) - log(Si(x_hat))|) )
    x, x_hat: [B, T]

    Extras (backward compatible):
    - If hf_boost>0 and hf_freq is set (e.g., 6000), upweight high-frequency bins in the log-mag term:
        weight(f) = 1 + hf_boost * ((max(f-hf_freq,0) / (sr/2 - hf_freq)) ** hf_power)
      This empirically helps recover >6 kHz细节而不显著破坏低频。
    """
    if configs is None:
        configs = [
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        ]
    loss = x.new_zeros(())
    eps = 1e-7
    for n_fft, hop, win in configs:
        X = stft_mag(x, n_fft, hop, win)
        Y = stft_mag(x_hat, n_fft, hop, win)
        # spectral convergence (per-sample Frobenius over [F, T], then mean over batch)
        diff = X - Y                       # [B, F, TT]
        num = torch.sqrt(torch.sum(diff * diff, dim=(1, 2)) + eps)   # [B]
        den = torch.sqrt(torch.sum(X * X, dim=(1, 2)) + eps)         # [B]
        sc = (num / (den + eps)).mean()
        # log magnitude
        log_abs = torch.abs(torch.log(X + eps) - torch.log(Y + eps))  # [B,F,TT]

        if hf_freq is not None and hf_boost > 0.0 and sr > 0:
            Fbins = X.shape[1]
            # Build/cached frequency weights [F] on the right device/dtype
            key = (X.device, X.dtype, int(Fbins), int(hf_freq), float(hf_boost), float(hf_power))
            w = _FREQ_WEIGHT_CACHE.get(key)
            if w is None:
                f = torch.linspace(0.0, float(sr) / 2.0, Fbins, device=X.device, dtype=X.dtype)  # [F]
                f0 = float(max(0, min(hf_freq, sr // 2 - 1)))
                denom = max(1e-6, (float(sr) / 2.0) - f0)
                ramp = torch.clamp((f - f0) / denom, min=0.0, max=1.0)
                w = 1.0 + float(hf_boost) * (ramp ** float(hf_power))  # [F]
                _FREQ_WEIGHT_CACHE[key] = w
            # Apply weight on F dimension; keep per-bin mean
            log_diff = (log_abs * w.view(1, -1, 1)).mean()
        else:
            log_diff = log_abs.mean()
        loss = loss + sc + log_diff
    return loss


def kl_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(N(mu, sigma^2) || N(0, I)) with logvar = log(sigma^2)
    """
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))
