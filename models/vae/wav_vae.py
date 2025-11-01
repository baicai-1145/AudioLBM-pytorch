#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolutional VAE for waveform compression (48 kHz):
- Analysis via Conv1d with stride=hop to produce latent frames ~ 100 Hz
- Synthesis via ConvTranspose1d to reconstruct waveform
- Latent: channels=C (default 64); per-frame Gaussian posterior (mu, logvar)

Defaults align with paper's Appendix B (frame_rate≈100 Hz, channels=64, KL≈0, latent scale s later in LBM):
- hop = 480 samples @48kHz -> 100 Hz frame rate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    channels: int = 64      # latent channels C
    hop: int = 512          # 48k / 512 ≈ 93.75 Hz (r_x=512)
    win: int = 1024         # ~2*hop
    hidden: int = 256       # feature width in analysis/synthesis path
    kl_weight: float = 0.0  # as per paper, small or zero


class AnalysisConv(nn.Module):
    def __init__(self, win: int, hop: int, in_ch: int = 1, out_ch: int = 256):
        super().__init__()
        # reflection pad improves border reconstruction
        pad = max(0, win - hop)
        self.pad_amt = pad // 2
        self.pad = nn.ReflectionPad1d(self.pad_amt)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=win, stride=hop, padding=0)
        self.norm = nn.GroupNorm(16, out_ch)
        self.act = nn.GELU()
        self.res1 = ResBlock1d(out_ch)
        self.res2 = ResBlock1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        x = self.pad(x)
        h = self.act(self.norm(self.conv(x)))
        h = self.res2(self.res1(h))
        return h  # [B, H, L]


class SynthesisConvT(nn.Module):
    def __init__(self, win: int, hop: int, in_ch: int = 256, out_ch: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=win, stride=hop, padding=0)
        self.res1 = ResBlock1d(in_ch)
        self.res2 = ResBlock1d(in_ch)
        self.pad_amt = max(0, win - hop) // 2  # crop amount to reverse reflection pad

    def forward(self, z: torch.Tensor, out_len: int) -> torch.Tensor:
        # z: [B, H, L]
        z = self.res2(self.res1(z))
        y = self.deconv(z)  # [B,1,T']
        # Crop/pad to out_len
        T = y.shape[-1]
        # compensate reflection pad (remove extra samples on both ends)
        if self.pad_amt > 0 and T > 2 * self.pad_amt:
            y = y[..., self.pad_amt: T - self.pad_amt]
            T = y.shape[-1]
        if T > out_len:
            y = y[..., :out_len]
        elif T < out_len:
            pad = out_len - T
            y = F.pad(y, (0, pad))
        return y


class ResBlock1d(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        self.gn1 = nn.GroupNorm(16, ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        self.gn2 = nn.GroupNorm(16, ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + x)


class WaveVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.channels
        H = cfg.hidden
        # Analysis stack
        self.ana = AnalysisConv(cfg.win, cfg.hop, in_ch=1, out_ch=H)
        self.proj_mu = nn.Conv1d(H, C, kernel_size=1)
        self.proj_lv = nn.Conv1d(H, C, kernel_size=1)
        # Synthesis stack
        self.syn_in = nn.Conv1d(C, H, kernel_size=1)
        self.syn = SynthesisConvT(cfg.win, cfg.hop, in_ch=H, out_ch=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, T]
        h = self.ana(x)
        mu = self.proj_mu(h)
        logvar = self.proj_lv(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor, out_len: int) -> torch.Tensor:
        h = self.syn_in(z)
        y = self.syn(h, out_len)
        return y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, T]
        returns x_hat, mu, logvar
        """
        B, _, T = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, out_len=T)
        return x_hat, mu, logvar
