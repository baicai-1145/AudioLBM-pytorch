#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class DiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int):
        super().__init__()
        self.conv = spectral_norm(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class SubDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        ch = base
        layers = [
            DiscBlock(in_ch, ch, k=15, s=1, p=7),
            DiscBlock(ch, ch, k=41, s=4, p=20),
            DiscBlock(ch, ch * 2, k=41, s=4, p=20),
            DiscBlock(ch * 2, ch * 4, k=41, s=4, p=20),
            DiscBlock(ch * 4, ch * 8, k=41, s=4, p=20),
            DiscBlock(ch * 8, ch * 16, k=5, s=1, p=2),
        ]
        self.feat = nn.ModuleList(layers)
        self.post = spectral_norm(nn.Conv1d(ch * 16, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats = []
        h = x
        for f in self.feat:
            h = f(h)
            feats.append(h)
        out = self.post(h)
        return out, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_discriminators: int = 3, base: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList([SubDiscriminator(1, base) for _ in range(n_discriminators)])
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)  # downsample between scales

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        outs = []
        h = x
        for d in self.discriminators:
            o, f = d(h)
            outs.append((o, f))
            # downsample for next scale
            h = self.pool(h)
        return outs
