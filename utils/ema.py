#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from typing import Iterable
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: str | None = None):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        if device is not None:
            self.shadow.to(device)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for (name, p), (_, sp) in zip(model.named_parameters(), self.shadow.named_parameters()):
            if p.requires_grad and sp.data.shape == p.data.shape:
                sp.data.mul_(d).add_(p.data, alpha=1.0 - d)

    def state_dict(self):
        return self.shadow.state_dict()

