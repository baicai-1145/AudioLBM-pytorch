#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn.functional as F


def hinge_d_loss(real_scores: List[torch.Tensor], fake_scores: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for r, f in zip(real_scores, fake_scores):
        loss = loss + torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
    return loss


def hinge_g_loss(fake_scores: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for f in fake_scores:
        loss = loss - torch.mean(f)
    return loss


def feature_matching_loss(real_feats: List[List[torch.Tensor]], fake_feats: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    real_feats/fake_feats: list over discriminators of list over feature maps
    """
    loss = 0.0
    for rf, ff in zip(real_feats, fake_feats):
        for r, f in zip(rf, ff):
            denom = torch.mean(torch.abs(r).detach()) + 1e-7
            loss = loss + torch.mean(torch.abs(r.detach() - f)) / denom
    return loss

