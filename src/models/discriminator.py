from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass
class DiscriminatorConfig:
    input_channels: int = 1
    base_channels: int = 32
    max_channels: int = 512
    num_layers: int = 6
    num_scales: int = 3
    kernel_size: int = 15
    stride: int = 2


class ScaleDiscriminator(nn.Module):
    def __init__(self, cfg: DiscriminatorConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = cfg.input_channels
        ch = cfg.base_channels
        for idx in range(cfg.num_layers):
            out_ch = min(cfg.max_channels, ch * (2 if idx > 0 else 1))
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride if idx < cfg.num_layers - 1 else 1,
                        padding=cfg.kernel_size // 2,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_ch = out_ch
            ch = out_ch
        self.layers = nn.ModuleList(layers)
        self.out = nn.Conv1d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        feats: List[Tensor] = []
        h = x
        for layer in self.layers:
            h = layer(h)
            feats.append(h)
        out = self.out(h)
        feats.append(out)
        return out, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, cfg: DiscriminatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.discriminators = nn.ModuleList([ScaleDiscriminator(cfg) for _ in range(cfg.num_scales)])
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: Tensor) -> List[Tuple[Tensor, List[Tensor]]]:
        outputs = []
        h = x
        for disc in self.discriminators:
            out, feats = disc(h)
            outputs.append((out, feats))
            h = self.downsample(h)
        return outputs


def discriminator_loss(real_outputs: Sequence[Tensor], fake_outputs: Sequence[Tensor]) -> Tensor:
    loss = 0.0
    for real, fake in zip(real_outputs, fake_outputs):
        loss += torch.mean(torch.relu(1.0 - real))
        loss += torch.mean(torch.relu(1.0 + fake))
    return loss / len(real_outputs)


def generator_adversarial_loss(fake_outputs: Sequence[Tensor]) -> Tensor:
    loss = 0.0
    for fake in fake_outputs:
        loss += -torch.mean(fake)
    return loss / len(fake_outputs)


def feature_matching_loss(
    real_feats: Sequence[Sequence[Tensor]],
    fake_feats: Sequence[Sequence[Tensor]],
) -> Tensor:
    loss = 0.0
    count = 0
    for real_scale, fake_scale in zip(real_feats, fake_feats):
        for real_f, fake_f in zip(real_scale, fake_scale):
            loss += torch.mean(torch.abs(real_f - fake_f))
            count += 1
    return loss / max(count, 1)
