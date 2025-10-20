from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """Conv → Norm → Act block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = _make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class DeconvBlock(nn.Module):
    """Transposed Conv → Norm → Act block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding=output_padding,
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = _make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.deconv(x)))


@dataclass
class AudioVAEConfig:
    input_channels: int = 1
    latent_channels: int = 64
    downsample_ratio: int = 512
    latent_scale: float = 0.25
    activation: str = "silu"

    def encoder_channels(self) -> Tuple[int, ...]:
        return (32, 64, 128, 256, 256, 512, 512, 512, 512)


class AudioVAE(nn.Module):
    """
    1D VAE with overall downsampling ratio 512 and latent frame rate ≈100 Hz for 48 kHz audio.
    """

    def __init__(
        self,
        config: Optional[AudioVAEConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or AudioVAEConfig()
        enc_channels = self.config.encoder_channels()
        if len(enc_channels) != 9:
            raise ValueError("Expect 9 encoder stages for overall stride 512.")

        layers = []
        in_ch = self.config.input_channels
        strides = []
        for ch in enc_channels:
            layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    activation=self.config.activation,
                )
            )
            in_ch = ch
            strides.append(2)

        total_stride = 1
        for s in strides:
            total_stride *= s
        if total_stride != self.config.downsample_ratio:
            raise ValueError(
                f"Downsample ratio mismatch: expected {self.config.downsample_ratio}, got {total_stride}"
            )

        self.encoder = nn.Sequential(*layers)
        latent_in = enc_channels[-1]
        self.to_mu = nn.Conv1d(latent_in, self.config.latent_channels, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv1d(latent_in, self.config.latent_channels, kernel_size=3, padding=1)

        dec_channels = list(reversed(enc_channels))
        dec_layers = []
        in_ch = self.config.latent_channels
        for ch in dec_channels:
            dec_layers.append(
                DeconvBlock(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    activation=self.config.activation,
                )
            )
            in_ch = ch

        self.decoder_blocks = nn.Sequential(*dec_layers)
        self.to_wave = nn.Conv1d(in_ch, self.config.input_channels, kernel_size=3, padding=1)

        self.latent_scale = self.config.latent_scale

    def encode(self, wave: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(wave)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        z = self._reparameterize(mu, logvar)
        z_scaled = z * self.latent_scale
        return mu, logvar, z_scaled

    def decode(self, latent: Tensor) -> Tensor:
        latent = latent / self.latent_scale
        h = latent
        h = self.decoder_blocks(h)
        return self.to_wave(h)

    def forward(self, wave: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar, z_scaled = self.encode(wave)
        recon = self.decode(z_scaled)
        return recon, mu, logvar, z_scaled

    @staticmethod
    def _reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        if torch.jit.is_scripting():
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
