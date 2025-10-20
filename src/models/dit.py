from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def sinusoidal_embedding(x: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1e-4), math.log(1.0), half, device=x.device, dtype=x.dtype)
    )
    args = x[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


@dataclass
class DiTConfig:
    latent_channels: int = 64
    hidden_dim: int = 768
    depth: int = 16
    num_heads: int = 16
    time_embed_dim: int = 512
    freq_embed_dim: int = 128
    dropout: float = 0.0


class FrequencyAwareDiT(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.latent_channels * 2
        self.input_proj = nn.Linear(input_dim, cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.depth)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(cfg.freq_embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.latent_channels)
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        z_prior: Tensor,
        f_prior: Tensor,
        f_target: Tensor,
    ) -> Tensor:
        b, c, l = z_t.shape
        z_cat = torch.cat([z_t, z_prior], dim=1).permute(0, 2, 1)  # B, L, 2C
        tokens = self.input_proj(z_cat)

        t_embed = sinusoidal_embedding(t, self.cfg.time_embed_dim)
        t_embed = self.time_mlp(t_embed)

        freq_scale = 24_000.0
        prior_norm = torch.clamp(f_prior / freq_scale, 0.0, 1.0)
        target_norm = torch.clamp(f_target / freq_scale, 0.0, 1.0)
        f_prior_embed = self.freq_proj(sinusoidal_embedding(prior_norm, self.cfg.freq_embed_dim))
        f_target_embed = self.freq_proj(sinusoidal_embedding(target_norm, self.cfg.freq_embed_dim))

        freq_tokens = torch.stack([f_prior_embed, f_target_embed], dim=1)

        x = torch.cat([freq_tokens, tokens], dim=1)
        x = x + t_embed[:, None, :]
        x = self.transformer(x)
        x = self.layer_norm(x)

        x = x[:, 2:, :]
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)
        return x
