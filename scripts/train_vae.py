#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AudioDatasetConfig, AudioSegmentDataset  # noqa: E402
from src.losses import MultiResolutionSTFTLoss  # noqa: E402
from src.models import AudioVAE, AudioVAEConfig  # noqa: E402
from src.models.discriminator import (  # noqa: E402
    DiscriminatorConfig,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Audio VAE for AudioLBM.")
    parser.add_argument("--data-root", type=Path, required=True, help="Processed audio root directory.")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--segment-seconds", type=float, default=5.12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--feat-weight", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=Path, default=Path("models/vae"))
    parser.add_argument("--save-interval", type=int, default=1000)
    return parser.parse_args()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = AudioSegmentDataset(
        AudioDatasetConfig(
            root=args.data_root,
            segment_seconds=args.segment_seconds,
            sample_rate=args.sample_rate,
        )
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    vae = AudioVAE(AudioVAEConfig(input_channels=1)).to(device)
    disc = MultiScaleDiscriminator(DiscriminatorConfig(input_channels=1)).to(device)

    opt_g = torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.99))

    stft_loss = MultiResolutionSTFTLoss().to(device)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    dataloader_iter = iter(dataloader)

    while global_step < args.max_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = batch.to(device)

        with torch.no_grad():
            recon, mu, logvar, _ = vae(batch)

        real_outputs = disc(batch)
        fake_outputs = disc(recon.detach())
        d_loss = discriminator_loss([o[0] for o in real_outputs], [o[0] for o in fake_outputs])

        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        recon, mu, logvar, _ = vae(batch)
        fake_outputs = disc(recon)
        real_outputs = disc(batch)

        adv_loss = generator_adversarial_loss([o[0] for o in fake_outputs]) * args.adv_weight
        feat_loss = feature_matching_loss(
            [o[1] for o in real_outputs],
            [o[1] for o in fake_outputs],
        ) * args.feat_weight
        spec_loss, log_loss = stft_loss(recon.squeeze(1), batch.squeeze(1))
        kl_loss = kl_divergence(mu, logvar) * args.kl_weight

        g_loss = spec_loss + log_loss + adv_loss + feat_loss + kl_loss

        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt_g.step()

        if global_step % args.log_interval == 0:
            tqdm.write(
                f"step {global_step:06d} | "
                f"D: {d_loss.item():.4f} | "
                f"G: {g_loss.item():.4f} | "
                f"spec: {spec_loss.item():.4f} | "
                f"log: {log_loss.item():.4f} | "
                f"adv: {adv_loss.item():.4f} | "
                f"feat: {feat_loss.item():.4f} | "
                f"kl: {kl_loss.item():.4f}"
            )

        if (global_step + 1) % args.save_interval == 0:
            torch.save(vae.state_dict(), args.save_dir / f"vae_step{global_step+1:07d}.pt")
            torch.save(disc.state_dict(), args.save_dir / f"disc_step{global_step+1:07d}.pt")

        global_step += 1


if __name__ == "__main__":
    main()
