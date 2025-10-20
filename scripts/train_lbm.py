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

from src.bridge import ExponentialBridgeSchedule  # noqa: E402
from src.data import AudioDatasetConfig, LBMAny48Config, LBMDataset  # noqa: E402
from src.models import AudioVAE, AudioVAEConfig  # noqa: E402
from src.models.dit import DiTConfig, FrequencyAwareDiT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frequency-aware LBM (any-to-48k).")
    parser.add_argument("--data-root", type=Path, required=True, help="Processed audio root directory.")
    parser.add_argument("--vae-checkpoint", type=Path, required=True, help="Path to trained VAE checkpoint.")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--segment-seconds", type=float, default=5.12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=Path, default=Path("models/lbm"))
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = LBMDataset(
        LBMAny48Config(
            data=AudioDatasetConfig(
                root=args.data_root,
                segment_seconds=args.segment_seconds,
                sample_rate=args.sample_rate,
            )
        )
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    vae = AudioVAE(AudioVAEConfig(input_channels=1))
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location="cpu"))
    vae = vae.to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    dit_cfg = DiTConfig()
    model = FrequencyAwareDiT(dit_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    schedule = ExponentialBridgeSchedule(g_min=0.031622, g_max=1.0)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    iterator = iter(dataloader)

    while global_step < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        wave_hr = batch["wave_hr"].to(device)
        wave_lr = batch["wave_lr"].to(device)
        f_prior = batch["f_prior"].to(device)
        f_target = batch["f_target"].to(device)

        with torch.no_grad():
            _, _, z_hr = vae.encode(wave_hr)
            _, _, z_lr = vae.encode(wave_lr)

        eps = 5e-4
        t = torch.rand(wave_hr.size(0), device=device) * (1 - 2 * eps) + eps
        z_t, target = schedule.forward_sample(z_hr, z_lr, t)

        pred = model(z_t, t, z_lr, f_prior, f_target)
        loss = torch.mean((pred - target) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if global_step % args.log_interval == 0:
            tqdm.write(f"step {global_step:06d} | loss {loss.item():.4f}")

        if (global_step + 1) % args.save_interval == 0:
            ckpt_path = args.save_dir / f"lbm_step{global_step+1:07d}.pt"
            torch.save({"model": model.state_dict()}, ckpt_path)

        global_step += 1


if __name__ == "__main__":
    main()
