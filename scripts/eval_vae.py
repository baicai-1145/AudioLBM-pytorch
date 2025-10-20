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
from src.models import AudioVAE, AudioVAEConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AudioVAE reconstruction metrics.")
    parser.add_argument("--data-root", type=Path, required=True, help="Processed audio root directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="VAE checkpoint path.")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--segment-seconds", type=float, default=5.12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-batches", type=int, default=50)
    return parser.parse_args()


def log_spectral_distance(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=reference.device)
    spec_ref = torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    spec_est = torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    power_ref = torch.abs(spec_ref) ** 2 + 1e-7
    power_est = torch.abs(spec_est) ** 2 + 1e-7
    diff = torch.log10(power_ref) - torch.log10(power_est)
    lsd_per_frame = torch.sqrt(torch.mean(diff**2, dim=1).clamp_min(1e-8))
    return lsd_per_frame.mean(dim=1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = AudioSegmentDataset(
        AudioDatasetConfig(
            root=args.data_root,
            segment_seconds=args.segment_seconds,
            sample_rate=args.sample_rate,
        )
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    vae = AudioVAE(AudioVAEConfig(input_channels=1))
    vae.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    vae = vae.to(device)
    vae.eval()

    mse_total = 0.0
    lsd_total = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx >= args.num_batches:
                break
            batch = batch.to(device)
            recon, _, _, _ = vae(batch)
            mse_total += torch.mean((batch - recon) ** 2).item()
            lsd_batch = log_spectral_distance(batch.squeeze(1), recon.squeeze(1))
            lsd_total += lsd_batch.mean().item()
            count += 1

    mse_avg = mse_total / max(count, 1)
    lsd_avg = lsd_total / max(count, 1)
    print(f"MSE: {mse_avg:.6f} | LSD: {lsd_avg:.6f}")


if __name__ == "__main__":
    main()
