#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bridge import ExponentialBridgeSchedule  # noqa: E402
from src.models import AudioVAE, AudioVAEConfig  # noqa: E402
from src.models.dit import DiTConfig, FrequencyAwareDiT  # noqa: E402
from src.preprocess.bandwidth import BandwidthConfig, apply_lowpass, estimate_effective_bandwidth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LBM sampler for audio SR.")
    parser.add_argument("--lr-audio", type=Path, required=True, help="Path to low-resolution audio waveform.")
    parser.add_argument("--vae-checkpoint", type=Path, required=True, help="Trained VAE checkpoint.")
    parser.add_argument("--lbm-checkpoint", type=Path, required=True, help="Trained LBM checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("lbm_output.wav"))
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--f-target", type=float, default=24_000.0)
    parser.add_argument("--f-prior", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wave_lr, sr = librosa.load(args.lr_audio, sr=args.sample_rate, mono=True)
    cfg = BandwidthConfig(target_sr=args.sample_rate)

    if args.f_prior is None:
        f_prior = estimate_effective_bandwidth(wave_lr, sr, cfg)
    else:
        f_prior = args.f_prior

    wave_filtered = apply_lowpass(wave_lr, sr, cutoff_hz=f_prior, cfg=cfg)

    vae = AudioVAE(AudioVAEConfig(input_channels=1))
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location="cpu"))
    vae = vae.to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    dit_cfg = DiTConfig()
    model = FrequencyAwareDiT(dit_cfg)
    state = torch.load(args.lbm_checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    schedule = ExponentialBridgeSchedule(g_min=0.031622, g_max=1.0)

    wave_tensor = torch.from_numpy(wave_filtered).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, z_lr = vae.encode(wave_tensor)

    z_s = z_lr.clone()
    steps = torch.linspace(1.0, 0.0, args.num_steps + 1, device=device)

    f_prior_tensor = torch.tensor([f_prior], device=device, dtype=torch.float32)
    f_target_tensor = torch.tensor([args.f_target], device=device, dtype=torch.float32)

    for i in tqdm(range(args.num_steps), desc="Sampling"):
        s = steps[i].item()
        t = steps[i + 1].item()
        s_tensor = torch.full((1,), s, device=device, dtype=torch.float32)
        with torch.no_grad():
            eps_pred = model(z_s, s_tensor, z_lr, f_prior_tensor, f_target_tensor)
        _, sigma_s, _, _ = schedule.coefficients(s_tensor)
        z_hat0 = z_s - sigma_s[:, None, None] * eps_pred
        z_s = schedule.step(z_s, z_hat0, t, s, noise=torch.zeros_like(z_s))

    z_0 = z_s
    with torch.no_grad():
        wave_hat = vae.decode(z_0).squeeze(0).squeeze(0).cpu().numpy()

    sf.write(args.output, wave_hat, args.sample_rate)
    print(f"Saved super-resolved audio to {args.output} (f_prior≈{f_prior:.1f} Hz)")


if __name__ == "__main__":
    main()
