#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export GT/Recon audio pairs from a trained WaveVAE checkpoint for quick listening.

Usage:
  python tools/vae_export_recon.py \
    --metadata preprocessed/VCTK48k_smoke/metadata.jsonl \
    --ckpt exp/vae48k_smoke/ckpt_40000.pt \
    --outdir exports/vae_recon --num 12 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# repo import path
import sys as _sys
if __package__ is None or __package__ == "":
    _sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.wav_hr_inmemory_dataset import WavHRInMemoryDataset
from models.vae.wav_vae import WaveVAE, VAEConfig


def main():
    ap = argparse.ArgumentParser("Export WaveVAE recon for listening")
    ap.add_argument("--metadata", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="exports/vae_recon")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    cfg = VAEConfig(
        channels=cfg_dict.get("channels", 64),
        hop=cfg_dict.get("hop", 480),
        win=cfg_dict.get("win", 960),
        hidden=cfg_dict.get("hidden", 256),
        kl_weight=cfg_dict.get("kl_weight", 0.0),
    )
    model = WaveVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # dataset
    ds = WavHRInMemoryDataset(args.metadata, target_sr=48000, segment_sec=5.12, normalize=True, preload_workers=8)
    N = len(ds)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # random picks
    idx = torch.randperm(N)[: args.num].tolist()
    for i, j in enumerate(idx):
        hr_np = ds.hr[j]  # [T]
        hr = torch.from_numpy(hr_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]
        with torch.no_grad():
            x_hat, _, _ = model(hr)
        gt = hr.squeeze().detach().cpu().numpy()
        recon = x_hat.squeeze().detach().cpu().numpy()
        sf.write(outdir / f"sample{i:03d}_gt.wav", gt, 48000)
        sf.write(outdir / f"sample{i:03d}_recon.wav", recon, 48000)
        print(f"[Saved] {outdir / f'sample{i:03d}_gt.wav'}  |  {outdir / f'sample{i:03d}_recon.wav'}")

    print(f"Done. Files are in: {outdir}")


if __name__ == "__main__":
    main()

