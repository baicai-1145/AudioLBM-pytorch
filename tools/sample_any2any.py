#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal sampler to sanity check AnyToAnyDataset:
- Loads dataset
- Iterates DataLoader (multi-worker) and dumps a few (LR, HR) pairs to disk
- Prints condition stats

Usage:
  python tools/sample_any2any.py \
    --metadata preprocessed/VCTK48k_smoke/metadata.jsonl \
    --outdir samples_any2any \
    --num 8 --batch 4 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from datasets.any2any_dataset import AnyToAnyDataset, DegradeCfg, collate_fn


def save_wav(path: str, y: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32, copy=False), sr)


def main():
    p = argparse.ArgumentParser("Sample AnyToAnyDataset")
    p.add_argument("--metadata", type=str, required=True)
    p.add_argument("--outdir", type=str, default="samples_any2any")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    ds = AnyToAnyDataset(metadata_path=args.metadata)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate_fn, drop_last=False)

    printed = 0
    fpriors = []
    for batch in dl:
        B = batch["lr"].shape[0]
        sr = int(batch["sr"].item())
        for i in range(B):
            if printed >= args.num:
                break
            lr = batch["lr"][i].numpy()
            hr = batch["hr"][i].numpy()
            f_prior = float(batch["f_prior_hz"][i].item())
            f_target_q = float(batch["f_target_q_hz"][i].item())
            fpriors.append(f_prior)
            save_wav(os.path.join(args.outdir, f"sample{printed:03d}_LR_{int(f_prior)}Hz.wav"), lr, sr)
            save_wav(os.path.join(args.outdir, f"sample{printed:03d}_HR.wav"), hr, sr)
            printed += 1
        if printed >= args.num:
            break
    if fpriors:
        fpriors = np.asarray(fpriors, dtype=np.float32)
        print("f_prior stats: min=", float(fpriors.min()),
              "p50=", float(np.percentile(fpriors, 50)),
              "p90=", float(np.percentile(fpriors, 90)),
              "max=", float(fpriors.max()))
        print("Saved samples to", args.outdir)
    else:
        print("No samples written; dataset may be empty.")


if __name__ == "__main__":
    main()

