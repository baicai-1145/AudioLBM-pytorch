#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCTK preprocessing pipeline (multi-process):
- scan wav files
- resample to 48 kHz
- segment into 5.12 s
- curvature-aware f_eff estimation and low-pass filtering at f_eff
- export HR segments and metadata jsonl for training any→48 kHz

Usage:
  python tools/preprocess_vctk.py \
      --root data/VCTK-Corpus-0.92 \
      --outdir preprocessed/VCTK48k \
      --sr 48000 \
      --segment-sec 5.12 \
      --jobs 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
	
# Make the script runnable both as module and as script
if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[1]))
	
from tools.audio_utils import read_audio, write_audio, lowpass
import soundfile as sf
from tools.curvature import compute_f_eff


@dataclass
class CurvatureCfg:
    sg_window: int = 101
    sg_poly: int = 3
    downsample: int = 4
    eps_sr: float = 1e-3
    energy_quantile: float = 0.97
    window_k: int = 5


@dataclass
class FilterCfg:
    type: str = "cheby1"   # cheby1 / butter / bessel / elliptic
    order: int = 8
    rp_db: float = 0.5
    rs_db: float = 60.0


def find_wavs(root: str) -> List[str]:
    exts = {".wav", ".flac"}
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def segment_indices(n_samples: int, seg_len: int, hop_len: int, offset: int = 0) -> List[Tuple[int, int]]:
    idx = []
    i = max(0, int(offset))
    hop = max(1, int(hop_len))
    while i + seg_len <= n_samples:
        idx.append((i, i + seg_len))
        i += hop
    return idx


def process_one(
    args: argparse.Namespace,
    wav_path: str,
    sr: int,
    seg_len: int,
    curv_cfg: CurvatureCfg,
    filt_cfg: FilterCfg,
    seed: int,
) -> List[Dict]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    # Original source SR filter (paper filters SR<32k for training sets)
    try:
        sr_orig = sf.info(wav_path).samplerate
    except Exception:
        sr_orig = 0
    if args.min_src_sr > 0 and sr_orig and sr_orig < args.min_src_sr:
        return []

    # Read + resample to target sr
    y, sr0 = read_audio(wav_path, target_sr=sr, mono=True)
    # Estimate f_eff and filter
    f_eff_hz, dbg = compute_f_eff(
        y, sr,
        sg_window=curv_cfg.sg_window,
        sg_poly=curv_cfg.sg_poly,
        downsample=curv_cfg.downsample,
        eps_sr=curv_cfg.eps_sr,
        energy_quantile=curv_cfg.energy_quantile,
        window_k=curv_cfg.window_k,
    )
    y_f = lowpass(y, sr, cutoff_hz=f_eff_hz,
                  filter_type=filt_cfg.type, order=filt_cfg.order,
                  rp_db=filt_cfg.rp_db, rs_db=filt_cfg.rs_db)
    # Segment
    hop_len = int(round(args.hop_sec * sr)) if args.hop_sec and args.hop_sec > 0 else seg_len
    offset = 0
    if args.random_offset:
        max_off = max(0, hop_len - 1)
        if len(y_f) > seg_len:
            offset = random.randint(0, min(max_off, len(y_f) - seg_len - 1))
    segs = segment_indices(len(y_f), seg_len, hop_len, offset)
    rel = os.path.relpath(wav_path, args.root)
    parts = Path(rel).parts
    # Heuristic: pick the first directory that looks like a VCTK speaker (e.g., p225)
    speaker = None
    for pdir in parts[:-1]:
        if len(pdir) >= 2 and (pdir[0] in ("p", "P")) and pdir[1:].isdigit():
            speaker = pdir
            break
    if speaker is None:
        # Fallback to parent directory if available, else 'spk'
        speaker = parts[-2] if len(parts) >= 2 else "spk"
    base = Path(rel).with_suffix("").name

    out_records: List[Dict] = []
    for si, (s, e) in enumerate(segs):
        seg = y_f[s:e].copy()
        if seg.shape[0] != seg_len:
            continue
        out_path = Path(args.outdir) / speaker / f"{base}_seg{si:04d}.wav"
        write_audio(str(out_path), seg, sr)
        rec = {
            "speaker": speaker,
            "utt": base,
            "seg_idx": si,
            "hr_path": str(out_path.as_posix()),
            "sr_hr": sr,
            "num_samples": seg_len,
            "t0": s / sr,
            "t1": e / sr,
            "f_eff_hz": float(f_eff_hz),
            "curvature": dbg,
            "prefilter": {
                "type": filt_cfg.type,
                "order": filt_cfg.order,
                "rp_db": filt_cfg.rp_db,
                "rs_db": filt_cfg.rs_db,
            },
            # Training-time any→any construction will sample SR_LR~U(0, SR_HR)
            "f_target_hz": sr / 2.0,
            "seed": seed,
            "src_rel": rel,
            "src_sr": sr0,
        }
        out_records.append(rec)
    return out_records


def main():
    p = argparse.ArgumentParser("VCTK preprocessing (multi-process)")
    p.add_argument("--root", type=str, required=True, help="VCTK root directory")
    p.add_argument("--outdir", type=str, required=True, help="Output dir for HR segments and metadata")
    p.add_argument("--sr", type=int, default=48000, help="Target sampling rate")
    p.add_argument("--segment-sec", type=float, default=5.12, help="Segment length in seconds")
    p.add_argument("--hop-sec", type=float, default=5.12, help="Hop length in seconds (<= segment-sec for overlap)")
    p.add_argument("--random-offset", action="store_true", help="Random starting offset per file within one hop")
    p.add_argument("--jobs", type=int, default=8, help="Number of worker processes")
    p.add_argument("--seed", type=int, default=1337, help="Global seed")
    p.add_argument("--min-src-sr", type=int, default=32000, help="Skip files whose original SR < this (paper: 32k)")
    # Curvature cfg
    p.add_argument("--sg-window", type=int, default=101)
    p.add_argument("--sg-poly", type=int, default=3)
    p.add_argument("--downsample", type=int, default=4)
    p.add_argument("--eps-sr", type=float, default=1e-3)
    p.add_argument("--energy-quantile", type=float, default=0.97)
    p.add_argument("--window-k", type=int, default=5)
    # Filter cfg
    p.add_argument("--filter-type", type=str, default="cheby1",
                   choices=["cheby1", "butter", "bessel", "elliptic"])
    p.add_argument("--filter-order", type=int, default=8)
    p.add_argument("--filter-rp-db", type=float, default=0.5)
    p.add_argument("--filter-rs-db", type=float, default=60.0)
    # Misc
    p.add_argument("--metadata", type=str, default="metadata.jsonl", help="Metadata filename under outdir")
    p.add_argument("--limit", type=int, default=0, help="Limit number of wavs for quick run")
    p.add_argument("--skip-existing", action="store_true", help="Skip processing if speaker folder exists")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)

    os.makedirs(args.outdir, exist_ok=True)
    wavs = find_wavs(args.root)
    if args.limit > 0:
        wavs = wavs[:args.limit]

    seg_len = int(round(args.segment_sec * args.sr))
    curv_cfg = CurvatureCfg(
        sg_window=args.sg_window, sg_poly=args.sg_poly, downsample=args.downsample,
        eps_sr=args.eps_sr, energy_quantile=args.energy_quantile, window_k=args.window_k
    )
    filt_cfg = FilterCfg(
        type=args.filter_type, order=args.filter_order, rp_db=args.filter_rp_db, rs_db=args.filter_rs_db
    )

    # Early exit if already processed and skip-existing
    if args.skip_existing:
        # Heuristic: if outdir contains a speaker directory with many files
        spk_dirs = [d for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir, d))]
        if spk_dirs:
            print(f"[Info] Found existing output in {args.outdir}, skip-existing enabled. Exiting.")
            return

    # Multiprocessing
    meta_path = os.path.join(args.outdir, args.metadata)
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex, open(meta_path, "w", encoding="utf-8") as fout:
        futures = []
        for idx, wp in enumerate(wavs):
            seed = (args.seed + idx * 9973) & 0x7FFFFFFF
            futures.append(ex.submit(
                process_one, args, wp, args.sr, seg_len, curv_cfg, filt_cfg, seed
            ))
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="preprocess"):
            try:
                recs = fut.result()
                for r in recs:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Warn] worker failed: {e}", file=sys.stderr)

    print(f"[Done] Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
