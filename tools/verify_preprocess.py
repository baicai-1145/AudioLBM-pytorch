#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify preprocessed metadata/audio:
- existence of hr_path
- samplerate == target sr
- duration ~= segment-sec (tolerance)
- f_eff range in [0, sr/2]
- speaker directory name sanity (e.g., p225)

Usage:
  python tools/verify_preprocess.py \
    --metadata preprocessed/VCTK48k_smoke/metadata.jsonl \
    --sr 48000 --segment-sec 5.12 --tol-sec 0.02 --jobs 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm


@dataclass
class Args:
    metadata: str
    sr: int
    segment_sec: float
    tol_sec: float
    jobs: int
    max_check: int
    outdir: Optional[str]
    speaker_regex: str


def load_metadata(path: str, max_check: int = 0) -> List[Dict]:
    recs: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
                recs.append(r)
            except Exception:
                continue
            if max_check > 0 and len(recs) >= max_check:
                break
    return recs


def infer_speaker(hr_path: str, outdir: Optional[str]) -> str:
    p = Path(hr_path)
    if outdir:
        try:
            rel = Path(hr_path).resolve().relative_to(Path(outdir).resolve())
            first = rel.parts[0] if len(rel.parts) > 0 else p.parent.name
        except Exception:
            first = p.parent.name
    else:
        first = p.parent.name
    return first


def check_one(r: Dict, a: Args, spk_re: re.Pattern) -> Tuple[bool, Dict]:
    ok = True
    problems: List[str] = []
    path = r.get("hr_path", "")
    if not path or not os.path.exists(path):
        ok = False
        problems.append("missing_file")
        return ok, {"path": path, "problems": problems}
    try:
        info = sf.info(path)
    except Exception:
        ok = False
        problems.append("sf_info_error")
        info = None
    if info is not None:
        if info.samplerate != a.sr:
            ok = False
            problems.append(f"sr_mismatch:{info.samplerate}")
        dur = info.frames / float(info.samplerate)
        if abs(dur - a.segment_sec) > a.tol_sec:
            ok = False
            problems.append(f"dur_mismatch:{dur:.5f}")
    f_eff = r.get("f_eff_hz", None)
    if isinstance(f_eff, (int, float)):
        if not (0.0 <= float(f_eff) <= a.sr * 0.5 + 1.0):
            ok = False
            problems.append(f"f_eff_out:{f_eff}")
    speaker = infer_speaker(path, a.outdir)
    if not spk_re.match(speaker):
        problems.append(f"speaker_suspicious:{speaker}")
    return ok, {"path": path, "speaker": speaker, "problems": problems}


def main():
    p = argparse.ArgumentParser("Verify preprocessed metadata/audio")
    p.add_argument("--metadata", type=str, required=True)
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--segment-sec", type=float, default=5.12)
    p.add_argument("--tol-sec", type=float, default=0.02)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--max-check", type=int, default=0, help="Limit number of records to check (0=all)")
    p.add_argument("--outdir", type=str, default=None, help="Expected root of hr_path for speaker inference")
    p.add_argument("--speaker-regex", type=str, default=r"^[pP]\d+$")
    args_ns = p.parse_args()
    args = Args(
        metadata=args_ns.metadata,
        sr=args_ns.sr,
        segment_sec=args_ns.segment_sec,
        tol_sec=args_ns.tol_sec,
        jobs=args_ns.jobs,
        max_check=args_ns.max_check,
        outdir=args_ns.outdir,
        speaker_regex=args_ns.speaker_regex,
    )
    spk_re = re.compile(args.speaker_regex)

    recs = load_metadata(args.metadata, args.max_check)
    if not recs:
        print("[Error] No records loaded from", args.metadata)
        raise SystemExit(2)

    summary = {
        "total": len(recs),
        "ok": 0,
        "missing_file": 0,
        "sr_mismatch": 0,
        "dur_mismatch": 0,
        "f_eff_out": 0,
        "sf_info_error": 0,
        "speaker_suspicious": 0,
    }
    speaker_counts: Dict[str, int] = {}
    samples_problem: List[Dict] = []

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(check_one, r, args, spk_re) for r in recs]
        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="verify"):
            ok, info = fut.result()
            if ok:
                summary["ok"] += 1
            else:
                samples_problem.append(info)
            probs = info.get("problems", [])
            for pr in probs:
                key = pr.split(":", 1)[0]
                if key in summary:
                    summary[key] += 1
            spk = info.get("speaker", None)
            if spk:
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    # f_eff stats
    fs = []
    with open(args.metadata, "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            try:
                fs.append(float(json.loads(l)["f_eff_hz"]))
            except Exception:
                continue
            if args.max_check > 0 and i >= args.max_check:
                break
    fs = np.array(fs) if fs else np.array([0.0])
    fstats = dict(
        count=int(fs.size),
        min=float(fs.min()),
        p50=float(np.percentile(fs, 50)),
        p90=float(np.percentile(fs, 90)),
        max=float(fs.max()),
    )

    print("\nSummary:", json.dumps(summary, ensure_ascii=False, indent=2))
    print("f_eff stats:", json.dumps(fstats, ensure_ascii=False))
    print("speakers:", len(speaker_counts), "examples:", dict(list(speaker_counts.items())[:10]))

    # Print a few problematic samples for quick inspection
    if samples_problem:
        print("\nExamples of problems (up to 10):")
        for s in samples_problem[:10]:
            print(s)

    # Exit with non-zero if hard errors exist
    hard_errors = summary["missing_file"] + summary["sf_info_error"] + summary["sr_mismatch"] + summary["dur_mismatch"] + summary["f_eff_out"]
    raise SystemExit(1 if hard_errors > 0 else 0)


if __name__ == "__main__":
    main()

