#!/usr/bin/env python3
"""
使用训练好的 VAE 对音频进行编码-解码重建，并将结果写入磁盘。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import AudioVAE, AudioVAEConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct audio with trained VAE.")
    parser.add_argument("--audio", type=Path, required=True, help="输入音频路径（任意长度，自动裁剪/补零）")
    parser.add_argument("--checkpoint", type=Path, required=True, help="VAE 权重路径")
    parser.add_argument("--output", type=Path, default=Path("vae_recon.wav"), help="输出音频路径")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="目标采样率")
    parser.add_argument("--segment-seconds", type=float, default=5.12, help="片段长度（秒），需与训练保持一致")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    return parser.parse_args()


def segment_waveform(wave: np.ndarray, segment_length: int) -> List[np.ndarray]:
    segments: List[np.ndarray] = []
    total = len(wave)
    offset = 0
    while offset < total:
        end = offset + segment_length
        chunk = wave[offset:end]
        if len(chunk) < segment_length:
            chunk = np.pad(chunk, (0, segment_length - len(chunk)))
        segments.append(chunk.astype(np.float32))
        offset = end
    if not segments:
        segments.append(np.zeros(segment_length, dtype=np.float32))
    return segments


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wave, sr = librosa.load(args.audio, sr=args.sample_rate, mono=True)
    segment_length = int(args.segment_seconds * args.sample_rate)
    segments = segment_waveform(wave, segment_length)

    vae = AudioVAE(AudioVAEConfig(input_channels=1))
    vae.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    vae = vae.to(device)
    vae.eval()

    recon_segments: List[np.ndarray] = []
    with torch.no_grad():
        for seg in segments:
            tensor = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0).to(device)
            recon, _, _, _ = vae(tensor)
            recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()
            recon_segments.append(recon_np)

    recon_wave = np.concatenate(recon_segments, axis=0)
    recon_wave = recon_wave[: len(wave)]  # 截断回与原始长度一致

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, recon_wave, args.sample_rate)
    print(f"Saved reconstructed audio to {args.output}")


if __name__ == "__main__":
    main()
