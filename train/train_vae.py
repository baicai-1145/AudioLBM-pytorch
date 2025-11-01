#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Unified VAE trainer (config-driven):
- Loads all hyper-parameters from a YAML config.
- Supports non-adv and adv+fm training via `adv.enabled`.
- Keeps defaults aligned with paper settings; enhanced HF options remain opt-in.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# repo path
import sys as _sys
if __package__ is None or __package__ == "":
    _sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml

from datasets.wav_hr_inmemory_dataset import WavHRInMemoryDataset
from models.vae.wav_vae import WaveVAE, VAEConfig
from models.vae.losses import mrstft_loss, kl_gaussian
from models.vae.discriminator import MultiScaleDiscriminator
from models.vae.adv_losses import hinge_d_loss, hinge_g_loss, feature_matching_loss


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return v if v is not None else default


def save_ckpt(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer, step: int, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "optim": optim.state_dict(), "config": cfg}, path)


def main():
    ap = argparse.ArgumentParser("Unified WaveVAE trainer (config.yaml)")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)

    # Sections with defaults
    data = C.get("data", {})
    model_cfg = C.get("model", {})
    train = C.get("train", {})
    optim_g = C.get("optim_g", {})
    optim_d = C.get("optim_d", {})
    loss = C.get("loss", {})
    adv = C.get("adv", {})
    sched = C.get("scheduler", {})

    device_str = str(_get(train, "device", "auto"))
    if device_str.lower() == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.backends.cudnn.benchmark = True

    # Dataset
    metadata = data["metadata"]
    target_sr = _get(data, "target_sr", 48000)
    segment_sec = _get(data, "segment_sec", 5.12)
    batch = _get(train, "batch", 16)
    ds = WavHRInMemoryDataset(metadata, target_sr=target_sr, segment_sec=segment_sec, normalize=True, preload_workers=_get(data, "preload_workers", 8))
    N = len(ds)

    # Model
    vae_conf = VAEConfig(
        channels=_get(model_cfg, "channels", 64),
        hop=_get(model_cfg, "hop", 512),
        win=_get(model_cfg, "win", 1024),
        hidden=_get(model_cfg, "hidden", 256),
        kl_weight=_get(loss, "kl_weight", 0.0),
    )
    model = WaveVAE(vae_conf).to(device)

    # Adversarial toggle
    use_adv = bool(_get(adv, "enabled", False))
    if use_adv:
        disc = MultiScaleDiscriminator(
            n_discriminators=_get(adv, "n_discriminators", 3),
            base=_get(adv, "disc_base", 32)
        ).to(device)
    else:
        disc = None

    # Optimizers
    def make_optim(module: torch.nn.Module, spec: Dict[str, Any], fallback_lr: float = 2e-4):
        typ = spec.get("type", "adamw").lower()
        lr = float(spec.get("lr", fallback_lr))
        b1 = float(spec.get("beta1", 0.9)); b2 = float(spec.get("beta2", 0.999))
        wd = float(spec.get("weight_decay", 0.0))
        if typ == "adam":
            return torch.optim.Adam(module.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
        else:
            return AdamW(module.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)

    opt_g = make_optim(model, optim_g, fallback_lr=_get(optim_g, "lr", _get(train, "lr", 2e-4)))
    if use_adv:
        opt_d = make_optim(disc, optim_d, fallback_lr=_get(optim_d, "lr", _get(train, "lr", 2e-4)))
    else:
        opt_d = None

    # Scheduler
    steps = int(_get(train, "steps", 200000))
    warmup = int(_get(train, "warmup", 2000))
    sched_type = _get(sched, "type", "cosine")
    if sched_type == "cosine":
        t_max = int(_get(sched, "t_max", max(10000, steps - warmup)))
        eta_min = float(_get(sched, "eta_min", 0.0))
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=t_max, eta_min=eta_min)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=t_max, eta_min=eta_min) if use_adv else None
    elif sched_type == "plateau":
        factor = float(_get(sched, "factor", 0.5))
        patience = int(_get(sched, "patience", 1000))
        threshold = float(_get(sched, "threshold", 1e-4))
        cooldown = int(_get(sched, "cooldown", 0))
        min_lr = float(_get(sched, "min_lr", 0.0))
        sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, mode="min", factor=factor, patience=patience, threshold=threshold,
            cooldown=cooldown, min_lr=min_lr, verbose=True)
        sched_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_d, mode="min", factor=factor, patience=patience, threshold=threshold,
            cooldown=cooldown, min_lr=min_lr, verbose=True) if use_adv else None
    else:
        sched_g = None; sched_d = None

    # Logging / saving
    outdir = Path(_get(train, "outdir", "exp/vae48k_cfg")); outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_log.jsonl"
    dump_every = int(_get(train, "dump_every", 0))
    dump_dir = Path(_get(train, "dump_dir", str(outdir / "samples")))
    save_every = int(_get(train, "save_every", 5000))
    log_every = int(_get(train, "log_every", 200))
    ema_enabled = bool(_get(train, "ema", False))
    if ema_enabled:
        from utils.ema import EMA
        ema = EMA(model, decay=float(_get(train, "ema_decay", 0.999)), device=str(device))
    else:
        ema = None

    # Resume
    step = 0
    resume = _get(train, "resume", None)
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location="cpu")
        try:
            model.load_state_dict(ckpt["model"], strict=False)
        except Exception:
            model.load_state_dict({k: v for k, v in ckpt["model"].items() if k in model.state_dict()}, strict=False)
        if "optim" in ckpt:
            try:
                opt_g.load_state_dict(ckpt["optim"])
            except Exception:
                for pg in opt_g.param_groups: pg["lr"] = float(_get(optim_g, "lr", 2e-4))
        step = int(ckpt.get("step", 0))
        print(f"[Resume] {resume} @ step={step}")

    # MR-STFT config
    preset = str(_get(loss, "mrstft_preset", "A")).upper()
    def mrstft_cfg(preset: str):
        if preset == "A":
            return [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]
        elif preset == "B":
            return [(1024, 256, 1024), (2048, 512, 2048), (4096, 1024, 4096)]
        else:  # C (optional, not paper)
            return [(128, 32, 128), (256, 64, 256), (512, 128, 512), (1024, 256, 1024)]

    # Loss weights
    l1_w = float(_get(loss, "l1_weight", 0.0))
    kl_w = float(_get(loss, "kl_weight", 0.0))
    hf_emphasis = float(_get(loss, "hf_emphasis", 0.0))
    hf_freq = int(_get(loss, "hf_freq", 6000))

    # Adv knobs
    prewarm = int(_get(adv, "prewarm", 5000))
    adv_w_max = float(_get(adv, "adv_weight", 0.5))
    fm_w_max = float(_get(adv, "fm_weight", 0.5))
    adv_ramp = int(_get(adv, "adv_ramp", 10000))
    fm_ramp = int(_get(adv, "fm_ramp", 10000))
    d_steps = int(_get(adv, "d_steps", 1))
    r1_gamma = float(_get(adv, "r1_gamma", 0.0))

    model.train(); 
    if disc is not None: disc.train()
    best_recon = float("inf")

    while step < steps:
        perm = torch.randperm(N)
        for i in range(0, N, batch):
            if step >= steps: break
            idx = perm[i:i+batch].tolist()
            hr_np = ds.hr[idx]
            hr = torch.from_numpy(hr_np).unsqueeze(1).to(device, non_blocking=True)  # [B,1,T]

            # 1) Discriminator (if enabled)
            if use_adv:
                with torch.no_grad():
                    x_det, _, _ = model(hr)
                for _ in range(max(1, d_steps)):
                    real_outs = disc(hr)
                    fake_outs = disc(x_det.detach())
                    real_scores = [s.mean(dim=-1) for (s, f) in real_outs]
                    fake_scores = [s.mean(dim=-1) for (s, f) in fake_outs]
                    d_loss = hinge_d_loss(real_scores, fake_scores)
                    if r1_gamma > 0.0:
                        hr.requires_grad_(True)
                        real_scores_full = disc(hr)[0][0].sum()
                        grads = torch.autograd.grad(real_scores_full, hr, create_graph=True, retain_graph=True)[0]
                        r1 = r1_gamma * 0.5 * torch.mean(grads.pow(2))
                        hr.requires_grad_(False)
                        d_loss = d_loss + r1
                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=5.0)
                    opt_d.step()
                real_feats = [[t.detach() for t in feats] for (_, feats) in real_outs]
            else:
                d_loss = torch.zeros([], device=device)
                real_feats = None

            # 2) Generator
            x_hat, mu, logvar = model(hr)
            l_recon = mrstft_loss(
                hr.squeeze(1), x_hat.squeeze(1),
                configs=mrstft_cfg(preset),
                sr=target_sr, hf_freq=(hf_freq if hf_emphasis > 0 else None), hf_boost=hf_emphasis,
            )
            l_time = torch.mean(torch.abs(hr - x_hat)) if l1_w > 0 else torch.zeros([], device=device)
            l_kl = kl_gaussian(mu.transpose(1, 2).reshape(mu.size(0), -1),
                               logvar.transpose(1, 2).reshape(logvar.size(0), -1)) * kl_w

            if use_adv:
                fake_outs_g = disc(x_hat)
                g_scores = [s.mean(dim=-1) for (s, f) in fake_outs_g]
                if step < prewarm:
                    adv_w = 0.0; fm_w = 0.0
                else:
                    adv_w = adv_w_max * min(1.0, (step - prewarm) / max(1, adv_ramp))
                    fm_w = fm_w_max * min(1.0, (step - prewarm) / max(1, fm_ramp))
                adv_loss = hinge_g_loss(g_scores) * adv_w
                fm_loss = feature_matching_loss(real_feats, [feats for (_, feats) in fake_outs_g]) * fm_w
            else:
                adv_loss = torch.zeros([], device=device)
                fm_loss = torch.zeros([], device=device)
                adv_w = 0.0; fm_w = 0.0

            g_loss = l_recon + l_time + l_kl + adv_loss + fm_loss
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt_g.step()
            if ema is not None:
                ema.update(model)
            step += 1

            # Warmup & LR schedulers
            if step <= warmup and warmup > 0:
                for pg in opt_g.param_groups: pg["lr"] = float(_get(optim_g, "lr", 2e-4)) * float(step) / float(warmup)
                if use_adv:
                    for pg in opt_d.param_groups: pg["lr"] = float(_get(optim_d, "lr", 2e-4)) * float(step) / float(warmup)
            if sched_g is not None and sched_type == "cosine":
                sched_g.step(); 
                if use_adv: sched_d.step()

            # Log
            if step % log_every == 0:
                with torch.no_grad():
                    z_std = float(torch.exp(0.5 * logvar).mean().detach().cpu())
                    kl_per_dim = float((l_kl.detach() / (mu.shape[1] * mu.shape[2])).cpu()) if kl_w > 0 else 0.0
                    best_recon = min(best_recon, float(l_recon.detach().cpu().item()))
                rec = {
                    "step": step,
                    "loss": float(g_loss.detach().cpu().item()),
                    "g_loss": float(g_loss.detach().cpu().item()),
                    "d_loss": float(d_loss.detach().cpu().item()) if use_adv else 0.0,
                    "recon": float(l_recon.detach().cpu().item()),
                    "adv": float(adv_loss.detach().cpu().item()) if use_adv else 0.0,
                    "fm": float(fm_loss.detach().cpu().item()) if use_adv else 0.0,
                    "adv_w": float(adv_w),
                    "fm_w": float(fm_w),
                    "kl": float(l_kl.detach().cpu().item()),
                    "kl_per_dim": kl_per_dim,
                    "z_std": z_std,
                    "lr_g": float(opt_g.param_groups[0]["lr"]),
                    "lr_d": float(opt_d.param_groups[0]["lr"]) if use_adv else 0.0,
                    "best_recon": best_recon,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(json.dumps(rec))

            # Dump audio
            if dump_every and step % dump_every == 0:
                dump_dir.mkdir(parents=True, exist_ok=True)
                import soundfile as sf
                sf.write(dump_dir / f"gt_step{step}.wav", hr[0, 0].detach().cpu().numpy(), target_sr)
                sf.write(dump_dir / f"recon_step{step}.wav", x_hat[0, 0].detach().cpu().numpy(), target_sr)

            # Save
            if step % save_every == 0:
                save_ckpt(str(outdir / f"ckpt_{step}.pt"), model, opt_g, step, C)
                if use_adv and opt_d is not None:
                    save_ckpt(str(outdir / f"ckptD_{step}.pt"), disc, opt_d, step, {"disc": True, **C})
                if ema is not None:
                    torch.save({"step": step, "model": ema.state_dict(), "config": C}, str(outdir / f"ckptEMA_{step}.pt"))

            if sched_g is not None and sched_type == "plateau" and step % log_every == 0:
                sched_g.step(l_recon.detach().cpu().item())
                if use_adv: sched_d.step(l_recon.detach().cpu().item())

    save_ckpt(str(outdir / f"ckpt_final.pt"), model, opt_g, step, C)


if __name__ == "__main__":
    main()

