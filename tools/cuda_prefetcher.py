#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA prefetcher to overlap H2D copy with compute using a dedicated CUDA stream.
Usage:
  for batch in cuda_prefetch(dataloader, device):
      # batch tensors are already on device (non_blocking transfer)
      ...
Notes:
  - Works best with pin_memory=True and reasonable batch/worker settings
  - When device is CPU, simply yields original batches
"""
from __future__ import annotations

from typing import Iterator, Dict, Any
import torch


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def cuda_prefetch(dataloader, device: torch.device) -> Iterator[Dict[str, Any]]:
    if device.type != "cuda":
        for b in dataloader:
            yield b
        return

    stream = torch.cuda.Stream(device=device)
    loader_iter = iter(dataloader)

    def preload():
        try:
            batch = next(loader_iter)
        except StopIteration:
            return None
        with torch.cuda.stream(stream):
            batch = _to_device(batch, device)
        return batch

    next_batch = preload()
    while next_batch is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        batch = next_batch
        next_batch = preload()
        yield batch

