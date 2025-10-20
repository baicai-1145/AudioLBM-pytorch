from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass
class STFTConfig:
    fft_size: int
    hop_size: int
    win_size: int
    window: Tensor


def _build_window(win_size: int, device: torch.device) -> Tensor:
    return torch.hann_window(win_size, periodic=True, device=device)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        configs: Iterable[Tuple[int, int, int]] = (
            (1024, 120, 600),
            (2048, 240, 1200),
            (512, 50, 240),
        ),
        log_weight: float = 1.0,
        spec_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.raw_configs = list(configs)
        self.log_weight = log_weight
        self.spec_weight = spec_weight
        self.register_buffer("dummy", torch.tensor(0.0), persistent=False)
        self._stft_configs: List[STFTConfig] = []

    def _ensure_configs(self, device: torch.device) -> None:
        if self._stft_configs and self._stft_configs[0].window.device == device:
            return
        self._stft_configs = []
        for fft_size, hop_size, win_size in self.raw_configs:
            window = _build_window(win_size, device)
            self._stft_configs.append(STFTConfig(fft_size, hop_size, win_size, window))

    def forward(self, input_wave: Tensor, target_wave: Tensor) -> Tuple[Tensor, Tensor]:
        device = input_wave.device
        self._ensure_configs(device)

        total_spec = input_wave.new_tensor(0.0)
        total_log = input_wave.new_tensor(0.0)
        for cfg in self._stft_configs:
            spec_input = torch.stft(
                input_wave,
                n_fft=cfg.fft_size,
                hop_length=cfg.hop_size,
                win_length=cfg.win_size,
                window=cfg.window,
                return_complex=True,
            )
            spec_target = torch.stft(
                target_wave,
                n_fft=cfg.fft_size,
                hop_length=cfg.hop_size,
                win_length=cfg.win_size,
                window=cfg.window,
                return_complex=True,
            )
            mag_input = torch.abs(spec_input)
            mag_target = torch.abs(spec_target)

            diff = torch.norm(mag_target - mag_input, p="fro")
            norm = torch.norm(mag_target, p="fro") + 1e-7
            total_spec = total_spec + diff / norm

            log_mag_input = torch.log(mag_input + 1e-7)
            log_mag_target = torch.log(mag_target + 1e-7)
            total_log = total_log + torch.mean(torch.abs(log_mag_target - log_mag_input))

        total_spec = total_spec * self.spec_weight / len(self._stft_configs)
        total_log = total_log * self.log_weight / len(self._stft_configs)
        return total_spec, total_log
