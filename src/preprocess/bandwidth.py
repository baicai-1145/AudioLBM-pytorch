from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import cheby1, filtfilt, savgol_filter

import librosa
import soundfile as sf

EPS = 1e-8


@dataclass
class BandwidthConfig:
    """Configuration for effective bandwidth estimation and filtering."""

    target_sr: int = 48000
    sg_window: int = 129
    sg_poly: int = 3
    downsample: int = 4
    curvature_window: int = 32
    curvature_threshold: float = 5e-4
    magnitude_threshold: float = -8.0  # in log-amplitude
    filter_order: int = 8
    ripple_db: float = 0.5


def load_wave(path: Path, target_sr: int) -> Tuple[NDArray[np.float32], int]:
    """Load mono waveform with librosa to ensure consistent resampling."""

    wave, sr = librosa.load(path, sr=target_sr, mono=True)
    return wave.astype(np.float32), sr


def compute_log_spectrum(
    wave: NDArray[np.float32],
    n_fft: Optional[int] = None,
) -> NDArray[np.float32]:
    """Return log-magnitude spectrum of waveform."""

    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(len(wave))))
    spectrum = np.fft.rfft(wave, n=n_fft)
    magnitude = np.abs(spectrum)
    log_mag = np.log(magnitude + EPS)
    return log_mag.astype(np.float32)


def smooth_and_downsample(
    log_mag: NDArray[np.float32],
    cfg: BandwidthConfig,
) -> NDArray[np.float32]:
    """Apply Savitzky-Golay smoothing and local downsampling."""

    smooth = savgol_filter(log_mag, cfg.sg_window, cfg.sg_poly, mode="interp")
    if cfg.downsample <= 1:
        return smooth

    length = len(smooth) // cfg.downsample
    trimmed = smooth[: length * cfg.downsample]
    reshaped = trimmed.reshape(length, cfg.downsample)
    return reshaped.mean(axis=1)


def estimate_effective_bandwidth(
    wave: NDArray[np.float32],
    sr: int,
    cfg: Optional[BandwidthConfig] = None,
) -> float:
    """Estimate effective bandwidth following Appendix A description."""

    cfg = cfg or BandwidthConfig(target_sr=sr)
    log_mag = compute_log_spectrum(wave)
    processed = smooth_and_downsample(log_mag, cfg)

    second_deriv = np.gradient(np.gradient(processed))

    lok = cfg.curvature_window
    max_idx = len(processed) - lok

    for idx in range(max_idx):
        window_curve = np.max(np.abs(second_deriv[idx : idx + lok]))
        if (
            window_curve < cfg.curvature_threshold
            and processed[idx] < cfg.magnitude_threshold
        ):
            effective_ratio = idx / len(processed)
            return effective_ratio * (sr / 2)

    return (sr / 2) * 0.95


def design_lowpass(
    cutoff_hz: float,
    sr: int,
    cfg: Optional[BandwidthConfig] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Design Chebyshev Type-I low-pass filter with sharp roll-off."""

    cfg = cfg or BandwidthConfig(target_sr=sr)
    nyquist = sr / 2
    normalized_cutoff = min(max(cutoff_hz / nyquist, 1e-4), 0.999)
    b, a = cheby1(
        N=cfg.filter_order,
        rp=cfg.ripple_db,
        Wn=normalized_cutoff,
        btype="lowpass",
    )
    return b, a


def apply_lowpass(
    wave: NDArray[np.float32],
    sr: int,
    cutoff_hz: float,
    cfg: Optional[BandwidthConfig] = None,
) -> NDArray[np.float32]:
    """Apply forward-backward Chebyshev filtering."""

    b, a = design_lowpass(cutoff_hz, sr, cfg)
    filtered = filtfilt(b, a, wave)
    return filtered.astype(np.float32)


def process_file(
    path: Path,
    output_dir: Path,
    cfg: Optional[BandwidthConfig] = None,
    dry_run: bool = False,
) -> dict:
    """Estimate bandwidth, apply filtering, and save processed waveform."""

    cfg = cfg or BandwidthConfig()
    wave, sr = load_wave(path, cfg.target_sr)

    bw_hz = estimate_effective_bandwidth(wave, sr, cfg)

    filtered = apply_lowpass(wave, sr, bw_hz, cfg)

    rel_path = path.relative_to(path.parents[1])
    out_path = output_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        sf.write(out_path, filtered, sr)

    return {
        "input": str(path),
        "output": str(out_path),
        "effective_bandwidth_hz": float(bw_hz),
        "sample_rate": sr,
    }


def list_audio_files(root: Path, exts: Iterable[str] = (".wav", ".flac")) -> Iterable[Path]:
    """Yield all audio files under root with selected extensions."""

    for ext in exts:
        yield from root.rglob(f"*{ext}")
