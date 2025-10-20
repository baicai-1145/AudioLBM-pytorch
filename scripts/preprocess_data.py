#!/usr/bin/env python3
"""预处理音频数据：估计有效带宽并应用低通滤波。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.bandwidth import (  # noqa: E402
    BandwidthConfig,
    list_audio_files,
    process_file,
)


def resolve_paths(dataset_root: Path, output_root: Optional[Path]) -> tuple[Path, Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if output_root is None:
        output_root = dataset_root.parent / "processed" / dataset_root.name

    output_root.mkdir(parents=True, exist_ok=True)
    return dataset_root, output_root


@click.command()
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path),
    required=True,
    help="原始数据集根目录，例如 data/VCTK-Corpus-0.92",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=None,
    help="预处理输出目录，默认写入 data/processed/<dataset>",
)
@click.option("--target-sr", type=int, default=48000, help="重采样目标采样率。")
@click.option("--dry-run", is_flag=True, help="只计算带宽但不写入音频。")
@click.option("--report-path", type=click.Path(path_type=Path), default=None, help="保存统计 JSON。")
def main(
    dataset_root: Path,
    output_root: Optional[Path],
    target_sr: int,
    dry_run: bool,
    report_path: Optional[Path],
) -> None:
    dataset_root, output_root = resolve_paths(dataset_root, output_root)
    cfg = BandwidthConfig(target_sr=target_sr)

    results = []
    audio_files = list(list_audio_files(dataset_root))
    if not audio_files:
        raise RuntimeError(f"No audio files found under {dataset_root}")

    for audio_path in tqdm(audio_files, desc="Preprocessing audio"):
        info = process_file(audio_path, output_root, cfg=cfg, dry_run=dry_run)
        results.append(info)

    if report_path is None:
        report_path = output_root / "preprocess_report.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    click.echo(f"Processed {len(results)} files. Report saved to {report_path}")


if __name__ == "__main__":
    main()
