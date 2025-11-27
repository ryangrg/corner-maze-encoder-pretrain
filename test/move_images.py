#!/usr/bin/env python3
"""
move_images.py

Copy images from `data/images/all-images` into the matching
`data/images/config-files/<prefix>` directory based on each file's
prefix (session_phase_start_arm_cue_wall). Non-image files are ignored.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import argparse

ROOT = Path(__file__).resolve().parents[1]
ALL_IMAGES_DIR = ROOT / "data/images/all-images"
CONFIG_FILES_DIR = ROOT / "data/images/config-files"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def determine_prefix(filename: str) -> str:
    """
    Extract the prefix (first four underscore-separated tokens) for a filename.
    Example: 'trl_n_x_xx_3_1_ne_l.png' -> 'trl_n_x_xx'
    """
    parts = filename.split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename '{filename}' does not contain a valid prefix.")
    return "_".join(parts[:4])


def gather_files(directory: Path) -> Iterable[Path]:
    return directory.rglob("*")


def copy_images(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy all supported image files from src_dir to the prefix-matched
    subdirectory in dst_dir.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    for path in gather_files(src_dir):
        if path.is_dir():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        prefix = determine_prefix(path.name)
        target_dir = dst_dir / prefix
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / path.name
        shutil.copy2(path, destination)
        print(f"[INFO] Copied {path.relative_to(src_dir)} -> {destination.relative_to(dst_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy images into config-specific folders based on filename prefix."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=ALL_IMAGES_DIR,
        help=f"Source directory containing images (default: {ALL_IMAGES_DIR})",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=CONFIG_FILES_DIR,
        help=f"Destination base directory (default: {CONFIG_FILES_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_images(args.src, args.dst)


if __name__ == "__main__":
    main()
