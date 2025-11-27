#!/usr/bin/env python3
"""
image_equalizer.py

Copy every image from `data/images/all-images` into
`data/images/all-images-equalized` after applying histogram equalization
to normalize per-image brightness. Alpha channels and non-image files
are preserved.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/images/test-compare"
DEFAULT_OUTPUT = ROOT / "data/images/test-compare-equalized"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _build_clahe() -> cv2.CLAHE:
    return cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))


CLAHE = _build_clahe()


def _equalize_single_channel(channel: np.ndarray) -> np.ndarray:
    return CLAHE.apply(channel)


def equalize_image(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE-based histogram equalization to the luminance channel.
    Supports grayscale, BGR, and BGRA images (alpha preserved).
    """
    if image.ndim == 2:
        return _equalize_single_channel(image)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    channels = image.shape[2]
    if channels == 1:
        return _equalize_single_channel(image[:, :, 0])[:, :, None]

    if channels == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        bgr = image[:, :, :3]
        alpha = None

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_equalized = _equalize_single_channel(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    equalized_bgr = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    if alpha is not None:
        return np.dstack((equalized_bgr, alpha))
    return equalized_bgr


def process_images(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy and equalize all supported images from src_dir to dst_dir,
    preserving relative paths. Non-image files are copied as-is.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    files: Iterable[Path] = src_dir.rglob("*")
    for path in files:
        relative = path.relative_to(src_dir)
        destination = dst_dir / relative

        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            shutil.copy2(path, destination)
            continue

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[WARN] Unable to read image: {path}")
            continue

        equalized = equalize_image(image)
        success = cv2.imwrite(str(destination), equalized)
        if not success:
            print(f"[WARN] Failed to write: {destination}")
        else:
            print(f"[INFO] Equalized {relative}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy images into a new directory after histogram equalization."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source image directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory for equalized images (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_images(args.src, args.dst)


if __name__ == "__main__":
    main()
