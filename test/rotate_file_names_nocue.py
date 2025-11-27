"""
Generate rotated/copy-renamed versions of config images without cues.

Update the constants below:
    - `SOURCE_DIR`: folder that currently holds the images for one config (e.g. trl_n_x_xx)
    - `ROTATION_DEGREES`: rotation angle, in degrees, around the board center (default 6,6)

When run, the script:
    1. Computes the rotated start-arm direction (and goal if present in the prefix)
    2. Creates a sibling folder reflecting that rotated direction (e.g. trl_e_x_xx)
    3. Copies each PNG from SOURCE_DIR into that folder and renames the copies so
       the prefix and coordinates reflect the rotation.

Example:
    SOURCE_DIR      = Path("/path/to/trl_n_x_xx")
    ROTATION_DEGREES = 90

    => Output folder: /path/to/trl_e_x_xx
    => Filename:      trl_e_x_xx_<x_rot>_<y_rot>_<rest>.png
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Tuple

# ========= USER PARAMETERS =========
SOURCE_DIR = Path("~/VS Code Local/corner-maze-encoder-pretrain/data/images/config-files-update/iti_w_x_xx")  # update to your source folder
ROTATION_DEGREES = 90                    # e.g., 90, 180, 270 (multiples of 45)
CENTER: Tuple[float, float] = (6.0, 6.0)  # rotation center
# ===================================

_DIRECTION_ORDER = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
_DIR_TO_INDEX = {value: idx for idx, value in enumerate(_DIRECTION_ORDER)}


def rotate_point(x: int, y: int, degrees: float, center: Tuple[float, float]) -> Tuple[int, int]:
    """Rotate point (x, y) about center by `degrees` degrees."""
    radians = math.radians(degrees)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    dx = x - center[0]
    dy = y - center[1]

    rotated_x = dx * cos_theta - dy * sin_theta + center[0]
    rotated_y = dx * sin_theta + dy * cos_theta + center[1]

    return int(round(rotated_x)), int(round(rotated_y))


def rotation_steps(degrees: float) -> int:
    """Return number of 45° steps represented by degrees."""
    steps = round(degrees / 45.0)
    if not math.isclose(steps * 45.0, degrees, abs_tol=1e-6):
        raise ValueError("ROTATION_DEGREES must be a multiple of 45.")
    return steps


def rotate_direction_token(token: str, degrees: float) -> str:
    token_norm = token.lower()
    if token_norm not in _DIR_TO_INDEX:
        raise ValueError(
            f"Unsupported direction token '{token}'. Expected one of {_DIRECTION_ORDER}."
        )
    steps = rotation_steps(degrees)
    idx = (_DIR_TO_INDEX[token_norm] + steps) % len(_DIRECTION_ORDER)
    return _DIRECTION_ORDER[idx]


def parse_filename(filename: str) -> Tuple[Tuple[str, ...], int, int, Tuple[str, ...], str]:
    """
    Parse filenames like 'trl_n_x_xx_1_1_ne_l.png'.

    Returns:
        prefix_parts   -> ('trl', 'n', 'x', 'xx')
        x, y           -> integer coordinates
        suffix_parts   -> ('ne', 'l')
        extension      -> '.png'
    """
    stem, ext = Path(filename).stem, Path(filename).suffix
    parts = stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")

    prefix_parts = tuple(parts[:4])
    try:
        x = int(parts[4])
        y = int(parts[5])
    except ValueError as exc:
        raise ValueError(f"Cannot parse coordinates in '{filename}'.") from exc

    suffix_parts = tuple(parts[6:])
    return prefix_parts, x, y, suffix_parts, ext


def build_rotated_prefix(prefix_parts: Tuple[str, ...], rotation: float) -> str:
    """Rotate the start-arm component in the prefix (second token)."""
    if len(prefix_parts) < 2:
        raise ValueError(f"Prefix {prefix_parts!r} is too short to contain a direction.")
    updated = list(prefix_parts)
    updated[1] = rotate_direction_token(updated[1], rotation)
    if len(updated) >= 4 and updated[3].lower() in _DIR_TO_INDEX:
        updated[3] = rotate_direction_token(updated[3], rotation)
    return "_".join(updated)


def ensure_output_dir(source_dir: Path, new_dir_name: str) -> Path:
    target_dir = source_dir.parent / new_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def process_directory(source_dir: Path, rotation: float) -> None:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Provided path '{source_dir}' is not a directory.")

    sample_files = list(sorted(source_dir.glob("*.png")))
    if not sample_files:
        print(f"No PNG files found in {source_dir}. Nothing to do.")
        return

    first_prefix, _, _, _, _ = parse_filename(sample_files[0].name)
    new_prefix = build_rotated_prefix(first_prefix, rotation)
    output_dir = ensure_output_dir(source_dir, new_prefix)

    for file_path in sample_files:
        prefix_parts, x, y, suffix_parts, ext = parse_filename(file_path.name)
        updated_prefix = build_rotated_prefix(prefix_parts, rotation)
        x_rot, y_rot = rotate_point(x, y, rotation, CENTER)
        rotated_suffix = list(suffix_parts)
        if rotated_suffix:
            rotated_suffix[0] = rotate_direction_token(rotated_suffix[0], rotation)
        suffix_segment = "_".join(rotated_suffix) if rotated_suffix else ""
        new_name = f"{updated_prefix}_{x_rot}_{y_rot}"
        if suffix_segment:
            new_name += f"_{suffix_segment}"
        new_name += ext

        destination = output_dir / new_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)
        print(f"Copied {file_path.name} -> {destination.relative_to(output_dir.parent)}")

    print(f"Completed rotation for {len(sample_files)} files.")


def main() -> None:
    if SOURCE_DIR == Path("/path/to/trl_n_x_xx"):
        raise ValueError("Please update SOURCE_DIR before running the script.")
    process_directory(SOURCE_DIR, ROTATION_DEGREES)


if __name__ == "__main__":
    main()
