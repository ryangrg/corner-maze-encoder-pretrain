"""
Build a stereo tensor dataset by pairing left/right PNG eye captures.

Usage:
    python test/stereo_tensor_stacker.py INPUT_DIR [--output PATH]

Inputs must follow names like `config1_2_2_east_left.png` where the final segment
indicates the eye. The script groups images that share the same base name
(`config1_2_2_east`) and stacks each pair into a tensor with the mirrored right
image placed above the left eye image (axis 0 order: right, left). Each frame is
reduced to a single grayscale channel prior to stacking. Every stacked sample
emits a string label in the form `"<config_number> <x> <y> <direction>"`.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StereoKey:
    base: str
    config: str
    x: int
    y: int
    direction: str | None


def _is_int_token(token: str) -> bool:
    stripped = token.lstrip("+-")
    return stripped.isdigit() and stripped != ""


def _parse_base_metadata(base: str) -> StereoKey:
    parts = base.split("_")
    numeric_indices = [idx for idx, part in enumerate(parts) if _is_int_token(part)]
    if len(numeric_indices) < 2:
        raise ValueError(f"Could not locate x/y coordinates in '{base}'.")

    x_idx, y_idx = numeric_indices[-2], numeric_indices[-1]
    x = int(parts[x_idx])
    y = int(parts[y_idx])

    config_parts = parts[:x_idx]
    if not config_parts:
        raise ValueError(f"Missing configuration prefix in '{base}'.")
    config = "_".join(config_parts)

    direction_parts = parts[y_idx + 1 :]
    direction = "_".join(direction_parts) if direction_parts else None

    return StereoKey(base=base, config=config, x=x, y=y, direction=direction)


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        grayscale = img.convert("L")  # collapse to one channel (black/white)
        return np.array(grayscale, dtype=np.uint8)


def _mirror_right_image(arr: np.ndarray) -> np.ndarray:
    # Mirror horizontally so folded-right coordinates align with the left eye.
    return np.flip(arr, axis=1)


def _config_label_token(config: str) -> str:
    match = re.search(r"(\d+)$", config)
    return match.group(1) if match else config


def discover_pairs(directory: Path) -> dict[StereoKey, dict[str, Path]]:
    pairs: dict[StereoKey, dict[str, Path]] = {}
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)

    for path in sorted(directory.glob("*.png")):
        stem = path.stem
        if "_" not in stem:
            continue

        eye_split = stem.rsplit("_", 1)
        if len(eye_split) != 2:
            continue

        base, eye = eye_split
        eye = eye.lower()
        if eye not in {"left", "right"}:
            continue

        key = _parse_base_metadata(base)
        grouped[key.base][eye] = path
        pairs[key] = grouped[key.base]

    return pairs


def stack_directory(directory: Path) -> tuple[np.ndarray, np.ndarray, list[StereoKey]]:
    pairs = discover_pairs(directory)
    if not pairs:
        raise FileNotFoundError(f"No PNG files found under {directory}")

    stacks: list[np.ndarray] = []
    labels: list[str] = []
    keys: list[StereoKey] = []

    for key, eyes in sorted(
        pairs.items(),
        key=lambda item: (
            item[0].config,
            item[0].x,
            item[0].y,
            item[0].direction or "",
        ),
    ):
        if "left" not in eyes or "right" not in eyes:
            continue

        left_arr = _load_image(eyes["left"])
        right_arr = _load_image(eyes["right"])
        if left_arr.shape != right_arr.shape:
            raise ValueError(
                f"Image shapes mismatch for {key.base}: {left_arr.shape} vs {right_arr.shape}"
            )

        mirrored_right = _mirror_right_image(right_arr)
        stack = np.stack([mirrored_right, left_arr], axis=0)

        stacks.append(stack)
        config_token = _config_label_token(key.config)
        label = f"{config_token} {key.x} {key.y} {(key.direction or '').strip()}".strip()
        labels.append(label)
        keys.append(key)

    if not stacks:
        raise RuntimeError("No complete stereo pairs were processed.")

    stacked_array = np.stack(stacks, axis=0)
    label_array = np.asarray(labels, dtype=object)
    return stacked_array, label_array, keys


def save_npz(
    output_path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    keys: Iterable[StereoKey],
) -> None:
    key_strings = np.array([key.base for key in keys], dtype=object)
    directions = np.array([key.direction or "" for key in keys], dtype=object)
    np.savez_compressed(
        output_path,
        images=images,
        labels=labels,
        bases=key_strings,
        directions=directions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stack paired left/right PNG eye images into a tensor dataset where the "
            "right image is horizontally mirrored before stacking."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing left/right PNG pairs with names like config1_x_y_dir_left.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stereo_tensor_stack.npz"),
        help="Destination .npz path (default: ./stereo_tensor_stack.npz).",
    )
    args = parser.parse_args()
    directory: Path = args.directory
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory.")

    images, labels, keys = stack_directory(directory)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, images, labels, keys)

    print(
        f"Saved {images.shape[0]} stacked pairs "
        f"({images.shape[1]} layers each) to {output_path}."
    )


if __name__ == "__main__":
    main()
