"""
View individual entries from a stereo tensor stack (.npz) produced by
stereo_tensor_stacker.py.

Examples:
    python test/tensor_viewer.py stack.npz 3 --eye left
    python test/tensor_viewer.py stack.npz 5 --eye overlay --save-dir exports/

Options:
    --eye left      View the mirrored left-eye frame.
    --eye right     View the mirrored-right frame as stored.
    --eye both      Show mirrored left and mirrored-right side by side.
    --eye overlay   Blend channels (red=mirrored-right, blue=left) to inspect alignment.
    --save PATH     Write the selected view to PATH.
    --save-dir DIR  Write to DIR with metadata-based filename (config_x_y_dir_eye.png).
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _render_pair(pair: np.ndarray, *, eye: str) -> Image.Image:
    if pair.ndim != 3 or pair.shape[0] != 2:
        raise ValueError(f"Expected pair shape (2, H, W); received {pair.shape!r}.")

    left, right = pair

    if eye == "left":
        array = np.flip(left, axis=1)
        return Image.fromarray(array.astype(np.uint8), mode="L")
    if eye == "right":
        return Image.fromarray(right.astype(np.uint8), mode="L")
    if eye == "both":
        array = np.concatenate([np.flip(left, axis=1), right], axis=1)
        return Image.fromarray(array.astype(np.uint8), mode="L")
    if eye == "overlay":
        rgb = np.zeros((*left.shape, 3), dtype=np.uint8)
        rgb[..., 0] = right  # mirrored-right -> red
        rgb[..., 2] = left   # raw left -> blue
        return Image.fromarray(rgb, mode="RGB")

    raise ValueError(f"Unknown eye option: {eye}")


def _open_with_system_viewer(path: Path, title: str) -> None:
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print(f"Could not launch system viewer ({exc!r}). Falling back to Pillow viewer.")
        Image.open(path).show(title=title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display or save a specific sample from a stereo tensor stack (.npz)."
    )
    parser.add_argument("npz_path", type=Path, help="Path to the .npz tensor stack.")
    parser.add_argument("index", type=int, help="Index of the sample to view.")
    parser.add_argument(
        "--eye",
        choices=("left", "right", "both", "overlay"),
        default="left",
        help=(
            "Which eye to render (default: left). "
            "Use 'both' for an aligned side-by-side view or 'overlay' to blend raw slices "
            "(red=mirrored-right, blue=left)."
        ),
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save",
        type=Path,
        help="Explicit output path. When omitted, the image is opened via the default viewer.",
    )
    save_group.add_argument(
        "--save-dir",
        type=Path,
        help=(
            "Directory to save using metadata-based filename "
            "(e.g. config1_2_2_east_left.png)."
        ),
    )
    args = parser.parse_args()

    arrays = _load_npz(args.npz_path)
    if "images" not in arrays:
        raise KeyError("NPZ archive missing 'images' array.")

    images = arrays["images"]
    if args.index < 0 or args.index >= len(images):
        raise IndexError(f"Index {args.index} out of range for images of length {len(images)}.")

    img = _render_pair(images[args.index], eye=args.eye)

    base_name = None
    if "bases" in arrays and args.index < len(arrays["bases"]):
        base_name = str(arrays["bases"][args.index]).strip()
    elif "labels" in arrays and args.index < len(arrays["labels"]):
        base_name = str(arrays["labels"][args.index]).replace(" ", "_")
    if "directions" in arrays and args.index < len(arrays["directions"]):
        direction_token = str(arrays["directions"][args.index]).strip()
        if base_name and direction_token and direction_token not in base_name:
            base_name = f"{base_name}_{direction_token}"

    suffix = args.eye
    if suffix == "both":
        suffix = "both_eyes"
    filename = (base_name or f"sample_{args.index}") + f"_{suffix}.png"

    if args.save or args.save_dir:
        if args.save_dir:
            target_dir = args.save_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / filename
        else:
            output_path = args.save
            output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"Saved {args.eye} eye sample {args.index} to {output_path}.")
    else:
        label = None
        if "labels" in arrays:
            labels = arrays["labels"]
            if args.index < len(labels):
                label = labels[args.index]
        meta = f" (label: {label})" if label is not None else ""
        print(
            f"Displaying sample {args.index} as '{filename}'{meta}. Close the viewer window to exit."
        )
        temp_dir = Path(tempfile.gettempdir()) / "tensor_viewer"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / filename
        img.save(temp_path)
        _open_with_system_viewer(temp_path, title=filename)


if __name__ == "__main__":
    main()
