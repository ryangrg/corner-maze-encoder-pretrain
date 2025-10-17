from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class StereoPair:
    base: str
    left_path: Path
    right_path: Path


def load_stereo_pairs(
    data_dir: Path,
    blur_radius: float = 1.5,
    return_metadata: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[StereoPair]]]:
    """
    Load left/right eye PNGs, convert to grayscale, blur, mirror right eye,
    and return a stacked tensor of shape (num_pairs, 2, H, W).
    Filenames must end with `_left.png` and `_right.png`.

    Set `return_metadata=False` to omit the metadata list if not needed.
    """
    png_paths = sorted(data_dir.glob("**/*.png"))

    if not png_paths:
        raise FileNotFoundError(f"No PNG files found in {data_dir.resolve()}")

    pairs: Dict[str, Dict[str, Path]] = {}
    for path in png_paths:
        stem = path.stem
        try:
            base, eye = stem.rsplit("_", 1)
        except ValueError:
            print(f"Skipping {path.name}: unable to parse eye suffix")
            continue

        eye = eye.lower()
        if eye not in {"left", "right"}:
            print(f"Skipping {path.name}: unexpected eye tag {eye!r}")
            continue

        pairs.setdefault(base, {})[eye] = path

    stacks: List[torch.Tensor] = []
    metadata: List[StereoPair] = []
    missing_pairs: List[str] = []

    for base, eyes in sorted(pairs.items()):
        if {"left", "right"} <= eyes.keys():
            left_tensor = _preprocess_image(eyes["left"], blur_radius)
            right_tensor = _preprocess_image(
                eyes["right"],
                blur_radius,
                mirror=True,
            )
            stacked = torch.cat([left_tensor, right_tensor], dim=0)
            stacks.append(stacked)
            metadata.append(
                StereoPair(
                    base=base,
                    left_path=eyes["left"],
                    right_path=eyes["right"],
                )
            )
        else:
            missing = {"left", "right"} - eyes.keys()
            missing_pairs.append(f"{base} missing {', '.join(sorted(missing))}")

    if missing_pairs:
        print("Skipped incomplete pairs:")
        for entry in missing_pairs:
            print(f"  - {entry}")

    if not stacks:
        raise RuntimeError("No complete left/right eye pairs found.")

    stacked_tensor = torch.stack(stacks)

    if return_metadata:
        return stacked_tensor, metadata

    return stacked_tensor


def _preprocess_image(
    image_path: Path,
    blur_radius: float,
    mirror: bool = False,
) -> torch.Tensor:
    """Convert image to grayscale, blur, optionally mirror, and return tensor."""
    with Image.open(image_path) as img:
        gray = img.convert("L")
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        if mirror:
            blurred = blurred.transpose(Image.FLIP_LEFT_RIGHT)
        array = np.array(blurred, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def visualize_stereo_pair(
    stack: torch.Tensor,
    index: int,
    metadata: Sequence[StereoPair],
) -> None:
    """
    Display overlap (left=blue, right=red), mirrored, and original right-eye views.
    """
    if stack.ndim != 4 or stack.shape[1] != 2:
        raise ValueError("Expected stack of shape (N, 2, H, W).")

    if not (0 <= index < stack.shape[0]):
        raise IndexError(f"Index {index} out of range for stack of size {stack.shape[0]}.")

    if len(metadata) != stack.shape[0]:
        raise ValueError("Metadata length must match number of stereo pairs.")

    sample = metadata[index]

    left = stack[index, 0].detach().cpu().numpy()
    right_mirrored = stack[index, 1].detach().cpu().numpy()
    right_original = np.fliplr(right_mirrored)

    def tint_blue(channel: np.ndarray) -> np.ndarray:
        colored = np.zeros((*channel.shape, 3), dtype=np.float32)
        colored[..., 2] = channel
        return colored

    def tint_red(channel: np.ndarray) -> np.ndarray:
        colored = np.zeros((*channel.shape, 3), dtype=np.float32)
        colored[..., 0] = channel
        return colored

    overlap = tint_blue(left) + tint_red(right_mirrored)
    overlap = np.clip(overlap, 0.0, 1.0)

    left_vs_mirrored = np.concatenate(
        (tint_blue(left), tint_red(right_mirrored)),
        axis=1,
    )
    left_vs_original = np.concatenate(
        (tint_blue(left), tint_red(right_original)),
        axis=1,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(overlap)
    axes[0].set_title("Overlap (Left=Blue, Right=Red)")
    axes[0].axis("off")

    axes[1].imshow(left_vs_mirrored)
    axes[1].set_title("Left vs Right (Mirrored)")
    axes[1].axis("off")

    axes[2].imshow(left_vs_original)
    axes[2].set_title("Left vs Right (Original)")
    axes[2].axis("off")

    fig.suptitle(f"Stereo Pair: {sample.base} (index {index})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_directory = Path("~/VS Code Local/corner-maze-encoder-pretrain/data/images/").expanduser().resolve()
    stack, meta = load_stereo_pairs(data_directory)
    print("Image stack shape:", tuple(stack.shape))
    visualize_stereo_pair(stack, index=3, metadata=meta)
