#!/usr/bin/env python3
"""
build_binocular_bundle.py

Processes binocular PNG images into a single .pt dataset bundle.

Each file should follow:
    configN_x_y_direction_eye.png

Where:
  - N: integer configuration ID
  - x, y: coordinates (ints)
  - direction: 0–7 or compass (north, NE, east, SE, south, SW, west, NW)
  - eye: 'left' or 'right'

Processing steps:
  • Convert to grayscale
  • Apply Gaussian blur (to model low visual acuity)
  • Mirror the right eye horizontally
  • Stack (left, right_mirrored) → shape (2, H, W)

Output:
  A single .pt file containing:
  {
      "x": Tensor [N, 2, H, W],
      "y": Tensor [N],
      "labels": [str],
      "label2idx": {str:int},
      "meta": {blur_radius, count, root}
  }

This version is notebook-friendly:
just edit the constants below and run the whole cell.
"""

# =======================
# === USER SETTINGS ====
# =======================

IMAGE_DIR = "/path/to/images"                 # directory containing PNGs
BLUR_RADIUS = 1.5                             # Gaussian blur radius
OUTPUT_NAME = "data/processed/rat_dataset.pt" # output path and filename

# =======================
# === SCRIPT LOGIC ======
# =======================

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF


# --- Filename pattern ---
FNAME_RE = re.compile(
    r"""^
        config(?P<config>\d+)
        _(?P<x>-?\d+)
        _(?P<y>-?\d+)
        _(?P<direction>(?:[0-7]|north|NE|east|SE|south|SW|west|NW))
        _(?P<eye>left|right)(?:-?eye)?
        \.png$
    """,
    re.IGNORECASE | re.VERBOSE,
)
DIR_TO_INT = {
    "NORTH": 0,
    "NE": 1,
    "EAST": 2,
    "SE": 3,
    "SOUTH": 4,
    "SW": 5,
    "WEST": 6,
    "NW": 7,
}


def key_to_label(key: Tuple[int, int, int, int]) -> str:
    cfg, x, y, d = key
    return f"{cfg}_{x}_{y}_{d}"


def index_pairs(root: Path) -> Dict[Tuple[int,int,int,int], Dict[str, Path]]:
    pairs: Dict[Tuple[int,int,int,int], Dict[str, Path]] = {}
    for p in root.glob("*.png"):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        cfg = int(m.group("config"))
        x = int(m.group("x"))
        y = int(m.group("y"))
        dir_raw = m.group("direction")
        eye = m.group("eye").lower()

        if dir_raw.isdigit():
            d = int(dir_raw)
        else:
            d = DIR_TO_INT.get(dir_raw.upper(), None)
            if d is None:
                continue

        key = (cfg, x, y, d)
        bucket = pairs.setdefault(key, {})
        bucket[eye] = p

    # Keep only complete pairs
    return {k: v for k, v in pairs.items() if "left" in v and "right" in v}


def load_and_process_pair(left_path: Path, right_path: Path, blur_radius: float) -> torch.Tensor:
    left_img = Image.open(left_path).convert("L")
    right_img = Image.open(right_path).convert("L")

    if blur_radius > 0:
        left_img = left_img.filter(ImageFilter.GaussianBlur(blur_radius))
        right_img = right_img.filter(ImageFilter.GaussianBlur(blur_radius))

    right_img = TF.hflip(right_img)

    left_t = TF.to_tensor(left_img).float()
    right_t = TF.to_tensor(right_img).float()
    return torch.cat([left_t, right_t], dim=0)  # (2, H, W)


def main(image_dir: str, blur_radius: float, output_name: str):
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    pairs = index_pairs(root)
    if not pairs:
        raise RuntimeError("No complete left/right pairs found in directory.")

    keys = sorted(pairs.keys())
    labels = [key_to_label(k) for k in keys]
    label2idx = {lbl: i for i, lbl in enumerate(labels)}

    print(f"[INFO] Found {len(keys)} complete pairs in {root}")

    xs, ys = [], []
    for i, key in enumerate(keys, 1):
        pair = pairs[key]
        t = load_and_process_pair(pair["left"], pair["right"], blur_radius)
        xs.append(t)
        ys.append(label2idx[key_to_label(key)])
        if i % 200 == 0 or i == len(keys):
            print(f"[INFO] Processed {i}/{len(keys)}")

    X = torch.stack(xs)
    Y = torch.tensor(ys, dtype=torch.long)

    payload = {
        "x": X,
        "y": Y,
        "labels": labels,
        "label2idx": label2idx,
        "meta": {
            "blur_radius": blur_radius,
            "count": len(labels),
        },
    }

    out_path = Path(output_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # <-- ensures dirs exist
    torch.save(payload, out_path)

    print(f"[DONE] Saved {out_path.resolve()}  ->  x{tuple(X.shape)}, y{tuple(Y.shape)}")
    print(f"[DONE] Example labels: {labels[:3]}")


# Automatically run when executed or pasted in a notebook cell
if __name__ == "__main__":
    main(IMAGE_DIR, BLUR_RADIUS, OUTPUT_NAME)
