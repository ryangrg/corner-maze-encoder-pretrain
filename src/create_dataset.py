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

IMAGE_DIR = "/Users/ryangrgurich/VS Code Local/corner-maze-encoder-pretrain/data/images/all-images"                 # directory containing PNGs
BLUR_RADIUS = 1.5                             # Gaussian blur radius
OUTPUT_NAME = "/Users/ryangrgurich/VS Code Local/corner-maze-encoder-pretrain/data/pt-files/all-images-dataset.pt" # output path and filename

# =======================
# === SCRIPT LOGIC ======
# =======================

import re
from collections import defaultdict
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

# Index all left/right image pairs in the given directory
# Returns a dict mapping (config, x, y, direction) -> {"left": Path, "right": Path}
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

# Here we are loading and processing each image pair and returning a tensor of size (2, H, W)
# Convert images to grayscale, apply Gaussian blur if specified, and mirror the right eye image
def load_and_process_pair(left_path: Path, right_path: Path, blur_radius: float) -> torch.Tensor:
    # Load images and convert to grayscale
    left_img = Image.open(left_path).convert("L")
    right_img = Image.open(right_path).convert("L")
    
    # Apply Gaussian blur if specified
    if blur_radius > 0:
        left_img = left_img.filter(ImageFilter.GaussianBlur(blur_radius))
        right_img = right_img.filter(ImageFilter.GaussianBlur(blur_radius))

    # Mirror the right image horizontally
    right_img = TF.hflip(right_img)
    
    # Convert to tensor and stack along new dimension going from [0,255] to [0.0,1.0]
    # Then concatenate left and right images along the channel dimension forming (2, H, W)
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
    label_names = [key_to_label(k) for k in keys]
    name_to_id = {name: i for i, name in enumerate(label_names)}

    print(f"[INFO] Found {len(keys)} complete pairs in {root}")

    xs: List[torch.Tensor] = []
    label_ids: List[int] = []
    for i, key in enumerate(keys, 1):
        pair = pairs[key]
        t = load_and_process_pair(pair["left"], pair["right"], blur_radius)
        xs.append(t)
        label_ids.append(name_to_id[key_to_label(key)])
        if i % 200 == 0 or i == len(keys):
            print(f"[INFO] Processed {i}/{len(keys)}")

    X = torch.stack(xs)
    Y = torch.tensor(label_ids, dtype=torch.long)

    catalog_builder: Dict[int, set] = defaultdict(set)
    for label_id, description in zip(label_ids, label_names):
        catalog_builder[label_id].add(description)
    label_catalog = {
        label_id: {"descriptions": sorted(descriptions)}
        for label_id, descriptions in catalog_builder.items()
    }
    label2idx = {
        description: label_id
        for label_id, info in label_catalog.items()
        for description in info["descriptions"]
    }
    idx2label = {
        label_id: info["descriptions"][0]
        for label_id, info in label_catalog.items()
    }

    payload = {
        "x": X,
        "y": Y,
        "labels": label_ids,
        "label_names": label_names,
        "label_catalog": label_catalog,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "meta": {
            "blur_radius": blur_radius,
            "count": len(label_names),
            "label_dtype": str(Y.dtype),
            "catalog_size": len(label_catalog),
        },
    }

    out_path = Path(output_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # <-- ensures dirs exist
    torch.save(payload, out_path)

    print(f"[DONE] Saved {out_path.resolve()}  ->  x{tuple(X.shape)}, y{tuple(Y.shape)}")
    print(f"[DONE] Example label_names: {label_names[:3]}")
    print(f"[DONE] Example label: {label_ids[:3]}")


# Automatically run when executed or pasted in a notebook cell
if __name__ == "__main__":
    main(IMAGE_DIR, BLUR_RADIUS, OUTPUT_NAME)
