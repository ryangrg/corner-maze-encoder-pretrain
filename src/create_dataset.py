#!/usr/bin/env python3

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF

import dataset_io

"""
create_dataset.py

Load stereoscopic PNG files, pre-process them (grayscale, blur, right-eye mirror),
and write a dataset bundle that downstream scripts (visualizer, classifier, embedding
export) can consume. The final bundle is persisted via `dataset_io.save_bundle(...)`
so tensors live in `data/datasets/...` alongside a JSON metadata file.

Filename template:
    <session>_<start>_<cue>_<goal>_<x>_<y>_<direction>_<eye>.png

Key conventions:
  - session_phase: `trl`, `iti`, etc.
  - start_arm / cue_wall / goal_zone: maze metadata tokens
  - x, y: integer maze coordinates
  - direction: n, e, s, w, ne, se, sw, nw
  - eye: l (left) or r (right)

Processing pipeline:
  • Convert both eyes to grayscale
  • Optionally apply Gaussian blur
  • Mirror the right eye so both channels align in the same orientation
  • Stack into tensors shaped (2, H, W)

Bundle payload schema:
    payload = {
        "x": X,
        "y": Y,
        "labels": label_ids,
        "label_names": label_names,
        "labels2label_names": labels2label_names,
        "meta": {
            "blur_radius": BLUR_RADIUS,
            "stack_len": X.shape[0],
            "label_names_count": len(label_names),
            "config_fields": list(CONFIG_FIELDS),
            "source_images": str(IMAGE_DIR),
            "per_channel_tolerance": None,
            "mean_tolerance": None,
        },
    }
  • x: Tensor shaped [N, 2, H, W] containing left/right grayscale channels (right eye mirrored).
  • y: Tensor shaped [N] with remapped label IDs (0..num_classes-1) aligned with x.
  • labels: Original integer label IDs (pre-remap) in dataset order (useful when reapplying updates).
  • label_names: Slugified filename strings (“config_x_y_direction”) – unique per stereo pair.
  • labels2label_names: Dict[label_id -> sorted list of associated label names].
  • meta: Auxiliary metadata dictionary.
      - blur_radius: Float blur radius applied to each eye before stacking.
      - stack_len: Number of stereo pairs (equals len(x)).
      - label_names_count: Number of label_name entries.
      - config_fields: Ordered list of config tokens (session, start, cue, goal) used in slug generation.
      - source_images: String path to the root directory scanned for PNGs.
      - per_channel_tolerance / mean_tolerance placeholders (filled after grouping if desired).
      - Additional keys can be added freely (dataset_io persists everything in JSON).

Edit the constants below and run the script/notebook cell to regenerate the bundle.
"""

# =======================
# === USER SETTINGS ====
# =======================
# Update these paths for your local setup as needed
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/images/corner-maze-render-base-images"
BLUR_RADIUS = 1.5
OUTPUT_DATASET_DIR = ROOT / "data/datasets/corner-maze-render-base-images"

# =======================
# === SCRIPT LOGIC ======
# =======================

# --- Filename pattern ---
FNAME_RE = re.compile(
    r"""^
        (?P<session_phase>[a-z0-9]+)
        _(?P<start_arm>[a-z0-9]+)
        _(?P<cue_wall>[a-z0-9]+)
        _(?P<goal_zone>[a-z0-9]+)
        _(?P<x>-?\d+)
        _(?P<y>-?\d+)
        _(?P<direction>(?:n|e|s|w|ne|se|sw|nw))
        _(?P<eye>l|r)
        \.png$
    """,
    re.IGNORECASE | re.VERBOSE,
)
DIR_TO_INT = {
    "NW": 0,
    "N": 0,
    "NE": 1,
    "E": 1,
    "SE": 2,
    "S": 2,
    "SW": 3,
    "W": 3,
}
CONFIG_FIELDS = ("session_phase", "start_arm", "cue_wall", "goal_zone")
ConfigTuple = Tuple[str, str, str, str]
PairKey = Tuple[ConfigTuple, int, int, int]


def normalize_config(match: re.Match) -> ConfigTuple:
    """Extract the config tuple (session, start, cue, goal) from a filename match."""
    return tuple(match.group(field).lower() for field in CONFIG_FIELDS)  # type: ignore[arg-type]


def config_to_slug(config: ConfigTuple) -> str:
    """Convert a config tuple into the slug used in label names."""
    return "_".join(config)


def config_to_metadata(config: ConfigTuple) -> Dict[str, str]:
    """Map config fields to a metadata dict stored alongside each label."""
    data = {field: value for field, value in zip(CONFIG_FIELDS, config)}
    data["config_slug"] = config_to_slug(config)
    # Additional metadata can be added here if needed
    return data


def key_to_label(key: PairKey) -> str:
    """Generate the canonical label string for a (config, x, y, direction) key."""
    cfg, x, y, d = key
    return f"{config_to_slug(cfg)}_{x}_{y}_{d}"

# Index all left/right image pairs in the given directory
# Returns a dict mapping (config tuple, x, y, direction) -> {"left": Path, "right": Path}
def index_pairs(root: Path) -> Dict[PairKey, Dict[str, Path]]:
    """Scan the image directory and collect complete left/right-eye pairs."""
    pairs: Dict[PairKey, Dict[str, Path]] = {}
    for p in root.rglob("*.png"):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        cfg = normalize_config(m)
        x = int(m.group("x"))
        y = int(m.group("y"))
        dir_raw = m.group("direction").upper()
        eye_token = m.group("eye").lower()
        eye = "left" if eye_token in {"l", "left"} else "right"

        d = DIR_TO_INT.get(dir_raw)
        if d is None:
            continue

        key: PairKey = (cfg, x, y, d)
        bucket = pairs.setdefault(key, {})
        bucket[eye] = p

    # Keep only complete pairs
    return {k: v for k, v in pairs.items() if "left" in v and "right" in v}

# Here we are loading and processing each image pair and returning a tensor of size (2, H, W)
# Convert images to grayscale, apply Gaussian blur if specified, and mirror the right eye image
def load_and_process_pair(left_path: Path, right_path: Path, blur_radius: float) -> torch.Tensor:
    """Load, grayscale, blur, and stack a single stereo pair."""
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

root = Path(IMAGE_DIR)
if not root.exists():
    raise FileNotFoundError(f"Image directory not found: {root}")

pairs = index_pairs(root)
if not pairs:
    raise RuntimeError("No complete left/right pairs found in directory.")

keys = sorted(pairs.keys())
label_names: List[str] = []
name_to_id: Dict[str, int] = {}
for idx, key in enumerate(keys):
    label = key_to_label(key)
    label_names.append(label)
    name_to_id[label] = idx

print(f"[INFO] Found {len(keys)} complete pairs in {root}")

xs: List[torch.Tensor] = []
label_ids: List[int] = []
for i, key in enumerate(keys, 1):
    pair = pairs[key]
    t = load_and_process_pair(pair["left"], pair["right"], BLUR_RADIUS)
    xs.append(t)
    label_ids.append(name_to_id[key_to_label(key)])
    if i % 200 == 0 or i == len(keys):
        print(f"[INFO] Processed {i}/{len(keys)}")

X = torch.stack(xs)
Y = torch.tensor(label_ids, dtype=torch.long)

catalog_builder: Dict[int, set] = defaultdict(set)
for label_id, description in zip(label_ids, label_names):
    catalog_builder[label_id].add(description)
labels2label_names: Dict[int, List[str]] = {
    label_id: sorted(descriptions)
    for label_id, descriptions in catalog_builder.items()
}

payload = {
    "x": X,
    "y": Y,
    "labels": label_ids,
    "label_names": label_names,
    "labels2label_names": labels2label_names,
    "meta": {
        "blur_radius": BLUR_RADIUS,
        "stack_len": X.shape[0],
        "label_names_count": len(label_names),
        "config_fields": list(CONFIG_FIELDS),
        "source_images": str(IMAGE_DIR),
        "per_channel_tolerance": None,
        "mean_tolerance": None,
    },
}

dataset_io.save_bundle(payload, OUTPUT_DATASET_DIR)

print(f"[DONE] Saved dataset to {OUTPUT_DATASET_DIR.resolve()}  ->  x{tuple(X.shape)}, y{tuple(Y.shape)}")
print(f"[DONE] Example label_names: {label_names[:3]}")
print(f"[DONE] Example label: {label_ids[:3]}")
