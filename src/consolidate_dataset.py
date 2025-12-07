#!/usr/bin/env python3
"""
create_median_group_dataset.py

Build a new dataset bundle by averaging (via per-pixel median) groups of labels
listed in a CSV. Each row of the CSV specifies label names that should be merged
into a single representative image.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import dataset_io

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT_DIR / "data/datasets/corner-maze-render-base-images-regrouped-ds"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data/datasets/corner-maze-render-base-images-consolidated-ds"


def _load_groups(csv_path: Path) -> List[List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Grouping CSV not found: {csv_path}")
    groups: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            labels = [token.strip() for token in row if token.strip()]
            if labels:
                groups.append(labels)
    return groups


def _median_image(images: torch.Tensor) -> torch.Tensor:
    # images shape: (k, 2, H, W)
    median_vals = torch.median(images, dim=0).values
    return median_vals


def build_median_bundle(
    csv_path: Path | None = None,
    source_dataset: Path = DEFAULT_DATASET_DIR,
    output_dataset: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Path | int | bool]:
    source_dir = Path(source_dataset).expanduser().resolve()
    payload = dataset_io.load_bundle(source_dir)
    x_tensor: torch.Tensor = payload["x"].float()
    label_names: Sequence[str] = [str(name) for name in payload["label_names"]]
    label_ids = payload["labels"]
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.tolist()
    label_ids = [int(v) for v in label_ids]

    name_to_index = {name: idx for idx, name in enumerate(label_names)}

    if csv_path is None:
        csv_path = source_dir / f"{source_dir.name.removesuffix("-ds")}.csv"

    csv_path = Path(csv_path).expanduser().resolve()
    raw_groups = _load_groups(csv_path)
    group_indices: List[List[int]] = []
    missing_labels: List[str] = []
    used_indices = set()

    for labels in raw_groups:
        indices = []
        for label in labels:
            idx = name_to_index.get(label)
            if idx is None:
                missing_labels.append(label)
                continue
            indices.append(idx)
        if indices:
            group_indices.append(indices)
            used_indices.update(indices)

    # Include leftover samples as singleton groups.
    for idx in range(len(label_names)):
        if idx not in used_indices:
            group_indices.append([idx])

    new_images: List[torch.Tensor] = []
    new_label_ids: List[int] = []
    new_label_names: List[str] = []
    group_meta: List[Dict[str, object]] = []

    for indices in group_indices:
        group_images = x_tensor[indices]
        median_img = _median_image(group_images)
        primary_idx = indices[0]
        primary_label = label_names[primary_idx]
        primary_id = label_ids[primary_idx]

        new_images.append(median_img)
        new_label_ids.append(primary_id)
        new_label_names.append(primary_label)
        group_meta.append(
            {
                "primary_label": primary_label,
                "label_id": primary_id,
                "members": [label_names[i] for i in indices],
                "count": len(indices),
            }
        )

    new_x = torch.stack(new_images)
    new_y = torch.tensor(new_label_ids, dtype=torch.long)

    existing_map = payload.get("labels2label_names", {})
    labels2label_names: Dict[int, set] = defaultdict(set)
    if isinstance(existing_map, dict):
        for key, value in existing_map.items():
            try:
                label_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, (list, tuple, set)):
                entries = value
            elif value is None:
                entries = []
            else:
                entries = [value]
            for item in entries:
                labels2label_names[label_id].add(str(item))

    for entry in group_meta:
        label_id = entry["label_id"]
        for desc in entry["members"]:
            labels2label_names[label_id].add(str(desc))

    labels2label_names = {
        label_id: sorted(descriptions)
        for label_id, descriptions in labels2label_names.items()
    }

    new_payload = {
        "x": new_x,
        "y": new_y,
        "labels": new_label_ids,
        "label_names": new_label_names,
        "labels2label_names": labels2label_names,
        "meta": {
            **payload.get("meta", {}),
            "source_bundle": str(source_dir),
            "group_meta": group_meta,
            "stack_len": len(new_images),
            "label_names_count": len(new_label_names),
            "csv_used": str(Path(csv_path).expanduser().resolve()),
        },
    }

    output_dir = Path(output_dataset).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_io.save_bundle(new_payload, output_dir)
    csv_target = output_dir / f"{output_dir.name}.csv"
    csv_target.write_text(Path(csv_path).expanduser().read_text())
    print(
        f"Median-grouped dataset saved to {output_dir} "
        f"(samples={len(new_images)}, skipped_missing={len(missing_labels)})"
    )
    return {
        "output_path": output_dir,
        "groups": len(group_indices),
        "missing_labels": missing_labels,
    }


if __name__ == "__main__":
    build_median_bundle()
