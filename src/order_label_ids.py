#!/usr/bin/env python3
"""order_label_ids.py

Generate a deterministic ordering of label IDs (based on pose position and config type)
for the embedding parquet table and write the ordered IDs to JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_PARQUET = ROOT_DIR / "data/tables/stereo-cnn-consolidated-acute-60-attentionNoRelu-embeddings.parquet"
OUTPUT_JSON = ROOT_DIR / f"data/json/{Path(EMBEDDINGS_PARQUET.stem).stem}.json"

# Pose ordering template for direction 0 (x, y pairs)
ROW_SEGMENTS = [
    [(1, 1)] + [(x, 2) for x in range(2, 11)] + [(11, 1)],
    [(x, 6) for x in range(2, 11)],
    [(1, 11)] + [(x, 10) for x in range(2, 11)] + [(11, 11)],
]

COLUMN_SEGMENTS = (
    [[(1, 1), (1, 6), (1, 11)]]
    + [[(x, 2), (x, 6), (x, 10)] for x in range(2, 11)]
    + [[(11, 1), (11, 6), (11, 11)]]
)

DIRECTIONS = [0, 1, 2, 3]  # clockwise order
PHASE_PRIORITY = ["trl_cued", "trl_no_cue", "exp", "iti", "other"]


def _phase_kind(label_name: str) -> str:
    label_lower = label_name.lower()
    if label_lower.startswith(("trl_n_n_xx", "trl_e_n_xx", "trl_w_n_xx", "trl_s_n_xx")):
        return "trl_cued"
    if label_lower.startswith("trl_n_x_xx"):
        return "trl_no_cue"
    if label_lower.startswith("exp_x_x_xx"):
        return "exp"
    if label_lower.startswith(("iti_w_x_xx", "iti_w_x_sw", "iti_w_x_nw")):
        return "iti"
    return "other"


def _parse_pose(label_name: str) -> Tuple[int, int, int] | None:
    tokens = label_name.split("_")
    if len(tokens) < 3:
        return None
    try:
        direction = int(tokens[-1])
        y = int(tokens[-2])
        x = int(tokens[-3])
    except ValueError:
        return None
    return x, y, direction


def _collect_entries(df: pd.DataFrame) -> Dict[Tuple[int, int, int], Dict[str, List[Tuple[int, str]]]]:
    pose_entries: Dict[Tuple[int, int, int], Dict[str, List[Tuple[int, str]]]] = {}
    for _, row in df.iterrows():
        label_name = str(row.get("label_name", "")).strip()
        pose = _parse_pose(label_name)
        if pose is None:
            continue
        label_id = int(row.get("label_id", len(pose_entries)))
        phase = _phase_kind(label_name)
        pose_entries.setdefault(pose, {}).setdefault(phase, []).append((label_id, label_name))
    return pose_entries


def _pose_sequence(direction: int) -> Iterable[Tuple[int, int, int]]:
    seen: set[Tuple[int, int]] = set()
    for segment in list(ROW_SEGMENTS) + list(COLUMN_SEGMENTS):
        for x, y in segment:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            yield (x, y, direction)


def _ordered_label_ids(df: pd.DataFrame) -> List[int]:
    pose_entries = _collect_entries(df)

    # Build a deterministic ordering index for every pose (x, y, direction)
    pose_order: Dict[Tuple[int, int, int], int] = {}
    order_idx = 0
    for direction in DIRECTIONS:
        for pose in _pose_sequence(direction):
            if pose not in pose_order:
                pose_order[pose] = order_idx
                order_idx += 1

    fallback_start = order_idx
    fallback_counter = 0

    # Collect entries per phase alongside their pose order index
    phase_entries: Dict[str, List[Tuple[int, str, int]]] = {}
    for pose, entries_by_phase in pose_entries.items():
        pose_idx = pose_order.get(pose)
        if pose_idx is None:
            pose_idx = fallback_start + fallback_counter
            fallback_counter += 1
        for phase, entries in entries_by_phase.items():
            if not entries:
                continue
            phase_entries.setdefault(phase, []).extend(
                (pose_idx, label_name, label_id) for label_id, label_name in entries
            )

    ordered_ids: List[int] = []
    seen_ids: set[int] = set()
    remaining_phases = [phase for phase in phase_entries if phase not in PHASE_PRIORITY]
    for phase in PHASE_PRIORITY + sorted(remaining_phases):
        entries = phase_entries.get(phase)
        if not entries:
            continue
        for pose_idx, label_name, label_id in sorted(entries, key=lambda item: (item[0], item[1])):
            if label_id in seen_ids:
                continue
            ordered_ids.append(label_id)
            seen_ids.add(label_id)

    return ordered_ids


def main() -> None:
    parquet_path = EMBEDDINGS_PARQUET.expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "label_id" not in df.columns or "label_name" not in df.columns:
        raise ValueError("Parquet file must contain 'label_id' and 'label_name' columns.")

    ordered_ids = _ordered_label_ids(df)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(ordered_ids, fh, indent=2)
    print(f"Wrote {len(ordered_ids)} label IDs to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
