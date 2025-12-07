#!/usr/bin/env python3
"""
visualize_image_stack.py

Interactive viewer dedicated to the duplicate groups discovered by `group_duplicates.py`.
This script only understands the new dataset_io bundle format (directory containing
metadata.json plus tensor files) and renders a single row of three panels:

    1. Overlap (left eye tinted blue, right eye tinted red or grayscale mix)
    2. Left eye by itself
    3. Right eye mirrored back to its original orientation

Use the Prev/Next buttons to walk through duplicate groups, optionally toggling a
grayscale view for a clearer comparison of subtle differences.

How to run CLI:
move to the repo root, make sure you have venv activated, then run:
python src/visualize_image_stack.py --bundle-dir data/datasets/dataset-name
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, TextBox

import dataset_io

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT_DIR / "data/datasets/corner-maze-render-base-images-consolidated-ds"
DISPLAY_DEVICE: str = "cpu"


def load_dataset(
    bundle_dir: Path,
    *,
    device: str | torch.device = DISPLAY_DEVICE,
) -> Tuple[torch.Tensor, List[str], List[int]]:
    """
    Load the dataset bundle directory created by dataset_io and return tensors + labels.
    """
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    payload = dataset_io.load_bundle(bundle_dir)

    stack = payload.get("x")
    if not isinstance(stack, torch.Tensor):
        raise TypeError("Dataset payload missing tensor 'x'.")
    stack = stack.detach().to(device)

    size = stack.shape[0]

    labels_map_raw = payload.get("labels2label_names", {})
    labels2label_names: Dict[int, List[str]] = {}
    if isinstance(labels_map_raw, dict):
        for key, value in labels_map_raw.items():
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
            labels2label_names[label_id] = [str(v) for v in entries]

    def _coerce_labels(values: Iterable) -> List[str] | None:
        data = list(values)
        if len(data) != size:
            return None
        return [str(item) for item in data]

    label_names = None
    raw_label_names = payload.get("label_names")
    if isinstance(raw_label_names, torch.Tensor):
        label_names = _coerce_labels(raw_label_names.detach().cpu().tolist())
    elif isinstance(raw_label_names, Sequence):
        label_names = _coerce_labels(raw_label_names)

    if label_names is None:
        raw_ids = payload.get("labels")
        id_list: List[int] | None = None
        if isinstance(raw_ids, torch.Tensor):
            id_list = [int(v) for v in raw_ids.detach().cpu().tolist()]
        elif isinstance(raw_ids, Sequence):
            try:
                id_list = [int(v) for v in raw_ids]
            except (TypeError, ValueError):
                id_list = None
        if id_list is not None and len(id_list) == size:
            label_names = [
                labels2label_names.get(label_id, [str(label_id)])[0]
                for label_id in id_list
            ]

    if label_names is None:
        label_names = [f"sample_{idx}" for idx in range(size)]

    if len(label_names) != size:
        raise ValueError("Label count does not match tensor length.")

    labels_tensor = payload.get("labels")
    if labels_tensor is None:
        label_ids = list(range(size))
    elif isinstance(labels_tensor, torch.Tensor):
        label_ids = [int(v) for v in labels_tensor.detach().cpu().tolist()]
    else:
        label_ids = [int(v) for v in labels_tensor]

    if len(label_ids) != size:
        raise ValueError("Label id count does not match tensor length.")

    return stack, label_names, label_ids


def _read_duplicate_csv(
    csv_path: Path,
    known_labels: Sequence[str],
) -> Tuple[List[List[str]], int]:
    """
    Load duplicate rows from CSV and filter out labels not present in the dataset.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Duplicate CSV not found: {csv_path}")

    label_to_index = {label: idx for idx, label in enumerate(known_labels)}
    groups: List[List[str]] = []
    skipped = 0

    with csv_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            tokens = [token.strip() for token in line.strip().split(",") if token.strip()]
            if not tokens:
                continue
            filtered = [token for token in tokens if token in label_to_index]
            if filtered:
                groups.append(filtered)
            else:
                skipped += 1

    if not groups:
        raise RuntimeError("No duplicate groups matched dataset labels.")

    return groups, skipped


def _groups_from_label_ids(
    label_names: Sequence[str],
    label_ids: Sequence[int],
) -> Tuple[List[List[str]], int]:
    """
    Derive duplicate groups directly from label IDs stored in the dataset bundle.
    """
    if len(label_names) != len(label_ids):
        raise ValueError("label_names and label_ids must be the same length.")

    groups: List[List[str]] = []
    seen: Dict[int, List[str]] = {}
    for label, label_id in zip(label_names, label_ids):
        seen.setdefault(int(label_id), []).append(label)

    for labels in seen.values():
        unique = list(dict.fromkeys(labels))
        if unique:
            groups.append(unique)

    if not groups:
        raise RuntimeError("No samples found in dataset bundle.")

    return groups, 0


def _load_duplicate_groups(
    duplicate_csv: Path | None,
    label_names: Sequence[str],
    label_ids: Sequence[int],
) -> Tuple[List[List[str]], int]:
    """
    Resolve duplicate groups from CSV when available, otherwise fall back to metadata.
    """
    if duplicate_csv is not None:
        csv_path = Path(duplicate_csv).expanduser().resolve()
        if csv_path.exists():
            return _read_duplicate_csv(csv_path, label_names)
        print(f"[WARN] Duplicate CSV not found at {csv_path}; falling back to bundle metadata.")
    return _groups_from_label_ids(label_names, label_ids)


def _default_duplicate_csv(bundle_dir: Path) -> Path | None:
    """
    Guess the duplicate CSV path that accompanies a dataset bundle.
    """
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    candidates = [bundle_dir / f"{bundle_dir.name}.csv"]
    name = bundle_dir.name
    if name.endswith("-ds"):
        candidates.append(bundle_dir / f"{name[:-3]}.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _tint_blue(channel: np.ndarray) -> np.ndarray:
    base = np.repeat(channel[..., None], 3, axis=-1).astype(np.float32)
    base[..., 2] = channel
    base[..., 0] *= 0.25
    base[..., 1] *= 0.25
    return np.clip(base, 0.0, 1.0)


def _tint_red(channel: np.ndarray) -> np.ndarray:
    base = np.repeat(channel[..., None], 3, axis=-1).astype(np.float32)
    base[..., 0] = channel
    base[..., 1] *= 0.25
    base[..., 2] *= 0.25
    return np.clip(base, 0.0, 1.0)


def _style_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor("#151515")
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#7f7f7f")
        spine.set_linewidth(1.0)


def _resize_axis(ax, width_px: float, height_px: float) -> None:
    """Resize an Axes to a specific pixel size while keeping it centered."""
    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches()
    width = width_px / (fig_w * fig.dpi)
    height = height_px / (fig_h * fig.dpi)
    bbox = ax.get_position()
    cx = bbox.x0 + bbox.width / 2
    y0 = bbox.y0 + (bbox.height - height) / 2
    x0 = cx - width / 2
    ax.set_position([x0, y0, width, height])


def show_duplicate_viewer(
    stack: torch.Tensor,
    label_names: Sequence[str],
    label_ids: Sequence[int],
    *,
    duplicate_csv: Path | None = None,
) -> None:
    """
    Launch the duplicate group viewer, stepping through CSV-defined groups.
    """
    groups, skipped = _load_duplicate_groups(duplicate_csv, label_names, label_ids)
    if skipped:
        print(f"Skipped {skipped} CSV rows that no longer exist in the dataset.")

    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    group_idx = 0
    member_idx = 0
    show_gray = False

    fig = plt.figure(figsize=(11, 7.0), constrained_layout=False, facecolor="#151515")
    fig.subplots_adjust(left=0.02, right=0.98)
    outer_grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.18, 0.14, 0.68],
        hspace=0.035,
    )

    info_grid = outer_grid[0, 0].subgridspec(1, 2, width_ratios=[0.7, 0.3], wspace=0.02)
    info_ax = fig.add_subplot(info_grid[0, 0])
    info_ax.axis("off")
    info_text = info_ax.text(
        0.0,
        0.5,
        "",
        va="center",
        ha="left",
        fontsize=11,
        color="#e0e0e0",
    )

    view_toggle_ax = fig.add_subplot(info_grid[0, 1])
    view_toggle_ax.set_xticks([])
    view_toggle_ax.set_yticks([])
    view_toggle_ax.set_facecolor("#f0f0f0")
    _resize_axis(view_toggle_ax, width_px=90, height_px=26)
    view_toggle_btn = Button(view_toggle_ax, "Show Gray")
    view_toggle_btn.label.set_fontsize(9)
    view_toggle_btn.label.set_color("#000000")

    button_grid = outer_grid[1, 0].subgridspec(
        1,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.4],
        wspace=0.02,
    )
    group_prev_ax = fig.add_subplot(button_grid[0, 0])
    group_next_ax = fig.add_subplot(button_grid[0, 1])
    item_prev_ax = fig.add_subplot(button_grid[0, 2])
    item_next_ax = fig.add_subplot(button_grid[0, 3])
    group_jump_ax = fig.add_subplot(button_grid[0, 4])
    for ax in (group_prev_ax, group_next_ax, item_prev_ax, item_next_ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f0f0f0")
        _resize_axis(ax, width_px=80, height_px=30)
    group_jump_ax.set_xticks([])
    group_jump_ax.set_yticks([])
    group_jump_ax.set_facecolor("#f0f0f0")
    _resize_axis(group_jump_ax, width_px=120, height_px=30)

    group_prev_btn = Button(group_prev_ax, "Prev Group")
    group_next_btn = Button(group_next_ax, "Next Group")
    item_prev_btn = Button(item_prev_ax, "Prev Item")
    item_next_btn = Button(item_next_ax, "Next Item")
    for btn in (group_prev_btn, group_next_btn, item_prev_btn, item_next_btn):
        btn.label.set_fontsize(9)
        btn.label.set_color("#000000")

    group_jump_box = TextBox(group_jump_ax, "Group #", initial="1")
    group_jump_box.label.set_fontsize(9)
    group_jump_box.label.set_color("#ffffff")
    group_jump_box.text_disp.set_fontsize(10)
    group_jump_box.text_disp.set_color("#000000")
    group_jump_box.cursor.set_color("#303030")

    jump_box_locked = False

    def _update_jump_box() -> None:
        nonlocal jump_box_locked
        jump_box_locked = True
        group_jump_box.set_val(str(group_idx + 1))
        jump_box_locked = False

    def _handle_jump_submit(text: str) -> None:
        nonlocal jump_box_locked
        if jump_box_locked:
            return
        stripped = text.strip()
        if not stripped:
            _update_jump_box()
            return
        try:
            target = int(stripped)
        except ValueError:
            print(f"[WARN] Invalid group number: {text!r}")
            _update_jump_box()
            return
        total = len(groups)
        if total == 0:
            return
        index = max(0, min(total - 1, target - 1))
        if index != group_idx:
            _set_group(index)
        else:
            _update_jump_box()

    group_jump_box.on_submit(_handle_jump_submit)

    image_grid = outer_grid[2, 0].subgridspec(1, 3, wspace=0.0)
    overlap_ax = fig.add_subplot(image_grid[0, 0])
    left_eye_ax = fig.add_subplot(image_grid[0, 1])
    right_eye_ax = fig.add_subplot(image_grid[0, 2])
    for ax in (overlap_ax, left_eye_ax, right_eye_ax):
        _style_axis(ax)

    def _fetch_images(label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = label_to_index[label]
        left = stack[idx, 0].detach().cpu().numpy()
        right_mirrored = stack[idx, 1].detach().cpu().numpy()
        right_original = np.fliplr(right_mirrored)
        return left, right_mirrored, right_original

    def _render(label: str) -> None:
        left, right_mirrored, right_original = _fetch_images(label)
        overlap_rgb = np.clip(_tint_blue(left) + _tint_red(right_mirrored), 0.0, 1.0)

        for ax in (overlap_ax, left_eye_ax, right_eye_ax):
            ax.cla()
            _style_axis(ax)

        if show_gray:
            mirrored_original = np.fliplr(right_original)
            mixed_gray = np.clip((left + mirrored_original) / 2.0, 0.0, 1.0)
            overlap_ax.imshow(mixed_gray, cmap="gray", vmin=0.0, vmax=1.0)
            overlap_ax.set_title("Overlap (Gray)", fontsize=12, color="#dcdcdc")
            left_eye_ax.imshow(left, cmap="gray", vmin=0.0, vmax=1.0)
            left_eye_ax.set_title("Left Eye (Gray)", fontsize=12, color="#dcdcdc")
            right_eye_ax.imshow(right_original, cmap="gray", vmin=0.0, vmax=1.0)
            right_eye_ax.set_title("Right Eye (Gray)", fontsize=12, color="#dcdcdc")
        else:
            overlap_ax.imshow(overlap_rgb)
            overlap_ax.set_title("Overlap (Left=Blue, Right=Red)", fontsize=12, color="#dcdcdc")
            left_eye_ax.imshow(_tint_blue(left))
            left_eye_ax.set_title("Left Eye (Blue)", fontsize=12, color="#dcdcdc")
            right_eye_ax.imshow(_tint_red(right_original))
            right_eye_ax.set_title("Right Eye (Red, Original)", fontsize=12, color="#dcdcdc")

    def _update_display() -> None:
        group = groups[group_idx]
        label = group[member_idx]
        _render(label)
        info_text.set_text(
            f"Group {group_idx + 1}/{len(groups)} (size {len(group)}) | "
            f"Item {member_idx + 1}/{len(group)}\n{label}"
        )
        _update_jump_box()
        fig.canvas.draw_idle()

    def _set_group(index: int) -> None:
        nonlocal group_idx, member_idx
        total = len(groups)
        if total == 0:
            return
        group_idx = max(0, min(total - 1, index))
        member_idx = 0
        _update_display()

    def _shift_group(delta: int) -> None:
        total = len(groups)
        target = (group_idx + delta) % total
        _set_group(target)

    def _shift_member(delta: int) -> None:
        nonlocal member_idx
        group = groups[group_idx]
        member_idx = (member_idx + delta) % len(group)
        _update_display()

    def _toggle_view(_: str) -> None:
        nonlocal show_gray
        show_gray = not show_gray
        view_toggle_btn.label.set_text("Show RGB" if show_gray else "Show Gray")
        _render(groups[group_idx][member_idx])
        fig.canvas.draw_idle()

    group_prev_btn.on_clicked(lambda _: _shift_group(-1))
    group_next_btn.on_clicked(lambda _: _shift_group(1))
    item_prev_btn.on_clicked(lambda _: _shift_member(-1))
    item_next_btn.on_clicked(lambda _: _shift_member(1))
    view_toggle_btn.on_clicked(_toggle_view)

    _update_display()
    plt.show()


def main(
    bundle_dir: Path = DEFAULT_DATASET_DIR,
    duplicate_csv: Path | None = None,
    device: str | torch.device = DISPLAY_DEVICE,
) -> None:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    stack, label_names, label_ids = load_dataset(bundle_dir, device=device)
    csv_path = duplicate_csv if duplicate_csv is not None else _default_duplicate_csv(bundle_dir)
    show_duplicate_viewer(
        stack,
        label_names,
        label_ids,
        duplicate_csv=csv_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize duplicate stereo groups.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to dataset_io bundle directory.",
    )
    parser.add_argument(
        "--duplicate-csv",
        type=Path,
        default=None,
        help="Optional CSV listing duplicate groups (metadata fallback when omitted).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DISPLAY_DEVICE,
        help="Device identifier for tensor loading (cpu, cuda, mps...).",
    )
    args = parser.parse_args()
    main(bundle_dir=args.bundle_dir, duplicate_csv=args.duplicate_csv, device=args.device)
