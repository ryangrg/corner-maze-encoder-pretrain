#!/usr/bin/env python3
"""
embedding_autocorrelogram.py

Utility script (non-CLI) to scan the data/tables directory for embedding tables
and display cosine-similarity correlogram heat maps (every ID vs every ID) using Matplotlib.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT_DIR / "data/tables"
TARGET_TABLE = TABLE_DIR / "corner-maze-render-base-images-consolidated-acute-ds-embeddings-32.parquet"
EMBEDDING_COLUMN = "embedding"
LABEL_NAME_COLUMN = "label_name"
LABEL_ID_COLUMN = "label_id"
LEFT_IMAGE_COLUMN = "left_eye_img"
RIGHT_IMAGE_COLUMN = "right_eye_img"
LABEL_LIST_PATH = TABLE_DIR / "corner-maze-render-base-images-consolidated-acute-ds-embedding-32-label-order.json"
with LABEL_LIST_PATH.open("r", encoding="utf-8") as fh:
    label_id_list = json.load(fh)
ORDER_LABEL_IDS: list[int] = label_id_list
APPLY_LABEL_ORDER: bool = False
MIN_VECTOR_NORM = 1e-9


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".parq", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".feather", ".ftr"}:
        return pd.read_feather(path)
    if suffix in {".pkl", ".pickle", ".pd"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table type for {path.name}")


def _coerce_image_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except Exception:
                pass
    arr: np.ndarray | None = None
    if isinstance(value, np.ndarray):
        try:
            arr = value.astype(np.float32, copy=False)
        except ValueError:
            arr = None
    if arr is None:
        try:
            arr = np.asarray(value, dtype=np.float32)
        except (ValueError, TypeError):
            arrays = []
            try:
                for entry in value:  # type: ignore[operator]
                    arrays.append(np.asarray(entry, dtype=np.float32))
            except Exception:
                return None
            if not arrays:
                return None
            try:
                arr = np.stack(arrays, axis=0)
            except ValueError:
                max_len = max(a.shape[0] for a in arrays)
                padded = []
                for a in arrays:
                    pad_width = [(0, max_len - a.shape[0])]
                    if a.ndim == 2:
                        pad_width.append((0, 0))
                    padded.append(np.pad(a, pad_width, mode="constant"))
                arr = np.stack(padded, axis=0)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        if side * side == arr.size:
            arr = arr.reshape(side, side)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    max_val = float(np.max(arr))
    if max_val > 1.0:
        arr = arr / 255.0
    return arr


def _load_embedding_matrix(path: Path) -> tuple[np.ndarray, list[str], list[int], dict[str, tuple[np.ndarray, np.ndarray]]]:
    df = _read_table(path)
    if EMBEDDING_COLUMN not in df.columns:
        raise ValueError(f"Table {path.name} missing '{EMBEDDING_COLUMN}' column.")
    column = df[EMBEDDING_COLUMN].to_list()
    if not column:
        raise ValueError(f"Table {path.name} does not contain any embeddings.")
    label_source = df[LABEL_NAME_COLUMN].tolist() if LABEL_NAME_COLUMN in df.columns else None
    id_source = df[LABEL_ID_COLUMN].tolist() if LABEL_ID_COLUMN in df.columns else None
    left_images = df[LEFT_IMAGE_COLUMN].tolist() if LEFT_IMAGE_COLUMN in df.columns else None
    right_images = df[RIGHT_IMAGE_COLUMN].tolist() if RIGHT_IMAGE_COLUMN in df.columns else None
    embeddings = []
    labels: list[str] = []
    label_ids: list[int] = []
    image_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for idx, entry in enumerate(column):
        array = np.asarray(entry, dtype=np.float32).reshape(1, -1)
        if array.size == 0:
            raise ValueError(f"Row {idx} in {path.name} has an empty embedding.")
        embeddings.append(array)
        label = None
        if label_source is not None and idx < len(label_source):
            raw = label_source[idx]
            if raw is not None:
                text = str(raw).strip()
                if text:
                    label = text
        if label is None:
            label = f"row_{idx}"
        labels.append(label)
        if id_source is not None and idx < len(id_source):
            try:
                label_ids.append(int(id_source[idx]))
            except (TypeError, ValueError):
                label_ids.append(idx)
        else:
            label_ids.append(idx)
        if left_images is not None and right_images is not None:
            left_arr = _coerce_image_array(left_images[idx])
            right_arr = _coerce_image_array(right_images[idx])
            if left_arr is not None and right_arr is not None:
                image_map[label] = (left_arr, right_arr)
    return np.vstack(embeddings), labels, label_ids, image_map


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings matrix must be 2-D (samples x dims).")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, MIN_VECTOR_NORM)
    normalized = embeddings / norms
    matrix = np.matmul(normalized, normalized.T)
    return np.clip(matrix, -1.0, 1.0)


def _apply_ordering(
    matrix: np.ndarray,
    labels: list[str],
    label_ids: list[int],
) -> tuple[np.ndarray, list[str], list[int]]:
    if not APPLY_LABEL_ORDER or not ORDER_LABEL_IDS:
        return matrix, labels, label_ids
    order_map = {label_id: idx for idx, label_id in enumerate(ORDER_LABEL_IDS)}
    indexed = list(range(len(labels)))
    indexed.sort(
        key=lambda i: (
            order_map.get(label_ids[i], len(order_map)),
            label_ids[i],
        )
    )
    permuted_matrix = matrix[np.ix_(indexed, indexed)]
    ordered_labels = [labels[i] for i in indexed]
    ordered_ids = [label_ids[i] for i in indexed]
    return permuted_matrix, ordered_labels, ordered_ids


def show_heatmap(
    labels: Sequence[str],
    matrix: np.ndarray,
    title: str,
    image_map: dict[str, tuple[np.ndarray, np.ndarray]],
) -> plt.Figure:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cosine similarity matrix must be square.")
    count = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cax = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("Sample (columns)")
    ax.set_ylabel("Sample (rows)")
    ax.set_title(title)
    ax.grid(False)
    if count <= 40:
        ticks = np.arange(count)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(cax, ax=ax, label="Cosine Similarity")

    info_ax = fig.add_axes([0.9, 0.8, 0.08, 0.18], facecolor="#f9f9f9")
    info_ax.set_xticks([])
    info_ax.set_yticks([])
    info_ax.set_frame_on(True)
    info_text = info_ax.text(
        0.05,
        0.9,
        "Click a cell to view label pair.",
        ha="left",
        va="top",
        fontsize=8,
        color="#202020",
        wrap=True,
    )

    highlight = ax.scatter([], [], marker="s", s=60, facecolors="none", edgecolors="#00ff00", linewidths=1.5)
    row_left_ax = fig.add_axes([0.9, 0.58, 0.08, 0.18])
    row_right_ax = fig.add_axes([0.9, 0.4, 0.08, 0.18])
    col_left_ax = fig.add_axes([0.9, 0.22, 0.08, 0.18])
    col_right_ax = fig.add_axes([0.9, 0.04, 0.08, 0.18])

    def _prepare_axis(axis, title: str) -> None:
        axis.cla()
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, fontsize=7)
        axis.axis("off")

    def _display_images(label: str, left_axis, right_axis) -> None:
        _prepare_axis(left_axis, f"{label}\nLeft")
        _prepare_axis(right_axis, f"{label}\nRight")
        pair = image_map.get(label)
        if pair is None:
            left_axis.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8, color="#888888")
            right_axis.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8, color="#888888")
            return
        left_img, right_img = pair
        left_axis.imshow(left_img, cmap="gray", vmin=0.0, vmax=1.0)
        right_axis.imshow(right_img, cmap="gray", vmin=0.0, vmax=1.0)
    current_row = [0]
    current_col = [0]

    def _update_info(row_idx: int, col_idx: int) -> None:
        if not (0 <= row_idx < count and 0 <= col_idx < count):
            return
        left_label = labels[row_idx]
        right_label = labels[col_idx]
        value = matrix[row_idx, col_idx]
        info_text.set_text(f"Row: {left_label}\nCol: {right_label}\nCosine: {value:.4f}")
        highlight.set_offsets(np.array([[col_idx, row_idx]], dtype=float))
        highlight.set_visible(True)
        _display_images(left_label, row_left_ax, row_right_ax)
        _display_images(right_label, col_left_ax, col_right_ax)
        current_row[0] = row_idx
        current_col[0] = col_idx

    def _on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        col = int(np.clip(round(event.xdata), 0, count - 1))
        row = int(np.clip(round(event.ydata), 0, count - 1))
        _update_info(row, col)
        fig.canvas.draw_idle()

    def _on_key(event) -> None:
        row = current_row[0]
        col = current_col[0]
        if event.key == "up":
            row = max(0, row - 1)
        elif event.key == "down":
            row = min(count - 1, row + 1)
        elif event.key == "left":
            col = max(0, col - 1)
        elif event.key == "right":
            col = min(count - 1, col + 1)
        else:
            return
        _update_info(row, col)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    fig.canvas.mpl_connect("button_press_event", _on_click)

    fig.tight_layout()
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    return fig


def iter_embedding_tables(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Table directory not found: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".parquet", ".parq", ".pq", ".feather", ".ftr", ".pkl", ".pickle", ".pd"}
    )


def _selected_tables() -> list[Path]:
    if TARGET_TABLE is None:
        return list(iter_embedding_tables(TABLE_DIR))
    target = TARGET_TABLE.expanduser()
    if not target.is_absolute():
        target = (TABLE_DIR / target).resolve()
    else:
        target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target table not found: {target}")
    if target.is_dir():
        return list(iter_embedding_tables(target))
    return [target]


def main() -> None:
    tables = _selected_tables()
    if not tables:
        raise RuntimeError(f"No embedding tables found in {TABLE_DIR}")
    figures: list[plt.Figure] = []
    for table_path in tables:
        embeddings, labels, label_ids, image_map = _load_embedding_matrix(table_path)
        if embeddings.shape[0] != len(labels):
            raise ValueError("Label count does not match embedding count.")
        matrix = cosine_similarity_matrix(embeddings)
        matrix, labels, label_ids = _apply_ordering(matrix, labels, label_ids)
        title = f"Cosine Similarity Correlogram: {table_path.stem}"
        fig = show_heatmap(labels, matrix, title, image_map)
        figures.append(fig)
        print(f"[INFO] Displayed correlogram for {table_path}")
    if figures:
        plt.show()


if __name__ == "__main__":
    main()
