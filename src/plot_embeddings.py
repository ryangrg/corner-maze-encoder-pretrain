#!/usr/bin/env python3
"""
plot_embeddings.py

Render precomputed embedding coordinates (Parquet) using the same interactive scatter
logic as umap_embeddings.py, but without recomputing UMAP.
"""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Callable, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PARQUET = ROOT / "data/tables/stereo-cnn-consolidated-acute-60-attention-embeddings-umap.parquet"
POSES_COLUMN = "poses"
HOVER_LABEL_COLUMN: str = "label_name"
COLOR_BY_POSE: bool = False
COLOR_BY_DIRECTION: bool = False
DIRECTION_COLOR_MAP = {
    0: (0.2, 0.4, 0.9),  # North
    1: (0.1, 0.7, 0.3),  # East
    2: (0.95, 0.45, 0.2),  # South
    3: (0.8, 0.2, 0.6),  # West
}
NEUTRAL_GRAY = (0.6, 0.6, 0.6)
PATH_LABEL_GROUPS: list = [
    # Example entries copied from umap_embeddings.py
    # ["trl_e_n_xx_1_1_3", "trl_e_n_xx_1_11_2", "trl_n_n_xx_1_11_2"],
    # ["trl_e_n_xx_6_10_3", "trl_e_n_xx_6_9_3", "trl_e_n_xx_6_8_3", "trl_e_n_xx_6_7_3", "trl_e_n_xx_6_6_3",
    #  "trl_e_n_xx_6_6_0", "trl_e_n_xx_7_6_0", "trl_e_n_xx_8_6_0", "trl_e_n_xx_9_6_0", "trl_e_n_xx_10_6_0",
    #  "trl_e_n_xx_10_6_3", "trl_e_n_xx_10_5_3", "trl_e_n_xx_10_4_3", "trl_e_n_xx_10_3_3", "trl_e_n_xx_10_2_3",
    #  "trl_e_n_xx_11_1_0"],
    #  ["trl_e_n_xx_6_10_3", "trl_e_n_xx_6_9_3", "trl_e_n_xx_6_8_3", "trl_e_n_xx_6_7_3", "trl_e_n_xx_6_6_3",
    #  "trl_e_n_xx_6_6_0", "trl_e_n_xx_7_6_0", "trl_e_n_xx_8_6_0", "trl_e_n_xx_9_6_0", "trl_e_n_xx_10_6_0",
    #  "trl_e_n_xx_10_6_1", "trl_e_n_xx_10_7_1", "trl_e_n_xx_10_8_1", "trl_e_n_xx_10_9_1", "trl_e_n_xx_10_10_1",
    #  "trl_e_n_xx_11_11_1"]
    # ["trl_e_n_xx_6_10_3", "trl_e_n_xx_6_9_3", "trl_e_n_xx_6_8_3", "trl_e_n_xx_6_7_3", "trl_e_n_xx_6_6_3",
    #  "trl_e_n_xx_6_6_2", "trl_e_n_xx_5_6_2", "trl_e_n_xx_4_6_2", "trl_e_n_xx_3_6_2", "trl_e_n_xx_2_6_2",
    #  "trl_e_n_xx_2_6_1","trl_e_n_xx_2_7_1", "trl_e_n_xx_2_8_1", "trl_e_n_xx_2_9_1", "trl_e_n_xx_2_10_1",
    #  "trl_e_n_xx_1_11_2"]
    # ["trl_e_n_xx_6_6_0", "trl_e_n_xx_6_6_1", "trl_e_n_xx_6_6_2", "trl_e_n_xx_6_6_3"],
    #  ["trl_s_n_xx_1_1_0", "trl_s_n_xx_2_2_0", "trl_s_n_xx_3_2_0", "trl_s_n_xx_4_2_0", "trl_s_n_xx_5_2_0",
    #  "trl_s_n_xx_6_2_0", "trl_s_n_xx_7_2_0", "trl_s_n_xx_8_2_0", "trl_s_n_xx_9_2_0", "trl_s_n_xx_10_2_0",
    #  "trl_s_n_xx_11_1_1"],
    #  ["trl_s_n_xx_2_6_0", "trl_s_n_xx_3_6_0", "trl_s_n_xx_4_6_0", "trl_s_n_xx_5_6_0", "trl_s_n_xx_6_6_0",
    #  "trl_s_n_xx_7_6_0", "trl_s_n_xx_8_6_0", "trl_s_n_xx_9_6_0", "trl_s_n_xx_10_6_0"],
    #  ["trl_s_n_xx_1_11_0", "trl_s_n_xx_2_10_0", "trl_s_n_xx_3_10_0", "trl_s_n_xx_4_10_0", "trl_s_n_xx_5_10_0",
    #  "trl_s_n_xx_6_10_0", "trl_s_n_xx_7_10_0", "trl_s_n_xx_8_10_0", "trl_s_n_xx_9_10_0", "trl_s_n_xx_10_10_0",
    #  "trl_s_n_xx_11_11_0"],
    #  ["trl_n_x_xx_6_2_0", "trl_n_x_xx_6_3_0", "trl_n_x_xx_6_4_0", "trl_n_x_xx_6_5_0", "trl_n_x_xx_6_6_0",
    #   "trl_n_x_xx_6_7_0", "trl_n_x_xx_6_8_0", "trl_n_x_xx_6_9_0", "trl_n_x_xx_6_10_0"], 
       ["trl_n_n_xx_2_2_0", "trl_n_n_xx_3_2_0", "trl_n_n_xx_4_2_0", "trl_n_n_xx_5_2_0", "trl_n_n_xx_6_2_0",
        "trl_n_n_xx_6_3_0", "trl_n_n_xx_6_4_0", "trl_n_n_xx_6_5_0", "trl_n_n_xx_6_6_0", "trl_n_n_xx_6_7_0",
        "trl_n_n_xx_6_8_0", "trl_n_n_xx_6_9_0", "trl_n_n_xx_6_10_0", "trl_n_n_xx_5_10_0", "trl_n_n_xx_4_10_0",
        "trl_n_n_xx_3_10_0", "trl_n_n_xx_2_10_0"],
        # ["trl_n_n_xx_2_2_2", "trl_n_n_xx_3_2_2", "trl_n_n_xx_4_2_2", "trl_n_n_xx_5_2_2", "trl_n_n_xx_6_2_2",
        # "trl_n_n_xx_6_3_2", "trl_n_n_xx_6_4_2", "trl_n_n_xx_6_5_2", "trl_n_n_xx_6_6_2", "trl_n_n_xx_6_7_2",
        # "trl_n_n_xx_6_8_2", "trl_n_n_xx_6_9_2", "trl_n_n_xx_6_10_2", "trl_n_n_xx_5_10_2", "trl_n_n_xx_4_10_2",
        # "trl_n_n_xx_3_10_2", "trl_n_n_xx_2_10_2"],
        ["exp_x_x_xx_2_2_0", "exp_x_x_xx_3_2_0", "exp_x_x_xx_4_2_0", "exp_x_x_xx_5_2_0", "exp_x_x_xx_6_2_0",
        "exp_x_x_xx_6_3_0", "exp_x_x_xx_6_4_0", "exp_x_x_xx_6_5_0", "exp_x_x_xx_6_6_0", "exp_x_x_xx_6_7_0",
        "exp_x_x_xx_6_8_0", "exp_x_x_xx_6_9_0", "exp_x_x_xx_6_10_0", "exp_x_x_xx_5_10_0", "exp_x_x_xx_4_10_0",
        "exp_x_x_xx_3_10_0", "exp_x_x_xx_2_10_0"],
        # ["exp_x_x_xx_2_2_2", "exp_x_x_xx_3_2_2", "exp_x_x_xx_4_2_2", "exp_x_x_xx_5_2_2", "exp_x_x_xx_6_2_2",
        # "exp_x_x_xx_6_3_2", "exp_x_x_xx_6_4_2", "exp_x_x_xx_6_5_2", "exp_x_x_xx_6_6_2", "exp_x_x_xx_6_7_2",
        # "exp_x_x_xx_6_8_2", "exp_x_x_xx_6_9_2", "exp_x_x_xx_6_10_2", "exp_x_x_xx_5_10_2", "exp_x_x_xx_4_10_2",
        # "exp_x_x_xx_3_10_2", "exp_x_x_xx_2_10_2"]
]

class PathInteraction:
    """Handle hover/click/keyboard navigation for predefined label trajectories."""

    def __init__(
        self,
        fig,
        ax,
        paths,
        base_nodes,
        plot_dims: int,
        on_select: Callable[[str], None] | None = None,
    ):
        self.fig = fig
        self.ax = ax
        self.paths = [path for path in paths if path]
        self.base_nodes = base_nodes or []
        self.plot_dims = plot_dims
        self.path_map = {path[0]["path_index"]: path for path in self.paths}
        self.nodes = [node for path in self.paths for node in path] + list(self.base_nodes)
        self.active_node = None
        self.hover_node = None
        self._on_select = on_select
        try:
            self.ax.set_autoscale_on(False)
        except AttributeError:
            pass

        if plot_dims == 2:
            self.highlight = ax.scatter([], [], s=60, facecolors="none", edgecolors="red", linewidths=1.5)
        else:
            self.highlight = ax.scatter([], [], [], s=60, facecolors="none", edgecolors="red", linewidths=1.5)

        self.info_text = fig.text(
            0.01,
            0.01,
            "",
            transform=fig.transFigure,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "fc": "w", "alpha": 0.8},
        )
        self.info_text.set_visible(False)
        self.highlight.set_visible(False)

        self.cid_move = fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.cid_click = fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.cid_key = fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _capture_view(self) -> dict:
        if self.plot_dims == 2:
            return {
                "xlim": self.ax.get_xlim(),
                "ylim": self.ax.get_ylim(),
            }
        return {
            "xlim": self.ax.get_xlim3d(),
            "ylim": self.ax.get_ylim3d(),
            "zlim": self.ax.get_zlim3d(),
            "azim": getattr(self.ax, "azim", None),
            "elev": getattr(self.ax, "elev", None),
        }

    def _restore_view(self, state: dict) -> None:
        if not state:
            return
        if self.plot_dims == 2:
            if "xlim" in state:
                self.ax.set_xlim(state["xlim"])
            if "ylim" in state:
                self.ax.set_ylim(state["ylim"])
            return
        elev = state.get("elev")
        azim = state.get("azim")
        if elev is not None or azim is not None:
            self.ax.view_init(elev=elev if elev is not None else self.ax.elev, azim=azim if azim is not None else self.ax.azim)
        if "xlim" in state:
            self.ax.set_xlim3d(state["xlim"])
        if "ylim" in state:
            self.ax.set_ylim3d(state["ylim"])
        if "zlim" in state:
            self.ax.set_zlim3d(state["zlim"])

    def _data_to_display(self, point: tuple[float, ...]) -> tuple[float, float]:
        if self.plot_dims == 2:
            return tuple(self.ax.transData.transform((point[0], point[1])))
        tx, ty, _tz = proj3d.proj_transform(point[0], point[1], point[2], self.ax.get_proj())
        return tuple(self.ax.transData.transform((tx, ty)))

    def _find_node_near(self, event, radius: float = 10.0):
        if event.inaxes != self.ax or event.x is None or event.y is None:
            return None
        mouse = np.array([event.x, event.y], dtype=float)
        closest = None
        best_dist = radius
        for node in self.nodes:
            disp = np.array(self._data_to_display(node["point"]))
            dist = float(np.linalg.norm(disp - mouse))
            if dist < best_dist:
                best_dist = dist
                closest = node
        return closest

    def _update_highlight(self, point: tuple[float, ...]) -> None:
        if self.plot_dims == 2:
            self.highlight.set_offsets(np.array(point[:2]).reshape(1, 2))
        else:
            self.highlight._offsets3d = ([point[0]], [point[1]], [point[2]])  # type: ignore[attr-defined]
        self.highlight.set_visible(True)

    def _display_node(self, node, make_active: bool) -> None:
        view_state = self._capture_view()
        point = node["point"]
        path_index = node.get("path_index")
        step_info = ""
        if path_index is not None:
            path_nodes = self.path_map.get(path_index, [])
            step_info = f" (path {path_index + 1}, step {node['node_index'] + 1}/{len(path_nodes)})"
        info_text = f"{node['label']}{step_info}"
        self.info_text.set_text(info_text)
        self.info_text.set_visible(True)
        self._update_highlight(point)
        if make_active:
            self.active_node = node
            if self._on_select is not None:
                self._on_select(node["label"])
        self._restore_view(view_state)
        self.fig.canvas.draw_idle()

    def _restore_active_display(self) -> None:
        if self.active_node is not None:
            self._display_node(self.active_node, make_active=True)
        else:
            self.info_text.set_visible(False)
            self.highlight.set_visible(False)
            self.fig.canvas.draw_idle()

    def _on_mouse_move(self, event) -> None:
        node = self._find_node_near(event)
        if node is not None:
            if node is not self.hover_node:
                self.hover_node = node
                self._display_node(node, make_active=False)
        else:
            self.hover_node = None
            self._restore_active_display()

    def _on_click(self, event) -> None:
        if event.button != 1:
            return
        node = self._find_node_near(event)
        if node is not None:
            self._display_node(node, make_active=True)

    def _step_active(self, direction: int) -> None:
        if self.active_node is None:
            return
        path_index = self.active_node.get("path_index")
        if path_index is None:
            return
        path_nodes = self.path_map.get(path_index, [])
        if not path_nodes:
            return
        next_idx = self.active_node["node_index"] + direction
        next_idx = max(0, min(len(path_nodes) - 1, next_idx))
        if next_idx == self.active_node["node_index"]:
            return
        next_node = path_nodes[next_idx]
        self._display_node(next_node, make_active=True)

    def _on_key_press(self, event) -> None:
        if event.key == "up":
            self._step_active(-1)
        elif event.key == "down":
            self._step_active(1)


def _iter_pose_labels(value: object) -> Iterable[str]:
    if isinstance(value, (list, tuple, set)):
        for entry in value:
            entry_str = str(entry).strip()
            if entry_str:
                yield entry_str
    elif isinstance(value, np.ndarray):
        for entry in value.tolist():
            entry_str = str(entry).strip()
            if entry_str:
                yield entry_str
    elif isinstance(value, str):
        entry = value.strip()
        if entry:
            yield entry


def _coerce_image_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
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
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _build_label_points(
    df: pd.DataFrame,
    plot_cols: list[str],
) -> Tuple[dict[str, tuple[float, ...]], dict[str, Tuple[np.ndarray, np.ndarray]]]:
    label_to_point: dict[str, tuple[float, ...]] = {}
    label_to_images: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if not all(col in df.columns for col in plot_cols):
        missing = ", ".join(plot_cols)
        raise ValueError(f"Parquet file missing required columns: {missing}")
    for _, row in df.iterrows():
        point = tuple(float(row[col]) for col in plot_cols)
        primary_label = str(row.get("label_name", ""))
        left_eye = _coerce_image_array(row.get("left_eye_img"))
        right_eye = _coerce_image_array(row.get("right_eye_img"))
        image_pair = None
        if left_eye is not None and right_eye is not None:
            image_pair = (left_eye, right_eye)

        if primary_label:
            label_to_point.setdefault(primary_label, point)
            if image_pair is not None:
                label_to_images.setdefault(primary_label, image_pair)
        poses = row.get(POSES_COLUMN)
        for pose_label in _iter_pose_labels(poses):
            label_to_point.setdefault(pose_label, point)
            if image_pair is not None:
                label_to_images.setdefault(pose_label, image_pair)
    return label_to_point, label_to_images


def _extract_pose_coords(label: str) -> Tuple[int, int] | None:
    tokens = str(label).split("_")
    if len(tokens) < 3:
        return None
    try:
        x = int(tokens[-3])
        y = int(tokens[-2])
    except ValueError:
        return None
    return x, y


def _pose_color_array(labels: Iterable[str]) -> np.ndarray | None:
    coords = [_extract_pose_coords(label) for label in labels]
    valid = [coord for coord in coords if coord is not None]
    if not valid:
        return None
    x_vals = np.array([coord[0] for coord in valid], dtype=float)
    y_vals = np.array([coord[1] for coord in valid], dtype=float)
    x_min, x_span = x_vals.min(), max(x_vals.max() - x_vals.min(), 1.0)
    y_min, y_span = y_vals.min(), max(y_vals.max() - y_vals.min(), 1.0)

    colors = []
    for coord in coords:
        if coord is None:
            colors.append((0.5, 0.5, 0.5))
            continue
        x_norm = (coord[0] - x_min) / x_span
        y_norm = (coord[1] - y_min) / y_span
        hue = x_norm
        saturation = 0.35 + 0.65 * (1.0 - abs(0.5 - y_norm) * 2.0)
        value = 0.5 + 0.5 * y_norm
        hsv = (
            hue,
            max(0.0, min(1.0, saturation)),
            max(0.0, min(1.0, value)),
        )
        colors.append(mcolors.hsv_to_rgb(hsv))
    return np.array(colors)


def _direction_color_array(labels: Iterable[str]) -> np.ndarray:
    colors = []
    for raw_label in labels:
        tokens = [token.strip().lower() for token in str(raw_label).split("_") if token]
        if len(tokens) < 4:
            colors.append(NEUTRAL_GRAY)
            continue
        phase = tokens[0]
        if phase != "trl":
            colors.append(NEUTRAL_GRAY)
            continue
        cue_token = tokens[2]
        if cue_token == "x":
            colors.append(NEUTRAL_GRAY)
            continue
        direction_token = tokens[-1]
        try:
            direction_idx = int(direction_token)
        except ValueError:
            colors.append(NEUTRAL_GRAY)
            continue
        colors.append(DIRECTION_COLOR_MAP.get(direction_idx, NEUTRAL_GRAY))
    return np.array(colors, dtype=float)


def main() -> None:
    parquet_path = EMBEDDINGS_PARQUET.expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Embedding coordinates Parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    umap_cols = sorted([col for col in df.columns if col.startswith("umap_")])
    if not umap_cols:
        raise ValueError("Parquet file does not contain columns prefixed with 'umap_'.")
    plot_dims = min(len(umap_cols), 3)
    plot_cols = [f"umap_{idx}" for idx in range(plot_dims)]

    label_to_point, label_to_images = _build_label_points(df, plot_cols)
    base_nodes = [{"label": label, "point": point, "path_index": None, "node_index": None} for label, point in label_to_point.items()]
    path_sequences = []

    fig = plt.figure(figsize=(12.5, 6.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 1.0], wspace=0.08)

    scatter_colors = None
    if COLOR_BY_DIRECTION and HOVER_LABEL_COLUMN in df.columns:
        scatter_colors = _direction_color_array(df[HOVER_LABEL_COLUMN])
    elif COLOR_BY_POSE and HOVER_LABEL_COLUMN in df.columns:
        scatter_colors = _pose_color_array(df[HOVER_LABEL_COLUMN])

    if plot_dims == 2:
        scatter_ax = fig.add_subplot(grid[0, 0])
        scatter_ax.scatter(
            df["umap_0"],
            df["umap_1"],
            s=5,
            alpha=0.7,
            c=scatter_colors,
        )
        scatter_ax.set_xlabel("UMAP 0")
        scatter_ax.set_ylabel("UMAP 1")
    else:
        scatter_ax = fig.add_subplot(grid[0, 0], projection="3d")
        scatter_ax.scatter(
            df["umap_0"],
            df["umap_1"],
            df["umap_2"],
            s=5,
            alpha=0.7,
            c=scatter_colors,
        )
        scatter_ax.set_xlabel("UMAP 0")
        scatter_ax.set_ylabel("UMAP 1")
        scatter_ax.set_zlabel("UMAP 2")

    image_left_ax = fig.add_subplot(grid[0, 1])
    image_right_ax = fig.add_subplot(grid[0, 2])
    for display_ax, title in ((image_left_ax, "Left Eye"), (image_right_ax, "Right Eye")):
        display_ax.set_xticks([])
        display_ax.set_yticks([])
        display_ax.set_title(title)
        display_ax.axis("off")

    scatter_ax.set_title("UMAP Projection")

    def _reset_image_axis(axis, title: str) -> None:
        axis.cla()
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.axis("off")

    def _update_eye_images(label: str | None) -> None:
        _reset_image_axis(image_left_ax, "Left Eye")
        _reset_image_axis(image_right_ax, "Right Eye")
        if not label:
            for axis in (image_left_ax, image_right_ax):
                axis.text(
                    0.5,
                    0.5,
                    "Select a point",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=9,
                    color="#cccccc",
                )
            fig.canvas.draw_idle()
            return
        pair = label_to_images.get(label)
        if pair is None:
            image_left_ax.text(
                0.5,
                0.5,
                "No images\navailable",
                ha="center",
                va="center",
                transform=image_left_ax.transAxes,
                fontsize=9,
                color="#cccccc",
            )
            image_right_ax.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                transform=image_right_ax.transAxes,
                fontsize=8,
                color="#aaaaaa",
            )
            fig.canvas.draw_idle()
            return
        left_eye, right_eye = pair
        image_left_ax.imshow(left_eye, cmap="gray", vmin=0.0, vmax=1.0)
        image_right_ax.imshow(right_eye, cmap="gray", vmin=0.0, vmax=1.0)
        fig.canvas.draw_idle()

    _update_eye_images(None)

    if PATH_LABEL_GROUPS:
        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for path_idx, entry in enumerate(PATH_LABEL_GROUPS):
            if isinstance(entry, str):
                labels = [token.strip() for token in entry.split(",") if token.strip()]
            elif isinstance(entry, (list, tuple, set)):
                labels = [str(token).strip() for token in entry if str(token).strip()]
            else:
                continue

            if not labels:
                path_sequences.append([])
                continue

            missing = []
            sequence = []
            for label in labels:
                point = label_to_point.get(label)
                if point is None:
                    missing.append(label)
                    continue
                sequence.append(
                    {
                        "label": label,
                        "point": point,
                        "path_index": path_idx,
                        "node_index": len(sequence),
                    }
                )

            path_sequences.append(sequence)

            if missing:
                print(f"[WARN] Path labels not found: {', '.join(missing)}")
            if len(sequence) >= 2:
                coords = [node["point"] for node in sequence]
                color = next(colors)
                if plot_dims == 2:
                    xs, ys = zip(*((pt[0], pt[1]) for pt in coords))
                    scatter_ax.plot(xs, ys, marker="o", linestyle="-", linewidth=2, color=color)
                else:
                    xs, ys, zs = zip(*((pt[0], pt[1], pt[2]) for pt in coords))
                    scatter_ax.plot(xs, ys, zs, marker="o", linestyle="-", linewidth=2, color=color)

    if path_sequences or base_nodes:
        interaction = PathInteraction(
            fig,
            scatter_ax,
            path_sequences,
            base_nodes,
            plot_dims,
            on_select=_update_eye_images,
        )
        setattr(fig, "_path_interaction", interaction)

    plt.show()


if __name__ == "__main__":
    main()
