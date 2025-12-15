#!/usr/bin/env python3
"""
umap_embeddings.py

User-configurable script to run UMAP on embeddings (Parquet) and produce
both a new Parquet file and a scatter plot image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from itertools import cycle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401
import numpy as np
import pandas as pd

# =======================
# === USER SETTINGS ====
# =======================

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PARQUET: Path = ROOT / "data/tables/stereo-cnn-consolidated-acute-60-attention-embeddings.parquet"
UMAP_COMPONENTS: int = 3
UMAP_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
UMAP_METRIC: str = "euclidean"
UMAP_SEED: int = 42
OUTPUT_PARQUET: Path = EMBEDDINGS_PARQUET.with_stem(EMBEDDINGS_PARQUET.stem + "-umap")
HOVER_LABEL_COLUMN: str = "label_name"
POSES_COLUMN: str = "poses"
NORMALIZE_EMBEDDINGS: bool = False
PATH_LABEL_GROUPS: list = [
    # Example label groups for paths (uncomment and customize as needed)
    # "config1_0_0_north, config1_0_1_north, config1_0_2_north",
    # ["config2_1_0_east", "config2_1_1_east", "config2_1_2_east"],
]


class PathInteraction:
    """Handle hover/click/keyboard navigation for predefined label trajectories."""

    def __init__(self, fig, ax, paths, base_nodes, plot_dims: int):
        self.fig = fig
        self.ax = ax
        self.paths = [path for path in paths if path]
        self.base_nodes = base_nodes or []
        self.plot_dims = plot_dims
        self.path_map = {path[0]["path_index"]: path for path in self.paths}
        self.nodes = [node for path in self.paths for node in path] + list(self.base_nodes)
        self.active_node = None
        self.hover_node = None
        if plot_dims == 2:
            self._fixed_limits = (ax.get_xlim(), ax.get_ylim(), None)
        else:
            self._fixed_limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
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

    def _apply_fixed_limits(self) -> None:
        xlim, ylim, zlim = self._fixed_limits
        if self.plot_dims == 2:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            self.ax.set_xlim3d(xlim)
            self.ax.set_ylim3d(ylim)
            if zlim is not None:
                self.ax.set_zlim3d(zlim)

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
        self._apply_fixed_limits()
        self.fig.canvas.draw_idle()

    def _restore_active_display(self) -> None:
        if self.active_node is not None:
            self._display_node(self.active_node, make_active=True)
        else:
            self.info_text.set_visible(False)
            self.highlight.set_visible(False)
            self._apply_fixed_limits()
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
        if event.key == "left":
            self._step_active(-1)
        elif event.key == "right":
            self._step_active(1)


def main() -> None:
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install 'umap-learn' with `pip install umap-learn`.") from exc

    embeddings_path = EMBEDDINGS_PARQUET.expanduser().resolve()
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    df = pd.read_parquet(embeddings_path)
    if "embedding" not in df.columns:
        raise ValueError("Parquet file missing 'embedding' column.")
    if POSES_COLUMN not in df.columns:
        raise ValueError(f"Parquet file missing '{POSES_COLUMN}' column required for pose data.")

    embedding_matrix = df["embedding"].apply(lambda v: pd.Series(v)).to_numpy()
    if NORMALIZE_EMBEDDINGS:
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding_matrix = embedding_matrix / norms

    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_SEED,
    )
    projected = reducer.fit_transform(embedding_matrix)

    for dim in range(UMAP_COMPONENTS):
        df[f"umap_{dim}"] = projected[:, dim]

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

    plot_dims = min(UMAP_COMPONENTS, 3)
    plot_cols = [f"umap_{dim}" for dim in range(plot_dims)]
    label_to_point: dict[str, tuple[float, ...]] = {}
    if all(col in df.columns for col in plot_cols):
        for _, row in df.iterrows():
            point = tuple(float(row[col]) for col in plot_cols)
            primary_label = str(row.get("label_name", ""))
            if primary_label:
                label_to_point.setdefault(primary_label, point)
            poses = row.get(POSES_COLUMN)
            for pose_label in _iter_pose_labels(poses):
                label_to_point.setdefault(pose_label, point)

    output_path = OUTPUT_PARQUET.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved UMAP data to {output_path}")

    base_nodes = []
    for label, point in label_to_point.items():
        base_nodes.append({"label": label, "point": point, "path_index": None, "node_index": None})

    path_sequences = []

    if plot_dims >= 2:
        if plot_dims == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df["umap_0"], df["umap_1"], s=5, alpha=0.7)
            ax.set_xlabel("UMAP 0")
            ax.set_ylabel("UMAP 1")
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(df["umap_0"], df["umap_1"], df["umap_2"], s=5, alpha=0.7)
            ax.set_xlabel("UMAP 0")
            ax.set_ylabel("UMAP 1")
            ax.set_zlabel("UMAP 2")

        ax.set_title("UMAP Projection")
        fig.tight_layout()

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
                        ax.plot(xs, ys, marker="o", linestyle="-", linewidth=2, color=color)
                    else:
                        xs, ys, zs = zip(*((pt[0], pt[1], pt[2]) for pt in coords))
                        ax.plot(xs, ys, zs, marker="o", linestyle="-", linewidth=2, color=color)

        if path_sequences or base_nodes:
            interaction = PathInteraction(fig, ax, path_sequences, base_nodes, plot_dims)
            setattr(fig, "_path_interaction", interaction)

        plt.show()


if __name__ == "__main__":
    main()
