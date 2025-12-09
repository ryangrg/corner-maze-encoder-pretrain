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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # ensures 3D proj registered
import numpy as np
import pandas as pd

# =======================
# === USER SETTINGS ====
# =======================

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PARQUET: Path = ROOT / "data/tables/corner-maze-render-base-images-consolidated-dull-ds-embeddings-20251208165058.parquet"
UMAP_COMPONENTS: int = 3
UMAP_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
UMAP_METRIC: str = "euclidean"
UMAP_SEED: int = 42
OUTPUT_PARQUET: Path = ROOT / "data/tables/corner-maze-render-base-images-consolidated-dull-ds-umap.parquet"
OUTPUT_PLOT: Path = ROOT / "data/plots/corner-maze-render-base-images-consolidated-dull-ds-umap.png"
HOVER_LABEL_COLUMN: str = "label_name"
POSES_COLUMN: str = "poses"
PATH_LABEL_GROUPS: list = [
    # Example entries:
    # ["trl_e_n_xx_1_1_3", "trl_e_n_xx_1_11_2", "trl_n_n_xx_1_11_2"],
    # "iti_w_x_nw_11_1_0, iti_w_x_sw_1_1_3, exp_x_x_xx_11_11_1",
    # ["exp_x_x_xx_2_2_0", "exp_x_x_xx_3_2_0", "exp_x_x_xx_4_2_0", "exp_x_x_xx_5_2_0",
    #  "exp_x_x_xx_6_2_0", "exp_x_x_xx_7_2_0", "exp_x_x_xx_8_2_0", "exp_x_x_xx_9_2_0",
    #  "exp_x_x_xx_10_2_0"],
     ["trl_n_n_xx_2_2_0", "trl_n_n_xx_3_2_0", "trl_n_n_xx_4_2_0", "trl_n_n_xx_5_2_0",
     "trl_n_n_xx_6_2_0", "trl_n_n_xx_7_2_0", "trl_n_n_xx_8_2_0", "trl_n_n_xx_9_2_0",
     "trl_n_n_xx_10_2_0"],
    #  ["trl_s_n_xx_2_10_0", "trl_s_n_xx_3_10_0", "trl_s_n_xx_4_10_0", "trl_s_n_xx_5_10_0",
    #  "trl_s_n_xx_6_10_0", "trl_s_n_xx_7_10_0", "trl_s_n_xx_8_10_0", "trl_s_n_xx_9_10_0",
    #  "trl_s_n_xx_10_10_0"],
    #  ["trl_s_n_xx_6_2_0", "trl_s_n_xx_6_3_0", "trl_s_n_xx_6_4_0", "trl_s_n_xx_6_5_0",
    #  "trl_s_n_xx_6_6_0", "trl_s_n_xx_6_7_0", "trl_s_n_xx_6_8_0", "trl_s_n_xx_6_9_0",
    #  "trl_s_n_xx_6_10_0"],
     ["trl_s_n_xx_2_6_0", "trl_s_n_xx_3_6_0", "trl_s_n_xx_4_6_0", "trl_s_n_xx_5_6_0",
     "trl_s_n_xx_6_6_0"]
     
]


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

    if plot_dims >= 2:
        plot_path = OUTPUT_PLOT.expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)

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
        fig.savefig(plot_path, dpi=200)
        print(f"Saved UMAP plot to {plot_path}")

        if PATH_LABEL_GROUPS:
            colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            for entry in PATH_LABEL_GROUPS:
                if isinstance(entry, str):
                    labels = [token.strip() for token in entry.split(",") if token.strip()]
                elif isinstance(entry, (list, tuple, set)):
                    labels = [str(token).strip() for token in entry if str(token).strip()]
                else:
                    continue
                if len(labels) < 2:
                    continue
                points = []
                missing = []
                for name in labels:
                    point = label_to_point.get(name)
                    if point is None:
                        missing.append(name)
                        continue
                    points.append(point)
                if missing:
                    print(f"[WARN] Path labels not found: {', '.join(missing)}")
                if len(points) >= 2:
                    color = next(colors)
                    if plot_dims == 2:
                        xs, ys = zip(*((p[0], p[1]) for p in points))
                        ax.plot(xs, ys, marker="o", linestyle="-", linewidth=2, color=color)
                    else:
                        xs, ys, zs = zip(*((p[0], p[1], p[2]) for p in points))
                        ax.plot(xs, ys, zs, marker="o", linestyle="-", linewidth=2, color=color)

        plt.show()


if __name__ == "__main__":
    main()
