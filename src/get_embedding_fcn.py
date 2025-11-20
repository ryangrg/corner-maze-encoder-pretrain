"""
Utility for loading a parquet of embeddings and providing a quick
pose→embedding lookup by the pose string used in the table
(e.g., "0_1_1_1").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import pandas as pd
import torch

# Default to the known export location; callers can override in load_pose_lookup.
DEFAULT_TABLE_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "tables"
    / "dedup-all-images-dataset-embeddings-20251116154229.parquet"
)


@dataclass
class PoseEmbeddingLookup:
    pose_to_embedding: Dict[str, torch.Tensor]
    valid_poses: Set[str]

    def __contains__(self, pose: object) -> bool:
        return str(pose) in self.pose_to_embedding

    def get(self, pose: str) -> torch.Tensor:
        """
        Return the embedding tensor for the given pose string.
        Raises KeyError if the pose is not present in the table or pose list.
        """
        pose_str = str(pose)
        if pose_str not in self.valid_poses:
            raise KeyError(f"Pose {pose_str!r} not found in pose list.")
        if pose_str not in self.pose_to_embedding:
            raise KeyError(f"Pose {pose_str!r} has no embedding in the table.")
        return self.pose_to_embedding[pose_str]

    def __call__(self, pose: str) -> torch.Tensor:
        return self.get(pose)


def _collect_valid_poses(series: Iterable[object]) -> Set[str]:
    known: Set[str] = set()
    for item in series:
        if isinstance(item, (list, tuple, set)):
            known.update(map(str, item))
        elif pd.isna(item):
            continue
        else:
            known.add(str(item))
    return known


def load_pose_lookup(
    table_path: Optional[Path | str] = None,
    device: Optional[str] = None,
    as_tensor: bool = True,
) -> PoseEmbeddingLookup:
    """
    Load the parquet table of embeddings and build a pose→embedding lookup.

    Args:
        table_path: Path to the parquet produced by export_embeddings.py.
        device: Optional device string ("cpu", "cuda", "mps") to place tensors (only applies if as_tensor=True).
        as_tensor: When True (default) convert embedding lists to torch.Tensor. When False, keep them as Python lists.
    """
    target = Path(table_path or DEFAULT_TABLE_PATH).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Parquet file not found: {target}")

    df = pd.read_parquet(target)
    if "label_name" not in df.columns or "embedding" not in df.columns:
        raise ValueError("Parquet must contain 'label_name' and 'embedding' columns.")

    pose_to_embedding: Dict[str, torch.Tensor] = {}
    pose_list_column = df["poses"] if "poses" in df.columns else None
    valid_poses = set(df["label_name"].astype(str).tolist())
    if pose_list_column is not None:
        valid_poses.update(_collect_valid_poses(pose_list_column.tolist()))

    for pose, embedding in zip(df["label_name"], df["embedding"]):
        if as_tensor:
            value = torch.tensor(embedding, dtype=torch.float32)
            if device:
                value = value.to(device)
        else:
            # Keep the raw list from the parquet (no device placement).
            value = list(embedding)
        pose_to_embedding[str(pose)] = value

    return PoseEmbeddingLookup(pose_to_embedding=pose_to_embedding, valid_poses=valid_poses)


if __name__ == "__main__":
    lookup = load_pose_lookup()
    sample_pose = next(iter(lookup.pose_to_embedding.keys()))
    vector = lookup.get(sample_pose)
    print(f"Example pose: {sample_pose}")
    print(f"Embedding shape: {tuple(vector.shape)} (device={vector.device})")
