#!/usr/bin/env python3
"""
dataset_io.py

Helper functions for saving/loading dataset bundles in a JSON + tensor format.
The expected payload schema mirrors create_dataset's output and primarily stores
tensor files plus basic metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import shutil


# Keys that should be stored via NumPy rather than torch.save to avoid pickle.
NUMPY_TENSOR_KEYS = {"x", "y"}
NPZ_KEY = "data"


def _tensor_filename(key: str) -> str:
    suffix = ".npz" if key in NUMPY_TENSOR_KEYS else ".pt"
    return f"{key}{suffix}"


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare metadata dict for JSON serialization."""
    return dict(metadata)


def _deserialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return metadata exactly as stored."""
    return dict(metadata)


def _clear_directory(path: Path) -> None:
    """Remove all contents from a directory in preparation for fresh output."""
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def save_bundle(payload: Dict[str, Any], target_dir: Path) -> None:
    """
    Save a payload dictionary into a directory containing tensor files and metadata.json.
    """
    target_dir = target_dir.expanduser().resolve()
    if target_dir.exists():
        if not target_dir.is_dir():
            raise NotADirectoryError(f"Target path exists and is not a directory: {target_dir}")
        _clear_directory(target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    tensor_keys = []
    metadata: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor_path = target_dir / _tensor_filename(key)
            if key in NUMPY_TENSOR_KEYS:
                np.savez_compressed(str(tensor_path), **{NPZ_KEY: value.detach().cpu().numpy()})
            else:
                torch.save(value, tensor_path)
            tensor_keys.append(key)
        else:
            metadata[key] = value

    metadata["_tensor_keys"] = tensor_keys
    metadata_path = target_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(_serialize_metadata(metadata), fh, indent=2)


def load_bundle(source_dir: Path) -> Dict[str, Any]:
    """
    Load a payload dictionary previously written by save_bundle.
    """
    source_dir = source_dir.expanduser().resolve()
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {source_dir}")

    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = _deserialize_metadata(json.load(fh))

    tensor_keys = metadata.pop("_tensor_keys", [])
    payload = dict(metadata)
    for key in tensor_keys:
        tensor_path = source_dir / _tensor_filename(key)
        if tensor_path.exists():
            if key in NUMPY_TENSOR_KEYS:
                with np.load(tensor_path, allow_pickle=False) as npz:
                    array = npz[NPZ_KEY]
                payload[key] = torch.from_numpy(array)
            else:
                payload[key] = torch.load(tensor_path, map_location="cpu")
            continue

        raise FileNotFoundError(f"Tensor file missing: {tensor_path}")
    return payload
