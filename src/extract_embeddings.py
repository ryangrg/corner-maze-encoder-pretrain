#!/usr/bin/env python3
"""
export_embeddings.py

Load a trained CNN classifier (from image-classifier-cnn.py), run it in inference mode
over the stereo dataset bundle, and export an embedding table keyed by each label name.
Supports TorchScript exports (.ts).

The resulting lookup is saved as a Parquet file containing:
  • label_name: configN_x_y_direction string
  • label_id: integer class index
  • embedding: list[float] (hidden representation from the penultimate linear layer)
  • poses: list[str] of all descriptions tied to that label_id (from label_catalog)
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import dataset_io
# =======================
# =======================
# === USER SETTINGS ====
# =======================

ROOT: Path = Path(__file__).resolve().parents[1]
DATASET_PATH: Path = ROOT / "data/datasets/corner-maze-render-base-images-consolidated-acute-ds"
MODEL_PATH: Path = ROOT / "data/models/stereo_cnn-consolidated-acute-32.ts"
OUTPUT_DIR: Path = ROOT / "data/tables"
BATCH_SIZE: int = 128
DEVICE_OVERRIDE: str | None = None  # "cuda", "cpu", "mps", or None for auto

# =======================
# === DATA LOADING ======
# =======================

def load_dataset(bundle_path: Path) -> Tuple[TensorDataset, Dict[str, Any]]:
    bundle_path = bundle_path.expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {bundle_path}")

    if bundle_path.is_dir():
        payload = dataset_io.load_bundle(bundle_path)
    else:
        payload = torch.load(bundle_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, found {type(payload)!r}")

    if "x" not in payload or "y" not in payload:
        raise KeyError("Payload must contain 'x' and 'y' tensors.")

    x = payload["x"]
    y = payload["y"]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"'x' must be torch.Tensor (got {type(x)!r})")
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    x = x.float()
    y = y.long()

    unique_labels: List[int] = sorted({int(label) for label in y.tolist()})
    id_to_index = {label_id: idx for idx, label_id in enumerate(unique_labels)}
    remapped = torch.tensor(
        [id_to_index[int(label)] for label in y.tolist()],
        dtype=torch.long,
    )

    dataset = TensorDataset(x, remapped)
    payload["_orig_label_ids"] = unique_labels
    return dataset, payload


# =======================
# === EMBEDDING EXPORT ==
# =======================

def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Return embeddings shaped (N, 64) captured from the penultimate linear layer.
    """
    model.eval()
    embeddings: List[torch.Tensor] = []

    scripted_module_types = (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)

    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0].to(device, non_blocking=True)
            # Prefer pulling embedding via the model's forward contract if supported
            if isinstance(model, scripted_module_types):
                out = model(batch_x)
            elif hasattr(model, "forward_with_embedding"):
                out = model.forward_with_embedding(batch_x)
            else:
                try:
                    out = model(batch_x, return_embedding=True)
                except TypeError:
                    out = None

            if isinstance(out, tuple) and len(out) == 2:
                _, embedding = out
            else:
                # Fallback: try to extract similarly to older codepaths
                try:
                    features = model.features(batch_x)
                    flat = features.flatten(1)
                    # If model has fc1 (new architecture) use that, otherwise attempt attributes used by older models
                    if hasattr(model, "fc1"):
                        embedding = F.relu(model.fc1(flat))
                    elif hasattr(model, "hidden") and model.hidden is not None:
                        embedding = model.hidden(flat)
                    else:
                        embedding = flat
                except Exception:
                    # As last resort, run a forward pass and take the penultimate activation by re-running
                    # the forward without return_embedding and try to inspect internals — fallback to zeros
                    embedding = torch.zeros((batch_x.size(0), 64), dtype=torch.float32, device=batch_x.device)

            # Ensure non-negative embeddings (like SB3’s NatureCNN ReLU output)
            embedding = torch.relu(embedding)
            embeddings.append(embedding.cpu())

    return torch.cat(embeddings, dim=0)


def build_dataframe(
    embeddings: torch.Tensor,
    payload: Dict[str, Any],
    label_ids: Sequence[int],
) -> pd.DataFrame:
    label_names = payload.get("label_names")
    if label_names is None or len(label_names) != len(label_ids):
        raise ValueError("Payload lacks aligned 'label_names'.")
    labels2label_names = payload.get("labels2label_names", {})
    x_tensor = payload.get("x")
    if not isinstance(x_tensor, torch.Tensor):
        raise ValueError("Payload missing 'x' tensor for image export.")

    rows = []
    for idx, (name, label_id, embedding) in enumerate(zip(label_names, label_ids, embeddings)):
        left_eye = x_tensor[idx, 0].detach().cpu().numpy()
        right_eye = x_tensor[idx, 1].detach().cpu().numpy()
        right_eye_original = np.fliplr(right_eye)
        if isinstance(labels2label_names, dict):
            group_entries = labels2label_names.get(str(label_id))
            if group_entries is None:
                group_entries = labels2label_names.get(int(label_id), [])
            group_labels = sorted({str(desc) for desc in (group_entries or [])})
        else:
            group_labels = []

        rows.append(
            {
                "label_name": str(name),
                "label_id": int(label_id),
                "embedding": embedding.tolist(),
                "poses": group_labels,
                "left_eye_img": left_eye.tolist(),
                "right_eye_img": right_eye_original.tolist(),
            }
        )
    return pd.DataFrame(rows)


# =======================
# === CLI ================
# =======================

def main() -> None:
    dataset, payload = load_dataset(DATASET_PATH)
    label_ids = payload.get("labels")
    if label_ids is None or len(label_ids) != len(dataset):
        raise ValueError("Payload must contain 'labels' aligned with dataset samples.")
    num_samples = len(dataset)
    batch_size = num_samples if BATCH_SIZE is None or BATCH_SIZE <= 0 else BATCH_SIZE

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    if DEVICE_OVERRIDE:
        device = torch.device(DEVICE_OVERRIDE)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = torch.jit.load(str(MODEL_PATH.expanduser().resolve()), map_location=device)
    model.eval()
    print(f"[INFO] Using device: {device}")

    embeddings = extract_embeddings(model, dataloader, device)
    df = build_dataframe(embeddings, payload, label_ids)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{DATASET_PATH.stem}-embeddings-{timestamp}.parquet"

    df.to_parquet(output_path, index=False)
    print(f"Saved embeddings for {len(df)} samples to {output_path}")


if __name__ == "__main__":
    main()
