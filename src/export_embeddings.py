#!/usr/bin/env python3
"""
export_embeddings.py

Load a trained CNN classifier (from image-classifier-cnn.py), run it in inference mode
over the stereo dataset bundle, and export an embedding table keyed by each label name.

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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# =======================
# === USER SETTINGS ====
# =======================

ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH: Path = ROOT / "data/pt-files/dedup-all-images-dataset.pt"
DEFAULT_MODEL_DIR: Path = ROOT / "data/models"
DEFAULT_MODEL_PATH: Path | None = ROOT / "data/models/cornermaze-cnn-model-20251031220236.pt"
DEFAULT_OUTPUT_DIR: Path = ROOT / "data/tables"

# =======================
# === MODEL DEFINITION ==
# =======================

class StereoConvNet(nn.Module):
    """Matches the architecture used in image-classifier-cnn.py (with optional hidden FC)."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_width: int | None = None,
        dropout_p: float = 0.0,
        linear_keys: Sequence[str] | None = None,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_p)

        if hidden_width is not None:
            self.hidden = nn.Linear(32, hidden_width)
            out_features = num_classes
            self.output = nn.Linear(hidden_width, out_features)
            self.hidden_key = (linear_keys or ["classifier.2", "classifier.3"])[0]
            self.output_key = (linear_keys or ["classifier.2", "classifier.3"])[1]
        else:
            self.hidden = None
            self.output = nn.Linear(32, num_classes)
            self.hidden_key = None
            self.output_key = (linear_keys or ["classifier.2"])[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        if self.hidden is not None:
            x = self.hidden(x)
        x = self.output(x)
        return x


# =======================
# === DATA LOADING ======
# =======================

def load_dataset(bundle_path: Path) -> Tuple[TensorDataset, Dict[str, Any]]:
    bundle_path = bundle_path.expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {bundle_path}")

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
    model: StereoConvNet,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Return embeddings shaped (N, 32) captured from the penultimate linear layer.
    """
    model.eval()
    embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0].to(device, non_blocking=True)
            features = model.features(batch_x)
            flat = model.flatten(features)
            drop = model.dropout(flat)
            if model.hidden is not None:
                embedding = model.hidden(drop)
            else:
                embedding = drop
            # Ensure non-negative embeddings (like SB3’s NatureCNN ReLU output)
            embedding = torch.relu(embedding)
            embeddings.append(embedding.cpu())

    return torch.cat(embeddings, dim=0)


def gather_pose_lists(label_catalog: Dict[Any, Any]) -> Dict[int, List[str]]:
    pose_lookup: Dict[int, List[str]] = {}
    for key, entry in label_catalog.items():
        try:
            label_id = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(entry, dict):
            descriptions = entry.get("descriptions") or []
            pose_lookup[label_id] = sorted({str(desc) for desc in descriptions})
    return pose_lookup


def build_dataframe(
    embeddings: torch.Tensor,
    payload: Dict[str, Any],
    label_ids: Sequence[int],
) -> pd.DataFrame:
    label_names = payload.get("label_names")
    if label_names is None or len(label_names) != len(label_ids):
        raise ValueError("Payload lacks aligned 'label_names'.")
    label_catalog = payload.get("label_catalog", {})
    pose_lookup = gather_pose_lists(label_catalog if isinstance(label_catalog, dict) else {})

    rows = []
    for name, label_id, embedding in zip(label_names, label_ids, embeddings):
        rows.append(
            {
                "label_name": str(name),
                "label_id": int(label_id),
                "embedding": embedding.tolist(),
                "poses": pose_lookup.get(int(label_id), []),
            }
        )
    return pd.DataFrame(rows)


# =======================
# === CLI ================
# =======================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CNN embeddings for each stereo label.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained model checkpoint (.pt). Defaults to the newest file in DEFAULT_MODEL_DIR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory or parquet file path.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference (<=0 → full batch).")
    parser.add_argument("--device", default=None, help="Override device (cuda/mps/cpu). Auto-detect when omitted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset, payload = load_dataset(args.bundle)
    x_tensor: torch.Tensor = dataset.tensors[0]
    label_ids = dataset.tensors[1].tolist()
    num_samples = len(dataset)
    in_channels = x_tensor.shape[1]
    num_classes = int(torch.unique(dataset.tensors[1]).numel())

    batch_size = num_samples if args.batch_size is None or args.batch_size <= 0 else args.batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model_path = args.model
    if model_path is None:
        if DEFAULT_MODEL_PATH is not None and DEFAULT_MODEL_PATH.exists():
            model_path = DEFAULT_MODEL_PATH
            print(f"[INFO] Using configured model checkpoint: {model_path}")
        else:
            candidate_files = sorted(DEFAULT_MODEL_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidate_files:
                raise FileNotFoundError(
                    f"No .pt files found in {DEFAULT_MODEL_DIR}. Specify --model explicitly or update DEFAULT_MODEL_PATH."
                )
            model_path = candidate_files[0]
            print(f"[INFO] Using latest model checkpoint in directory: {model_path}")
    else:
        model_path = model_path.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    hidden_width = None
    model = StereoConvNet(in_channels=in_channels, num_classes=num_classes, hidden_width=hidden_width)
    classifier_weight_keys = [key for key in state_dict if key.startswith("classifier.") and key.endswith(".weight")]
    classifier_indices = sorted({int(key.split(".")[1]) for key in classifier_weight_keys})

    hidden_idx = None
    output_idx = None
    if classifier_indices:
        if len(classifier_indices) == 2:
            hidden_idx = classifier_indices[0]
            output_idx = classifier_indices[1]
            hidden_width = state_dict[f"classifier.{hidden_idx}.weight"].shape[0]
            model = StereoConvNet(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_width=hidden_width,
            )
        elif len(classifier_indices) == 1:
            output_idx = classifier_indices[0]
            model = StereoConvNet(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_width=None,
            )
        else:
            raise RuntimeError(f"Unrecognized classifier configuration: {classifier_indices}")
    else:
        model = StereoConvNet(in_channels=in_channels, num_classes=num_classes)

    renamed_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("classifier."):
            idx = int(key.split(".")[1])
            suffix = key.split(".", 2)[-1]
            if hidden_idx is not None and idx == hidden_idx:
                renamed_state[f"hidden.{suffix}"] = value
            elif output_idx is not None and idx == output_idx:
                renamed_state[f"output.{suffix}"] = value
            else:
                continue
        else:
            renamed_state[key] = value

    model.load_state_dict(renamed_state, strict=True)
    model.to(device)

    embeddings = extract_embeddings(model, dataloader, device)
    df = build_dataframe(embeddings, payload, label_ids)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_target = args.output.expanduser().resolve()
    if output_target.suffix.lower() == ".parquet":
        output_dir = output_target.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_target.stem}-{timestamp}.parquet"
    else:
        output_dir = output_target
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.bundle.stem}-embeddings-{timestamp}.parquet"

    df.to_parquet(output_path, index=False)
    print(f"Saved embeddings for {len(df)} samples to {output_path}")


if __name__ == "__main__":
    main()
