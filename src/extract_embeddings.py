#!/usr/bin/env python3
"""
export_embeddings.py

Load a trained CNN classifier (from image-classifier-cnn.py), run it in inference mode
over the stereo dataset bundle, and export an embedding table keyed by each label name.
Supports PyTorch checkpoints (.pt) as well as ONNX exports created by train_classifier.py.

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
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =======================
# === USER SETTINGS ====
# =======================

ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH: Path = ROOT / "data/pt-files/corner-maze-render-base-images-dataset-median-groups.pt"
DEFAULT_MODEL_DIR: Path = ROOT / "data/models"
DEFAULT_MODEL_PATH: Path | None = ROOT / "data/models/stereo_cnn-median-merge-20251202-201506.onnx"
DEFAULT_OUTPUT_DIR: Path = ROOT / "data/tables"

# =======================
# === MODEL DEFINITION ==
# =======================

class StereoConvNet(nn.Module):
    """
    Lightweight CNN matching `image-classifier-cnn.py`'s implementation.
    Supports `forward(x, return_embedding=True)` to return (logits, embedding).
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Coming in with (N, 2, 128, 128)
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(inplace=True),
        )
        # Fully-connected classifier
        self.fc1 = nn.Linear(64, 64)     # hidden embedding layer
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = self.features(x)             # (N, 64, 1, 1)
        x = x.flatten(1)                 # (N, 64)

        h = F.relu(self.fc1(x))          # (N, 64)  ← embedding

        logits = self.fc2(h)             # (N, num_classes)

        if return_embedding:
            return logits, h

        return logits


class OnnxStereoWrapper(nn.Module):
    """
    Wrap an ONNX Runtime session to mimic the StereoConvNet interface.
    """

    def __init__(self, session: ort.InferenceSession):
        super().__init__()
        self._session = session
        inputs = session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no inputs.")
        self._input_name = inputs[0].name
        self._output_names = [output.name for output in session.get_outputs()]

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        batch = x.detach().cpu().numpy()
        outputs = self._session.run(self._output_names, {self._input_name: batch})
        if not outputs:
            raise RuntimeError("ONNX model produced no outputs.")

        logits_np = outputs[0]
        logits = torch.from_numpy(logits_np)
        if len(outputs) > 1:
            embedding = torch.from_numpy(outputs[1])
        else:
            embedding = logits

        if return_embedding:
            return logits, embedding
        return logits


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


def _remap_legacy_classifier_keys(model: StereoConvNet, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap checkpoints saved with legacy classifier.* keys to the fc1/fc2 layout.
    """
    renamed_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("classifier."):
            suffix = key.split(".", 2)[-1]
            if not isinstance(value, torch.Tensor):
                continue
            try:
                if value.dim() == 2:
                    if tuple(value.shape) == tuple(model.fc1.weight.shape):
                        renamed_state[f"fc1.{suffix}"] = value
                    elif tuple(value.shape) == tuple(model.fc2.weight.shape):
                        renamed_state[f"fc2.{suffix}"] = value
                    else:
                        if value.shape[1] == model.fc1.weight.shape[1]:
                            if value.shape[0] == model.fc2.weight.shape[0]:
                                renamed_state[f"fc2.{suffix}"] = value
                            else:
                                renamed_state[f"fc1.{suffix}"] = value
                        else:
                            continue
                elif value.dim() == 1:
                    if tuple(value.shape) == tuple(model.fc1.bias.shape):
                        renamed_state[f"fc1.{suffix}"] = value
                    elif tuple(value.shape) == tuple(model.fc2.bias.shape):
                        renamed_state[f"fc2.{suffix}"] = value
                    else:
                        if value.shape[0] == model.fc2.bias.shape[0]:
                            renamed_state[f"fc2.{suffix}"] = value
                        elif value.shape[0] == model.fc1.bias.shape[0]:
                            renamed_state[f"fc1.{suffix}"] = value
                        else:
                            continue
            except Exception:
                continue
        else:
            renamed_state[key] = value
    return renamed_state


def _load_pytorch_checkpoint(
    model_path: Path,
    in_channels: int,
    num_classes: int,
    device: torch.device,
) -> StereoConvNet:
    state = torch.load(model_path, map_location="cpu")
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    model = StereoConvNet(in_channels=in_channels, num_classes=num_classes)
    renamed_state = _remap_legacy_classifier_keys(model, state_dict)

    load_info = model.load_state_dict(renamed_state, strict=False)
    if getattr(load_info, "missing_keys", None):
        print(f"[WARN] Missing keys when loading checkpoint: {load_info.missing_keys}")
    if getattr(load_info, "unexpected_keys", None):
        print(f"[WARN] Unexpected keys in checkpoint: {load_info.unexpected_keys}")
    model.to(device)
    return model


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

    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0].to(device, non_blocking=True)
            # Prefer pulling embedding via the model's forward contract if supported
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
    x_tensor = payload.get("x")
    if not isinstance(x_tensor, torch.Tensor):
        raise ValueError("Payload missing 'x' tensor for image export.")

    rows = []
    for idx, (name, label_id, embedding) in enumerate(zip(label_names, label_ids, embeddings)):
        left_eye = x_tensor[idx, 0].detach().cpu().numpy()
        right_eye = x_tensor[idx, 1].detach().cpu().numpy()
        right_eye_original = np.fliplr(right_eye)
        rows.append(
            {
                "label_name": str(name),
                "label_id": int(label_id),
                "embedding": embedding.tolist(),
                "poses": pose_lookup.get(int(label_id), []),
                "left_eye_img": left_eye.tolist(),
                "right_eye_img": right_eye_original.tolist(),
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
        help="Path to trained model checkpoint (.onnx preferred, .pt still supported). Defaults to the newest file in DEFAULT_MODEL_DIR.",
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

    requested_device = torch.device(args.device) if args.device else None

    model_path = args.model
    if model_path is None:
        if DEFAULT_MODEL_PATH is not None and DEFAULT_MODEL_PATH.exists():
            model_path = DEFAULT_MODEL_PATH
            print(f"[INFO] Using configured model checkpoint: {model_path}")
        else:
            model_path = None
            for pattern in ("*.onnx", "*.pt"):
                candidate_files = sorted(
                    DEFAULT_MODEL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
                )
                if candidate_files:
                    model_path = candidate_files[0]
                    print(f"[INFO] Using latest {pattern} checkpoint in directory: {model_path}")
                    break
            if model_path is None:
                raise FileNotFoundError(
                    f"No .onnx or .pt files found in {DEFAULT_MODEL_DIR}. Specify --model explicitly or update DEFAULT_MODEL_PATH."
                )
    else:
        model_path = model_path.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    use_onnx = model_path.suffix.lower() == ".onnx"
    if use_onnx:
        device = torch.device("cpu")
        if requested_device and requested_device.type != "cpu":
            print(f"[WARN] Overriding requested device {requested_device} with CPU for ONNX Runtime.")
    else:
        if requested_device:
            device = requested_device
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    if use_onnx:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model: nn.Module = OnnxStereoWrapper(session)
    else:
        model = _load_pytorch_checkpoint(model_path, in_channels, num_classes, device)

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
