#!/usr/bin/env python3
"""
train_classifier_cnn_attention.py

Variant of the barebones classifier that adds a lightweight spatial-attention
residual block after the final convolution so the network can keep track of
where monitors/cues appear even after downsampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import dataset_io

# =======================
# === USER SETTINGS ====
# =======================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data/datasets/base-images-consolidated-partitioned-52-1-acute-ds"
MAX_EPOCHS: int = 1_000_000
BATCH_SIZE: int = 0  # 0 or negative → full batch (memorization)
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 0.0
TARGET_ACCURACY: float = 1.0
PRINT_EVERY: int = 1
SEED: int = 42
DEVICE: str | None = None  # "cuda", "cpu", "mps", or None for auto
EMBEDDING_DIM: int = 60
MODEL_SUFFIX: str = f"consolidated-acute-{EMBEDDING_DIM}-attentionNoRelu"
MODEL_OUTPUT_DIR = ROOT_DIR / "data/models/"


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
        raise TypeError(f"'x' must be a torch.Tensor (got {type(x)!r})")
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    x = x.float()
    y = y.long()

    unique_labels: List[int] = sorted({int(label) for label in y.tolist()})
    id_to_index = {label_id: idx for idx, label_id in enumerate(unique_labels)}
    remapped = torch.tensor([id_to_index[int(label)] for label in y.tolist()], dtype=torch.long)

    dataset = TensorDataset(x, remapped)
    payload["_orig_label_ids"] = unique_labels
    return dataset, payload


class SpatialAttention(nn.Module):
    """Single-head self-attention over spatial locations with residual scaling."""

    def __init__(self, channels: int):
        super().__init__()
        reduced = max(1, channels // 8)
        self.query = nn.Conv2d(channels, reduced, kernel_size=1)
        self.key = nn.Conv2d(channels, reduced, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        q = self.query(x).view(b, -1, n)            # (B, c_q, N)
        k = self.key(x).view(b, -1, n)              # (B, c_q, N)
        v = self.value(x).view(b, -1, n)            # (B, c,   N)

        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)  # (B, N, N)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return self.gamma * out + x


class StereoConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            #nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            #nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
        )
        self.conv3 = nn.Conv2d(32, EMBEDDING_DIM, kernel_size=3, padding=1)
        self.attn = SpatialAttention(EMBEDDING_DIM)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.fc2 = nn.Linear(EMBEDDING_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        #x = F.leaky_relu(self.conv3(x), inplace=True)
        x = self.conv3(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        h = self.fc1(x)
        logits = self.fc2(h)
        return logits

    @torch.jit.export
    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.leaky_relu(self.conv3(x), inplace=True)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        h = self.fc1(x)
        logits = self.fc2(h)
        return logits, h


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    preds = pred.argmax(dim=1)
    return (preds == target).float().mean().item()


def train(
    model: nn.Module,
    dataloader: DataLoader,
    full_dataset: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    target_accuracy: float,
    print_every: int,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            running_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            running_total += batch_x.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:05d}: loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

        if epoch_acc >= target_accuracy - 1e-6:
            print(f"Target accuracy {target_accuracy:.0%} reached at epoch {epoch}.")
            break


def ensure_all_samples(dataset: TensorDataset) -> TensorDataset:
    return dataset


def save_torchscript_model(model: nn.Module, suffix: str = "") -> Path:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ts_path = MODEL_OUTPUT_DIR / f"stereo-cnn{suffix}.ts"

    class _ExportWrapper(nn.Module):
        def __init__(self, wrapped: nn.Module):
            super().__init__()
            self.core = wrapped

        def forward(self, x: torch.Tensor):
            return self.core.forward_with_embedding(x)

    scripted = torch.jit.script(_ExportWrapper(model.eval()))
    scripted.save(str(ts_path))
    print(f"Model exported to {ts_path}")
    return ts_path


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset, payload = load_dataset(DATASET_PATH)
    dataset = ensure_all_samples(dataset)
    data_x, data_y = dataset.tensors

    in_channels = data_x.shape[1]
    num_classes = int(torch.unique(data_y).numel())

    batch_size = len(dataset) if BATCH_SIZE <= 0 else BATCH_SIZE
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if DEVICE:
        device = torch.device(DEVICE)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = StereoConvNet(in_channels, num_classes)
    train(
        model,
        dataloader,
        (data_x, data_y),
        device,
        MAX_EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        TARGET_ACCURACY,
        PRINT_EVERY,
    )

    save_torchscript_model(model, suffix=MODEL_SUFFIX)


if __name__ == "__main__":
    main()
