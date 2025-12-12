#!/usr/bin/env python3
"""
image_classifier_cnn_01.py

Train a compact StereoConvNet on the full stereo dataset bundle, ensuring every
image remains in the training set. This script is notebook-friendly and bundles
all necessary definitions locally (no imports from image_classifier_cnn.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import dataset_io

# =======================
# === USER SETTINGS ====
# =======================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data/datasets/corner-maze-render-base-images-consolidated-acute-ds"
MAX_EPOCHS: int = 1000000
BATCH_SIZE: int = 0  # 0 or negative → full batch (memorization)
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 0.0
TARGET_ACCURACY: float = 1.0
PRINT_EVERY: int = 1
SEED: int = 42
DEVICE: str | None = None  # "cuda", "cpu", "mps", or None for auto
MODEL_SUFFIX: str = "consolidated-acute-32"
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
    remapped = torch.tensor(
        [id_to_index[int(label)] for label in y.tolist()],
        dtype=torch.long,
    )

    dataset = TensorDataset(x, remapped)
    payload["_orig_label_ids"] = unique_labels
    return dataset, payload


class StereoConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
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
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits

    @torch.jit.export
    def forward_with_embedding(self, x):
        x = self.features(x)
        x = x.flatten(1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits, h


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    preds = pred.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.numel()


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
    total_steps = epochs * max(1, len(dataloader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.1
    )

    model.to(device)
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            scheduler.step()
            step += 1

            running_loss += loss.item() * batch_x.size(0)
            running_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            running_total += batch_x.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: loss={epoch_loss:.4f} acc={epoch_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if epoch_acc >= target_accuracy - 1e-6:
            model.eval()
            with torch.no_grad():
                eval_logits = model(full_dataset[0].to(device))
                eval_acc = accuracy(eval_logits, full_dataset[1].to(device))
            model.train()
            if eval_acc >= target_accuracy - 1e-6:
                print(f"Target accuracy {target_accuracy:.0%} reached at epoch {epoch}.")
                break
    else:
        print(
            f"Reached max epochs ({epochs}) with final accuracy {epoch_acc:.4f}. "
            "Increase epochs or adjust hyperparameters if needed."
        )
def ensure_all_samples(dataset: TensorDataset) -> TensorDataset:
    return dataset


def save_torchscript_model(model: nn.Module, suffix: str = "") -> Path:
    """
    Script the trained classifier (retaining logits/embedding behavior) and save it.
    """
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ts_path = MODEL_OUTPUT_DIR / f"stereo_cnn{suffix}.ts"

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

    keep_mask = []
    seen = set()
    for idx, label in enumerate(data_y.tolist()):
        if label not in seen:
            seen.add(label)
            keep_mask.append(idx)

    data_x = data_x[keep_mask]
    data_y = data_y[keep_mask]
    dataset = TensorDataset(data_x, data_y)

    num_samples = len(dataset)
    in_channels = data_x.shape[1]
    num_classes = int(torch.unique(data_y).numel())

    batch_size = num_samples if BATCH_SIZE <= 0 else BATCH_SIZE
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if DEVICE:
        device = torch.device(DEVICE)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    model = StereoConvNet(in_channels=in_channels, num_classes=num_classes)

    print(
        f"Dataset: {DATASET_PATH} (samples={num_samples}, classes={num_classes}, "
        f"shape={tuple(data_x.shape[1:])})"
    )
    print(
        f"Training configuration: epochs={MAX_EPOCHS}, batch_size={batch_size}, "
        f"lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
        f"target_acc={TARGET_ACCURACY:.2f}"
    )

    train(
        model=model,
        dataloader=dataloader,
        full_dataset=(data_x, data_y),
        device=device,
        epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        target_accuracy=TARGET_ACCURACY,
        print_every=max(1, PRINT_EVERY),
    )

    model.eval()
    with torch.no_grad():
        preds = model(data_x.to(device))
        final_accuracy = accuracy(preds, data_y.to(device))
    _ = final_accuracy  # kept for potential logging; TorchScript export does not need metadata.
    save_torchscript_model(model.cpu(), suffix=MODEL_SUFFIX)


if __name__ == "__main__":
    main()
