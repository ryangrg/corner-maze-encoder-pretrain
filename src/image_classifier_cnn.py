#!/usr/bin/env python3
"""
image-classifier.py

Train a compact VGG-style convolutional network (three Conv2d→ReLU blocks,
global average pooling, fully connected classifier) on a serialized stereo
dataset bundle (.pt file produced by create_dataset.py or remove_duplicates.py),
optimizing directly on the training set until it memorizes every sample
(100% accuracy) or a maximum epoch budget is reached.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
# =======================
# === USER SETTINGS ====
# =======================

DATASET_PATH: Path = Path(
    "/Users/ryangrgurich/VS Code Local/corner-maze-encoder-pretrain/data/pt-files/dedup-all-images-dataset.pt"
)
MAX_EPOCHS: int = 10000
BATCH_SIZE: int = 0  # 0 or negative → full batch (good for memorising)
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 0
TARGET_ACCURACY: float = 1.0
PRINT_EVERY: int = 1
SEED: int = 42
DEVICE: str | None = None  # "cuda", "cpu", "mps", or None (auto-detect)


def load_dataset(bundle_path: Path) -> Tuple[TensorDataset, Dict[str, Any]]:
    """
    Load the dataset bundle and wrap tensors in a TensorDataset.
    """
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
    """
    Lightweight CNN that is expressive enough to memorise the training set quickly.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Coming in with (N, 2, 128, 128)
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), # (N, 16, 128, 128)
            nn.ReLU(inplace=True), # no change in size just do ReLU on each pixel
            nn.MaxPool2d(kernel_size=2, stride=2), # Now we shrink 1/2 (N, 16, 64, 64)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Increase channels (N, 32, 64, 64)
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), # shrink (N, 32, 32, 32)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Increase channels (N, 32, 64, 64)
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), # shrink (N, 64, 16, 16)

            nn.AdaptiveAvgPool2d(1), # 
            nn.ReLU(inplace=True), # keep values positive
        )
        # Fully-connected classifier
        self.fc1 = nn.Linear(64, 64)     # hidden embedding layer
        self.fc2 = nn.Linear(64, num_classes) # size of number of classes

    def forward(self, x, return_embedding=False):
        x = self.features(x)             # (N, 64, 1, 1)
        x = x.flatten(1)                 # (N, 64)

        h = F.relu(self.fc1(x))          # (N, 64)  ← this is your embedding
        
        logits = self.fc2(h)             # (N, num_classes)

        if return_embedding:
            return logits, h             # return both logits + embedding
        
        return logits


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
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}: loss={epoch_loss:.4f} acc={epoch_acc:.4f} lr={current_lr:.2e}"
            )

        if epoch_acc >= target_accuracy - 1e-6:
            # Verify accuracy in eval mode before stopping
            model.eval()
            with torch.no_grad():
                eval_logits = model(full_dataset[0].to(device))
                eval_acc = accuracy(eval_logits, full_dataset[1].to(device))
            model.train()

            if eval_acc >= target_accuracy - 1e-6:
                print(
                    f"Target accuracy {target_accuracy:.0%} reached at epoch {epoch}. "
                    "Stopping training."
                )
                break
    else:
        print(
            f"Reached maximum epochs ({epochs}) with final accuracy {epoch_acc:.4f}. "
            "Increase epochs or adjust hyperparameters if higher fit is required."
        )

def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    dataset, payload = load_dataset(DATASET_PATH)
    x_tensor: torch.Tensor = dataset.tensors[0]
    num_samples = len(dataset)
    in_channels = x_tensor.shape[1]
    num_classes = int(torch.unique(dataset.tensors[1]).numel())

    if BATCH_SIZE is None or BATCH_SIZE <= 0:
        batch_size = num_samples
    else:
        batch_size = BATCH_SIZE

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    if DEVICE:
        device = torch.device(DEVICE)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    elif device.type == "mps":
        print("Using Apple GPU via MPS backend.")
    else:
        print(f"Using device: {device}")

    model = StereoConvNet(in_channels=in_channels, num_classes=num_classes)

    print(
        f"Dataset: {DATASET_PATH} (samples={num_samples}, classes={num_classes}, "
        f"shape={tuple(x_tensor.shape[1:])})"
    )
    print(
        f"Training configuration: epochs={MAX_EPOCHS}, batch_size={batch_size}, "
        f"lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, target_acc={TARGET_ACCURACY:.2f}"
    )

    train(
        model=model,
        dataloader=dataloader,
        full_dataset=(dataset.tensors[0], dataset.tensors[1]),
        device=device,
        epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        target_accuracy=TARGET_ACCURACY,
        print_every=max(1, PRINT_EVERY),
    )
    model.eval()
    with torch.no_grad():
        preds = model(dataset.tensors[0].to(device))
        final_accuracy = accuracy(preds, dataset.tensors[1].to(device))
    if final_accuracy >= 1.0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_dir = Path("data/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"cornermaze-cnn-model-{timestamp}.pt"
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "model_class": model.__class__.__name__,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "config": {
                    "epochs": MAX_EPOCHS,
                    "batch_size": batch_size,
                    "lr": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                },
            },
            model_path,
        )
        print(f"Saved trained model to {model_path}")
    else:
        print(f"Final accuracy {final_accuracy:.4f} < 1.0; model not saved.")


if __name__ == "__main__":
    main()
