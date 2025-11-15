#!/usr/bin/env python3
"""
image-classifier-resnet.py

Train a ResNet-style classifier (torchvision ResNet-18 backbone adapted to stereo
inputs) on a serialized dataset bundle (.pt file produced by create_dataset.py or
remove_duplicates.py). The network is allowed to overfit intentionally so it can
memorise the training set (reach ~100% accuracy) when given enough capacity and
epochs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

# =======================
# === USER SETTINGS ====
# =======================

DATASET_PATH: Path = Path(
    "/Users/ryangrgurich/VS Code Local/corner-maze-encoder-pretrain/data/pt-files/dedup-all-images-dataset.pt"
)
MAX_EPOCHS: int = 300
BATCH_SIZE: int = 64
LEARNING_RATE: float = 5e-4
WEIGHT_DECAY: float = 0.0
TARGET_ACCURACY: float = 1.0
PRINT_EVERY: int = 1
SEED: int = 42
WARMUP_EPOCHS: int = 1


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


class StereoResNet(nn.Module):
    """
    ResNet-18 backbone adapted to accept stereo (C=2) inputs with configurable head.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        # Adapt the first convolution to match the stereo channel count.
        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    preds = pred.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.numel()


def train(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    target_accuracy: float,
    print_every: int,
    warmup_epochs: int,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(dataloader))
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.05
    )

    model.to(device)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            global_step += 1

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            if epoch > warmup_epochs:
                scheduler.step()

            running_loss += loss.item() * batch_x.size(0)
            running_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            running_total += batch_x.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total

        if epoch % print_every == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:03d}: loss={epoch_loss:.4f} acc={epoch_acc:.4f} lr={current_lr:.2e}"
            )

        if epoch_acc >= target_accuracy - 1e-6:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-based classifier on the stereo dataset bundle."
    )
    parser.add_argument(
        "--bundle",
        default=DATASET_PATH,
        type=Path,
        help="Path to the dataset bundle (.pt).",
    )
    parser.add_argument(
        "--epochs",
        default=MAX_EPOCHS,
        type=int,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        default=BATCH_SIZE,
        type=int,
        help="Samples per batch. Use 0/negative for full-batch training.",
    )
    parser.add_argument(
        "--lr",
        default=LEARNING_RATE,
        type=float,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        default=WEIGHT_DECAY,
        type=float,
        help="L2 weight decay coefficient.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device (e.g., 'cuda', 'mps', 'cpu'). Auto-detects when omitted.",
    )
    parser.add_argument(
        "--target-acc",
        default=TARGET_ACCURACY,
        type=float,
        help="Stop training once this accuracy is achieved (0-1 scale).",
    )
    parser.add_argument(
        "--seed",
        default=SEED,
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--print-every",
        default=PRINT_EVERY,
        type=int,
        help="Print metrics every N epochs.",
    )
    parser.add_argument(
        "--warmup-epochs",
        default=WARMUP_EPOCHS,
        type=int,
        help="Number of warmup epochs before applying LR cosine annealing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset, payload = load_dataset(args.bundle)
    x_tensor: torch.Tensor = dataset.tensors[0]
    num_samples = len(dataset)
    in_channels = x_tensor.shape[1]
    num_classes = int(torch.unique(dataset.tensors[1]).numel())

    if args.batch_size is None or args.batch_size <= 0:
        batch_size = num_samples
    else:
        batch_size = args.batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    if args.device:
        device = torch.device(args.device)
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

    model = StereoResNet(in_channels=in_channels, num_classes=num_classes)

    print(
        f"Dataset: {args.bundle} (samples={num_samples}, classes={num_classes}, "
        f"input_shape={tuple(x_tensor.shape[1:])})"
    )
    print(
        f"Training configuration: epochs={args.epochs}, batch_size={batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, warmup_epochs={args.warmup_epochs}"
    )

    train(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        target_accuracy=args.target_acc,
        print_every=max(1, args.print_every),
        warmup_epochs=max(0, args.warmup_epochs),
    )

    model_path = args.bundle.with_name(args.bundle.stem + "-resnet-classifier.pt")
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_class": model.__class__.__name__,
            "num_classes": num_classes,
            "in_channels": in_channels,
            "config": {
                "epochs": args.epochs,
                "batch_size": batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "warmup_epochs": max(0, args.warmup_epochs),
            },
            "orig_label_ids": payload.get("_orig_label_ids"),
        },
        model_path,
    )
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
