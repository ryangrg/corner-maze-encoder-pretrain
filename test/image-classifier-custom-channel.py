#!/usr/bin/env python3
"""
image-classifier-custom-channel.py

Train a compact residual CNN where you control the channel widths for each stage.
The network is designed to overfit the provided stereo dataset (.pt bundle) so
you can experiment with different channel configurations and find the minimum
capacity that still reaches ~100% training accuracy.

Channel rules: defaults follow the standard ResNet pattern (e.g. [64, 128, 256, 512]),
but you can plug in any positive integer sequence. Common practice uses powers of two,
and later stages are usually >= earlier ones (e.g. 32→64→128→256 or 48→96→192→384).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# =======================
# === USER SETTINGS ====
# =======================

DATASET_PATH: Path = Path(
    "/Users/ryangrgurich/VS Code Local/corner-maze-encoder-pretrain/data/pt-files/dedup-all-images-dataset.pt"
)
MAX_EPOCHS: int = 400
BATCH_SIZE: int = 0  # 0/negative → full batch
LEARNING_RATE: float = 5e-4
WEIGHT_DECAY: float = 0.0
TARGET_ACCURACY: float = 1.0
PRINT_EVERY: int = 1
SEED: int = 42
WARMUP_EPOCHS: int = 0

# Channel configuration (per residual stage). Provide four integers >0.
# Example: [32, 64, 128, 192] or [64, 96, 128, 160].
CHANNEL_LIST: List[int] = [4, 8, 16, 32]


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class StereoCustomResNet(nn.Module):
    """
    Residual network with customizable channel widths per stage.
    """

    def __init__(self, in_channels: int, num_classes: int, channels: List[int]):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("CHANNEL_LIST must contain exactly four integers.")
        if any(c <= 0 for c in channels):
            raise ValueError("All channel values must be positive.")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_stage(channels[0], channels[0], stride=1, blocks=2)
        self.layer2 = self._make_stage(channels[0], channels[1], stride=2, blocks=2)
        self.layer3 = self._make_stage(channels[1], channels[2], stride=2, blocks=2)
        self.layer4 = self._make_stage(channels[2], channels[3], stride=2, blocks=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, stride: int, blocks: int) -> nn.Sequential:
        layers: List[nn.Module] = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


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
            current_lr = scheduler.get_last_lr()[0] if epoch > warmup_epochs else lr
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
        description="Train a custom-channel residual classifier on the stereo dataset bundle."
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

    model = StereoCustomResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        channels=CHANNEL_LIST,
    )

    print(
        f"Dataset: {args.bundle} (samples={num_samples}, classes={num_classes}, "
        f"input_shape={tuple(x_tensor.shape[1:])})"
    )
    print(
        f"Training configuration: epochs={args.epochs}, batch_size={batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, warmup_epochs={args.warmup_epochs}, "
        f"channels={CHANNEL_LIST}"
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

    model_path = args.bundle.with_name(args.bundle.stem + "-custom-resnet-classifier.pt")
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
                "channels": list(CHANNEL_LIST),
            },
            "orig_label_ids": payload.get("_orig_label_ids"),
        },
        model_path,
    )
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
