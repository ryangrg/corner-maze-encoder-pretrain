#!/usr/bin/env python3
"""
image-classifier-mlp.py

Train a fully connected multilayer perceptron (flatten stereo stack → dense
layers → logits) on a serialized stereo dataset bundle (.pt file produced by
create_dataset.py or remove_duplicates.py), optimising directly on the
training set until it memorises every sample (100% accuracy) or a maximum
epoch budget is reached.
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
MAX_EPOCHS: int = 1000
BATCH_SIZE: int = 0  # 0 or negative → full batch (good for memorising)
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 0.0
TARGET_ACCURACY: float = 1.0
PRINT_EVERY: int = 1
SEED: int = 42
HIDDEN_WIDTH: int = 2048
HIDDEN_DEPTH: int = 2  # number of hidden Linear layers (>=1)


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


class StereoMLP(nn.Module):
    """
    Flatten stereo stacks into a vector and push through several dense layers.
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        num_classes: int,
        hidden_width: int,
        hidden_depth: int,
    ):
        super().__init__()
        c, h, w = in_shape
        self.flatten_dim = c * h * w

        layers: List[nn.Module] = [nn.Flatten(), nn.Linear(self.flatten_dim, hidden_width), nn.GELU()]
        for _ in range(max(1, hidden_depth - 1)):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.GELU()])
        layers.append(nn.Linear(hidden_width, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

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

            running_loss += loss.item() * batch_x.size(0)
            running_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            running_total += batch_x.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

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
        description="Train an MLP on the stereo dataset bundle until it memorises the data."
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
        "--hidden-width",
        default=HIDDEN_WIDTH,
        type=int,
        help="Number of units in hidden layers.",
    )
    parser.add_argument(
        "--hidden-depth",
        default=HIDDEN_DEPTH,
        type=int,
        help="How many hidden Linear layers to stack (>=1).",
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
    in_shape = tuple(x_tensor.shape[1:])  # (C, H, W)
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

    model = StereoMLP(
        in_shape=in_shape,
        num_classes=num_classes,
        hidden_width=args.hidden_width,
        hidden_depth=max(1, args.hidden_depth),
    )

    print(
        f"Dataset: {args.bundle} (samples={num_samples}, classes={num_classes}, "
        f"flatten_dim={model.flatten_dim})"
    )
    print(
        f"Training configuration: epochs={args.epochs}, batch_size={batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, "
        f"hidden_width={args.hidden_width}, hidden_depth={max(1, args.hidden_depth)}"
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
    )

    model_path = args.bundle.with_name(args.bundle.stem + "-mlp-classifier.pt")
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_class": model.__class__.__name__,
            "in_shape": in_shape,
            "num_classes": num_classes,
            "config": {
                "epochs": args.epochs,
                "batch_size": batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "hidden_width": args.hidden_width,
                "hidden_depth": max(1, args.hidden_depth),
            },
            "orig_label_ids": payload.get("_orig_label_ids"),
        },
        model_path,
    )
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
