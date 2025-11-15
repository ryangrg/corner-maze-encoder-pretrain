#!/usr/bin/env python3
"""
image-calssifier-vac.py

Variational auto-encoder for the stereo maze dataset bundles (.pt files produced by
create_dataset.py / remove_duplicates.py). The model encodes each stereo pair into a
latent Gaussian distribution, reconstructs the original image stack, and can be used to
generate embeddings or sampled reconstructions.
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

ROOT: Path = Path(__file__).resolve().parents[1]
DATASET_PATH: Path = ROOT / "data/pt-files/dedup-all-images-dataset.pt"
OUTPUT_DIR: Path = ROOT / "outputs/vae"

LATENT_DIM: int = 128
HIDDEN_CHANNELS: List[int] = [32, 64, 128, 256]
MAX_EPOCHS: int = 300
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 0.0
TARGET_LOSS: float = 1e-4  # stop early if total loss drops below this
PRINT_EVERY: int = 1
SEED: int = 42
CLAMP_OUTPUT: bool = True  # keep reconstructions in [0, 1]


def load_dataset(bundle_path: Path) -> Tuple[TensorDataset, Dict[str, Any]]:
    bundle_path = bundle_path.expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {bundle_path}")

    payload = torch.load(bundle_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, found {type(payload)!r}")
    if "x" not in payload:
        raise KeyError("Payload missing 'x' tensor.")

    x = payload["x"]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"'x' must be a torch.Tensor (got {type(x)!r})")
    x = x.float()

    dataset = TensorDataset(x)
    return dataset, payload


class ConvVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_channels: List[int]):
        super().__init__()
        if any(c <= 0 for c in hidden_channels):
            raise ValueError("Hidden channels must all be positive.")

        encoder_layers: List[nn.Module] = []
        current_c = in_channels
        for out_c in hidden_channels:
            encoder_layers.extend(
                [
                    nn.Conv2d(current_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            )
            current_c = out_c

        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            enc_out = self.encoder(dummy)
            enc_shape = enc_out.shape  # (1, C, H, W)

        self._enc_feat_shape = enc_shape[1:]
        enc_flat_dim = int(torch.numel(enc_out))

        self.fc_mu = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, enc_flat_dim)

        decoder_layers: List[nn.Module] = []
        hidden_rev = list(reversed(hidden_channels))
        for idx, out_c in enumerate(hidden_rev[1:] + [in_channels]):
            in_c = hidden_rev[idx]
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_c,
                        out_c,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_c) if idx != len(hidden_rev) - 1 else nn.Identity(),
                    nn.ReLU(inplace=True) if idx != len(hidden_rev) - 1 else nn.Identity(),
                ]
            )

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        flat = features.flatten(start_dim=1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(z.size(0), *self._enc_feat_shape)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        if CLAMP_OUTPUT:
            recon = recon.clamp(0.0, 1.0)
        return recon, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl) / x.size(0)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    target_loss: float,
    print_every: int,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch_x)
            loss = vae_loss(recon, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        history.append(epoch_loss)

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={epoch_loss:.6f}")

        if epoch_loss <= target_loss:
            print(f"Target loss {target_loss:.2e} reached at epoch {epoch}. Stopping early.")
            break
    else:
        print(
            f"Reached maximum epochs ({epochs}) with final loss {history[-1]:.6f}. "
            "Increase epochs or adjust hyperparameters if lower loss is required."
        )

    return history


def save_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    count: int = 16,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            recon, _, _ = model(batch_x)
            for idx in range(batch_x.size(0)):
                if saved >= count:
                    return
                orig = batch_x[idx].cpu()
                rec = recon[idx].cpu()
                torch.save({"original": orig, "reconstruction": rec}, output_dir / f"pair_{saved:03d}.pt")
                saved += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a variational auto-encoder on the stereo dataset bundle.")
    parser.add_argument("--bundle", default=DATASET_PATH, type=Path, help="Path to the dataset bundle (.pt).")
    parser.add_argument("--output", default=OUTPUT_DIR, type=Path, help="Directory to store VAE artifacts.")
    parser.add_argument("--latent-dim", default=LATENT_DIM, type=int, help="Dimensionality of the latent space.")
    parser.add_argument(
        "--hidden-channels",
        default=",".join(map(str, HIDDEN_CHANNELS)),
        help="Comma-separated list of encoder channel sizes (four values recommended).",
    )
    parser.add_argument("--epochs", default=MAX_EPOCHS, type=int, help="Maximum training epochs.")
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int, help="Samples per batch (<=0 → full batch).")
    parser.add_argument("--lr", default=LEARNING_RATE, type=float, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", default=WEIGHT_DECAY, type=float, help="L2 weight decay.")
    parser.add_argument("--target-loss", default=TARGET_LOSS, type=float, help="Early stop once loss <= target.")
    parser.add_argument("--device", default=None, help="Training device ('cuda', 'mps', 'cpu'). Auto-detect when None.")
    parser.add_argument("--print-every", default=PRINT_EVERY, type=int, help="Print metrics every N epochs.")
    parser.add_argument("--seed", default=SEED, type=int, help="Random seed.")
    parser.add_argument(
        "--save-recon",
        default=16,
        type=int,
        help="Number of reconstructions to save after training (0 disables).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hidden_channels = [int(v) for v in str(args.hidden_channels).split(",") if v.strip()]
    if len(hidden_channels) < 1:
        raise ValueError("Provide at least one hidden channel value.")

    dataset, _ = load_dataset(args.bundle)
    x_tensor: torch.Tensor = dataset.tensors[0]
    num_samples = len(dataset)
    in_channels = x_tensor.shape[1]

    batch_size = num_samples if args.batch_size is None or args.batch_size <= 0 else args.batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

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

    model = ConvVAE(in_channels=in_channels, latent_dim=args.latent_dim, hidden_channels=hidden_channels)

    print(
        f"Dataset: {args.bundle} (samples={num_samples}, channels={in_channels}, "
        f"input_shape={tuple(x_tensor.shape[1:])})"
    )
    print(
        f"Training configuration: epochs={args.epochs}, batch_size={batch_size}, lr={args.lr}, "
        f"hidden_channels={hidden_channels}, latent_dim={args.latent_dim}"
    )

    history = train(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        target_loss=args.target_loss,
        print_every=max(1, args.print_every),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / (args.bundle.stem + "-vae.pt")
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "latent_dim": args.latent_dim,
            "hidden_channels": hidden_channels,
            "history": history,
            "config": {
                "epochs": args.epochs,
                "batch_size": batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            },
        },
        model_path,
    )
    print(f"Saved trained VAE to {model_path}")

    if args.save_recon > 0:
        save_reconstructions(
            model=model.to(device),
            dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=False),
            device=device,
            output_dir=args.output / "reconstructions",
            count=args.save_recon,
        )
        print(f"Saved {args.save_recon} reconstruction pairs to {args.output / 'reconstructions'}")


if __name__ == "__main__":
    main()
