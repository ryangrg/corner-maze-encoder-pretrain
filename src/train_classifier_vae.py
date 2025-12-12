#!/usr/bin/env python3
"""
train_classifier_vae.py

Train a simple convolutional VAE on a dataset bundle until it perfectly reconstructs
every sample (within a tolerance) and save both the model weights and extracted embeddings.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import dataset_io

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/datasets/corner-maze-render-base-images-consolidated-acute-ds"
MODEL_OUTPUT = ROOT / "data/models/stereo_vae.pt"
EMBEDDINGS_OUTPUT_DIR = ROOT / "data/tables"
LATENT_DIM = 64
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 2000
RECON_TOLERANCE = 1e-2
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def load_dataset(bundle_path: Path) -> Tuple[TensorDataset, Dict[str, Any]]:
    bundle_path = bundle_path.expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {bundle_path}")

    payload = dataset_io.load_bundle(bundle_path)
    x = payload.get("x")
    if not isinstance(x, torch.Tensor):
        raise TypeError("Bundle missing tensor 'x'.")
    y = payload.get("y")
    if y is None:
        labels = payload.get("labels")
        if labels is None:
            labels = torch.arange(x.shape[0], dtype=torch.long)
        if isinstance(labels, torch.Tensor):
            y = labels.long()
        else:
            y = torch.tensor(list(labels), dtype=torch.long)
    elif not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(x.float(), y.long())
    payload["_orig_labels"] = y.tolist()
    return dataset, payload


class StereoVAE(nn.Module):
    def __init__(self, sample_shape: Tuple[int, int, int], latent_dim: int) -> None:
        super().__init__()
        channels, height, width = sample_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        encoder_out = 64 * height * width
        self.fc_mu = nn.Linear(encoder_out, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, encoder_out)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, height, width)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_input(z)
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = torch.nn.functional.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.shape[0]
    return recon_loss + kl


def train_vae(
    model: StereoVAE,
    dataloader: DataLoader,
    device: torch.device,
    max_epochs: int,
    tolerance: float,
) -> None:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss = vae_loss(recon, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        meets_tol = evaluate_reconstruction(model, dataloader, device, tolerance)
        status = "met" if meets_tol else "not met"
        print(f"[Epoch {epoch}] loss={avg_loss:.6f} | reconstruction tolerance {status} (≤ {tolerance})")
        if meets_tol:
            print("Reached reconstruction tolerance across dataset; stopping early.")
            break


def evaluate_reconstruction(
    model: StereoVAE,
    dataloader: DataLoader,
    device: torch.device,
    tolerance: float,
) -> bool:
    model.eval()
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            recon, _, _ = model(batch_x)
            if not torch.allclose(recon, batch_x, atol=tolerance):
                return False
    return True


def extract_embeddings(
    model: StereoVAE,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, List[List[List[float]]], List[List[List[float]]]]:
    model.eval()
    embeddings: List[torch.Tensor] = []
    recon_left: List[List[List[float]]] = []
    recon_right: List[List[List[float]]] = []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            recon, mu, _ = model(batch_x)
            embeddings.append(mu.cpu())
            recon_cpu = recon.cpu().numpy()
            for sample in recon_cpu:
                recon_left.append(sample[0].tolist())
                recon_right.append(sample[1].tolist())
    return torch.cat(embeddings, dim=0), recon_left, recon_right


def build_dataframe(
    payload: Dict[str, Any],
    embeddings: torch.Tensor,
    recon_left: Sequence,
    recon_right: Sequence,
) -> pd.DataFrame:
    label_names = payload.get("label_names")
    label_ids_raw = payload.get("labels")
    if label_names is None or label_ids_raw is None:
        raise ValueError("Bundle must contain 'label_names' and 'labels'.")
    label_names = [str(name) for name in label_names]
    if isinstance(label_ids_raw, torch.Tensor):
        label_ids = [int(v) for v in label_ids_raw.tolist()]
    else:
        label_ids = [int(v) for v in label_ids_raw]
    labels2label_names = payload.get("labels2label_names", {})
    x_tensor = payload.get("x")
    if not isinstance(x_tensor, torch.Tensor):
        raise TypeError("Bundle missing tensor 'x' for image export.")

    rows = []
    for idx, (name, label_id, embedding) in enumerate(zip(label_names, label_ids, embeddings)):
        left_eye = x_tensor[idx, 0].detach().cpu().numpy().tolist()
        right_eye = x_tensor[idx, 1].detach().cpu().numpy().tolist()
        poses = labels2label_names.get(str(label_id)) or labels2label_names.get(int(label_id)) or []
        rows.append(
            {
                "label_name": str(name),
                "label_id": int(label_id),
                "embedding": embedding.tolist(),
                "poses": list(poses),
                "left_eye_img": left_eye,
                "right_eye_img": right_eye,
                "recon_left_eye_img": recon_left[idx],
                "recon_right_eye_img": recon_right[idx],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VAE to reconstruct stereo images and export embeddings.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to dataset_io bundle.")
    parser.add_argument("--model-output", type=Path, default=MODEL_OUTPUT, help="Path to save trained model (.pt).")
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_OUTPUT_DIR, help="Directory for embeddings parquet.")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--tolerance", type=float, default=RECON_TOLERANCE)
    args = parser.parse_args()

    dataset, payload = load_dataset(args.dataset)
    sample_shape = tuple(dataset[0][0].shape)
    model = StereoVAE(sample_shape, args.latent_dim)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_vae(model, dataloader, torch.device(DEVICE), args.max_epochs, args.tolerance)

    model_output = args.model_output.expanduser().resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output)
    print(f"Saved VAE weights to {model_output}")

    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    embeddings, recon_left, recon_right = extract_embeddings(model, full_loader, torch.device(DEVICE))

    df = build_dataframe(payload, embeddings, recon_left, recon_right)
    args.embeddings_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    parquet_path = args.embeddings_dir / f"{args.dataset.stem}-vae-embeddings.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved embeddings to {parquet_path}")


if __name__ == "__main__":
    main()
