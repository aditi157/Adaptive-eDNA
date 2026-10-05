"""
Autoencoder representation-learning stage.

Fix applied relative to the original prototype: the original trained on
ALL sequences and then re-used the SAME sequences to produce embeddings,
which is not a valid held-out evaluation. This module explicitly splits
features into train/test, trains only on the training subset, and reports
reconstruction MSE on the untouched test subset.

Terminology (see project notes):
  - The autoencoder is TRAINED (train/test split, train MSE, test MSE).
  - K-Means is FITTED on embeddings, not "trained" (see
    baseline_clustering.py).
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from scripts.config import (
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_HIDDEN_DIM,
    AE_LATENT_DIM,
    AE_LEARNING_RATE,
    AE_TEST_SIZE,
    RANDOM_SEED,
)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = AE_HIDDEN_DIM,
                 latent_dim: int = AE_LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def split_features(
    features: np.ndarray, test_size: float = AE_TEST_SIZE, seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split feature rows into train/test subsets. Returns
    (train_features, test_features, train_idx, test_idx) where the index
    arrays let the caller map rows back to sequence ids.
    """
    n = features.shape[0]
    idx = np.arange(n)

    if n < 5:
        # Too few sequences for a meaningful held-out split; train on
        # everything and reuse it for a (non-rigorous) sanity-check MSE.
        return features, features, idx, idx

    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed
    )
    return features[train_idx], features[test_idx], train_idx, test_idx


def train_autoencoder(
    train_features: np.ndarray,
    epochs: int = AE_EPOCHS,
    batch_size: int = AE_BATCH_SIZE,
    lr: float = AE_LEARNING_RATE,
    seed: int = RANDOM_SEED,
) -> Tuple[Autoencoder, list]:
    """
    Train the autoencoder on train_features only. Returns the trained
    model and the per-epoch mean training loss (MSE) history.
    """
    torch.manual_seed(seed)

    train_tensor = torch.tensor(train_features, dtype=torch.float)
    model = Autoencoder(train_tensor.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    effective_batch_size = min(batch_size, len(train_tensor))
    dataloader = DataLoader(
        TensorDataset(train_tensor), batch_size=effective_batch_size, shuffle=True
    )

    model.train()
    loss_history = []
    for _ in range(epochs):
        total_loss, n_batches = 0.0, 0
        for (batch,) in dataloader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        loss_history.append(total_loss / max(n_batches, 1))

    return model, loss_history


def evaluate_autoencoder(model: Autoencoder, features: np.ndarray) -> float:
    """
    Return reconstruction MSE of `model` on `features` (no gradient
    updates). Intended for use on the held-out test subset.
    """
    model.eval()
    tensor = torch.tensor(features, dtype=torch.float)
    criterion = nn.MSELoss()
    with torch.no_grad():
        recon, _ = model(tensor)
        mse = criterion(recon, tensor).item()
    return mse


def get_embeddings(model: Autoencoder, features: np.ndarray) -> np.ndarray:
    """Return the latent-space embeddings for `features`."""
    model.eval()
    tensor = torch.tensor(features, dtype=torch.float)
    with torch.no_grad():
        _, z = model(tensor)
    return z.numpy()


def save_model(model: Autoencoder, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str, input_dim: int) -> Autoencoder:
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
