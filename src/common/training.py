"""
common.training
===============
Shared PyTorch training loop with early stopping, used by FNN, MDN, and CVAE.

Provides:
  - temporal_val_split() — split arrays into train/val by temporal order
  - EarlyStopper         — early stopping tracker
  - train_loop()         — generic training loop with early stopping
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import (
    BATCH_SIZE,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    VAL_FRAC,
    WEIGHT_DECAY,
)


# ===================================================================
# TEMPORAL VALIDATION SPLIT
# ===================================================================
def temporal_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = VAL_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into train/val by temporal order (last val_frac rows = val).

    Returns (X_train, y_train, X_val, y_val)
    """
    n = len(X)
    n_val = max(1, int(n * val_frac))
    n_tr = n - n_val
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


# ===================================================================
# EARLY STOPPER
# ===================================================================
class EarlyStopper:
    """Track best validation loss and stop when patience runs out."""

    def __init__(self, patience: int = PATIENCE):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.wait = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        Call after each epoch with the validation loss.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load best checkpoint into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ===================================================================
# GENERIC TRAINING LOOP
# ===================================================================
def train_loop(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    train_step_fn: Callable,
    val_loss_fn: Callable,
    device: torch.device,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    val_frac: float = VAL_FRAC,
    seed: int = RANDOM_SEED,
    epoch_callback: Optional[Callable] = None,
) -> float:
    """
    Generic training loop with early stopping.

    Parameters
    ----------
    model : nn.Module
        The model to train (already on device).
    X_train, y_train : np.ndarray
        Full training data (will be split into train/val temporally).
    train_step_fn : Callable(model, xb, yb, optimizer, **kwargs) -> loss
        One training step. Called for each batch.  Should do forward,
        backward, optimizer step and return loss value.
    val_loss_fn : Callable(model, X_val_tensor, y_val_tensor, **kwargs) -> float
        Compute validation loss.  Called once per epoch after train phase.
    device : torch.device
    lr, weight_decay, batch_size, max_epochs, patience, val_frac, seed :
        Standard hyperparameters.
    epoch_callback : optional Callable(epoch: int) -> dict
        Called at start of each epoch. Return dict is passed as **kwargs
        to train_step_fn and val_loss_fn (e.g. for β-warmup).

    Returns
    -------
    best_val_loss : float
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Temporal split
    Xtr_np, ytr_np, Xva_np, yva_np = temporal_val_split(
        X_train, y_train, val_frac
    )

    Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
    ytr = torch.tensor(ytr_np, dtype=torch.float32)
    Xva = torch.tensor(Xva_np, dtype=torch.float32)
    yva = torch.tensor(yva_np, dtype=torch.float32)

    train_dl = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    stopper = EarlyStopper(patience=patience)

    for epoch in range(max_epochs):
        # Optional per-epoch callback (e.g. compute beta for CVAE)
        extra_kwargs = {}
        if epoch_callback is not None:
            extra_kwargs = epoch_callback(epoch) or {}

        # ── Train phase ──
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            train_step_fn(model, xb, yb, optimiser, **extra_kwargs)

        # ── Validation phase ──
        model.eval()
        with torch.no_grad():
            val_loss = val_loss_fn(
                model, Xva.to(device), yva.to(device), **extra_kwargs
            )

        # Handle NaN / inf validation loss
        import math
        if math.isnan(val_loss) or math.isinf(val_loss):
            continue

        scheduler.step(val_loss)

        if stopper.step(val_loss, model):
            break

    stopper.restore(model)
    model.eval()
    return stopper.best_loss
