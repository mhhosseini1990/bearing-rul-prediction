"""Training utilities for LSTM and GRU RUL regression."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> float:
    """
    Run one training or validation epoch.

    If optimizer is provided, update model parameters.
    Otherwise, run validation without gradient updates.

    Returns mean loss per sequence.
    """

    device = torch.device(device)
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:

        features = features.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            predictions = model(features)

            if predictions.shape != targets.shape:
                raise ValueError(
                    "Prediction and target shapes do not match."
                )

            loss = criterion(predictions, targets)

        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss encountered.")

        if training:
            loss.backward()
            optimizer.step()

        batch_size = features.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Dataloader cannot be empty.")

    return total_loss / total_samples


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    epochs: int = 50,
    learning_rate: float = 0.001,
    patience: int = 7,
    min_delta: float = 0.0,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    target_scale: float | None = None,
) -> dict:
    """
    Train a RUL regression model using early stopping.

    The model is trained on scaled RUL targets.

    The best model is selected using validation MSE.
    Its weights are restored before returning.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if patience < 1:
        raise ValueError("patience must be positive.")

    if min_delta < 0:
        raise ValueError("min_delta cannot be negative.")

    if target_scale is not None:
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("target_scale must be positive and finite.")

    device = torch.device(device)
    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

        if val_loss < best_val_loss - min_delta:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            if checkpoint_path is not None:

                checkpoint_path = Path(checkpoint_path)

                checkpoint_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                checkpoint = {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "target_scale": target_scale,
                    "model_class": type(model).__name__,
                }

                torch.save(
                    checkpoint,
                    checkpoint_path,
                )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )

            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)
    model.eval()

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history["train_loss"]),
        "checkpoint_path": (
            str(checkpoint_path)
            if checkpoint_path is not None
            else None
        ),
    }