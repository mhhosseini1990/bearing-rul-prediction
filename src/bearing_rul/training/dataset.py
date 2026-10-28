"""PyTorch dataset utilities and train-only RUL target scaling."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RULTargetScaler:
    """
    Scale RUL targets using the maximum RUL in training data.

    The scaling factor is fitted only on training targets.
    Validation and test targets use the same factor.
    """

    def __init__(self):
        self.scale_: float | None = None

    def fit(self, y_train):
        """Fit the scaling factor using training targets only."""

        y_train = np.asarray(y_train, dtype=np.float64)

        if y_train.ndim != 1:
            raise ValueError("Training targets must be one-dimensional.")

        if y_train.size == 0:
            raise ValueError("Training targets cannot be empty.")

        if not np.isfinite(y_train).all():
            raise ValueError("Training targets contain NaN or infinity.")

        if np.any(y_train < 0):
            raise ValueError("RUL targets cannot be negative.")

        maximum = float(np.max(y_train))

        # Avoid division by zero if all training targets are zero.
        self.scale_ = maximum if maximum > 0 else 1.0

        return self

    def transform(self, y):
        """Convert RUL minutes into scaled targets."""

        if self.scale_ is None:
            raise RuntimeError("Target scaler must be fitted first.")

        y = np.asarray(y, dtype=np.float64)

        if not np.isfinite(y).all():
            raise ValueError("Targets contain NaN or infinity.")

        if np.any(y < 0):
            raise ValueError("RUL targets cannot be negative.")

        return (y / self.scale_).astype(np.float32)

    def inverse_transform(self, y_scaled):
        """Convert scaled predictions back into RUL minutes."""

        if self.scale_ is None:
            raise RuntimeError("Target scaler must be fitted first.")

        y_scaled = np.asarray(y_scaled, dtype=np.float64)

        if not np.isfinite(y_scaled).all():
            raise ValueError("Predictions contain NaN or infinity.")

        return y_scaled * self.scale_

    def fit_transform(self, y_train):
        """Fit on training targets and transform them."""

        return self.fit(y_train).transform(y_train)


class BearingRULDataset(Dataset):
    """
    PyTorch dataset for bearing RUL sequence regression.

    X shape: (n_sequences, sequence_length, n_features)
    y shape: (n_sequences,)
    """

    def __init__(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                "X must have shape (samples, sequence_length, features)."
            )

        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same sample count.")

        if len(X) == 0:
            raise ValueError("Dataset cannot be empty.")

        if X.shape[1] == 0 or X.shape[2] == 0:
            raise ValueError(
                "Sequence length and feature count must be positive."
            )

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity.")

        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinity.")

        if np.any(y < 0):
            raise ValueError("RUL targets cannot be negative.")

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_dataloader(
    X,
    y,
    batch_size: int = 64,
    shuffle: bool = False,
):
    """Create a PyTorch DataLoader for sequence regression."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    dataset = BearingRULDataset(X, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def prepare_rul_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size: int = 64,
):
    """
    Fit target scaling on training data and build three DataLoaders.

    Input features must already be scaled by the train-fitted
    feature StandardScaler from the preprocessing pipeline.
    """

    target_scaler = RULTargetScaler()

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    train_loader = create_dataloader(
        X_train,
        y_train_scaled,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        X_val,
        y_val_scaled,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = create_dataloader(
        X_test,
        y_test_scaled,
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
        "target_scaler": target_scaler,
    }