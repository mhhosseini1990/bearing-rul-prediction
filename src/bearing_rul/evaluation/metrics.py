"""Evaluation metrics for rolling-bearing RUL prediction."""

from __future__ import annotations

import numpy as np


def validate_predictions(y_true, y_pred):
    """Validate and return one-dimensional numeric arrays."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "Targets and predictions must be one-dimensional."
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation arrays cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Targets and predictions must have equal lengths."
        )

    if not np.isfinite(y_true).all():
        raise ValueError("Targets contain NaN or infinity.")

    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain NaN or infinity.")

    if np.any(y_true < 0):
        raise ValueError("Actual RUL cannot be negative.")

    return y_true, y_pred


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""

    y_true, y_pred = validate_predictions(y_true, y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""

    y_true, y_pred = validate_predictions(y_true, y_pred)

    return float(
        np.sqrt(np.mean((y_true - y_pred) ** 2))
    )


def calculate_metrics(y_true, y_pred):
    """Return MAE and RMSE in the same unit as the inputs."""

    y_true, y_pred = validate_predictions(y_true, y_pred)

    return {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "n_samples": len(y_true),
    }


def calculate_per_bearing_metrics(
    y_true,
    y_pred,
    bearing_ids,
):
    """Calculate MAE and RMSE separately for each bearing."""

    y_true, y_pred = validate_predictions(y_true, y_pred)

    bearing_ids = np.asarray(bearing_ids)

    if bearing_ids.ndim != 1:
        raise ValueError("Bearing IDs must be one-dimensional.")

    if len(bearing_ids) != len(y_true):
        raise ValueError(
            "Bearing IDs must match the number of predictions."
        )

    results = []

    for bearing_id in np.unique(bearing_ids):

        mask = bearing_ids == bearing_id

        metrics = calculate_metrics(
            y_true[mask],
            y_pred[mask],
        )

        results.append({
            "bearing_id": str(bearing_id),
            "mae_minutes": metrics["mae"],
            "rmse_minutes": metrics["rmse"],
            "n_sequences": metrics["n_samples"],
        })

    return results