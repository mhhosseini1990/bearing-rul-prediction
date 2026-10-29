"""Model evaluation utilities for bearing RUL prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bearing_rul.models.lstm import LSTMRegressor
from bearing_rul.models.gru import GRURegressor


MODEL_CLASSES = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
}


def load_trained_model(
    model_name,
    checkpoint_path,
    *,
    input_size=8,
    hidden_size=64,
    num_layers=1,
    device="cpu",
):
    """Load a trained LSTM or GRU model."""

    model_name = model_name.lower()

    if model_name not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    expected_class = MODEL_CLASSES[model_name].__name__

    if checkpoint.get("model_class") != expected_class:
        raise ValueError(
            "Checkpoint model class does not match "
            f"the requested model: {model_name}"
        )

    target_scale = checkpoint.get("target_scale")

    if (
        target_scale is None
        or not np.isfinite(target_scale)
        or target_scale <= 0
    ):
        raise ValueError(
            "Checkpoint contains an invalid target scale."
        )

    model = MODEL_CLASSES[model_name](
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, float(target_scale)


def predict_rul(
    model,
    dataloader,
    target_scale,
    *,
    device="cpu",
    clip_negative=True,
):
    """
    Predict RUL and convert outputs back to minutes.

    Returns:
        y_true_minutes
        y_pred_minutes
    """

    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Invalid target scale.")

    device = torch.device(device)

    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for features, targets in dataloader:

            features = features.to(device)

            predictions = model(features)

            all_predictions.append(
                predictions.cpu().numpy().reshape(-1)
            )

            all_targets.append(
                targets.cpu().numpy().reshape(-1)
            )

    if not all_predictions:
        raise ValueError("Evaluation DataLoader is empty.")

    y_pred_scaled = np.concatenate(all_predictions)
    y_true_scaled = np.concatenate(all_targets)

    y_pred_minutes = (
        y_pred_scaled.astype(np.float64) * target_scale
    )

    y_true_minutes = (
        y_true_scaled.astype(np.float64) * target_scale
    )

    if clip_negative:
        y_pred_minutes = np.maximum(
            y_pred_minutes,
            0.0,
        )

    return y_true_minutes, y_pred_minutes