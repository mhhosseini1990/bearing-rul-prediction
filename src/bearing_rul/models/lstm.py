"""LSTM regressor for direct rolling-bearing RUL prediction from fixed48_v1 features.

Expected input: (batch, sequence_length, 48), e.g. (32, 20, 48).
Expected output: (batch, 1), predicted RUL in the target's training units.
Input normalization and target scaling belong to the training pipeline.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LSTMRegressor(nn.Module):
    """Predict a continuous RUL value from a sequence of fixed vibration features."""

    def __init__(
        self,
        input_size: int = 48,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        if output_activation not in {"linear", "softplus"}:
            raise ValueError("output_activation must be linear or softplus")
        self.output_activation = output_activation
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one continuous RUL estimate for each sequence."""
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, sequence, features)")
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, received {x.shape[2]}"
            )
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Batch and sequence length must be positive")
        if not torch.is_floating_point(x):
            raise ValueError("Input must be a floating-point tensor")
        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN or infinity")

        sequence_output, _ = self.lstm(x)
        last_hidden_state = sequence_output[:, -1, :]
        prediction = self.output_layer(last_hidden_state)
        return F.softplus(prediction) if self.output_activation == "softplus" else prediction
