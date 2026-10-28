"""LSTM regression model for rolling-bearing RUL prediction."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Predict RUL from a sequence of wavelet features."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()

        if input_size < 1:
            raise ValueError("input_size must be positive")

        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one RUL prediction per input sequence."""

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, sequence, features)"
            )

        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {x.shape[2]}"
            )

        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Batch and sequence length must be positive")

        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN or infinity")

        sequence_output, _ = self.lstm(x)

        last_hidden_state = sequence_output[:, -1, :]

        prediction = self.output_layer(last_hidden_state)

        return prediction