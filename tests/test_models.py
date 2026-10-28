"""Tests for LSTM and GRU RUL prediction architectures."""

import numpy as np
import pytest
import torch

from bearing_rul.models.lstm import LSTMRegressor
from bearing_rul.models.gru import GRURegressor


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a01_correct_output_shape(model_class):
    """A-01: models return one RUL prediction per sequence."""

    model = model_class(
        input_size=8,
        hidden_size=64,
        num_layers=1,
    )

    x = torch.randn(4, 20, 8)

    output = model(x)

    assert output.shape == (4, 1)


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a02_single_sequence(model_class):
    """A-02: batch size one is supported."""

    model = model_class(
        input_size=8,
        hidden_size=64,
    )

    x = torch.randn(1, 20, 8)

    output = model(x)

    assert output.shape == (1, 1)


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a03_output_is_finite(model_class):
    """A-03: outputs must contain finite values."""

    model = model_class(
        input_size=8,
        hidden_size=64,
    )

    x = torch.randn(4, 20, 8)

    output = model(x)

    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a04_gradient_flow(model_class):
    """A-04: model parameters receive gradients."""

    model = model_class(
        input_size=8,
        hidden_size=64,
    )

    x = torch.randn(4, 20, 8)
    targets = torch.tensor(
        [[10.0], [20.0], [30.0], [40.0]]
    )

    predictions = model(x)

    loss = torch.nn.functional.mse_loss(
        predictions,
        targets,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a05_reject_wrong_input_shape(model_class):
    """A-05: models reject inputs without three dimensions."""

    model = model_class(input_size=8)

    invalid_input = torch.randn(20, 8)

    with pytest.raises(ValueError):
        model(invalid_input)


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a06_reject_wrong_feature_count(model_class):
    """A-06: models require eight features per snapshot."""

    model = model_class(input_size=8)

    invalid_input = torch.randn(4, 20, 7)

    with pytest.raises(ValueError):
        model(invalid_input)


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a07_reject_nonfinite_input(model_class):
    """A-07: NaN and infinity must not enter the models."""

    model = model_class(input_size=8)

    x = torch.randn(4, 20, 8)
    x[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(x)


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a08_deterministic_inference(model_class):
    """A-08: identical input produces identical output in eval mode."""

    torch.manual_seed(42)

    model = model_class(
        input_size=8,
        hidden_size=64,
    )

    model.eval()

    x = torch.randn(4, 20, 8)

    with torch.no_grad():
        first = model(x)
        second = model(x)

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_a09_invalid_hyperparameters(model_class):
    """A-09: invalid model configuration is rejected."""

    with pytest.raises(ValueError):
        model_class(input_size=0)

    with pytest.raises(ValueError):
        model_class(input_size=8, hidden_size=0)

    with pytest.raises(ValueError):
        model_class(input_size=8, num_layers=0)


def test_a10_lstm_and_gru_have_different_parameter_counts():
    """A-10: recurrent architectures have different parameter counts."""

    lstm = LSTMRegressor(
        input_size=8,
        hidden_size=64,
        num_layers=1,
    )

    gru = GRURegressor(
        input_size=8,
        hidden_size=64,
        num_layers=1,
    )

    lstm_parameters = sum(
        parameter.numel()
        for parameter in lstm.parameters()
    )

    gru_parameters = sum(
        parameter.numel()
        for parameter in gru.parameters()
    )

    assert lstm_parameters > gru_parameters

    print(f"\nLSTM parameters: {lstm_parameters}")
    print(f"GRU parameters: {gru_parameters}")