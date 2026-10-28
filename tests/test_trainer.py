"""Tests for the RUL model training pipeline."""

import numpy as np
import pytest
import torch

from torch import nn

from bearing_rul.models.lstm import LSTMRegressor
from bearing_rul.models.gru import GRURegressor

from bearing_rul.training.dataset import create_dataloader

from bearing_rul.training.trainer import (
    set_seed,
    run_epoch,
    train_model,
)


def make_loaders():
    """Create small synthetic datasets for training tests."""

    rng = np.random.default_rng(42)

    X_train = rng.normal(
        size=(24, 20, 8)
    ).astype(np.float32)

    y_train = rng.uniform(
        0,
        1,
        size=24,
    ).astype(np.float32)

    X_val = rng.normal(
        size=(8, 20, 8)
    ).astype(np.float32)

    y_val = rng.uniform(
        0,
        1,
        size=8,
    ).astype(np.float32)

    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=8,
        shuffle=False,
    )

    val_loader = create_dataloader(
        X_val,
        y_val,
        batch_size=4,
        shuffle=False,
    )

    return train_loader, val_loader


def test_tr01_seed_reproducibility():
    set_seed(42)
    first = torch.randn(5)

    set_seed(42)
    second = torch.randn(5)

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_tr02_training_epoch_updates_weights(model_class):

    train_loader, _ = make_loaders()

    model = model_class(
        input_size=8,
        hidden_size=8,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    loss = run_epoch(
        model,
        train_loader,
        nn.MSELoss(),
        optimizer=optimizer,
    )

    assert np.isfinite(loss)

    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_tr03_validation_does_not_update_weights(model_class):

    _, val_loader = make_loaders()

    model = model_class(
        input_size=8,
        hidden_size=8,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    loss = run_epoch(
        model,
        val_loader,
        nn.MSELoss(),
        optimizer=None,
    )

    assert np.isfinite(loss)

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            before[name],
            parameter,
        )


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_tr04_training_returns_history(model_class):

    train_loader, val_loader = make_loaders()

    model = model_class(
        input_size=8,
        hidden_size=8,
    )

    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=3,
        patience=2,
    )

    assert len(result["history"]["train_loss"]) >= 1

    assert len(result["history"]["val_loss"]) >= 1

    assert result["best_epoch"] >= 1


@pytest.mark.parametrize(
    "model_class",
    [LSTMRegressor, GRURegressor],
)
def test_tr05_checkpoint_is_saved(model_class, tmp_path):

    train_loader, val_loader = make_loaders()

    model = model_class(
        input_size=8,
        hidden_size=8,
    )

    checkpoint_path = tmp_path / "best_model.pt"

    train_model(
        model,
        train_loader,
        val_loader,
        epochs=2,
        patience=2,
        checkpoint_path=checkpoint_path,
        target_scale=2500.0,
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert "model_state_dict" in checkpoint
    assert "best_epoch" in checkpoint
    assert checkpoint["target_scale"] == 2500.0


def test_tr06_invalid_epochs():

    train_loader, val_loader = make_loaders()

    model = LSTMRegressor(hidden_size=8)

    with pytest.raises(ValueError):
        train_model(
            model,
            train_loader,
            val_loader,
            epochs=0,
        )


def test_tr07_invalid_learning_rate():

    train_loader, val_loader = make_loaders()

    model = LSTMRegressor(hidden_size=8)

    with pytest.raises(ValueError):
        train_model(
            model,
            train_loader,
            val_loader,
            learning_rate=0,
        )


def test_tr08_invalid_patience():

    train_loader, val_loader = make_loaders()

    model = GRURegressor(hidden_size=8)

    with pytest.raises(ValueError):
        train_model(
            model,
            train_loader,
            val_loader,
            patience=0,
        )


def test_tr09_invalid_target_scale():

    train_loader, val_loader = make_loaders()

    model = GRURegressor(hidden_size=8)

    with pytest.raises(ValueError):
        train_model(
            model,
            train_loader,
            val_loader,
            target_scale=0,
        )


def test_tr10_empty_dataloader_rejected():

    model = LSTMRegressor(hidden_size=8)

    empty_loader = []

    with pytest.raises(ValueError):
        run_epoch(
            model,
            empty_loader,
            nn.MSELoss(),
        )


def test_tr11_early_stopping():

    train_loader, val_loader = make_loaders()

    model = LSTMRegressor(hidden_size=8)

    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=10,
        patience=2,
        min_delta=1e6,
    )

    # A huge min_delta prevents improvements after epoch 1.
    assert result["epochs_completed"] == 3
    assert result["best_epoch"] == 1


def test_tr12_best_model_is_restored(tmp_path):

    train_loader, val_loader = make_loaders()

    model = GRURegressor(hidden_size=8)

    checkpoint_path = tmp_path / "best_gru.pt"

    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=4,
        patience=2,
        min_delta=1e6,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    for name, parameter in result["model"].state_dict().items():
        torch.testing.assert_close(
            parameter.cpu(),
            checkpoint["model_state_dict"][name],
        )