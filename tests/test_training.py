"""Tests for RUL target scaling and PyTorch datasets."""

import numpy as np
import pytest
import torch

from bearing_rul.training.dataset import (
    RULTargetScaler,
    BearingRULDataset,
    create_dataloader,
    prepare_rul_dataloaders,
)


def make_sequences(n=10):
    """Create synthetic sequences for unit testing."""

    rng = np.random.default_rng(42)

    X = rng.normal(size=(n, 20, 8)).astype(np.float32)
    y = np.linspace(0, 100, n, dtype=np.float32)

    return X, y


def test_t01_scaler_fits_training_maximum():
    scaler = RULTargetScaler()

    scaler.fit(np.array([0, 100, 200, 500]))

    assert scaler.scale_ == 500.0


def test_t02_scaler_transforms_targets():
    scaler = RULTargetScaler().fit([0, 100, 200])

    result = scaler.transform([0, 50, 100, 200])

    np.testing.assert_allclose(
        result,
        [0.0, 0.25, 0.5, 1.0],
    )


def test_t03_scaler_inverse_transform():
    scaler = RULTargetScaler().fit([0, 100, 200])

    result = scaler.inverse_transform([0.0, 0.5, 1.0])

    np.testing.assert_allclose(
        result,
        [0.0, 100.0, 200.0],
    )


def test_t04_scaler_round_trip():
    y = np.array([0, 25, 100, 200], dtype=np.float32)

    scaler = RULTargetScaler()

    scaled = scaler.fit_transform(y)
    restored = scaler.inverse_transform(scaled)

    np.testing.assert_allclose(restored, y)


def test_t05_scaler_rejects_transform_before_fit():
    scaler = RULTargetScaler()

    with pytest.raises(RuntimeError):
        scaler.transform([10, 20])


def test_t06_scaler_rejects_empty_training_targets():
    scaler = RULTargetScaler()

    with pytest.raises(ValueError):
        scaler.fit([])


def test_t07_scaler_rejects_invalid_training_targets():
    scaler = RULTargetScaler()

    with pytest.raises(ValueError):
        scaler.fit([10, np.nan, 30])


def test_t08_scaler_rejects_negative_rul():
    scaler = RULTargetScaler()

    with pytest.raises(ValueError):
        scaler.fit([-1, 10, 20])


def test_t09_scaler_handles_zero_maximum():
    scaler = RULTargetScaler().fit([0, 0, 0])

    assert scaler.scale_ == 1.0

    np.testing.assert_allclose(
        scaler.transform([0, 0]),
        [0, 0],
    )


def test_t10_dataset_length():
    X, y = make_sequences(10)

    dataset = BearingRULDataset(X, y)

    assert len(dataset) == 10


def test_t11_dataset_tensor_shapes():
    X, y = make_sequences(10)

    dataset = BearingRULDataset(X, y)

    features, target = dataset[0]

    assert features.shape == (20, 8)
    assert target.shape == (1,)


def test_t12_dataset_uses_float32():
    X, y = make_sequences(10)

    dataset = BearingRULDataset(X, y)

    features, target = dataset[0]

    assert features.dtype == torch.float32
    assert target.dtype == torch.float32


def test_t13_dataset_rejects_wrong_input_shape():
    X = np.zeros((10, 8))
    y = np.zeros(10)

    with pytest.raises(ValueError):
        BearingRULDataset(X, y)


def test_t14_dataset_rejects_mismatched_lengths():
    X = np.zeros((10, 20, 8))
    y = np.zeros(9)

    with pytest.raises(ValueError):
        BearingRULDataset(X, y)


def test_t15_dataset_rejects_nonfinite_features():
    X, y = make_sequences(10)

    X[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        BearingRULDataset(X, y)


def test_t16_dataloader_batch_shapes():
    X, y = make_sequences(10)

    loader = create_dataloader(
        X,
        y,
        batch_size=4,
    )

    features, targets = next(iter(loader))

    assert features.shape == (4, 20, 8)
    assert targets.shape == (4, 1)


def test_t17_dataloader_rejects_invalid_batch_size():
    X, y = make_sequences(10)

    with pytest.raises(ValueError):
        create_dataloader(X, y, batch_size=0)


def test_t18_train_only_target_scaling():
    X_train, y_train = make_sequences(10)
    X_val, _ = make_sequences(4)
    X_test, _ = make_sequences(3)

    y_val = np.array([100, 150, 200, 300], dtype=np.float32)
    y_test = np.array([200, 400, 600], dtype=np.float32)

    result = prepare_rul_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=4,
    )

    scaler = result["target_scaler"]

    # The scale must come only from training targets.
    assert scaler.scale_ == 100.0

    # Validation and test must not change the scale.
    assert scaler.transform([300])[0] == pytest.approx(3.0)
    assert scaler.transform([600])[0] == pytest.approx(6.0)


def test_t19_all_dataloaders_created():
    X_train, y_train = make_sequences(10)
    X_val, y_val = make_sequences(4)
    X_test, y_test = make_sequences(3)

    result = prepare_rul_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    assert len(result["train"].dataset) == 10
    assert len(result["validation"].dataset) == 4
    assert len(result["test"].dataset) == 3