"""Tests for XJTU-SY wavelet feature extraction."""

import numpy as np
import pytest

from bearing_rul.features.wavelet_features import (
    extract_wavelet_features,
    FEATURE_NAMES,
)


def test_f01_output_has_eight_features():
    """F-01: two vibration channels produce eight features."""
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(32768, 2))

    features = extract_wavelet_features(signal)

    assert isinstance(features, np.ndarray)
    assert features.shape == (8,)
    assert len(FEATURE_NAMES) == 8


def test_f02_features_are_finite_and_nonnegative():
    """F-02: coefficient energies must be finite and nonnegative."""
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(1024, 2))

    features = extract_wavelet_features(signal)

    assert np.isfinite(features).all()
    assert (features >= 0).all()


def test_f03_zero_signal_has_zero_energy():
    """F-03: zero vibration produces zero wavelet energy."""
    signal = np.zeros((1024, 2))

    features = extract_wavelet_features(signal)

    np.testing.assert_allclose(features, np.zeros(8), atol=1e-12)


def test_f04_channel_independence():
    """F-04: changing horizontal data must not change vertical features."""
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(1024, 2))

    original = extract_wavelet_features(signal)

    modified = signal.copy()
    modified[:, 0] *= 3

    changed = extract_wavelet_features(modified)

    np.testing.assert_allclose(original[4:], changed[4:])
    assert not np.allclose(original[:4], changed[:4])


def test_f05_energy_scales_quadratically():
    """F-05: multiplying amplitude by 2 multiplies energy by 4."""
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(1024, 2))

    original = extract_wavelet_features(signal)
    scaled = extract_wavelet_features(signal * 2)

    np.testing.assert_allclose(scaled, original * 4, rtol=1e-10)


def test_f06_reject_invalid_input_shape():
    """F-06: inputs must have exactly two channels."""
    with pytest.raises(ValueError):
        extract_wavelet_features(np.zeros(1024))

    with pytest.raises(ValueError):
        extract_wavelet_features(np.zeros((1024, 1)))

    with pytest.raises(ValueError):
        extract_wavelet_features(np.zeros((1024, 3)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_f07_reject_nonfinite_values(bad_value):
    """F-07: NaN and infinity must be rejected."""
    signal = np.zeros((1024, 2))
    signal[0, 0] = bad_value

    with pytest.raises(ValueError):
        extract_wavelet_features(signal)


def test_f08_deterministic_output():
    """F-08: identical input produces identical features."""
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(1024, 2))

    first = extract_wavelet_features(signal)
    second = extract_wavelet_features(signal)

    np.testing.assert_array_equal(first, second)


def test_f09_reject_short_signal():
    """F-09: insufficient samples for level-3 db4 are rejected."""
    signal = np.zeros((8, 2))

    with pytest.raises(ValueError):
        extract_wavelet_features(signal)


def test_f10_feature_names_follow_channel_and_band_order():
    """F-10: feature names match the documented output order."""
    assert FEATURE_NAMES == (
        "horizontal_A3_energy",
        "horizontal_D3_energy",
        "horizontal_D2_energy",
        "horizontal_D1_energy",
        "vertical_A3_energy",
        "vertical_D3_energy",
        "vertical_D2_energy",
        "vertical_D1_energy",
    )


def test_f11_input_is_not_modified():
    """F-11: extraction must not modify the original signal."""
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(1024, 2))
    original = signal.copy()

    extract_wavelet_features(signal)

    np.testing.assert_array_equal(signal, original)