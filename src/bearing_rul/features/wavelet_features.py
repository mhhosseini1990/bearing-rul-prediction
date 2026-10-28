"""Wavelet feature extraction for XJTU-SY vibration snapshots.

Configuration:
    Wavelet: db4
    Transform: Discrete Wavelet Transform (DWT)
    Level: 3
    Extension mode: symmetric

Each vibration snapshot has two channels:
    0: Horizontal
    1: Vertical

For each channel, four mean squared coefficient energies
are calculated in this order:
    A3, D3, D2, D1

Output:
    One NumPy vector containing eight features.

This module processes each snapshot independently.
It does not use future measurements or RUL labels.
"""

import numpy as np
import pywt


WAVELET = "db4"
LEVEL = 3
MODE = "symmetric"

FEATURE_NAMES = (
    "horizontal_A3_energy",
    "horizontal_D3_energy",
    "horizontal_D2_energy",
    "horizontal_D1_energy",
    "vertical_A3_energy",
    "vertical_D3_energy",
    "vertical_D2_energy",
    "vertical_D1_energy",
)


def extract_wavelet_features(
    signal: np.ndarray,
) -> np.ndarray:
    """Extract eight wavelet energy features from one snapshot.

    Args:
        signal:
            Two-dimensional vibration array with shape
            (number_of_samples, 2).

    Returns:
        A float64 NumPy array with shape (8,).

    Raises:
        ValueError:
            If input dimensions, channel count, signal length,
            or numeric values are invalid.
    """
    try:
        values = np.asarray(signal, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Vibration signal must contain numeric values"
        ) from exc

    if values.ndim != 2:
        raise ValueError(
            "Expected a 2D vibration array"
        )

    if values.shape[1] != 2:
        raise ValueError(
            "Expected exactly two vibration channels"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Vibration signal contains NaN or infinity"
        )

    wavelet = pywt.Wavelet(WAVELET)

    maximum_level = pywt.dwt_max_level(
        data_len=values.shape[0],
        filter_len=wavelet.dec_len,
    )

    if maximum_level < LEVEL:
        raise ValueError(
            f"Signal is too short for level-{LEVEL} "
            f"{WAVELET} decomposition"
        )

    features = []

    for channel_index in range(2):
        channel = values[:, channel_index]

        coefficients = pywt.wavedec(
            channel,
            wavelet=WAVELET,
            mode=MODE,
            level=LEVEL,
        )

        for coefficient_array in coefficients:
            energy = np.mean(
                np.square(coefficient_array)
            )

            features.append(float(energy))

    result = np.asarray(features, dtype=np.float64)

    if result.shape != (8,):
        raise ValueError(
            f"Unexpected feature shape: {result.shape}"
        )

    if not np.isfinite(result).all():
        raise ValueError(
            "Extracted features contain NaN or infinity"
        )

    return result