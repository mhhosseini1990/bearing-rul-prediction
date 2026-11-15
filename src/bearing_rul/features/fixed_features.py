"""Fixed, deterministic 48-feature registry for XJTU-SY vibration snapshots.

This module consolidates feature extraction into one file; it NEVER performs
feature selection. Every snapshot always gets the same 48 features, in the same
order, whether it belongs to training, validation, or test.

Feature groups:
  *  8 DWT mean-square coefficient features: 4 bands x 2 channels
  * 16 WPT level-3 RMS features: 8 frequency-ordered nodes x 2 channels
  * 16 time-domain features: 8 statistics x 2 channels
  *  8 frequency-domain features: 4 statistics x 2 channels

Wavelet: db4; decomposition level: 3; extension mode: symmetric.
Channels: column 0 horizontal; column 1 vertical.
Default sampling frequency: 25,600 Hz.

Note: `*_energy` is an inherited legacy column name; the actual quantity is
mean-square DWT coefficient amplitude, not a sum of squared coefficients.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from typing import Final

import numpy as np
import pywt

FEATURE_SET_VERSION: Final[str] = "fixed48_v1"
WAVELET: Final[str] = "db4"
LEVEL: Final[int] = 3
MODE: Final[str] = "symmetric"
DEFAULT_SAMPLING_FREQUENCY_HZ: Final[float] = 25_600.0
CHANNELS: Final[tuple[str, str]] = ("horizontal", "vertical")
DWT_BANDS: Final[tuple[str, ...]] = ("A3", "D3", "D2", "D1")
TIME_FEATURES: Final[tuple[str, ...]] = (
    "rms",
    "standard_deviation",
    "absolute_peak",
    "peak_to_peak",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
)
FREQUENCY_FEATURES: Final[tuple[str, ...]] = (
    "spectrum_mean",
    "spectrum_standard_deviation",
    "spectrum_kurtosis",
    "frequency_centroid_hz",
)


def _group_names() -> OrderedDict[str, tuple[str, ...]]:
    return OrderedDict(
        (
            ("dwt", tuple(f"{ch}_{band}_energy" for ch in CHANNELS for band in DWT_BANDS)),
            (
                "wpt", tuple(f"{ch}_wpt_L3_{i:02d}_rms" for ch in CHANNELS for i in range(8))),
            ("time", tuple(f"{ch}_time_{name}" for ch in CHANNELS for name in TIME_FEATURES)),
            (
                "frequency",
                tuple(f"{ch}_frequency_{name}" for ch in CHANNELS for name in FREQUENCY_FEATURES),
            ),
        )
    )


FEATURE_GROUPS: Final[OrderedDict[str, tuple[str, ...]]] = _group_names()
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(name for group in FEATURE_GROUPS.values() for name in group)
INPUT_SIZE: Final[int] = len(FEATURE_NAMES)
FEATURE_SET_SHA256: Final[str] = sha256(
    (FEATURE_SET_VERSION + "|" + WAVELET + "|" + str(LEVEL) + "|" + MODE + "|" + "|".join(FEATURE_NAMES))
    .encode("utf-8")
).hexdigest()

assert INPUT_SIZE == 48, f"Expected exactly 48 fixed features, got {INPUT_SIZE}"
assert len(set(FEATURE_NAMES)) == INPUT_SIZE, "Feature names must be unique"


def feature_names() -> tuple[str, ...]:
    """Return the stable, immutable 48-feature column list."""
    return FEATURE_NAMES


def _validate_vibration(vibration: np.ndarray) -> np.ndarray:
    try:
        x = np.asarray(vibration, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Vibration must contain numeric data") from exc
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Expected vibration with shape (number_of_samples, 2)")
    if x.shape[0] < 16:
        raise ValueError("At least 16 samples per snapshot are required")
    if not np.isfinite(x).all():
        raise ValueError("Vibration contains NaN or infinity")
    maximum_level = pywt.dwt_max_level(x.shape[0], pywt.Wavelet(WAVELET).dec_len)
    if maximum_level < LEVEL:
        raise ValueError(f"Signal too short for level-{LEVEL} {WAVELET} DWT")
    return x


def _standardized_moment(values: np.ndarray, order: int) -> float:
    centered = values - values.mean()
    std = float(np.sqrt(np.mean(centered * centered)))
    if std <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.mean((centered / std) ** order))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > np.finfo(np.float64).eps else 0.0


def _time_features(x: np.ndarray) -> dict[str, float]:
    absx = np.abs(x)
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(absx.max())
    mean_abs = float(absx.mean())
    root_amplitude = float(np.mean(np.sqrt(absx)) ** 2)
    return {
        "rms": rms,
        "standard_deviation": float(np.std(x, ddof=0)),
        "absolute_peak": peak,
        "peak_to_peak": float(np.ptp(x)),
        "kurtosis": _standardized_moment(x, 4),  # Pearson kurtosis, not excess kurtosis
        "crest_factor": _safe_ratio(peak, rms),
        "impulse_factor": _safe_ratio(peak, mean_abs),
        "clearance_factor": _safe_ratio(peak, root_amplitude),
    }


def _frequency_features(x: np.ndarray, fs: float) -> dict[str, float]:
    magnitude = np.abs(np.fft.rfft(x)) / x.size
    # Single-sided amplitude: preserve DC and (when present) Nyquist.
    if x.size % 2 == 0:
        magnitude[1:-1] *= 2.0
    else:
        magnitude[1:] *= 2.0
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / fs)
    total_magnitude = float(magnitude.sum())
    centroid = (
        float(np.dot(frequencies, magnitude) / total_magnitude)
        if total_magnitude > np.finfo(np.float64).eps
        else 0.0
    )
    return {
        "spectrum_mean": float(magnitude.mean()),
        "spectrum_standard_deviation": float(np.std(magnitude, ddof=0)),
        "spectrum_kurtosis": _standardized_moment(magnitude, 4),
        "frequency_centroid_hz": centroid,
    }


def extract_fixed_features(
    vibration: np.ndarray,
    sampling_frequency_hz: float = DEFAULT_SAMPLING_FREQUENCY_HZ,
) -> OrderedDict[str, float]:
    """Extract exactly 48 fixed features from one two-channel snapshot.

    No fit, training-set statistics, labels, condition IDs, feature ranking,
    or feature selection are involved. The same pure transformation applies
    separately to training, validation, and test snapshots.
    """
    x = _validate_vibration(vibration)
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be positive and finite")

    result: OrderedDict[str, float] = OrderedDict()

    # 1) The eight original DWT features are always included.
    for col, channel in enumerate(CHANNELS):
        coeffs = pywt.wavedec(x[:, col], WAVELET, mode=MODE, level=LEVEL)
        for band, coefficients in zip(DWT_BANDS, coeffs, strict=True):
            result[f"{channel}_{band}_energy"] = float(np.mean(coefficients ** 2))

    # 2) All sixteen level-3 WPT RMS band features, in frequency order.
    for col, channel in enumerate(CHANNELS):
        packet = pywt.WaveletPacket(data=x[:, col], wavelet=WAVELET, mode=MODE, maxlevel=LEVEL)
        nodes = packet.get_level(LEVEL, order="freq")
        if len(nodes) != 8:
            raise RuntimeError("Expected eight level-3 WPT terminal nodes")
        for band_index, node in enumerate(nodes):
            coefficients = np.asarray(node.data, dtype=np.float64)
            result[f"{channel}_wpt_L3_{band_index:02d}_rms"] = float(
                np.sqrt(np.mean(coefficients ** 2))
            )

    # 3) Eight selected time-domain statistics for each vibration channel.
    for col, channel in enumerate(CHANNELS):
        time_values = _time_features(x[:, col])
        for name in TIME_FEATURES:
            result[f"{channel}_time_{name}"] = time_values[name]

    # 4) Four selected frequency-domain descriptors per channel.
    for col, channel in enumerate(CHANNELS):
        frequency_values = _frequency_features(x[:, col], sampling_frequency_hz)
        for name in FREQUENCY_FEATURES:
            result[f"{channel}_frequency_{name}"] = frequency_values[name]

    if tuple(result) != FEATURE_NAMES:
        raise RuntimeError("Fixed feature order/registry invariant failed")
    if not np.isfinite(np.fromiter(result.values(), dtype=np.float64)).all():
        raise RuntimeError("Extracted feature vector contains NaN or infinity")
    return result


def extract_feature_vector(
    vibration: np.ndarray,
    sampling_frequency_hz: float = DEFAULT_SAMPLING_FREQUENCY_HZ,
) -> np.ndarray:
    """Get the fixed 48-feature vector in FEATURE_NAMES order."""
    return np.fromiter(
        extract_fixed_features(vibration, sampling_frequency_hz).values(),
        dtype=np.float64,
        count=INPUT_SIZE,
    )


def fixed_feature_matrix(frame: "pd.DataFrame") -> np.ndarray:
    """Get only the 48 fixed columns from an already-prepared DataFrame.

    Columns are ordered explicitly. Missing, NaN and infinite values fail fast.
    This helper deliberately does NOT normalize or fit a scaler; the model
    pipeline must fit its scaler on TRAINING bearings only.
    """
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing fixed feature columns: {missing}")
    matrix = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Fixed feature matrix contains NaN or infinity")
    return matrix


if __name__ == "__main__":
    print(f"Feature set: {FEATURE_SET_VERSION}; count: {INPUT_SIZE}")
    print(f"Registry SHA256: {FEATURE_SET_SHA256}")
    for group, names in FEATURE_GROUPS.items():
        print(f"{group} ({len(names)}):")
        for name in names:
            print(f"  {name}")
