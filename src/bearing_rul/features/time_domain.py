"""Basic features from one vibration file at a time (no future look-ahead)."""
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    f"{axis}_{name}"
    for axis in ("horizontal", "vertical")
    for name in ("mean", "std", "rms", "peak", "ptp", "crest_factor", "skewness", "kurtosis")
]


def _signal_features(signal: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Vibration signal must contain finite numeric samples")

    mean = float(values.mean())
    centered = values - mean
    std = float(np.sqrt(np.mean(centered ** 2)))
    rms = float(np.sqrt(np.mean(values ** 2)))
    peak = float(np.max(np.abs(values)))
    skew = float(np.mean((centered / std) ** 3)) if std > 0 else 0.0
    kurt = float(np.mean((centered / std) ** 4)) if std > 0 else 0.0

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_ptp": float(np.ptp(values)),
        f"{prefix}_crest_factor": (peak / rms) if rms > 0 else 0.0,
        f"{prefix}_skewness": skew,
        f"{prefix}_kurtosis": kurt,
    }


def extract_features(horizontal: np.ndarray, vertical: np.ndarray) -> dict[str, float]:
    """Return per-channel statistical features for a single recording."""
    return {
        **_signal_features(horizontal, "horizontal"),
        **_signal_features(vertical, "vertical"),
    }


def extract_features_from_csv(path: str | Path) -> dict[str, float]:
    """Read two acceleration columns (with or without a header).

    Header text is converted to NaN and removed. Check the channel order and
    sampling method for the specific source files before scientific reporting.
    """
    raw = pd.read_csv(path, header=None, usecols=[0, 1])
    numeric = raw.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if numeric.empty:
        raise ValueError(f"No two-column numeric vibration data found: {path}")
    return extract_features(numeric.iloc[:, 0].to_numpy(), numeric.iloc[:, 1].to_numpy())


def feature_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Combine manifest and per-recording feature values."""
    features = [extract_features_from_csv(path) for path in manifest["path"]]
    return pd.concat([manifest.reset_index(drop=True), pd.DataFrame(features)], axis=1)
