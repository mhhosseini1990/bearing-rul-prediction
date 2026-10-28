"""Tests for XJTU-SY wavelet feature matrix generation."""

import numpy as np
import pandas as pd
import pytest

from bearing_rul.features.build_feature_matrix import build_feature_matrix
from bearing_rul.features.wavelet_features import FEATURE_NAMES


COLUMNS = [
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
]


def write_snapshot(path, value):
    """Create a small, valid, two-channel vibration CSV."""
    rng = np.random.default_rng(42)
    signal = rng.normal(loc=value, size=(128, 2))

    pd.DataFrame(signal, columns=COLUMNS).to_csv(path, index=False)


def test_m01_one_row_per_snapshot(tmp_path):
    """M-01: each CSV produces exactly one feature row."""
    bearing = tmp_path / "35Hz12kN" / "Bearing1_1"
    bearing.mkdir(parents=True)

    for index in range(1, 4):
        write_snapshot(bearing / f"{index}.csv", index)

    result = build_feature_matrix(tmp_path)

    assert len(result) == 3
    assert result["step"].tolist() == [0, 1, 2]
    assert result["rul_minutes"].tolist() == [2, 1, 0]


def test_m02_feature_columns_are_present(tmp_path):
    """M-02: all eight wavelet features must be present."""
    bearing = tmp_path / "35Hz12kN" / "Bearing1_1"
    bearing.mkdir(parents=True)

    write_snapshot(bearing / "1.csv", 1)

    result = build_feature_matrix(tmp_path)

    assert all(name in result.columns for name in FEATURE_NAMES)
    assert result.loc[:, FEATURE_NAMES].shape == (1, 8)


def test_m03_multiple_bearings_remain_separate(tmp_path):
    """M-03: bearing identities and RUL histories stay separate."""
    first = tmp_path / "35Hz12kN" / "Bearing1_1"
    second = tmp_path / "37.5Hz11kN" / "Bearing2_1"

    first.mkdir(parents=True)
    second.mkdir(parents=True)

    for index in range(1, 4):
        write_snapshot(first / f"{index}.csv", index)

    for index in range(1, 3):
        write_snapshot(second / f"{index}.csv", index)

    result = build_feature_matrix(tmp_path)

    assert len(result) == 5
    assert result["bearing_id"].nunique() == 2

    first_rows = result[
        result["bearing_id"] == "35Hz12kN/Bearing1_1"
    ]
    second_rows = result[
        result["bearing_id"] == "37.5Hz11kN/Bearing2_1"
    ]

    assert first_rows["rul_minutes"].tolist() == [2, 1, 0]
    assert second_rows["rul_minutes"].tolist() == [1, 0]


def test_m04_features_are_finite(tmp_path):
    """M-04: generated features contain no NaN or infinity."""
    bearing = tmp_path / "40Hz10kN" / "Bearing3_1"
    bearing.mkdir(parents=True)

    write_snapshot(bearing / "1.csv", 1)

    result = build_feature_matrix(tmp_path)

    values = result.loc[:, FEATURE_NAMES].to_numpy()

    assert np.isfinite(values).all()


def test_m05_chronological_order(tmp_path):
    """M-05: snapshots are ordered numerically."""
    bearing = tmp_path / "35Hz12kN" / "Bearing1_1"
    bearing.mkdir(parents=True)

    for index in [10, 2, 1, 3, 4, 5, 6, 7, 8, 9]:
        write_snapshot(bearing / f"{index}.csv", index)

    result = build_feature_matrix(tmp_path)

    assert result["filename"].tolist() == [
        f"{index}.csv" for index in range(1, 11)
    ]
    assert result["step"].tolist() == list(range(10))


def test_m06_missing_snapshot_rejected(tmp_path):
    """M-06: missing snapshot numbers must not be silently ignored."""
    bearing = tmp_path / "35Hz12kN" / "Bearing1_1"
    bearing.mkdir(parents=True)

    write_snapshot(bearing / "1.csv", 1)
    write_snapshot(bearing / "3.csv", 3)

    with pytest.raises(ValueError):
        build_feature_matrix(tmp_path)


def test_m07_invalid_csv_rejected(tmp_path):
    """M-07: invalid vibration CSVs must raise an error."""
    bearing = tmp_path / "35Hz12kN" / "Bearing1_1"
    bearing.mkdir(parents=True)

    (bearing / "1.csv").write_text(
        "wrong,columns\n1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        build_feature_matrix(tmp_path)