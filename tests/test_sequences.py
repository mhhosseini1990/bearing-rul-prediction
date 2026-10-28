"""Tests for leakage-safe temporal sequence construction."""

import numpy as np
import pandas as pd
import pytest

from bearing_rul.data.sequences import build_sequences
from bearing_rul.features.wavelet_features import FEATURE_NAMES


FEATURE_COLUMNS = list(FEATURE_NAMES)


def make_bearing(bearing_id, length, offset=0.0):
    """Create deterministic synthetic bearing measurements."""

    rows = []

    for step in range(length):
        row = {
            "bearing_id": bearing_id,
            "bearing_name": bearing_id.split("/")[-1],
            "step": step,
            "rul_minutes": length - 1 - step,
        }

        for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
            row[feature_name] = offset + step * 10 + feature_index

        rows.append(row)

    return pd.DataFrame(rows)


def test_s01_correct_sequence_shape():
    """S-01: output shape is (samples, sequence_length, features)."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    X, y, metadata = build_sequences(frame, sequence_length=20)

    assert X.shape == (6, 20, 8)
    assert y.shape == (6,)
    assert len(metadata) == 6


def test_s02_target_matches_last_snapshot():
    """S-02: target is aligned with the final sequence snapshot."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    X, y, metadata = build_sequences(frame, sequence_length=20)

    assert y.tolist() == [5, 4, 3, 2, 1, 0]

    assert metadata["target_step"].tolist() == [
        19, 20, 21, 22, 23, 24
    ]


def test_s03_no_future_information():
    """S-03: each sequence uses only steps up to its target step."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    X, y, metadata = build_sequences(frame, sequence_length=20)

    for index, row in metadata.iterrows():

        target_step = row["target_step"]

        expected_steps = list(
            range(target_step - 19, target_step + 1)
        )

        observed_steps = (
            X[index, :, 0] / 10
        ).astype(int).tolist()

        assert observed_steps == expected_steps


def test_s04_no_cross_bearing_sequences():
    """S-04: sequences must not mix different bearings."""

    first = make_bearing(
        "35Hz12kN/Bearing1_1",
        22,
        offset=0,
    )

    second = make_bearing(
        "37.5Hz11kN/Bearing2_1",
        23,
        offset=10000,
    )

    frame = pd.concat([first, second], ignore_index=True)

    X, y, metadata = build_sequences(frame, sequence_length=20)

    assert len(X) == 7

    assert metadata["bearing_id"].nunique() == 2

    for index, row in metadata.iterrows():

        values = X[index, :, 0]

        if row["bearing_id"] == "35Hz12kN/Bearing1_1":
            assert np.all(values < 10000)
        else:
            assert np.all(values >= 10000)


def test_s05_short_bearing_produces_no_sequences():
    """S-05: bearings shorter than sequence_length are skipped."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 10)

    with pytest.raises(ValueError, match="No sequences"):
        build_sequences(frame, sequence_length=20)


def test_s06_exact_sequence_length():
    """S-06: exactly 20 snapshots produce one sequence."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 20)

    X, y, metadata = build_sequences(frame, sequence_length=20)

    assert X.shape == (1, 20, 8)
    assert y.tolist() == [0]
    assert metadata["target_step"].tolist() == [19]


def test_s07_stride_two():
    """S-07: stride controls the sliding-window movement."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    X, y, metadata = build_sequences(
        frame,
        sequence_length=20,
        stride=2,
    )

    assert X.shape == (3, 20, 8)

    assert metadata["target_step"].tolist() == [19, 21, 23]


def test_s08_reject_duplicate_steps():
    """S-08: duplicate bearing/step combinations are rejected."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    frame = pd.concat(
        [frame, frame.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_sequences(frame, sequence_length=20)


def test_s09_reject_missing_steps():
    """S-09: missing intermediate snapshots are rejected."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    frame = frame[frame["step"] != 10]

    with pytest.raises(ValueError, match="Missing|consecutive"):
        build_sequences(frame, sequence_length=20)


def test_s10_input_not_modified():
    """S-10: original DataFrame must remain unchanged."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    original = frame.copy(deep=True)

    build_sequences(frame, sequence_length=20)

    pd.testing.assert_frame_equal(frame, original)


def test_s11_reject_nonfinite_features():
    """S-11: NaN and infinity must be rejected."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    frame.loc[0, FEATURE_COLUMNS[0]] = np.nan

    with pytest.raises(ValueError, match="Non-finite"):
        build_sequences(frame, sequence_length=20)


def test_s12_reject_invalid_parameters():
    """S-12: sequence length and stride must be positive."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    with pytest.raises(ValueError):
        build_sequences(frame, sequence_length=0)

    with pytest.raises(ValueError):
        build_sequences(frame, sequence_length=20, stride=0)


def test_s13_sequences_are_chronological():
    """S-13: unordered rows are sorted by snapshot step."""

    frame = make_bearing("35Hz12kN/Bearing1_1", 25)

    shuffled = frame.sample(frac=1, random_state=42)

    X, y, metadata = build_sequences(
        shuffled,
        sequence_length=20,
    )

    assert metadata["target_step"].tolist() == [
        19, 20, 21, 22, 23, 24
    ]

    assert X[0, 0, 0] == 0
    assert X[0, -1, 0] == 190