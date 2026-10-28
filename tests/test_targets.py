"""Unit tests for XJTU-SY Remaining Useful Life (RUL) targets."""

import pandas as pd
import pytest

from bearing_rul.data.rul_labels import (
    SAMPLING_INTERVAL_MINUTES,
    add_remaining_recordings,
)


def test_t01_correct_rul_calculation():
    """T-01: RUL decreases by one minute per snapshot."""
    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"] * 5,
        "step": [0, 1, 2, 3, 4],
    })

    result = add_remaining_recordings(frame)

    assert result["rul_minutes"].tolist() == [4, 3, 2, 1, 0]
    assert result["rul_remaining_recordings"].tolist() == [
        4, 3, 2, 1, 0
    ]


def test_t02_reject_duplicate_steps():
    """T-02: duplicate snapshot indices must be rejected."""
    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"] * 3,
        "step": [0, 1, 1],
    })

    with pytest.raises(ValueError, match="unique"):
        add_remaining_recordings(frame)


def test_t02_reject_missing_steps():
    """T-02: missing snapshot indices must be rejected."""
    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"] * 3,
        "step": [0, 1, 3],
    })

    with pytest.raises(ValueError, match="non-consecutive"):
        add_remaining_recordings(frame)


def test_t03_independent_bearing_targets():
    """T-03: RUL must be calculated separately for each bearing."""
    frame = pd.DataFrame({
        "bearing_id": [
            "Bearing1_1",
            "Bearing2_1",
            "Bearing1_1",
            "Bearing2_1",
            "Bearing1_1",
        ],
        "step": [0, 0, 1, 1, 2],
    })

    result = add_remaining_recordings(frame)

    bearing_1 = result.loc[
        result["bearing_id"] == "Bearing1_1",
        "rul_minutes",
    ].tolist()

    bearing_2 = result.loc[
        result["bearing_id"] == "Bearing2_1",
        "rul_minutes",
    ].tolist()

    assert bearing_1 == [2, 1, 0]
    assert bearing_2 == [1, 0]


def test_t04_rul_unit_is_minutes():
    """T-04: recording intervals are converted into minutes."""
    assert SAMPLING_INTERVAL_MINUTES == 1

    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"] * 3,
        "step": [0, 1, 2],
    })

    result = add_remaining_recordings(frame)

    assert result["rul_minutes"].tolist() == [2, 1, 0]


def test_t05_final_snapshot_has_zero_rul():
    """T-05: the final snapshot has zero remaining minutes."""
    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"] * 4,
        "step": [0, 1, 2, 3],
    })

    result = add_remaining_recordings(frame)

    assert result["rul_minutes"].iloc[-1] == 0


def test_t06_original_dataframe_unchanged():
    """T-06: target generation must not mutate the input."""
    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"] * 3,
        "step": [0, 1, 2],
    })

    original = frame.copy(deep=True)

    add_remaining_recordings(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_t07_reject_missing_columns():
    """T-07: required columns must exist."""
    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        add_remaining_recordings(frame)


def test_t08_reject_negative_steps():
    """T-08: negative snapshot indices must be rejected."""
    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"] * 3,
        "step": [-1, 0, 1],
    })

    with pytest.raises(ValueError, match="negative"):
        add_remaining_recordings(frame)


def test_t09_preserve_input_row_order():
    """T-09: labels are correct even when input rows are shuffled."""
    frame = pd.DataFrame({
        "bearing_id": ["Bearing1_1"] * 3,
        "step": [2, 0, 1],
    })

    result = add_remaining_recordings(frame)

    assert result["step"].tolist() == [2, 0, 1]
    assert result["rul_minutes"].tolist() == [0, 2, 1]


def test_t10_reject_empty_dataframe():
    """T-10: an empty DataFrame must be rejected."""
    frame = pd.DataFrame(columns=["bearing_id", "step"])

    with pytest.raises(ValueError, match="empty"):
        add_remaining_recordings(frame)