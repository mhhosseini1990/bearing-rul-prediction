"""Tests for RUL evaluation metrics."""

import numpy as np
import pytest

from bearing_rul.evaluation.metrics import (
    validate_predictions,
    calculate_mae,
    calculate_rmse,
    calculate_metrics,
    calculate_per_bearing_metrics,
)


def test_ev01_perfect_predictions():
    y = [0, 100, 200]

    assert calculate_mae(y, y) == 0.0
    assert calculate_rmse(y, y) == 0.0


def test_ev02_mae():
    actual = [100, 200, 300]
    predicted = [90, 180, 330]

    assert calculate_mae(actual, predicted) == pytest.approx(20.0)


def test_ev03_rmse():
    actual = [100, 200, 300]
    predicted = [90, 180, 330]

    expected = np.sqrt((100 + 400 + 900) / 3)

    assert calculate_rmse(actual, predicted) == pytest.approx(expected)


def test_ev04_metrics_dictionary():
    result = calculate_metrics(
        [100, 200, 300],
        [90, 180, 330],
    )

    assert result["mae"] == pytest.approx(20.0)
    assert result["rmse"] > 20.0
    assert result["n_samples"] == 3


def test_ev05_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_metrics([100, 200], [100])


def test_ev06_reject_empty_arrays():
    with pytest.raises(ValueError):
        calculate_metrics([], [])


def test_ev07_reject_nan():
    with pytest.raises(ValueError):
        calculate_metrics([100, np.nan], [100, 200])


def test_ev08_reject_infinity():
    with pytest.raises(ValueError):
        calculate_metrics([100, 200], [100, np.inf])


def test_ev09_reject_negative_actual_rul():
    with pytest.raises(ValueError):
        calculate_metrics([-10, 100], [0, 90])


def test_ev10_allow_negative_predictions():
    # Raw regression outputs can be negative.
    result = calculate_metrics([0, 100], [-10, 90])

    assert result["mae"] == pytest.approx(10.0)


def test_ev11_reject_two_dimensional_arrays():
    with pytest.raises(ValueError):
        validate_predictions(
            [[100], [200]],
            [[90], [190]],
        )


def test_ev12_per_bearing_metrics():
    actual = np.array([100, 50, 200, 100])
    predicted = np.array([90, 40, 180, 80])

    bearings = np.array([
        "Bearing1_5",
        "Bearing1_5",
        "Bearing2_5",
        "Bearing2_5",
    ])

    results = calculate_per_bearing_metrics(
        actual,
        predicted,
        bearings,
    )

    assert len(results) == 2

    assert results[0]["mae_minutes"] == pytest.approx(10.0)
    assert results[1]["mae_minutes"] == pytest.approx(20.0)


def test_ev13_reject_mismatched_bearing_ids():
    with pytest.raises(ValueError):
        calculate_per_bearing_metrics(
            [100, 200],
            [90, 190],
            ["Bearing1_5"],
        )


def test_ev14_rmse_not_less_than_mae():
    actual = [100, 200, 300]
    predicted = [80, 250, 270]

    result = calculate_metrics(actual, predicted)

    assert result["rmse"] >= result["mae"]


def test_ev15_metrics_are_in_minutes():
    actual = [1000, 2000]
    predicted = [900, 1800]

    result = calculate_metrics(actual, predicted)

    assert result["mae"] == pytest.approx(150.0)