"""Tests for bearing-level splitting and leakage-safe scaling."""

import numpy as np
import pandas as pd
import pytest

from bearing_rul.data.preprocessing import (
    load_split_config,
    split_feature_matrix,
    scale_feature_partitions,
)

from bearing_rul.features.wavelet_features import FEATURE_NAMES


@pytest.fixture
def split_config():
    """Small bearing-level split for isolated unit tests."""
    return {
        "split_unit": "bearing",
        "rul_unit": "minutes",
        "train": ["Bearing1_1", "Bearing2_1"],
        "validation": ["Bearing1_2"],
        "test": ["Bearing2_2"],
    }


@pytest.fixture
def feature_matrix():
    """Create synthetic features with clearly separated distributions."""

    rows = []

    specifications = [
        ("35Hz12kN", "Bearing1_1", 1.0),
        ("37.5Hz11kN", "Bearing2_1", 2.0),
        ("35Hz12kN", "Bearing1_2", 100.0),
        ("37.5Hz11kN", "Bearing2_2", 200.0),
    ]

    for condition, bearing, base_value in specifications:
        for step in range(3):

            record = {
                "bearing_id": f"{condition}/{bearing}",
                "bearing_name": bearing,
                "operating_condition": condition,
                "step": step,
                "rul_minutes": 2 - step,
            }

            for feature_index, name in enumerate(FEATURE_NAMES):
                record[name] = (
                    base_value + step + feature_index
                )

            rows.append(record)

    return pd.DataFrame(rows)


def test_p01_load_valid_yaml(tmp_path, split_config):
    """P-01: valid split YAML is loaded correctly."""
    import yaml

    path = tmp_path / "split.yaml"

    path.write_text(
        yaml.safe_dump(split_config),
        encoding="utf-8",
    )

    result = load_split_config(path)

    assert result["train"] == split_config["train"]
    assert result["rul_unit"] == "minutes"


def test_p02_reject_bearing_overlap(tmp_path, split_config):
    """P-02: a bearing cannot appear in multiple partitions."""
    import yaml

    invalid = split_config.copy()
    invalid["validation"] = ["Bearing1_1"]

    path = tmp_path / "invalid.yaml"

    path.write_text(
        yaml.safe_dump(invalid),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="more than one split"):
        load_split_config(path)


def test_p03_correct_bearing_split(feature_matrix, split_config):
    """P-03: rows are assigned to the correct bearing partitions."""
    partitions = split_feature_matrix(
        feature_matrix,
        split_config,
    )

    assert partitions["train"]["bearing_name"].nunique() == 2
    assert partitions["validation"]["bearing_name"].nunique() == 1
    assert partitions["test"]["bearing_name"].nunique() == 1

    assert len(partitions["train"]) == 6
    assert len(partitions["validation"]) == 3
    assert len(partitions["test"]) == 3


def test_p04_no_bearing_overlap(feature_matrix, split_config):
    """P-04: train, validation and test bearing IDs are disjoint."""
    partitions = split_feature_matrix(
        feature_matrix,
        split_config,
    )

    train = set(partitions["train"]["bearing_id"])
    validation = set(partitions["validation"]["bearing_id"])
    test = set(partitions["test"]["bearing_id"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_p05_reject_unassigned_bearings(feature_matrix, split_config):
    """P-05: every bearing must be assigned to a split."""
    invalid = split_config.copy()
    invalid["test"] = ["Bearing9_9"]

    with pytest.raises(ValueError):
        split_feature_matrix(feature_matrix, invalid)


def test_p06_scaler_fitted_on_train_only(feature_matrix, split_config):
    """P-06: scaler statistics must use training rows only."""
    partitions = split_feature_matrix(
        feature_matrix,
        split_config,
    )

    scaled, scaler = scale_feature_partitions(partitions)

    expected_mean = partitions["train"][
        list(FEATURE_NAMES)
    ].mean().to_numpy()

    np.testing.assert_allclose(
        scaler.mean_,
        expected_mean,
    )

    train_values = scaled["train"][
        list(FEATURE_NAMES)
    ].to_numpy()

    np.testing.assert_allclose(
        train_values.mean(axis=0),
        np.zeros(8),
        atol=1e-12,
    )


def test_p07_validation_test_do_not_change_scaler(
    feature_matrix,
    split_config,
):
    """P-07: changing held-out features cannot change scaler fit."""
    partitions = split_feature_matrix(
        feature_matrix,
        split_config,
    )

    _, first_scaler = scale_feature_partitions(partitions)

    modified = {
        name: frame.copy()
        for name, frame in partitions.items()
    }

    for split_name in ("validation", "test"):
        modified[split_name].loc[
            :, list(FEATURE_NAMES)
        ] *= 1000

    _, second_scaler = scale_feature_partitions(modified)

    np.testing.assert_allclose(
        first_scaler.mean_,
        second_scaler.mean_,
    )

    np.testing.assert_allclose(
        first_scaler.scale_,
        second_scaler.scale_,
    )


def test_p08_targets_and_metadata_unchanged(
    feature_matrix,
    split_config,
):
    """P-08: scaling must not modify RUL or bearing metadata."""
    partitions = split_feature_matrix(
        feature_matrix,
        split_config,
    )

    scaled, _ = scale_feature_partitions(partitions)

    protected_columns = [
        "bearing_id",
        "bearing_name",
        "operating_condition",
        "step",
        "rul_minutes",
    ]

    for split_name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(
            partitions[split_name][protected_columns],
            scaled[split_name][protected_columns],
        )


def test_p09_original_data_unchanged(feature_matrix, split_config):
    """P-09: preprocessing must not mutate the original data."""
    original = feature_matrix.copy(deep=True)

    partitions = split_feature_matrix(
        feature_matrix,
        split_config,
    )

    scale_feature_partitions(partitions)

    pd.testing.assert_frame_equal(
        feature_matrix,
        original,
    )


def test_p10_reject_nonfinite_features(feature_matrix, split_config):
    """P-10: NaN or infinity must not enter StandardScaler."""
    partitions = split_feature_matrix(
        feature_matrix,
        split_config,
    )

    partitions["train"].loc[
        0, FEATURE_NAMES[0]
    ] = np.nan

    with pytest.raises(ValueError, match="Non-finite"):
        scale_feature_partitions(partitions)