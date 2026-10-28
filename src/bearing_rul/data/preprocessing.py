"""Bearing-level splitting and leakage-safe feature scaling.

Dataset: XJTU-SY
Models: LSTM and GRU

The split is read from the project's YAML configuration.
StandardScaler is fitted only on training bearing snapshots.

RUL targets and metadata are never used to fit the scaler.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import StandardScaler

from bearing_rul.features.wavelet_features import FEATURE_NAMES


SPLIT_NAMES = ("train", "validation", "test")


def load_split_config(config_path: str | Path) -> dict:
    """Load and validate the bearing-level split configuration."""

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Split configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Split configuration must be a YAML mapping")

    if config.get("split_unit") != "bearing":
        raise ValueError("Split unit must be 'bearing'")

    if config.get("rul_unit") != "minutes":
        raise ValueError("RUL unit must be 'minutes'")

    all_bearings = []

    for split_name in SPLIT_NAMES:
        bearings = config.get(split_name)

        if not isinstance(bearings, list) or not bearings:
            raise ValueError(
                f"Missing or empty split: {split_name}"
            )

        if not all(
            isinstance(bearing, str) and bearing.strip()
            for bearing in bearings
        ):
            raise ValueError(
                f"Invalid bearing identifiers in {split_name}"
            )

        all_bearings.extend(bearings)

    if len(all_bearings) != len(set(all_bearings)):
        raise ValueError(
            "A bearing appears in more than one split"
        )

    return config


def split_feature_matrix(
    frame: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Split feature rows by bearing identity.

    YAML bearing names are matched against the bearing_name column.
    Every bearing must belong to exactly one partition.

    Returns:
        Dictionary containing train, validation and test DataFrames.
    """

    required = {
        "bearing_id",
        "bearing_name",
        "operating_condition",
        "step",
        "rul_minutes",
        *FEATURE_NAMES,
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing feature matrix columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Feature matrix is empty")

    if frame["bearing_id"].isna().any():
        raise ValueError("Missing bearing identifiers")

    if frame["bearing_name"].isna().any():
        raise ValueError("Missing bearing names")

    if frame.duplicated(["bearing_id", "step"]).any():
        raise ValueError("Duplicate bearing/step pairs")

    # Ensure a short bearing name does not refer to different bearings.
    identities = frame.groupby("bearing_name")["bearing_id"].nunique()

    if (identities > 1).any():
        raise ValueError(
            "A bearing_name maps to multiple bearing IDs"
        )

    assigned_names = set()

    for split_name in SPLIT_NAMES:
        assigned_names.update(config[split_name])

    available_names = set(frame["bearing_name"].unique())

    missing_bearings = assigned_names - available_names
    unexpected_bearings = available_names - assigned_names

    if missing_bearings:
        raise ValueError(
            f"Configured bearings missing from feature matrix: "
            f"{sorted(missing_bearings)}"
        )

    if unexpected_bearings:
        raise ValueError(
            f"Unassigned bearings in feature matrix: "
            f"{sorted(unexpected_bearings)}"
        )

    partitions = {}

    for split_name in SPLIT_NAMES:
        selected = frame[
            frame["bearing_name"].isin(config[split_name])
        ].copy()

        if selected.empty:
            raise ValueError(
                f"Split {split_name} contains no rows"
            )

        partitions[split_name] = selected.reset_index(drop=True)

    train_ids = set(partitions["train"]["bearing_id"])
    validation_ids = set(partitions["validation"]["bearing_id"])
    test_ids = set(partitions["test"]["bearing_id"])

    if train_ids & validation_ids:
        raise ValueError("Train/validation bearing overlap")

    if train_ids & test_ids:
        raise ValueError("Train/test bearing overlap")

    if validation_ids & test_ids:
        raise ValueError("Validation/test bearing overlap")

    return partitions


def scale_feature_partitions(
    partitions: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], StandardScaler]:
    """Fit StandardScaler on training features only.

    Only the eight Wavelet feature columns are transformed.

    Metadata, bearing IDs, step indices and RUL targets
    are preserved without modification.
    """

    for split_name in SPLIT_NAMES:
        if split_name not in partitions:
            raise ValueError(
                f"Missing partition: {split_name}"
            )

        if partitions[split_name].empty:
            raise ValueError(
                f"Empty partition: {split_name}"
            )

    feature_columns = list(FEATURE_NAMES)

    for split_name in SPLIT_NAMES:
        missing = set(feature_columns) - set(
            partitions[split_name].columns
        )

        if missing:
            raise ValueError(
                f"Missing feature columns in {split_name}: "
                f"{sorted(missing)}"
            )

        values = partitions[split_name][
            feature_columns
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite feature values in {split_name}"
            )

    scaler = StandardScaler()

    train_features = partitions["train"][
        feature_columns
    ].to_numpy(dtype=np.float64)

    scaler.fit(train_features)

    scaled_partitions = {}

    for split_name in SPLIT_NAMES:
        result = partitions[split_name].copy()

        values = result[
            feature_columns
        ].to_numpy(dtype=np.float64)

        result.loc[:, feature_columns] = scaler.transform(values)

        scaled_partitions[split_name] = result

    return scaled_partitions, scaler


def prepare_dataset(
    feature_matrix_path: str | Path,
    split_config_path: str | Path,
) -> tuple[dict[str, pd.DataFrame], StandardScaler]:
    """Load, split and scale the XJTU-SY feature matrix."""

    feature_matrix_path = Path(feature_matrix_path)

    if not feature_matrix_path.is_file():
        raise FileNotFoundError(
            f"Feature matrix not found: {feature_matrix_path}"
        )

    frame = pd.read_csv(feature_matrix_path)

    config = load_split_config(split_config_path)

    partitions = split_feature_matrix(frame, config)

    scaled_partitions, scaler = scale_feature_partitions(
        partitions
    )

    return scaled_partitions, scaler