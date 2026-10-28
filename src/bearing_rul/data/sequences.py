"""Leakage-safe temporal sequence construction for LSTM and GRU.

Each sequence contains only past and current snapshots
from a single bearing.

The target is the RUL of the final snapshot in the window.
"""

import numpy as np
import pandas as pd

from bearing_rul.features.wavelet_features import FEATURE_NAMES


def build_sequences(
    frame: pd.DataFrame,
    sequence_length: int = 20,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build sliding-window sequences from bearing feature data.

    Args:
        frame:
            DataFrame containing bearing_id, step, rul_minutes
            and eight wavelet feature columns.

        sequence_length:
            Number of consecutive snapshots per sequence.

        stride:
            Number of snapshots between consecutive window starts.

    Returns:
        X:
            Array of shape (n_sequences, sequence_length, 8).

        y:
            RUL targets of shape (n_sequences,).

        metadata:
            DataFrame with bearing_id, start_step and target_step.
    """

    if not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")

    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")

    feature_columns = list(FEATURE_NAMES)

    required_columns = {
        "bearing_id",
        "step",
        "rul_minutes",
        *feature_columns,
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if frame.empty:
        raise ValueError("Input DataFrame is empty")

    if frame[["bearing_id", "step", "rul_minutes"]].isna().any().any():
        raise ValueError("Missing bearing ID, step or RUL target")

    if frame.duplicated(["bearing_id", "step"]).any():
        raise ValueError("Duplicate bearing/step combinations")

    feature_values = frame[
        feature_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite feature values")

    if not np.isfinite(
        frame["rul_minutes"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Non-finite RUL targets")

    X_sequences = []
    y_targets = []
    metadata_rows = []

    for bearing_id, bearing_frame in frame.groupby(
        "bearing_id",
        sort=True,
    ):

        bearing_frame = bearing_frame.sort_values(
            "step"
        ).reset_index(drop=True)

        steps = bearing_frame["step"].to_numpy()

        expected_steps = np.arange(
            steps[0],
            steps[0] + len(steps),
        )

        if not np.array_equal(steps, expected_steps):
            raise ValueError(
                f"Missing or non-consecutive steps for {bearing_id}"
            )

        if len(bearing_frame) < sequence_length:
            continue

        features = bearing_frame[
            feature_columns
        ].to_numpy(dtype=np.float64)

        targets = bearing_frame[
            "rul_minutes"
        ].to_numpy(dtype=np.float64)

        for start in range(
            0,
            len(bearing_frame) - sequence_length + 1,
            stride,
        ):

            end = start + sequence_length

            sequence = features[start:end]

            target = targets[end - 1]

            X_sequences.append(sequence)
            y_targets.append(target)

            metadata_rows.append({
                "bearing_id": bearing_id,
                "start_step": int(steps[start]),
                "target_step": int(steps[end - 1]),
            })

    if not X_sequences:
        raise ValueError(
            "No sequences could be constructed"
        )

    X = np.stack(X_sequences).astype(np.float32)

    y = np.asarray(y_targets, dtype=np.float32)

    metadata = pd.DataFrame(metadata_rows)

    return X, y, metadata