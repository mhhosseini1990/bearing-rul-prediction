"""Generate RUL targets for complete XJTU-SY run-to-failure histories.

Each consecutive vibration snapshot is recorded one minute apart.

RUL is measured in minutes remaining until the final recorded
snapshot of the same bearing.

The final snapshot is treated as the experimental end-of-life
endpoint according to the dataset protocol.

The known endpoint is used only to generate supervised labels.
It must never be used as an inference-time input feature.
"""

import pandas as pd


SAMPLING_INTERVAL_MINUTES = 1


def add_remaining_recordings(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add remaining recording counts and RUL in minutes.

    Required columns:
        bearing_id: Unique identifier for each bearing.
        step: Zero-based chronological snapshot index.

    Added columns:
        rul_remaining_recordings: Number of future snapshots.
        rul_minutes: Remaining time in minutes.

    The function preserves input row order and does not modify
    the original DataFrame.
    """
    required = {"bearing_id", "step"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Input DataFrame is empty")

    if frame["bearing_id"].isna().any():
        raise ValueError("Missing bearing identifiers")

    if frame["step"].isna().any():
        raise ValueError("Missing snapshot indices")

    if not pd.api.types.is_integer_dtype(frame["step"]):
        raise ValueError("Snapshot indices must be integers")

    if (frame["step"] < 0).any():
        raise ValueError("Snapshot indices cannot be negative")

    if frame.duplicated(["bearing_id", "step"]).any():
        raise ValueError(
            "Each bearing/step pair must be unique"
        )

    result = frame.copy()

    for bearing_id, group in result.groupby("bearing_id"):
        steps = sorted(group["step"].tolist())
        expected = list(range(len(steps)))

        if steps != expected:
            raise ValueError(
                f"Missing or non-consecutive steps "
                f"for bearing {bearing_id}"
            )

    last_step = (
        result.groupby("bearing_id")["step"]
        .transform("max")
    )

    result["rul_remaining_recordings"] = (
        last_step - result["step"]
    ).astype(int)

    result["rul_minutes"] = (
        result["rul_remaining_recordings"]
        * SAMPLING_INTERVAL_MINUTES
    ).astype(int)

    return result