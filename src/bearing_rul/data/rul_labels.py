"""Create simple remaining-recording targets for completed run-to-failure tests."""
import pandas as pd


def add_remaining_recordings(frame: pd.DataFrame) -> pd.DataFrame:
    """Add RUL as number of *future recordings* until the final file.

    Assumes the final file for each bearing is its known end-of-life measurement.
    This information is for supervised training/evaluation only; it must never
    be an input feature at inference time.
    """
    required = {"bearing_id", "step"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.duplicated(["bearing_id", "step"]).any():
        raise ValueError("Each bearing/step pair must be unique")
    result = frame.copy()
    last_step = result.groupby("bearing_id")["step"].transform("max")
    result["rul_remaining_recordings"] = (last_step - result["step"]).astype(int)
    return result
