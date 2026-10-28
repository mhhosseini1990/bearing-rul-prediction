"""Offline historical replay, ordered by each test bearing's measurement step."""
import pandas as pd
from bearing_rul.features.time_domain import FEATURE_NAMES
from bearing_rul.models.random_forest import TARGET


def historical_replay(model, test_frame: pd.DataFrame) -> pd.DataFrame:
    """Make point-in-time predictions from already-extracted current-recording features.

    The model is trained separately. The actual target is copied into the output
    only for evaluation and is never included among input columns.
    """
    ordered = test_frame.sort_values(["bearing_id", "step"]).copy()
    output = ordered[["bearing_id", "step", TARGET]].copy()
    output["predicted_rul"] = model.predict(ordered[FEATURE_NAMES])
    output["predicted_rul"] = output["predicted_rul"].clip(lower=0)
    return output.reset_index(drop=True)
