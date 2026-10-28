"""First baseline: train on individual-recording time-domain features."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from bearing_rul.features.time_domain import FEATURE_NAMES

TARGET = "rul_remaining_recordings"


def fit_random_forest(train: pd.DataFrame, random_state: int = 42) -> RandomForestRegressor:
    if train.empty:
        raise ValueError("Training data is empty")
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_NAMES], train[TARGET])
    return model
