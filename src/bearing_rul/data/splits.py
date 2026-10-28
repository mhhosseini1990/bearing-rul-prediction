"""Bearing-level split to avoid same-machine leakage across train and test."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_bearing(
    frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame["bearing_id"].nunique() < 2:
        raise ValueError("Need at least two different bearings for a group split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["bearing_id"]))
    train = frame.iloc[train_idx].copy()
    test = frame.iloc[test_idx].copy()
    assert set(train["bearing_id"]).isdisjoint(test["bearing_id"])
    return train, test
